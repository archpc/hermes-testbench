"""GnuCOBOL verification: compile, run, and check against a Decimal oracle.

This module is the Phase 1B verification layer for the existing ``cobol``
adapter. It does not add a second runner: it emits the ``TB-PASS`` /
``TB-FAIL`` marker lines that ``parsers.parse_cobol`` already attributes, so
every failure reaches ``SuiteResult.failures`` and the normal reporter with an
actionable excerpt.

Two independent layers:

* the *reference oracle* -- a stdlib ``decimal`` model of COBOL fixed-width
  signed-decimal arithmetic: exact sums, ``ROUNDED`` (half away from zero),
  implicit-decimal truncation, and PICTURE overflow. It computes the expected
  output without asking the compiler what it thinks;
* *verification* -- compile each case with ``cobc``, run the binary, and compare
  exact stdout, the exit status, or an expected reference rejection. The first
  difference is reported, never a vague "output differed".

A program that compiles cleanly and exits 0 still fails verification when its
computed result disagrees with the oracle. That is the point: exit status alone
proves nothing about arithmetic.

Usage from a suite::

    python -m hermes_testbench.adapters.cobol_check --spec cases.json

where ``cases.json`` is ``{"cases": [{"name": ..., "source": ...,
"expected_stdout": ..., "expected_exit": 0}, ...]}``. The process exits 1 when
any case fails and prints the aggregate ``N passed, 0 skipped, M failed`` line.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "CobolCase",
    "CompileResult",
    "Diagnostic",
    "Field",
    "FixedRecordLayout",
    "LedgerSummary",
    "MalformedRecordError",
    "PictureOverflowError",
    "ReferenceLedger",
    "RunResult",
    "VerificationReport",
    "cobol_round",
    "cobol_truncate",
    "compile_source",
    "enforce_picture",
    "fits_picture",
    "find_cobc",
    "format_edited",
    "format_ledger_summary",
    "main",
    "parse_diagnostics",
    "parse_fixed_records",
    "run_program",
    "verify_case",
    "verify_cases",
]

_MAX_DETAIL_LINES = 12
_MAX_DETAIL_CHARS = 400


# --------------------------------------------------------------------------
# Compiler diagnostics
# --------------------------------------------------------------------------

_COBC_DIAG_RE = re.compile(
    r"^(?P<file>\S+\.(?:cob|cbl|cpy)):(?P<line>\d+):\s*"
    r"(?P<severity>error|warning|note):\s*(?P<message>.+)$"
)


@dataclass(frozen=True)
class Diagnostic:
    """One ``cobc`` diagnostic, rendered exactly as the compiler phrased it."""

    file: str
    line: Optional[int]
    severity: str
    message: str

    def render(self) -> str:
        where = f"{self.file}:{self.line}" if self.line is not None else self.file
        return f"{where}: {self.severity}: {self.message}"


def parse_diagnostics(text: str) -> Tuple[Diagnostic, ...]:
    """Extract ``file:line: error|warning|note: message`` lines from cobc output."""
    found: List[Diagnostic] = []
    for raw in (text or "").splitlines():
        match = _COBC_DIAG_RE.match(raw.strip())
        if not match:
            continue
        found.append(
            Diagnostic(
                file=match.group("file"),
                line=int(match.group("line")),
                severity=match.group("severity"),
                message=match.group("message").strip(),
            )
        )
    return tuple(found)


# --------------------------------------------------------------------------
# compile / run
# --------------------------------------------------------------------------


def find_cobc() -> Optional[str]:
    """Locate the GnuCOBOL compiler, including Homebrew's prefix when needed."""
    found = shutil.which("cobc")
    if found:
        return found
    for extra in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
        candidate = Path(extra) / "cobc"
        if candidate.is_file():
            return str(candidate)
    return None


@dataclass(frozen=True)
class CompileResult:
    ok: bool
    binary: Optional[Path]
    diagnostics: Tuple[Diagnostic, ...]
    command: Tuple[str, ...]
    output: str

    @property
    def errors(self) -> Tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "error")


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        return (self.stdout or "") + (self.stderr or "")


def compile_source(
    source: Union[str, Path],
    workdir: Union[str, Path],
    *,
    cobc: Optional[str] = None,
    free_form: bool = True,
    optimize: bool = True,
    timeout_s: float = 120.0,
) -> CompileResult:
    """Compile one COBOL source into ``workdir`` and capture its diagnostics."""
    compiler = cobc or find_cobc()
    if not compiler:
        raise FileNotFoundError("cobc (GnuCOBOL) is not installed")
    src = Path(source)
    out = Path(workdir)
    out.mkdir(parents=True, exist_ok=True)
    binary = out / (src.stem + ".bin")
    command = [compiler, "-x"]
    if free_form:
        command.append("-free")
    if optimize:
        command.append("-O2")
    command += ["-o", str(binary), str(src)]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0 and binary.is_file()
    return CompileResult(
        ok=ok,
        binary=binary if ok else None,
        diagnostics=parse_diagnostics(combined),
        command=tuple(command),
        output=combined,
    )


def run_program(
    binary: Union[str, Path],
    *,
    argv: Sequence[str] = (),
    stdin_text: Optional[str] = None,
    timeout_s: float = 120.0,
) -> RunResult:
    """Run a compiled COBOL binary once with optional argv and stdin."""
    proc = subprocess.run(
        [str(binary), *argv],
        input=stdin_text or "",
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return RunResult(exit_code=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")


# --------------------------------------------------------------------------
# Reference oracle: COBOL fixed-width signed-decimal semantics on Decimal
# --------------------------------------------------------------------------


class MalformedRecordError(ValueError):
    """A record does not match the fixed-width layout it is supposed to follow."""


class PictureOverflowError(ArithmeticError):
    """A value does not fit the declared PICTURE (the silent-wrap COBOL hazard)."""


def _quantum(places: int) -> Decimal:
    return Decimal(1).scaleb(-places)


def cobol_round(value: Decimal, places: int) -> Decimal:
    """COBOL ``ROUNDED``: half away from zero at ``places`` decimals."""
    return Decimal(value).quantize(_quantum(places), rounding=ROUND_HALF_UP)


def cobol_truncate(value: Decimal, places: int) -> Decimal:
    """COBOL arithmetic without ``ROUNDED``: truncation toward zero."""
    return Decimal(value).quantize(_quantum(places), rounding=ROUND_DOWN)


def enforce_picture(
    value: Decimal, *, int_digits: int, scale: int, signed: bool = True
) -> Decimal:
    """Validate that ``value`` fits ``PIC [S]9(int_digits)V9(scale)``."""
    limit = Decimal(10) ** int_digits - _quantum(scale)
    magnitude = abs(Decimal(value))
    if magnitude > limit or (not signed and value < 0):
        picture = ("S" if signed else "") + f"9({int_digits})"
        if scale:
            picture += f"V9({scale})"
        raise PictureOverflowError(f"{value} does not fit PIC {picture} (limit {limit})")
    return value


def fits_picture(value: Decimal, *, int_digits: int, scale: int, signed: bool = True) -> bool:
    try:
        enforce_picture(value, int_digits=int_digits, scale=scale, signed=signed)
    except PictureOverflowError:
        return False
    return True


def format_edited(
    value: Decimal, *, int_digits: int, scale: int, signed: bool = True
) -> str:
    """Render a value the way ``PIC S9(n)V9(m) SIGN IS LEADING SEPARATE`` does.

    ``Decimal('7.30')`` with ``int_digits=7, scale=2`` becomes ``"+0000007.30"``,
    byte-for-byte what COBOL's ``DISPLAY`` emits, so comparisons are exact.
    """
    quantized = enforce_picture(
        cobol_truncate(value, scale), int_digits=int_digits, scale=scale, signed=signed
    )
    digits = abs(quantized.scaleb(scale).to_integral_value())
    text = str(int(digits)).rjust(int_digits + scale, "0")
    sign = ""
    if signed:
        sign = "-" if quantized < 0 else "+"
    if scale:
        return f"{sign}{text[:-scale]}.{text[-scale:]}"
    return f"{sign}{text}"


@dataclass(frozen=True)
class Field:
    """One fixed-width field of a record layout."""

    name: str
    width: int
    kind: str = "text"  # text | int | decimal | sign
    scale: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"field {self.name!r} must have a positive width")
        if self.kind not in ("text", "int", "decimal", "sign"):
            raise ValueError(f"field {self.name!r} has unknown kind {self.kind!r}")
        if self.kind in ("text", "sign") and self.scale:
            raise ValueError(f"field {self.name!r} of kind {self.kind!r} cannot scale")
        if self.scale < 0 or self.scale > self.width:
            raise ValueError(f"field {self.name!r} scale {self.scale} exceeds width {self.width}")
        if self.kind == "sign" and self.width != 1:
            raise ValueError(f"sign field {self.name!r} must be one character wide")


@dataclass(frozen=True)
class FixedRecordLayout:
    """A record layout whose fields are contiguous and fixed-width by definition."""

    fields: Tuple[Field, ...]

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("layout needs at least one field")
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names in layout: {names}")

    @property
    def width(self) -> int:
        return sum(f.width for f in self.fields)

    def parse_record(self, record: str, *, index: int = 0) -> Dict[str, Union[str, Decimal]]:
        if len(record) != self.width:
            raise MalformedRecordError(
                f"record {index + 1} has {len(record)} characters; "
                f"layout expects {self.width}: {record!r}"
            )
        values: Dict[str, Union[str, Decimal]] = {}
        offset = 0
        for spec in self.fields:
            chunk = record[offset : offset + spec.width]
            offset += spec.width
            where = f"record {index + 1} field {spec.name}"
            if spec.kind == "text":
                values[spec.name] = chunk
                continue
            if spec.kind == "sign":
                if chunk not in ("+", "-"):
                    raise MalformedRecordError(
                        f"{where}: sign must be '+' or '-' (got {chunk!r}): {record!r}"
                    )
                values[spec.name] = chunk
                continue
            if not re.fullmatch(r"[0-9]+", chunk):
                raise MalformedRecordError(
                    f"{where}: {spec.kind} field must be {spec.width} digits "
                    f"(got {chunk!r}): {record!r}"
                )
            values[spec.name] = Decimal(int(chunk)).scaleb(-spec.scale)
        return values


def _record_lines(text: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(text, str):
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()  # a single trailing newline is not a record
        return lines
    return list(text)


def parse_fixed_records(
    text: Union[str, Iterable[str]], layout: FixedRecordLayout
) -> List[Dict[str, Union[str, Decimal]]]:
    """Parse fixed-width records, rejecting anything off-layout as malformed."""
    return [layout.parse_record(line, index=i) for i, line in enumerate(_record_lines(text))]


@dataclass(frozen=True)
class LedgerSummary:
    count: int
    total: Decimal


@dataclass
class ReferenceLedger:
    """Exact-decimal model of "read fixed records and accumulate a signed total".

    It mirrors the fixture programs: a sign field plus an implicit-decimal
    amount, accumulated into ``PIC S9(total_int_digits)V9(scale)``.
    """

    layout: FixedRecordLayout
    amount_field: str = "amount"
    sign_field: str = "sign"
    total_int_digits: int = 7
    scale: int = 2
    signed: bool = True

    def summarize(self, text: Union[str, Iterable[str]]) -> LedgerSummary:
        total = Decimal(0)
        count = 0
        for record in parse_fixed_records(text, self.layout):
            amount = record[self.amount_field]
            sign = record.get(self.sign_field)
            value = Decimal(amount)  # type: ignore[arg-type]
            if sign == "-":
                value = -value
            total = enforce_picture(
                total + value,
                int_digits=self.total_int_digits,
                scale=self.scale,
                signed=self.signed,
            )
            count += 1
        return LedgerSummary(count=count, total=total)


def format_ledger_summary(
    summary: LedgerSummary,
    *,
    count_width: int = 3,
    int_digits: int = 7,
    scale: int = 2,
    labels: Tuple[str, str] = ("COUNT", "TOTAL"),
) -> str:
    """Render the fixture's two DISPLAY lines, matching COBOL's edited format."""
    count_label, total_label = labels
    return (
        f"{count_label}={summary.count:0{count_width}d}\n"
        f"{total_label}={format_edited(summary.total, int_digits=int_digits, scale=scale)}\n"
    )


# --------------------------------------------------------------------------
# Verification against the oracle
# --------------------------------------------------------------------------


@dataclass
class CobolCase:
    """One verification case: a source, its input, and the oracle's verdict."""

    name: str
    source: Path
    expected_stdout: Optional[str] = None
    expected_exit: int = 0
    stdin_text: str = ""
    argv: Tuple[str, ...] = ()
    expected_error: str = ""
    free_form: bool = True
    description: str = ""


@dataclass
class CaseOutcome:
    name: str
    ok: bool
    lines: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class VerificationReport:
    cases: List[CaseOutcome]

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.ok)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.cases if not c.ok)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.passed > 0

    def text(self) -> str:
        lines: List[str] = []
        for case in self.cases:
            lines.extend(case.lines)
        lines.append(f"{self.passed} passed, 0 skipped, {self.failed} failed")
        return "\n".join(lines)


def _detail(lines: List[str], text: str, *, indent: str = "    ") -> None:
    body = [ln.rstrip() for ln in (text or "").splitlines()]
    if not body:
        body = ["<empty>"]
    for chunk in body[:_MAX_DETAIL_LINES]:
        lines.append(indent + "| " + chunk[:_MAX_DETAIL_CHARS])


def _first_difference(expected: str, actual: str) -> str:
    exp = expected.splitlines()
    act = actual.splitlines()
    for idx in range(min(len(exp), len(act))):
        if exp[idx] != act[idx]:
            return (
                f"first difference at output line {idx + 1}: "
                f"expected {exp[idx]!r} but got {act[idx]!r}"
            )
    if len(exp) > len(act):
        return (
            f"output ended early at line {len(act) + 1}: "
            f"expected {exp[len(act)]!r}, got end of output"
        )
    if len(act) > len(exp):
        return (
            f"unexpected extra output at line {len(exp) + 1}: "
            f"expected end of output, got {act[len(exp)]!r}"
        )
    if expected != actual:
        return "output differs only in trailing whitespace or line endings"
    return "outputs match"


def verify_case(
    case: CobolCase,
    workdir: Union[str, Path],
    *,
    cobc: Optional[str] = None,
    timeout_s: float = 120.0,
) -> CaseOutcome:
    """Compile, run, and judge one case; returns marker lines and a verdict."""
    stem = f"cobol:{case.name}"
    source = Path(case.source)
    case_dir = Path(workdir) / re.sub(r"[^A-Za-z0-9._-]+", "_", case.name)

    if not source.is_file():
        return CaseOutcome(
            name=case.name,
            ok=False,
            lines=[f"TB-FAIL {stem} source", f"  source not found: {source}"],
            reason=f"source not found: {source}",
        )

    compiled = compile_source(
        source,
        case_dir,
        cobc=cobc,
        free_form=case.free_form,
        timeout_s=timeout_s,
    )
    if not compiled.ok:
        lines = [f"TB-FAIL {stem} compile"]
        errors = compiled.errors
        for diag in (errors or compiled.diagnostics)[:6]:
            lines.append("  " + diag.render())
        if not errors:
            _detail(lines, compiled.output)
        reason = (
            f"cobc error: {errors[0].render()}"
            if errors
            else f"cobc failed to build {source.name}"
        )
        lines.append("  " + reason)
        return CaseOutcome(
            name=case.name,
            ok=False,
            lines=lines,
            reason=reason,
        )

    assert compiled.binary is not None
    run = run_program(
        compiled.binary,
        argv=case.argv,
        stdin_text=case.stdin_text,
        timeout_s=timeout_s,
    )

    if case.expected_error:
        observed = run.combined
        if run.exit_code == 0:
            lines = [
                f"TB-FAIL {stem} reject",
                f"  the reference oracle rejects this input with {case.expected_error!r},",
                "  but the program exited 0 and accepted it",
                "  actual stdout:",
            ]
            _detail(lines, run.stdout, indent="      ")
            lines.append(
                f"  expected a nonzero exit reporting {case.expected_error!r}, got exit 0"
            )
            return CaseOutcome(
                name=case.name,
                ok=False,
                lines=lines,
                reason=f"exit 0; expected rejection reporting {case.expected_error!r}",
            )
        if case.expected_error not in observed:
            lines = [
                f"TB-FAIL {stem} reject",
                f"  the reference oracle rejects this input with {case.expected_error!r},",
                f"  but the program exited {run.exit_code} without reporting it",
                "  actual output:",
            ]
            _detail(lines, observed, indent="      ")
            lines.append(f"  expected rejection text {case.expected_error!r} was not printed")
            return CaseOutcome(
                name=case.name,
                ok=False,
                lines=lines,
                reason=f"exit {run.exit_code} without {case.expected_error!r}",
            )
        return CaseOutcome(name=case.name, ok=True, lines=[f"TB-PASS {stem}"], reason="")

    if run.exit_code != case.expected_exit:
        lines = [
            f"TB-FAIL {stem} rc={run.exit_code}",
            f"  expected exit {case.expected_exit}, got {run.exit_code}",
        ]
        if run.stderr.strip():
            lines.append("  stderr:")
            _detail(lines, run.stderr, indent="      ")
        lines.append(f"  process exited with status {run.exit_code} (expected {case.expected_exit})")
        return CaseOutcome(
            name=case.name,
            ok=False,
            lines=lines,
            reason=f"exit {run.exit_code}, expected {case.expected_exit}",
        )

    if case.expected_stdout is not None and run.stdout != case.expected_stdout:
        lines = [
            f"TB-FAIL {stem} output",
            "  process exited 0 but its computed output disagrees with the reference oracle",
            "  expected stdout:",
        ]
        _detail(lines, case.expected_stdout, indent="      ")
        lines.append("  actual stdout:")
        _detail(lines, run.stdout, indent="      ")
        difference = _first_difference(case.expected_stdout, run.stdout)
        lines.append("  " + difference)
        return CaseOutcome(name=case.name, ok=False, lines=lines, reason=difference)

    return CaseOutcome(name=case.name, ok=True, lines=[f"TB-PASS {stem}"], reason="")


def verify_cases(
    cases: Sequence[CobolCase],
    workdir: Union[str, Path],
    *,
    cobc: Optional[str] = None,
    timeout_s: float = 120.0,
) -> VerificationReport:
    """Verify every case in one workdir and aggregate the outcomes."""
    outcomes = [
        verify_case(case, workdir, cobc=cobc, timeout_s=timeout_s) for case in cases
    ]
    return VerificationReport(cases=outcomes)


# --------------------------------------------------------------------------
# CLI: run a JSON case file through the same markers a suite would show
# --------------------------------------------------------------------------


def _cases_from_spec(path: Path) -> List[CobolCase]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    base = path.resolve().parent
    cases: List[CobolCase] = []
    for raw in spec.get("cases", []):
        source = Path(raw["source"])
        if not source.is_absolute():
            source = base / source
        expected = raw.get("expected_stdout")
        cases.append(
            CobolCase(
                name=str(raw["name"]),
                source=source,
                expected_stdout=None if expected is None else str(expected),
                expected_exit=int(raw.get("expected_exit", 0)),
                stdin_text=str(raw.get("stdin", "")),
                argv=tuple(str(a) for a in raw.get("argv", [])),
                expected_error=str(raw.get("expected_error", "")),
                free_form=bool(raw.get("free_form", True)),
                description=str(raw.get("description", "")),
            )
        )
    return cases


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cobol_check",
        description="Compile and verify GnuCOBOL programs against expected output.",
    )
    parser.add_argument("--spec", required=True, help="JSON file listing verification cases")
    parser.add_argument("--workdir", default="", help="build directory (default: a temp dir)")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-step timeout (seconds)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cases = _cases_from_spec(Path(args.spec))
    if not cases:
        print("TB-FAIL cobol:spec no cases declared")
        print("0 passed, 0 skipped, 1 failed")
        return 1

    if args.workdir:
        report = verify_cases(cases, Path(args.workdir), timeout_s=args.timeout)
    else:
        from ..isolation import default_scratch_root

        with tempfile.TemporaryDirectory(
            prefix="tb-cobol-", dir=str(default_scratch_root())
        ) as tmpdir:
            report = verify_cases(cases, Path(tmpdir), timeout_s=args.timeout)
    print(report.text())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI test
    raise SystemExit(main())
