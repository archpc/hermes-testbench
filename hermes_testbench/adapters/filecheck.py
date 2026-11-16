"""Output-contract assertions on captured program output, backed by LLVM FileCheck.

A suite that shells out to a compiler, a CLI, or a service often needs to assert on
its *output* rather than on an exit status: "this diagnostic must appear", "these
lines must appear in this order", "this warning must not appear", "this string must
appear exactly twice". This module expresses those four intents as a tiny immutable
``OutputContract`` and hands the actual matching to an existing LLVM ``FileCheck``
binary:

* ``required`` patterns become one ``CHECK-DAG`` group (each must appear, any order);
* ``ordered`` patterns become successive ``CHECK`` directives in one invocation;
* ``forbidden`` patterns become ``--implicit-check-not`` arguments (absent anywhere);
* ``counts`` become ``CHECK-COUNT-N`` followed by ``CHECK-NOT`` for the same pattern,
  so extra occurrences past N also fail.

Each independent intent runs as its own FileCheck invocation because FileCheck treats
multiple prefixes as one interleaved sequence (a trailing ``CHECK-DAG`` group also
suppresses implicit-NOT regions). One generated ``/bin/sh`` script runs all steps and
is then executed through TestBench's existing ``shell`` adapter (``execute_suite``), so
isolation, timeouts, evidence capture, redaction, exit-status accounting, and reporting
are reused unchanged and several failing assertions are reported in a single run. No
matching language is implemented here: FileCheck owns pattern semantics (including its
``{{regex}}`` and ``[[var]]`` extensions).

``FileCheck`` is optional. When it is not installed (for example a CI image without
LLVM), ``execute_output_contract`` falls back to ``check_text``: a deliberately
minimal, pure-stdlib literal-substring evaluator for the same four intents. The
fallback exists only so output contracts stay usable without a toolchain; it is
line-oriented, literal (no FileCheck regex/variables), and its diagnostics are shaped
like FileCheck's so the reporting path is identical. Prefer FileCheck whenever it is
present. Count assertions need a FileCheck with ``CHECK-COUNT`` support; the fallback
has no version floor.

``parse_filecheck_diagnostics`` is a pure text parser (no I/O, no spawning) that turns
raw FileCheck error output into ``FailureRecord``-shaped dicts, so the coordinator can
also wire plain ``FileCheck`` invocations into ``parse_shell`` without this module's
``execute_output_contract`` wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..schema import (
    CaseResult,
    FailureKind,
    FailureRecord,
    ProfileName,
    RunStatus,
    SuiteResult,
    SuiteSpec,
    utc_now_iso,
)

__all__ = [
    "OutputCheckStep",
    "OutputContract",
    "build_check_steps",
    "check_text",
    "execute_output_contract",
    "filecheck_command",
    "find_filecheck",
    "parse_filecheck_diagnostics",
]

# FileCheck's default ``CHECK`` prefix also accepts ``CHECK-DAG``, ``CHECK-COUNT-N`` and
# ``CHECK-NOT`` directives, so each independent assertion needs no custom prefix.

_DISABLE_VALUES = frozenset({"", "0", "off", "none", "false", "no"})

_MAX_MESSAGE = 800
_MAX_EXCERPT = 800


@dataclass(frozen=True)
class OutputContract:
    """Declarative output assertions; an empty contract is rejected.

    Patterns are passed to FileCheck verbatim (so its ``{{regex}}`` and
    ``[[variable]]`` syntax keeps working). The literal fallback treats them as
    plain substrings and does not interpret FileCheck syntax.
    """

    required: Tuple[str, ...] = ()
    ordered: Tuple[str, ...] = ()
    forbidden: Tuple[str, ...] = ()
    counts: Tuple[Tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        required = _normalise_patterns(self.required, "required")
        ordered = _normalise_patterns(self.ordered, "ordered")
        forbidden = _normalise_patterns(self.forbidden, "forbidden")
        counts = _normalise_counts(self.counts)
        asserted = set(required) | set(ordered) | {pat for pat, _n in counts}
        overlap = sorted(asserted & set(forbidden))
        if overlap:
            raise ValueError(
                "pattern(s) both asserted and forbidden: " + ", ".join(repr(p) for p in overlap)
            )
        if not (required or ordered or forbidden or counts):
            raise ValueError("an OutputContract needs at least one assertion")
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "ordered", ordered)
        object.__setattr__(self, "forbidden", forbidden)
        object.__setattr__(self, "counts", counts)

    def assertion_count(self) -> int:
        return len(self.required) + len(self.ordered) + len(self.forbidden) + len(self.counts)


def _validate_pattern(pattern: Any, field: str) -> str:
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError(f"{field} contains an empty or non-string pattern: {pattern!r}")
    if "\n" in pattern or "\r" in pattern:
        raise ValueError(f"{field} pattern must be a single line: {pattern!r}")
    return pattern


def _normalise_patterns(patterns: Iterable[str], field: str) -> Tuple[str, ...]:
    if isinstance(patterns, str):
        raise ValueError(f"{field} must be a sequence of patterns, not a string")
    normalised: List[str] = []
    for pattern in patterns:
        validated = _validate_pattern(pattern, field)
        if validated not in normalised:
            normalised.append(validated)
    return tuple(normalised)


def _normalise_counts(counts: Iterable[Tuple[str, int]]) -> Tuple[Tuple[str, int], ...]:
    normalised: List[Tuple[str, int]] = []
    seen: set[str] = set()
    for item in counts:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"counts entries must be (pattern, count) pairs: {item!r}")
        pattern, raw_count = item
        pattern = _validate_pattern(pattern, "counts")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError(f"count for {pattern!r} must be a non-negative int: {raw_count!r}")
        if pattern in seen:
            raise ValueError(f"counts declares {pattern!r} twice")
        seen.add(pattern)
        normalised.append((pattern, raw_count))
    return tuple(normalised)


# --------------------------------------------------------------------------------------
# FileCheck discovery (optional tool, never required)


def _brew_prefixes() -> List[Path]:
    prefixes: List[Path] = []
    env_prefix = os.environ.get("HOMEBREW_PREFIX", "").strip()
    if env_prefix:
        prefixes.append(Path(env_prefix).expanduser())
    brew = shutil.which("brew")
    if brew:
        prefixes.append(Path(brew).resolve().parent.parent)
    unique: List[Path] = []
    for prefix in prefixes:
        if prefix not in unique:
            unique.append(prefix)
    return unique


def _llvm_filecheck_candidates() -> List[str]:
    """Absolute FileCheck paths from keg-only/Linux LLVM installs.

    Built from Homebrew prefixes and standard Linux keg directories; no
    machine-specific user path is embedded. Cached by the caller's module state.
    """
    found: List[str] = []
    for prefix in _brew_prefixes():
        opt = prefix / "opt"
        if opt.is_dir():
            for cand in sorted(opt.glob("llvm*/bin/FileCheck"), key=lambda p: (p.parent.parent.name != "llvm", p.parent.parent.name)):
                if cand.is_file() and os.access(cand, os.X_OK):
                    found.append(str(cand))
        plain = prefix / "bin" / "FileCheck"
        if plain.is_file() and os.access(plain, os.X_OK):
            found.append(str(plain))
    for cand in sorted(Path("/usr/lib").glob("llvm-*/bin/FileCheck")):
        if cand.is_file() and os.access(cand, os.X_OK):
            found.append(str(cand))
    unique: List[str] = []
    for path in found:
        if path not in unique:
            unique.append(path)
    return unique


_CANDIDATE_CACHE: Optional[List[str]] = None


def find_filecheck() -> Optional[str]:
    """Return a usable FileCheck path, or ``None`` when the tool is absent.

    Order: ``TESTBENCH_FILECHECK`` override, ``PATH`` (``FileCheck`` then
    ``FileCheck-30``..``FileCheck-10``), then keg-only Homebrew / Linux LLVM dirs.
    An override of ``0``/``off``/``none`` disables the tool explicitly.
    """
    override = os.environ.get("TESTBENCH_FILECHECK", "")
    if override.strip():
        stripped = override.strip()
        if stripped.lower() in _DISABLE_VALUES:
            return None
        cand = Path(stripped).expanduser()
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
        resolved = shutil.which(stripped)
        return resolved

    for name in ["FileCheck"] + [f"FileCheck-{n}" for n in range(30, 9, -1)]:
        path = shutil.which(name)
        if path:
            return path

    global _CANDIDATE_CACHE
    if _CANDIDATE_CACHE is None:
        _CANDIDATE_CACHE = _llvm_filecheck_candidates()
    return _CANDIDATE_CACHE[0] if _CANDIDATE_CACHE else None


# --------------------------------------------------------------------------------------
# Check-file generation and command construction


@dataclass(frozen=True)
class OutputCheckStep:
    """One FileCheck invocation: a check file plus its extra command-line flags."""

    case_name: str
    check_text: str
    extra_args: Tuple[str, ...] = ()


def build_check_steps(contract: OutputContract) -> Tuple[OutputCheckStep, ...]:
    """Split a contract into independent FileCheck invocations.

    FileCheck evaluates multiple ``--check-prefix`` groups as *one interleaved
    sequence* (verified on LLVM 23): a trailing DAG group also suppresses
    ``--implicit-check-not`` regions. Keeping each intent in its own invocation is
    therefore the only way to make the four assertions independent. The steps are
    executed by a single generated ``/bin/sh`` script under one isolated suite, so
    several failing assertions still produce several diagnostics in one run.
    """
    steps: List[OutputCheckStep] = []

    if contract.required:
        body = "".join(f"CHECK-DAG: {pattern}\n" for pattern in contract.required)
        steps.append(OutputCheckStep(case_name="CHECK-DAG", check_text=body))

    if contract.ordered:
        body = "".join(f"CHECK: {pattern}\n" for pattern in contract.ordered)
        steps.append(OutputCheckStep(case_name="CHECK", check_text=body))

    for pattern, count in contract.counts:
        if count == 0:
            steps.append(
                OutputCheckStep(case_name=f"CHECK-NOT: {pattern}", check_text=f"CHECK-NOT: {pattern}\n")
            )
            continue
        # COUNT walks from the start of the input and stops after the Nth match, so the
        # trailing NOT (its region runs to EOF) proves there is no (N+1)th occurrence.
        body = f"CHECK-COUNT-{count}: {pattern}\nCHECK-NOT: {pattern}\n"
        steps.append(OutputCheckStep(case_name=f"CHECK-COUNT-{count}: {pattern}", check_text=body))

    for pattern in contract.forbidden:
        # A zero-width anchor at the start makes the implicit NOT region cover the
        # whole input, including the first line.
        steps.append(
            OutputCheckStep(
                case_name=f"IMPLICIT-CHECK-NOT: {pattern}",
                check_text="CHECK: {{^}}\n",
                extra_args=(f"--implicit-check-not={pattern}",),
            )
        )

    return tuple(steps)


def filecheck_command(
    step: OutputCheckStep,
    *,
    input_path: Path,
    check_file: Path,
    executable: str,
) -> List[str]:
    """Build the FileCheck argv for one check step over an on-disk input file."""
    return [
        str(executable),
        *step.extra_args,
        f"--input-file={input_path}",
        str(check_file),
    ]


# --------------------------------------------------------------------------------------
# Pure FileCheck diagnostic parsing


_DIAG_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<sev>error|warning):\s*(?P<rest>.+)$"
)
_BARE_ERROR_RE = re.compile(r"^(?P<sev>error|warning):\s*(?P<rest>.+)$")
_DIRECTIVE_RE = re.compile(r"^(?P<directive>[A-Z][A-Z0-9_-]*):\s*(?P<msg>.*)$")
_SNIPPET_RE = re.compile(r"^\s*(?://|;|#)?\s*(?P<directive>[A-Za-z][A-Za-z0-9_-]*):\s*(?P<pattern>.+?)\s*$")
_IMPLICIT_SNIPPET_RE = re.compile(
    r"-implicit-check-not=(?:'(?P<single>[^']*)'|\"(?P<double>[^\"]*)\"|(?P<bare>\S+))"
)
_PSEUDO_FILES = frozenset({"command line", "<command line>", "command-line"})


def _display_directive(directive: str) -> str:
    """Return the directive name exactly as FileCheck printed it."""
    return directive


def _window(lines: Sequence[str], start: int) -> List[str]:
    """Error line plus the following non-blank detail lines (bounded)."""
    window = [lines[start]]
    for line in lines[start + 1 : start + 8]:
        if not line.strip():
            break
        window.append(line)
    return window


def _pattern_from_window(window: Sequence[str], directive: str) -> Tuple[str, str]:
    """Return (display directive, pattern) from an error's source snippet."""
    if directive == "IMPLICIT-CHECK-NOT" or directive.endswith("IMPLICIT-CHECK-NOT"):
        for line in window[1:]:
            match = _IMPLICIT_SNIPPET_RE.search(line)
            if match:
                pattern = match.group("single") or match.group("double") or match.group("bare") or ""
                return directive, pattern
    fallback: Tuple[str, str] = (directive, "")
    for line in window[1:]:
        match = _SNIPPET_RE.match(line)
        if not match:
            continue
        full = match.group("directive")
        if full.lower() in ("note", "error", "warning"):
            continue
        # Prefer a snippet whose directive family agrees with the reported error.
        candidate = (_display_directive(full), match.group("pattern").strip())
        if directive == "CHECK-COUNT" or "-COUNT" in full.upper():
            return candidate
        if full.upper().startswith(directive.split("-")[0]):
            return candidate
        if not fallback[1]:
            fallback = candidate
    return fallback


def parse_filecheck_diagnostics(text: str) -> List[Dict[str, Any]]:
    """Extract actionable failure dicts from raw FileCheck error output.

    Every returned dict matches ``FailureRecord`` fields except ``suite``:
    ``case_name``, ``kind`` (always ``"assertion"``), ``message``, ``file``,
    ``line``, ``excerpt``. Pure: no I/O, no spawning, no re-runs.
    """
    lines = (text or "").splitlines()
    results: List[Dict[str, Any]] = []
    seen: set[Tuple[str, Optional[str], Optional[int], str]] = set()
    index = 0
    while index < len(lines):
        raw = lines[index]
        match = _DIAG_RE.match(raw)
        bare = None if match else _BARE_ERROR_RE.match(raw.strip())
        if not match and not bare:
            index += 1
            continue
        if match is not None:
            file_field = match.group("file").strip()
            line_no: Optional[int] = int(match.group("line"))
            rest = match.group("rest").strip()
        elif bare is not None:
            file_field = ""
            line_no = None
            rest = bare.group("rest").strip()
        else:
            index += 1
            continue
        source_file = None if file_field.lower() in _PSEUDO_FILES else (file_field or None)
        if bare is not None:
            line_no = None

        directive_match = _DIRECTIVE_RE.match(rest)
        directive = directive_match.group("directive") if directive_match else "FILECHECK"
        window = _window(lines, index)
        shown, pattern = _pattern_from_window(window, directive)
        shown = _display_directive(shown)
        if pattern:
            case_name = f"{shown}: {pattern}"
        else:
            case_name = shown
        case_name = case_name[:200]

        record = {
            "case_name": case_name,
            "kind": FailureKind.ASSERTION,
            "message": rest[:_MAX_MESSAGE],
            "file": source_file,
            "line": line_no if source_file is not None else None,
            "excerpt": "\n".join(part.rstrip() for part in window)[:_MAX_EXCERPT],
        }
        key = (record["case_name"], record["file"], record["line"], record["message"])
        if key not in seen:
            seen.add(key)
            results.append(record)
        index += max(1, len(window))
    return results


# --------------------------------------------------------------------------------------
# Minimal literal fallback (only used when FileCheck is unavailable)


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _context(text: str, offset: int) -> str:
    if offset < 0:
        return ""
    lines = text.splitlines()
    line_index = min(max(0, _line_of(text, offset) - 1), max(0, len(lines) - 1))
    if not lines:
        return ""
    lo = max(0, line_index - 1)
    hi = min(len(lines), line_index + 2)
    return "\n".join(lines[lo:hi])[:_MAX_EXCERPT]


def _literal_failure(
    case_name: str,
    message: str,
    text: str,
    offset: Optional[int],
    source_label: Optional[str],
) -> Dict[str, Any]:
    if offset is not None and offset >= 0:
        line = _line_of(text, offset)
        excerpt = _context(text, offset)
    else:
        line = None
        excerpt = ""
    return {
        "case_name": case_name,
        "kind": FailureKind.ASSERTION,
        "message": message[:_MAX_MESSAGE],
        "file": source_label,
        "line": line,
        "excerpt": excerpt[:_MAX_EXCERPT],
    }


def _literal_cases(
    contract: OutputContract, text: str, source_label: Optional[str]
) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    """Evaluate every assertion literally; return (case_name, failure|None) pairs."""
    cases: List[Tuple[str, Optional[Dict[str, Any]]]] = []

    for pattern in contract.required:
        offset = text.find(pattern)
        failure = None
        if offset < 0:
            failure = _literal_failure(
                f"CHECK-DAG: {pattern}", "expected string not found in input", text, None, source_label
            )
        cases.append((f"CHECK-DAG: {pattern}", failure))

    cursor = 0
    for pattern in contract.ordered:
        offset = text.find(pattern, cursor)
        failure = None
        if offset < 0:
            failure = _literal_failure(
                f"CHECK: {pattern}",
                "expected string not found in input (ordered after the previous match)",
                text,
                None,
                source_label,
            )
        else:
            cursor = offset + len(pattern)
        cases.append((f"CHECK: {pattern}", failure))

    for pattern in contract.forbidden:
        offset = text.find(pattern)
        failure = None
        if offset >= 0:
            failure = _literal_failure(
                f"IMPLICIT-CHECK-NOT: {pattern}", "excluded string found in input", text, offset, source_label
            )
        cases.append((f"IMPLICIT-CHECK-NOT: {pattern}", failure))

    for pattern, expected in contract.counts:
        found = text.count(pattern)
        failure = None
        if found != expected:
            if expected == 0:
                name = f"CHECK-NOT: {pattern}"
            else:
                name = f"CHECK-COUNT-{expected}: {pattern}"
            failure = _literal_failure(
                name,
                f"expected exactly {expected} occurrence(s) of pattern, found {found}",
                text,
                text.find(pattern) if found else None,
                source_label,
            )
        else:
            name = f"CHECK-NOT: {pattern}" if expected == 0 else f"CHECK-COUNT-{expected}: {pattern}"
        cases.append((name, failure))

    return cases


def check_text(
    contract: OutputContract, text: str, *, source_label: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Literal in-process evaluation of ``contract`` against ``text``.

    This is the FileCheck-less fallback: plain substring matching with 1-based line
    numbers. It intentionally supports no FileCheck regex/variable syntax.
    """
    return [failure for _name, failure in _literal_cases(contract, text, source_label) if failure]


# --------------------------------------------------------------------------------------
# Execution through the existing shell adapter


def _safe_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "contract").strip("._") or "contract"
    return slug[:60]


def _read_captured(result: SuiteResult) -> str:
    parts: List[str] = []
    for raw in (result.stdout_path, result.stderr_path):
        if not raw:
            continue
        try:
            parts.append(Path(raw).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(parts)


def _apply_contract_failures(result: SuiteResult, diagnostics: List[Dict[str, Any]]) -> None:
    records = [
        FailureRecord(
            suite=result.name,
            case_name=item["case_name"],
            kind=FailureKind.ASSERTION,
            message=item.get("message", ""),
            file=item.get("file"),
            line=item.get("line"),
            excerpt=item.get("excerpt", ""),
        )
        for item in diagnostics
    ]
    result.failures = records
    result.fail_count = max(1, len(records))
    result.pass_count = 0
    result.skip_count = 0
    result.error_count = 0
    result.status = RunStatus.FAIL
    result.total_count = max(result.total_count, result.fail_count)
    first = records[0].message if records else "output contract not satisfied"
    summary = first if len(records) == 1 else f"{first} (+{len(records) - 1} more)"
    result.summary_excerpt = summary[:_MAX_MESSAGE]


def _literal_result(
    name: str,
    contract: OutputContract,
    *,
    project_root: Path,
    profile: str,
    input_path: Path,
) -> SuiteResult:
    try:
        text = input_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result = SuiteResult(
            name=name,
            adapter="shell",
            profile=profile,
            command=[],
            cwd=str(project_root),
            status=RunStatus.ERROR,
            exit_code=None,
            started_at=utc_now_iso(),
            finished_at=utc_now_iso(),
            total_count=1,
            error_count=1,
            summary_excerpt=f"cannot read output file: {exc}",
            failures=[
                FailureRecord(
                    suite=name,
                    case_name="<input>",
                    kind=FailureKind.HARNESS_ERROR,
                    message=str(exc),
                )
            ],
        )
        result.metrics = {
            "output_contract_engine": "literal-fallback",
            "output_contract_assertions": contract.assertion_count(),
            "filecheck_used": False,
        }
        return result

    cases: List[CaseResult] = []
    failures: List[FailureRecord] = []
    for case_name, failure in _literal_cases(contract, text, str(input_path)):
        if failure is None:
            cases.append(CaseResult(name=case_name, suite=name, status=RunStatus.PASS))
            continue
        failures.append(
            FailureRecord(
                suite=name,
                case_name=failure["case_name"],
                kind=failure["kind"],
                message=failure["message"],
                file=failure.get("file"),
                line=failure.get("line"),
                excerpt=failure.get("excerpt", ""),
            )
        )
        cases.append(
            CaseResult(
                name=case_name,
                suite=name,
                status=RunStatus.FAIL,
                message=failure["message"],
                file=failure.get("file"),
                line=failure.get("line"),
            )
        )

    total = len(cases)
    failed = len(failures)
    started = utc_now_iso()
    result = SuiteResult(
        name=name,
        adapter="shell",
        profile=profile,
        command=[],
        cwd=str(Path(project_root).expanduser().resolve()),
        status=RunStatus.FAIL if failed else RunStatus.PASS,
        exit_code=None,
        started_at=started,
        finished_at=utc_now_iso(),
        total_count=total,
        pass_count=total - failed,
        fail_count=failed,
        summary_excerpt=(
            failures[0].message
            if failures
            else f"{total} output assertion(s) satisfied (literal fallback; FileCheck unavailable)"
        )[: _MAX_MESSAGE],
        failures=failures,
        cases=cases,
    )
    result.metrics = {
        "output_contract_engine": "literal-fallback",
        "output_contract_assertions": contract.assertion_count(),
        "filecheck_used": False,
    }
    return result


def execute_output_contract(
    name: str,
    *,
    contract: OutputContract,
    project_root: Path,
    evidence_dir: Path,
    run_id: str,
    input_text: Optional[str] = None,
    input_path: Optional[Path] = None,
    profile: str = ProfileName.UNIT,
    timeout_s: float = 60.0,
    executable: Optional[str] = None,
    prefer_filecheck: bool = True,
    scratch_root: Optional[Path] = None,
) -> SuiteResult:
    """Run one output contract, preferring FileCheck through the ``shell`` adapter.

    Exactly one of ``input_text`` / ``input_path`` supplies the output to check.
    ``prefer_filecheck=False`` forces the literal fallback (useful for deterministic
    CI without LLVM). An explicit ``executable`` wins over discovery.
    """
    if (input_text is None) == (input_path is None):
        raise ValueError("pass exactly one of input_text or input_path")

    project_root = Path(project_root).expanduser().resolve()
    evidence_dir = Path(evidence_dir)
    work_dir = evidence_dir / "filecheck"
    work_dir.mkdir(parents=True, exist_ok=True)
    slug = _safe_slug(name)

    if input_text is not None:
        output_file = work_dir / f"{slug}.input.txt"
        output_file.write_text(input_text, encoding="utf-8")
    else:
        output_file = Path(str(input_path)).expanduser().resolve()
        if not output_file.is_file():
            raise FileNotFoundError(f"output file not found: {output_file}")

    filecheck = str(executable) if executable else None
    if filecheck is None and prefer_filecheck:
        filecheck = find_filecheck()
    if filecheck is not None:
        candidate = Path(filecheck)
        usable = (candidate.is_file() and os.access(candidate, os.X_OK)) or bool(shutil.which(filecheck))
        if not usable:
            filecheck = None

    if filecheck is None:
        return _literal_result(name, contract, project_root=project_root, profile=profile, input_path=output_file)

    check_dir = work_dir / slug
    check_dir.mkdir(parents=True, exist_ok=True)
    steps = build_check_steps(contract)
    script_lines = [
        "#!/bin/sh",
        "# Generated by hermes_testbench.adapters.filecheck; do not edit.",
        "# One FileCheck invocation per independent assertion; keep going to collect all.",
        "rc=0",
    ]
    for index, step in enumerate(steps):
        check_file = check_dir / f"step{index}.check"
        check_file.write_text(step.check_text, encoding="utf-8")
        argv = filecheck_command(step, input_path=output_file, check_file=check_file, executable=filecheck)
        script_lines.append(f"{shlex.join(argv)} || rc=1")
    script_lines.append("exit $rc")
    script = check_dir / "run.sh"
    script.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
    script.chmod(0o700)

    suite = SuiteSpec(
        name=name,
        adapter="shell",
        profile=profile,
        command=["/bin/sh", str(script)],
        cwd=str(project_root),
        timeout_s=timeout_s,
        required_binaries=[filecheck],
        description=(
            f"output contract ({contract.assertion_count()} assertion(s), "
            f"{len(steps)} FileCheck step(s))"
        ),
    )

    # Local import keeps adapters/__init__ free to import this module later without a cycle.
    from . import execute_suite

    result, _usage = execute_suite(
        suite,
        project_root=project_root,
        evidence_dir=evidence_dir,
        run_id=run_id,
        scratch_root=scratch_root,
    )
    diagnostics = parse_filecheck_diagnostics(_read_captured(result))
    if diagnostics:
        _apply_contract_failures(result, diagnostics)
    result.metrics.update(
        {
            "output_contract_engine": "filecheck",
            "output_contract_assertions": contract.assertion_count(),
            "output_contract_steps": len(steps),
            "filecheck_binary": filecheck,
            "check_dir": str(check_dir),
        }
    )
    return result
