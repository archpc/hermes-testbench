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
is executed through TestBench's existing ``shell`` adapter (``execute_suite``), so
isolation, Seatbelt confinement, timeouts, evidence capture, in-stream secret
redaction, exit-status accounting, and ``SKIP`` gating when ``FileCheck`` is absent
are reused unchanged. No custom matching engine or FileCheck clone is implemented.

``parse_filecheck_diagnostics`` is a pure text parser (no I/O, no spawning) wired
into ``parsers.parse_shell`` so both ``execute_output_contract`` and direct
``FileCheck`` pipelines in ``shell`` suites produce ``FailureRecord`` entries with
``kind="assertion"``, source file, line number, and diagnostic excerpt.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..schema import FailureKind, ProfileName, SuiteResult, SuiteSpec

__all__ = [
    "OutputCheckStep",
    "OutputContract",
    "build_check_steps",
    "execute_output_contract",
    "filecheck_command",
    "find_filecheck",
    "parse_filecheck_diagnostics",
]

_DISABLE_VALUES = frozenset({"", "0", "off", "none", "false", "no"})

_MAX_MESSAGE = 800
_MAX_EXCERPT = 800


@dataclass(frozen=True)
class OutputContract:
    """Declarative output assertions; an empty contract is rejected.

    Patterns are passed to LLVM FileCheck verbatim, so FileCheck's native
    ``{{regex}}`` and ``[[variable]]`` directives remain available.
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
# FileCheck discovery (optional tool; missing binary triggers standard Gate 3 SKIP)


def _brew_prefixes() -> List[Path]:
    prefixes: List[Path] = []
    env_prefix = os.environ.get("HOMEBREW_PREFIX", "").strip()
    if env_prefix:
        prefixes.append(Path(env_prefix).expanduser())
    brew = shutil.which("brew")
    if brew:
        prefixes.append(Path(brew).resolve().parent.parent)
    for std_prefix in ("/opt/homebrew", "/usr/local"):
        cand = Path(std_prefix)
        if cand.is_dir():
            prefixes.append(cand)
    unique: List[Path] = []
    for prefix in prefixes:
        if prefix not in unique:
            unique.append(prefix)
    return unique


def _llvm_filecheck_candidates() -> List[str]:
    """Absolute FileCheck paths from keg-only Homebrew and Linux LLVM installs."""
    found: List[str] = []
    for prefix in _brew_prefixes():
        opt = prefix / "opt"
        if opt.is_dir():
            for cand in sorted(
                opt.glob("llvm*/bin/FileCheck"),
                key=lambda p: (p.parent.parent.name != "llvm", p.parent.parent.name),
            ):
                if cand.is_file() and os.access(cand, os.X_OK):
                    found.append(str(cand))
        plain = prefix / "bin" / "FileCheck"
        if plain.is_file() and os.access(plain, os.X_OK):
            found.append(str(plain))
    for cand in sorted(Path("/usr/lib").glob("llvm-*/bin/FileCheck"), reverse=True):
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
        return shutil.which(stripped)

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
    therefore required for independent assertions. All steps execute inside one
    generated ``/bin/sh`` script in a single isolated ``execute_suite`` call.
    """
    steps: List[OutputCheckStep] = []

    if contract.required:
        body = "".join(f"CHECK-DAG: {pattern}\n" for pattern in contract.required)
        steps.append(
            OutputCheckStep(case_name="CHECK-DAG", check_text=body, extra_args=("--allow-empty",))
        )

    if contract.ordered:
        body = "".join(f"CHECK: {pattern}\n" for pattern in contract.ordered)
        steps.append(
            OutputCheckStep(case_name="CHECK", check_text=body, extra_args=("--allow-empty",))
        )

    for pattern, count in contract.counts:
        if count == 0:
            steps.append(
                OutputCheckStep(
                    case_name=f"CHECK-NOT: {pattern}",
                    check_text=f"CHECK-NOT: {pattern}\n",
                    extra_args=("--allow-empty",),
                )
            )
            continue
        body = f"CHECK-COUNT-{count}: {pattern}\nCHECK-NOT: {pattern}\n"
        steps.append(
            OutputCheckStep(
                case_name=f"CHECK-COUNT-{count}: {pattern}",
                check_text=body,
                extra_args=("--allow-empty",),
            )
        )

    for pattern in contract.forbidden:
        steps.append(
            OutputCheckStep(
                case_name=f"IMPLICIT-CHECK-NOT: {pattern}",
                check_text="CHECK: {{^}}\n",
                extra_args=("--allow-empty", f"--implicit-check-not={pattern}"),
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
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s*error:\s*(?P<rest>.+)$"
)
_BARE_ERROR_RE = re.compile(r"^(?:FileCheck\s+)?error:\s*(?P<rest>.+)$")
_FC_DIRECTIVE_RE = re.compile(
    r"^(?P<directive>(?:[A-Z][A-Z0-9_-]*-)?(?:CHECK(?:-(?:NEXT|SAME|NOT|DAG|COUNT(?:-\d+)?|LABEL|EMPTY))?|IMPLICIT-CHECK-NOT"
    r"|NEXT|SAME|NOT|DAG|COUNT(?:-\d+)?|LABEL|EMPTY)):\s*(?P<msg>.*)$"
)
_FC_GLOBAL_ERR_RE = re.compile(
    r"(?:check strings found with prefix|FileCheck error:|invalid check|unable to open check-file"
    r"|supplied check-prefix|found 'CHECK-)",
    re.IGNORECASE,
)
_SNIPPET_RE = re.compile(
    r"^\s*(?://|;|#)?\s*(?P<directive>[A-Za-z][A-Za-z0-9_-]*):\s*(?P<pattern>.+?)\s*$"
)
_IMPLICIT_SNIPPET_RE = re.compile(
    r"-implicit-check-not=(?:'(?P<single>[^']*)'|\"(?P<double>[^\"]*)\"|(?P<bare>\S+))"
)
_PSEUDO_FILES = frozenset({"command line", "<command line>", "command-line"})


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
                pattern = (
                    match.group("single") or match.group("double") or match.group("bare") or ""
                )
                return directive, pattern
    fallback: Tuple[str, str] = (directive, "")
    for line in window[1:]:
        match = _SNIPPET_RE.match(line)
        if not match:
            continue
        full = match.group("directive")
        if full.lower() in ("note", "error", "warning", "input", "check"):
            if full != "CHECK":
                continue
        candidate = (full, match.group("pattern").strip())
        if directive == "CHECK-COUNT" or "-COUNT" in full.upper():
            return candidate
        if full.upper().startswith(directive.split("-")[0]):
            return candidate
        if not fallback[1]:
            fallback = candidate
    return fallback


def parse_filecheck_diagnostics(text: str) -> List[Dict[str, Any]]:
    """Extract actionable failure dicts from LLVM FileCheck error output.

    Matches only genuine FileCheck ``error:`` diagnostics (never ``warning:`` or
    unrelated compiler errors) so ``parsers.parse_shell`` can invoke this safely on
    every shell suite's captured output.
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
        else:
            assert bare is not None
            file_field = ""
            line_no = None
            rest = bare.group("rest").strip()

        directive_match = _FC_DIRECTIVE_RE.match(rest)
        is_fc_error = bool(directive_match) or bool(_FC_GLOBAL_ERR_RE.search(rest))
        if not is_fc_error:
            index += 1
            continue

        source_file = None if file_field.lower() in _PSEUDO_FILES else (file_field or None)
        directive = directive_match.group("directive") if directive_match else "FILECHECK"
        window = _window(lines, index)
        shown, pattern = _pattern_from_window(window, directive)
        case_name = f"{shown}: {pattern}" if pattern else shown
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
# Execution through the existing shell adapter


def _safe_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "contract").strip("._") or "contract"
    return slug[:60]


def execute_output_contract(
    name: str,
    *,
    contract: OutputContract,
    project_root: Path,
    evidence_dir: Path,
    run_id: str,
    input_text: Optional[str] = None,
    input_path: Optional[Path] = None,
    producer_command: Optional[Sequence[str]] = None,
    profile: str = ProfileName.UNIT,
    timeout_s: float = 60.0,
    executable: Optional[str] = None,
    scratch_root: Optional[Path] = None,
) -> SuiteResult:
    """Run one output contract through ``execute_suite`` and TestBench's ``shell`` adapter.

    Pass exactly one input source: ``input_text``, ``input_path``, or ``producer_command``
    (executed once inside the isolated sandbox with combined stdout/stderr checked).
    When ``FileCheck`` is unavailable, ``execute_suite`` Gate 3 returns ``RunStatus.SKIP``.
    """
    sources = sum(x is not None for x in (input_text, input_path, producer_command))
    if sources != 1:
        raise ValueError("pass exactly one of input_text, input_path, or producer_command")
    if producer_command is not None and not list(producer_command):
        raise ValueError("producer_command must not be empty")

    project_root = Path(project_root).expanduser().resolve()
    evidence_dir = Path(evidence_dir)
    work_dir = evidence_dir / "filecheck"
    work_dir.mkdir(parents=True, exist_ok=True)
    slug = _safe_slug(name)

    output_file: Optional[Path] = None
    if input_text is not None:
        output_file = work_dir / f"{slug}.input.txt"
        output_file.write_text(input_text, encoding="utf-8")
    elif input_path is not None:
        output_file = Path(str(input_path)).expanduser().resolve()
        if not output_file.is_file():
            raise FileNotFoundError(f"output file not found: {output_file}")

    filecheck = str(executable) if executable else find_filecheck()
    required_bin = filecheck if filecheck else "FileCheck"

    check_dir = work_dir / slug
    check_dir.mkdir(parents=True, exist_ok=True)
    steps = build_check_steps(contract)
    script_lines = [
        "#!/bin/sh",
        "# Generated by hermes_testbench.adapters.filecheck; do not edit.",
        "set -u",
        "rc=0",
    ]
    if producer_command is not None:
        script_lines.append('input_file="$TMPDIR/fc_captured.txt"')
        script_lines.append(f'{shlex.join(list(producer_command))} >"$input_file" 2>&1 || rc=$?')
        input_ref = "$input_file"
    else:
        assert output_file is not None
        input_ref = str(output_file)

    exec_str = filecheck if filecheck else "FileCheck"
    for index, step in enumerate(steps):
        check_file = check_dir / f"step{index}.check"
        check_file.write_text(step.check_text, encoding="utf-8")
        if producer_command is not None:
            extra = (" " + shlex.join(step.extra_args)) if step.extra_args else ""
            script_lines.append(
                f'{shlex.quote(exec_str)}{extra} --input-file="{input_ref}" {shlex.quote(str(check_file))} || rc=1'
            )
        else:
            assert output_file is not None
            argv = filecheck_command(
                step, input_path=output_file, check_file=check_file, executable=exec_str
            )
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
        required_binaries=[required_bin],
        description=(
            f"output contract ({contract.assertion_count()} assertion(s), "
            f"{len(steps)} FileCheck step(s))"
        ),
    )

    from . import execute_suite

    result, _usage = execute_suite(
        suite,
        project_root=project_root,
        evidence_dir=evidence_dir,
        run_id=run_id,
        scratch_root=scratch_root,
    )
    if result.status != "SKIP":
        result.metrics.update(
            {
                "output_contract_engine": "filecheck",
                "output_contract_assertions": contract.assertion_count(),
                "output_contract_steps": len(steps),
                "filecheck_binary": exec_str,
                "check_dir": str(check_dir),
            }
        )
    return result
