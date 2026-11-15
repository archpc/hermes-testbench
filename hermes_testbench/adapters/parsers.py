"""Pure stdout/stderr parsers for every native runner adapter.

Each parser consumes already-captured text only: no I/O, no spawning, no re-runs.
Parsers never invent counts or model usage: anything not present in the captured
text stays zero/``None`` so the caller can reconcile against the exit status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Tuple

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

_COUNT_PATTERNS: Dict[str, re.Pattern[str]] = {
    "passed": re.compile(r"(\d+)\s+(?:tests?\s+)?passed\b", re.IGNORECASE),
    "failed": re.compile(r"(\d+)\s+failed\b", re.IGNORECASE),
    "skipped": re.compile(r"(\d+)\s+(?:skipped|ignored)\b", re.IGNORECASE),
    "errors": re.compile(r"(\d+)\s+errors?\b", re.IGNORECASE),
}

# ``pytest -q`` prints its final summary without the ``=====`` rule, e.g.
# "1 failed, 2 passed, 1 skipped, 1 xfailed, 1 error in 0.01s". Match the whole
# line against pytest's grammar so prose that merely mentions counts never qualifies.
_PYTEST_QUIET_SUMMARY_RE = re.compile(
    r"^\d+ (?:passed|failed|skipped|deselected|xfailed|xpassed|warnings?|errors?|rerun)"
    r"(?:, \d+ (?:passed|failed|skipped|deselected|xfailed|xpassed|warnings?|errors?|rerun))*"
    r" in \d+(?:\.\d+)?s(?: \([0-9:.]+\))?$"
)

_SUMMARY_HINT_RE = re.compile(r"\d+\s+(?:tests?\s+)?(?:passed|failed|skipped|errors?)", re.IGNORECASE)
_PASS_MARKER_RE = re.compile(r"^(?:PASS|TB-PASS|OK)(?::|\s)\s*(.*)$")
_BARE_PASS_RE = re.compile(r"^(?:PASS|TB-PASS|OK)\s*$")
_FAIL_MARKER_RE = re.compile(r"^(?:FAIL|TB-FAIL|FAILED)(?::|\s)\s*(.*)$")
_BARE_FAIL_RE = re.compile(r"^(?:FAIL|TB-FAIL|FAILED)\s*$")
_SKIP_MARKER_RE = re.compile(r"^SKIP(?::|\s)\s*(.*)$")
_BARE_SKIP_RE = re.compile(r"^SKIP\s*$")

# A list item must carry an explicit bullet or number: an indented continuation
# line (traceback frame, assertion detail) is part of one failure, not a new one.
_BULLET_ITEM_RE = re.compile(r"^(?:[-*]|\d+[.)])\s+(\S.*)$")
_ANY_MARKER_RE = re.compile(r"^(?:PASS|TB-PASS|OK|FAIL|TB-FAIL|FAILED|SKIP)(?::|\s|$)")
_EXC_LINE_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_.]*\.)*[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Failure|Exit)(?::|\b)"
)

_SANITIZER_DIAG_RES = (
    re.compile(r"^\s*(?:==\d+==)?ERROR:\s*AddressSanitizer:\s*(?P<msg>.*)$"),
    re.compile(r"^\s*(?:==\d+==)?ERROR:\s*LeakSanitizer:\s*(?P<msg>.*)$"),
    re.compile(r"^\s*.*?runtime error:\s*(?P<msg>.*)$"),
)


@dataclass
class ParseOutcome:
    """Counts and detail extracted from one captured run."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    failures: List[Dict[str, Any]] = field(default_factory=list)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    summary_line: str = ""
    usage: Optional[Dict[str, Any]] = None
    structured: bool = False


def _clean(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def _failure(case_name: str, kind: str, message: str, excerpt: str = "") -> Dict[str, Any]:
    return {
        "case_name": case_name or "<run>",
        "kind": kind,
        "message": message.strip()[:800],
        "file": None,
        "line": None,
        "excerpt": (excerpt or message).strip()[:800],
    }


def _summarize_markers(text: str) -> Tuple[int, int, int, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Count PASS/FAIL/SKIP marker lines and build failure/case records."""
    passed = failed = skipped = 0
    failures: List[Dict[str, Any]] = []
    cases: List[Dict[str, Any]] = []
    raw_lines = text.splitlines()
    idx = 0
    while idx < len(raw_lines):
        raw_line = raw_lines[idx]
        line = raw_line.strip()
        if not line:
            idx += 1
            continue
        match = _FAIL_MARKER_RE.match(line)
        if match or _BARE_FAIL_RE.match(line):
            inline = (match.group(1) if match else "").strip()
            if not inline:
                block: List[str] = []
                look = idx + 1
                while look < len(raw_lines):
                    cand = raw_lines[look]
                    if not cand.strip() or not cand[:1].isspace():
                        break
                    # Indented markers are their own cases, never excerpt text.
                    if _ANY_MARKER_RE.match(cand.strip()):
                        break
                    block.append(cand.strip())
                    look += 1
                bullets = [bm.group(1).strip() for bm in map(_BULLET_ITEM_RE.match, block) if bm]
                # A list only when the block opens with a bullet item; bullet-shaped
                # lines later in a block (diff "- "/"+ " lines, stack frames) are excerpt.
                if bullets and _BULLET_ITEM_RE.match(block[0]):
                    for item in bullets:
                        failed += 1
                        cname = f"failure_{failed}"
                        failures.append(_failure(cname, "assertion", item))
                        cases.append({"name": cname, "status": "FAIL", "message": item})
                    idx = look
                    continue
                if block:
                    excerpt = "\n".join(block)
                    msg = next((part for part in reversed(block) if _EXC_LINE_RE.match(part)), block[-1])
                    name = f"case_{failed + 1}"
                    failed += 1
                    failures.append(_failure(name, "assertion", msg, excerpt))
                    cases.append({"name": name, "status": "FAIL", "message": msg})
                    idx = look
                    continue
            name = inline or f"case_{failed + 1}"
            msg = inline or line
            failed += 1
            failures.append(_failure(name, "assertion", msg))
            cases.append({"name": name, "status": "FAIL", "message": msg})
            idx += 1
            continue
        match = _SKIP_MARKER_RE.match(line)
        if match or _BARE_SKIP_RE.match(line):
            name = (match.group(1) if match else "").strip() or f"case_{skipped + 1}"
            skipped += 1
            cases.append({"name": name, "status": "SKIP", "message": name})
            idx += 1
            continue
        match = _PASS_MARKER_RE.match(line)
        if match or _BARE_PASS_RE.match(line):
            name = (match.group(1) if match else "").strip() or f"case_{passed + 1}"
            passed += 1
            cases.append({"name": name, "status": "PASS", "message": ""})
        idx += 1
    return passed, failed, skipped, 0, failures, cases


def _summary_counts(text: str) -> Tuple[Dict[str, int], str]:
    """Return counts from the last summary-looking line, plus that line."""
    best: Dict[str, int] = {}
    best_line = ""
    for raw_line in text.splitlines():
        line = _ANSI_RE.sub("", raw_line).strip()
        if not line or not _SUMMARY_HINT_RE.search(line):
            continue
        found: Dict[str, int] = {}
        for key, pattern in _COUNT_PATTERNS.items():
            match = pattern.search(line)
            if match:
                found[key] = int(match.group(1))
        if found:
            best, best_line = found, line
    return best, best_line


def _counts_from_summary(counts: Dict[str, int]) -> Tuple[int, int, int, int, int]:
    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0)
    skipped = counts.get("skipped", 0)
    errors = counts.get("errors", 0)
    total = passed + failed + skipped + errors
    return total, passed, failed, skipped, errors


def _apply_summary(outcome: ParseOutcome, counts: Dict[str, int], line: str) -> ParseOutcome:
    total, passed, failed, skipped, errors = _counts_from_summary(counts)
    outcome.total = max(outcome.total, total)
    outcome.passed = max(outcome.passed, passed)
    outcome.failed = max(outcome.failed, failed)
    outcome.skipped = max(outcome.skipped, skipped)
    outcome.errors = max(outcome.errors, errors)
    if line:
        outcome.summary_line = line
    outcome.structured = True
    return outcome


def _apply_markers(outcome: ParseOutcome, text: str) -> bool:
    passed, failed, skipped, errors, failures, cases = _summarize_markers(text)
    if not (passed or failed or skipped or errors):
        return False
    outcome.passed += passed
    outcome.failed += failed
    outcome.skipped += skipped
    outcome.errors += errors
    outcome.total += passed + failed + skipped + errors
    outcome.failures.extend(failures)
    outcome.cases.extend(cases)
    return True


def parse_python_unittest(text: str) -> ParseOutcome:
    text = _clean(text)
    outcome = ParseOutcome()
    ran = re.search(r"^Ran (\d+) tests? in [^\r\n]+", text, re.MULTILINE)
    total = int(ran.group(1)) if ran else 0

    ok_match = re.search(r"^OK(?: \(([^)]*)\))?\s*$", text, re.MULTILINE)
    failed_match = re.search(r"^FAILED(?: \(([^)]*)\))?\s*$", text, re.MULTILINE)
    if not ran and not ok_match and not failed_match:
        return outcome

    clause = ""
    if ok_match:
        clause = ok_match.group(1) or ""
        outcome.summary_line = ok_match.group(0).strip()
    elif failed_match:
        clause = failed_match.group(1) or ""
        outcome.summary_line = failed_match.group(0).strip()
    if ran:
        outcome.summary_line = (ran.group(0).strip() + (" " + outcome.summary_line if outcome.summary_line else "")).strip()

    def _clause_int(key: str) -> int:
        match = re.search(rf"{key}=(\d+)", clause)
        return int(match.group(1)) if match else 0

    skipped = _clause_int("skipped")
    failed = _clause_int("failures")
    errors = _clause_int("errors")
    outcome.total = total
    outcome.skipped = skipped
    outcome.failed = failed
    outcome.errors = errors
    outcome.passed = max(0, total - skipped - failed - errors)

    header_re = re.compile(r"^(FAIL|ERROR): (\S+) \((.*)\)\s*$")
    separator_re = re.compile(r"^(?:={5,}|-{5,})$")
    verbose_re = re.compile(
        r"^(\S+) \(([^)]+)\) \.\.\. (ok|FAIL|ERROR|skipped(?:\s+(?:['\"].*['\"]|\S.*))?|expected failure|unexpected success)\s*$"
    )
    lines = text.splitlines()
    fail_details: Dict[str, Tuple[str, str]] = {}
    ordered_fail_names: List[str] = []
    for idx, line in enumerate(lines):
        header = header_re.match(line.strip())
        if not header:
            continue
        status, func_name, loc = header.groups()
        display = f"{func_name} ({loc})"
        end = idx + 1
        while end < len(lines):
            nxt = lines[end].strip()
            if header_re.match(nxt) or re.match(r"^(?:Ran \d+ tests?|OK|FAILED)\b", nxt):
                break
            end += 1
        block_lines = [part for part in lines[idx + 1 : end] if part.strip() and not separator_re.match(part.strip())]
        block = "\n".join(block_lines)
        exc_line = next((part.strip() for part in reversed(block_lines) if _EXC_LINE_RE.match(part.strip())), "")
        message = exc_line or (block_lines[-1].strip() if block_lines else status)
        kind = "assertion" if status == "FAIL" else "harness_error"
        outcome.failures.append(_failure(display, kind, message, block or line))
        fail_details[display] = (status, message)
        ordered_fail_names.append(display)

    seen_cases: set[str] = set()
    for line in lines:
        vm = verbose_re.match(line.strip())
        if not vm:
            continue
        func_name, loc, raw_state = vm.groups()
        display = f"{func_name} ({loc})"
        if display in seen_cases:
            continue
        seen_cases.add(display)
        if display in fail_details:
            st, msg = fail_details[display]
            outcome.cases.append({"name": display, "status": st, "message": msg})
        elif raw_state.startswith("skipped"):
            reason = raw_state[len("skipped") :].strip().strip("'\"")
            outcome.cases.append({"name": display, "status": "SKIP", "message": reason})
        elif raw_state == "FAIL":
            outcome.cases.append({"name": display, "status": "FAIL", "message": "FAIL"})
        elif raw_state == "ERROR":
            outcome.cases.append({"name": display, "status": "ERROR", "message": "ERROR"})
        else:
            outcome.cases.append({"name": display, "status": "PASS", "message": ""})

    for display in ordered_fail_names:
        if display not in seen_cases:
            seen_cases.add(display)
            st, msg = fail_details[display]
            outcome.cases.append({"name": display, "status": st, "message": msg})

    outcome.structured = True
    return outcome


def parse_python_pytest(text: str) -> ParseOutcome:
    text = _clean(text)
    outcome = ParseOutcome()
    counts: Dict[str, int] = {}
    summary_line = ""
    for line in text.splitlines():
        stripped = line.strip()
        delimited = stripped.startswith("=") and stripped.endswith("=") and " in " in stripped
        if not delimited and not _PYTEST_QUIET_SUMMARY_RE.match(stripped):
            continue
        found: Dict[str, int] = {}
        for key, pattern in _COUNT_PATTERNS.items():
            match = pattern.search(stripped)
            if match:
                found[key] = int(match.group(1))
        if found:
            counts, summary_line = found, stripped
    if counts:
        _apply_summary(outcome, counts, summary_line)

    seen_cases: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"^(FAILED|ERROR) (\S+)(?: - (.*))?$", line.strip())
        if not match:
            continue
        status, name, message = match.groups()
        kind = "assertion" if status == "FAILED" else "harness_error"
        norm_status = "FAIL" if status == "FAILED" else "ERROR"
        outcome.failures.append(_failure(name, kind, message or status))
        if name not in seen_cases:
            seen_cases.add(name)
            outcome.cases.append({"name": name, "status": norm_status, "message": message or ""})
        outcome.structured = True
        if status == "FAILED":
            outcome.failed = max(outcome.failed, 1)
        else:
            outcome.errors = max(outcome.errors, 1)

    verbose_pt_re = re.compile(r"^(\S+::\S+)\s+(PASSED|SKIPPED)(?:\s+\[[^\]]*\])?\s*$")
    for line in text.splitlines():
        vm = verbose_pt_re.match(line.strip())
        if not vm:
            continue
        name, raw_st = vm.groups()
        if name in seen_cases:
            continue
        seen_cases.add(name)
        outcome.cases.append({"name": name, "status": "PASS" if raw_st == "PASSED" else "SKIP", "message": ""})

    if (outcome.failed or outcome.errors) and outcome.total < outcome.failed + outcome.errors:
        outcome.total = outcome.failed + outcome.errors
    return outcome


def parse_cargo(text: str) -> ParseOutcome:
    text = _clean(text)
    outcome = ParseOutcome()
    block_re = re.compile(
        r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;"
    )
    blocks = 0
    for line in text.splitlines():
        match = block_re.match(line.strip())
        if not match:
            continue
        blocks += 1
        passed, failed, ignored = (int(g) for g in match.groups())
        outcome.passed += passed
        outcome.failed += failed
        outcome.skipped += ignored
        outcome.total += passed + failed + ignored
    case_re = re.compile(r"^test (\S+) \.\.\. (ok|FAILED|ignored)$")
    for line in text.splitlines():
        match = case_re.match(line.strip())
        if match:
            name, state = match.groups()
            if state == "FAILED":
                outcome.failures.append(_failure(name, "assertion", name))
                outcome.cases.append({"name": name, "status": "FAIL", "message": name})
            elif state == "ignored":
                outcome.cases.append({"name": name, "status": "SKIP", "message": ""})
            else:
                outcome.cases.append({"name": name, "status": "PASS", "message": ""})
    if blocks:
        outcome.structured = True
        outcome.summary_line = f"{blocks} cargo test result block(s) aggregated"
    elif re.search(r"^error(?:\[E\d+\])?: could not compile", text, re.MULTILINE):
        outcome.failures.append(
            _failure("cargo-compile", "harness_error", "cargo build failed before tests ran")
        )
        outcome.total = max(outcome.total, 1)
        outcome.failed = max(outcome.failed, 1)
        outcome.structured = True
        outcome.summary_line = "cargo compile failure"
    return outcome


def parse_cobol(text: str) -> ParseOutcome:
    cleaned = _clean(text)
    if re.search(r"^Ran \d+ tests? in ", cleaned, re.M):
        return parse_python_unittest(cleaned)
    text = cleaned
    outcome = ParseOutcome()
    markers = _apply_markers(outcome, text)
    success_re = re.compile(r"\b(?:ALL TESTS PASSED|TESTS PASSED|SELFTEST OK)\b")
    success = bool(success_re.search(text))
    counts, line = _summary_counts(text)
    if counts:
        _apply_summary(outcome, counts, line)
    if not markers and success and not outcome.failed:
        outcome.total = max(outcome.total, 1)
        outcome.passed = max(outcome.passed, 1)
    if success and outcome.failed:
        outcome.summary_line = "cobol selftest reported success but also failures"
    outcome.structured = outcome.structured or success or markers
    return outcome


def parse_c_sanitizer(text: str) -> ParseOutcome:
    text = _clean(text)
    outcome = ParseOutcome()
    diagnostics = 0
    for line in text.splitlines():
        stripped = line.strip()
        for pattern in _SANITIZER_DIAG_RES:
            match = pattern.match(stripped)
            if not match:
                continue
            diagnostics += 1
            message = stripped[:300]
            outcome.failures.append(_failure("sanitizer", "harness_error", message, stripped))
            outcome.cases.append({"name": "sanitizer", "status": "FAIL", "message": message})
            break
    if diagnostics:
        outcome.failed = max(outcome.failed, diagnostics)
        outcome.total = max(outcome.total, outcome.failed)

    markers = _apply_markers(outcome, text)
    counts, line = _summary_counts(text)
    if counts:
        _apply_summary(outcome, counts, line)
    success = bool(re.search(r"\b(?:SELFTEST OK|ALL TESTS PASSED|TESTS PASSED)\b", text))
    if not markers and not counts and success and not outcome.failed:
        outcome.total = max(outcome.total, 1)
        outcome.passed = max(outcome.passed, 1)
    if diagnostics:
        outcome.summary_line = outcome.summary_line or "sanitizer diagnostics detected"
        outcome.structured = True
    outcome.structured = outcome.structured or success or markers
    return outcome


def parse_shell(text: str) -> ParseOutcome:
    text = _clean(text)
    outcome = ParseOutcome()
    markers = _apply_markers(outcome, text)
    counts, line = _summary_counts(text)
    if counts:
        _apply_summary(outcome, counts, line)
    outcome.structured = outcome.structured or markers
    return outcome


def parse_re_project(text: str) -> ParseOutcome:
    text = _clean(text)
    outcome = ParseOutcome()
    markers = _apply_markers(outcome, text)
    counts, line = _summary_counts(text)
    if counts:
        if markers:
            # Marker lines are ground truth for per-case results; adopt the
            # summary total only when it declares more cases than we saw.
            declared_total, _p, _f, _s, _e = _counts_from_summary(counts)
            if declared_total > outcome.total:
                outcome.total = declared_total
            outcome.summary_line = line
            outcome.structured = True
        else:
            _apply_summary(outcome, counts, line)
    outcome.structured = outcome.structured or markers
    return outcome


def parse_sqlite_probe(text: str) -> ParseOutcome:
    text = _clean(text)
    outcome = ParseOutcome()
    markers = _apply_markers(outcome, text)
    counts, line = _summary_counts(text)
    if counts:
        _apply_summary(outcome, counts, line)
    outcome.structured = outcome.structured or markers
    return outcome


def parse_pty_tmux(text: str) -> ParseOutcome:
    text = _clean(text)
    outcome = ParseOutcome()
    markers = _apply_markers(outcome, text)
    counts, line = _summary_counts(text)
    if counts:
        _apply_summary(outcome, counts, line)
    outcome.structured = outcome.structured or markers
    return outcome


_USAGE_RE = re.compile(r"TESTBENCH_MODEL_USAGE=(\{.*\})\s*$")


def parse_agent_eval(text: str) -> ParseOutcome:
    text = _clean(text)
    outcome = ParseOutcome()
    for line in text.splitlines():
        match = _USAGE_RE.search(line.strip())
        if not match:
            continue
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            outcome.usage = payload  # only ever taken verbatim from the run
    markers = _apply_markers(outcome, text)
    counts, line = _summary_counts(text)
    if counts:
        _apply_summary(outcome, counts, line)
    outcome.structured = outcome.structured or markers
    return outcome


def parse_generic(text: str) -> ParseOutcome:
    text = _clean(text)
    outcome = ParseOutcome()
    markers = _apply_markers(outcome, text)
    counts, line = _summary_counts(text)
    if counts:
        _apply_summary(outcome, counts, line)
    if not outcome.structured and "Traceback (most recent call last)" in text:
        outcome.total = max(outcome.total, 1)
        outcome.failed = max(outcome.failed, 1)
        outcome.failures.append(_failure("traceback", "harness_error", "unhandled traceback"))
        outcome.structured = True
    outcome.structured = outcome.structured or markers
    return outcome


ADAPTER_PARSERS = {
    "agent_eval": parse_agent_eval,
    "c_sanitizer": parse_c_sanitizer,
    "cargo": parse_cargo,
    "cobol": parse_cobol,
    "pty_tmux": parse_pty_tmux,
    "python_pytest": parse_python_pytest,
    "python_unittest": parse_python_unittest,
    "re_project": parse_re_project,
    "shell": parse_shell,
    "sqlite_probe": parse_sqlite_probe,
}
