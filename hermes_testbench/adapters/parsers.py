"""Pure stdout/stderr parsers for every native runner adapter.

Each parser consumes already-captured text only: no I/O, no spawning, no re-runs.
Parsers never invent counts or model usage: anything not present in the captured
text stays zero/``None`` so the caller can reconcile against the exit status.

The one promise a parser makes is ``ParseOutcome.structured``: ``True`` means the
text contained something this parser recognises as a runner summary or marker set,
so ``adapters.execute_suite`` trusts the counts and only cross-checks them against
the exit status (see ``adapters._finalize_counts``). ``False`` means "nothing here
I understand" and the caller must fall back to the exit status alone. Returning
``structured=True`` with no cases, or raising on odd input, breaks that contract.

Shared marker grammar (``_summarize_markers``)
---------------------------------------------
``PASS`` / ``TB-PASS`` / ``OK``, ``FAIL`` / ``TB-FAIL`` / ``FAILED`` and ``SKIP``
at the start of a stripped line, case-sensitive on purpose: an ``ok:`` from a JSON
log or a lowercase ``skip`` is not a verdict. An inline marker is one complete
case; a bare marker owns the indented block under it, and that block becomes
several failures only when it *opens* with a bullet or numbered item.

What the shared marker scanner silently drops (by design, worth knowing before reading a
run): TAP output (``ok 1 - name``), ``ERROR`` marker lines (``errors`` is filled from a
numeric summary, or by ``parse_python_pytest`` from its own ``ERROR`` lines), lowercase
``ok``/``skip``, and any count line that does not match ``_COUNT_PATTERNS``. Dropped
detail never becomes a fabricated pass — it becomes "unstructured", i.e. an exit-status
verdict.

Counting is monotonic per field: ``_apply_summary`` takes the maximum of what
markers already found and what the summary declares. ``total`` is therefore not
guaranteed to equal ``pass + fail + skip + error`` when a log contains text from
two different runs; treat the per-field counts as the more specific truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..schema import FailureKind, RunStatus

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
    """Counts and detail extracted from one captured run.

    ``total``/``passed``/``failed``/``skipped``/``errors`` are what lands on
    ``SuiteResult``; ``failures`` and ``cases`` are dicts (not schema objects) so a
    parser can stay free of imports from ``..schema``; the caller materialises them.
    ``summary_line`` becomes ``SuiteResult.summary_excerpt`` when non-empty, which is
    why parsers prefer the runner's own words over a synthesised sentence.

    ``usage`` is only ever the verbatim ``TESTBENCH_MODEL_USAGE=<json>`` payload a
    billable runner printed (``parse_agent_eval``); nothing estimates it.

    ``buckets``/``baseline_failed`` exist for the two mutation adapters only. They
    are kept separate from the five count fields so a mutation run can expose
    ``metrics["mutation"]`` without pretending ``invalid``/``timeout`` are skips in
    the generic sense (``_sync_mutation_counts`` maps them afterwards).
    """

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
    # Mutation adapters: canonical buckets plus a baseline-failure marker.
    buckets: Dict[str, int] = field(default_factory=dict)
    baseline_failed: bool = False


def _clean(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def _failure(case_name: str, kind: str, message: str, excerpt: str = "") -> Dict[str, Any]:
    """Materialise one failure dict, bounded to 800 chars per text field.

    ``file``/``line`` are always present-and-``None`` here; parsers that know a
    source location (FileCheck, ``cobc``) build their dicts directly instead.
    """
    return {
        "case_name": case_name or "<run>",
        "kind": kind,
        "message": message.strip()[:800],
        "file": None,
        "line": None,
        "excerpt": (excerpt or message).strip()[:800],
    }


def _summarize_markers(text: str) -> Tuple[int, int, int, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Count PASS/FAIL/SKIP marker lines and build failure/case records.

    Returns ``(passed, failed, skipped, errors, failures, cases)``. ``errors`` is
    always 0: this scanner has no ``ERROR`` marker, so harness errors are only
    visible through a numeric summary line. A bare FAIL owns the following indented
    non-blank block; if the block starts with a bullet/numbered item each item is a
    failure, otherwise the whole block is one failure's excerpt and the message is
    the most specific line in it (a reversal scan for ``*Error:``-shaped text, else
    the last line). Case names are synthesised (``failure_N``/``case_N``) when the
    runner supplied none, so they are stable per position, not per test identity.
    """
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
    """Return counts from the last summary-looking line, plus that line.

    "Last, not first" is deliberate: a suite that runs the real command twice (a
    smoke pass then a strict pass) should be judged by its final verdict. The line
    must contain a ``_SUMMARY_HINT_RE`` hit — ``<n> passed|failed|skipped|errors`` —
    which is what keeps prose ("3 tests failed to start") from being read as a
    summary. A line that mentions only ``warnings`` yields no counts at all.
    """
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
    """Parse ``unittest`` output: ``Ran N tests``, ``OK``/``FAILED (...)``, verbose lines.

    Counts come from the ``Ran`` header plus the ``OK``/``FAILED`` clause; ``passed``
    is *derived* (total - skipped - failures - errors), so a runner whose header and
    clause disagree still yields a self-consistent set.

    Failure detail is read twice on purpose: from ``FAIL: name (loc)`` / ``ERROR: ...``
    headers (which own the traceback block below them) and from verbose
    ``name (loc) ... ok|FAIL|skipped 'reason'`` lines. Cases are keyed by
    ``name (loc)`` and headers win over verbose lines, which is what makes
    ``-v`` output produce one case per test rather than two.

    Dropped: anything before the first header, ``expected failure`` /
    ``unexpected success`` verbose states (they land in the PASS branch, so an
    unexpected success is counted as a pass), and runner text that is not one of the
    two header forms.
    """
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
    """Parse ``pytest`` output: the final summary rule, ``FAILED``/``ERROR`` lines, ``-v``.

    The summary is accepted from either the ``===== 1 failed, 2 passed in 0.1s =====``
    rule or a bare ``-q`` line, and only when the whole line matches pytest's own
    grammar (``_PYTEST_QUIET_SUMMARY_RE``) — a log line that merely mentions counts
    does not qualify. The last matching line wins.

    ``FAILED path::test - msg`` / ``ERROR ...`` short-summary lines are the only
    source of failure records, and each one raises ``failed``/``errors`` to at least
    1 so a summary that undercounts cannot report zero failures. ``-v`` output adds
    PASS/SKIP cases; plain output produces failure cases only.

    Dropped: ``no tests ran in ...`` (no digits to match, so the run is unstructured
    and the exit status 5 decides), ``xfail``/``xpass`` markers, warning blocks, and
    the traceback body — the short summary line is the message, not the traceback.
    """
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
    """Parse ``cargo test`` output by summing every ``test result:`` block.

    Cargo prints one block per test binary (lib, each integration test), so counts
    are *summed* rather than taken from the last block; ``blocks`` drives
    ``summary_line`` and ``structured``. ``ignored`` is the skip bucket — cargo has
    no separate "skipped" vocabulary.

    ``test <name> ... FAILED`` lines add the failure records, but a compile error
    (``error[E0xxx]: could not compile``) with no result block is also a structured
    outcome with one synthetic ``cargo-compile`` failure: a build failure must not
    degrade to "exit status 1" with no attributable reason.

    Dropped: doctest blocks (``test src/lib.rs - foo (line 3) ... ok`` has extra text
    between the name and ``...``, so it does not match the per-case regex),
    ``failures:`` sections, and any ``test result:`` line whose shape drifts from
    ``ok|FAILED. N passed; N failed; N ignored;`` — that last one is the version-drift
    risk: nextest-style summaries land in the unstructured bucket instead of being
    mis-counted.
    """
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


_COBOL_DIAG_RE = re.compile(
    r"^(?P<file>\S+\.(?:cob|cbl|cpy)):(?P<line>\d+):\s*"
    r"(?P<severity>error|warning|note):\s*(?P<message>.+)$"
)


def _cobol_compile_errors(text: str) -> List[Dict[str, Any]]:
    """``cobc`` errors as ``file:line: error: message`` records.

    Only used when no harness marker supplied per-case results, so a harness that
    prints its own TB-PASS/TB-FAIL lines stays the single source of truth.
    """
    found: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        match = _COBOL_DIAG_RE.match(raw.strip())
        if not match or match.group("severity") != "error":
            continue
        found.append(
            {
                "file": match.group("file"),
                "line": int(match.group("line")),
                "message": match.group("message").strip(),
                "rendered": raw.strip()[:300],
            }
        )
    return found


def _cobol_inline_details(text: str) -> Dict[str, List[str]]:
    """Map an inline ``TB-FAIL <label>`` marker to the indented block under it.

    The shared marker scanner treats an inline marker as a complete one-line
    failure, which is right for ``cargo``/``shell`` output. GnuCOBOL verification
    writes the label inline and the evidence (expected vs actual, compiler
    diagnostics) as an indented block, so the block is folded back into the same
    failure here instead of being lost.
    """
    details: Dict[str, List[str]] = {}
    pending: Optional[str] = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if raw[:1].isspace():
            if pending is None:
                continue
            if _ANY_MARKER_RE.match(stripped):
                pending = None
                continue
            details.setdefault(pending, []).append(stripped)
            continue
        match = _FAIL_MARKER_RE.match(stripped)
        pending = match.group(1).strip() if match and match.group(1).strip() else None
    return details


def _apply_cobol_details(outcome: ParseOutcome, text: str) -> None:
    """Fold indented evidence into the failure whose inline marker owns it."""
    details = _cobol_inline_details(text)
    if not details:
        return
    for failure in outcome.failures:
        block = details.get(str(failure.get("case_name", "")))
        if not block:
            continue
        failure["excerpt"] = "\n".join(block)[:800]
        # The most specific line of the block is the actionable reason.
        failure["message"] = block[-1][:800]


def parse_cobol(text: str) -> ParseOutcome:
    """Parse GnuCOBOL verification output: harness markers first, compiler text second.

    Precedence is the whole contract here, because ``cobol_check`` deliberately
    prints its own ``TB-PASS``/``TB-FAIL`` lines (the "flagship" path) while a plain
    ``cobc``/selftest script prints only compiler output:

    1. ``Ran N tests in ...`` means the suite ran a Python unittest harness —
       delegate wholesale to ``parse_python_unittest``.
    2. Harness markers win. When any marker is present, ``cobc`` diagnostics are
       *not* mined (``_cobol_compile_errors`` is skipped) so one failure is not
       reported twice: once as a marker and once as a compiler error.
    3. Marker failure excerpts are refolded from the indented block under an inline
       ``TB-FAIL <label>`` (``_apply_cobol_details``), because the shared scanner
       treats an inline marker as a single-line failure.
    4. No markers: a ``ALL TESTS PASSED``/``TESTS PASSED``/``SELFTEST OK`` banner is
       one pass; ``cobc`` ``file:line: error:`` lines become attributable failures
       with the diagnostic as the message.

    Dropped: ``cobc`` warnings/notes (severity filter), diagnostics that do not
    start with a ``.cob/.cbl/.cpy`` path, and everything past the tenth compile
    error — ``failed`` still counts all of them, so ``failed`` can exceed
    ``len(failures)`` on a very broken build.
    """
    cleaned = _clean(text)
    if re.search(r"^Ran \d+ tests? in ", cleaned, re.M):
        return parse_python_unittest(cleaned)
    text = cleaned
    outcome = ParseOutcome()
    markers = _apply_markers(outcome, text)
    if markers:
        _apply_cobol_details(outcome, text)
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
    compile_errors = [] if markers else _cobol_compile_errors(text)
    if compile_errors and not outcome.failed:
        # A bare cobc failure has no runner summary: turn its diagnostics into
        # attributable failures instead of a generic "exit status 1".
        for diag in compile_errors[:10]:
            outcome.failures.append(
                _failure(
                    f"cobol:{diag['file']}:{diag['line']}",
                    "harness_error",
                    f"cobc error: {diag['message']}",
                    diag["rendered"],
                )
            )
            outcome.cases.append(
                {
                    "name": f"{diag['file']}:{diag['line']}",
                    "status": "FAIL",
                    "message": f"cobc error: {diag['message']}",
                }
            )
        outcome.failed = max(outcome.failed, len(compile_errors))
        outcome.total = max(outcome.total, outcome.failed)
        outcome.summary_line = f"cobc compile failed: {compile_errors[0]['rendered']}"
    outcome.structured = outcome.structured or success or markers or bool(compile_errors)
    return outcome


def parse_c_sanitizer(text: str) -> ParseOutcome:
    """Parse AddressSanitizer/LeakSanitizer/UBSan output plus optional markers.

    Each diagnostic line is one failure; ``failed`` is raised to the diagnostic
    count so a program that printed a sanitizer report and exited 0 still fails.
    Marker lines and a summary line are still honoured, which is what lets the
    generated C suite report per-source passes.

    Dropped: sanitizer stack frames and ``SUMMARY:`` blocks (the ``ERROR:`` line is
    the record), and the third pattern matches any line containing
    ``runtime error:`` — including a program's own echo of that phrase, which is
    why this parser is only wired to the ``c_sanitizer`` adapter.
    """
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
    """Parse a ``shell``/``shell_test`` suite: markers, else FileCheck, else summary.

    Order matters. Markers are authoritative, and when any marker is present the
    FileCheck diagnostic miner is skipped entirely — a script that both prints
    ``TB-PASS`` and pipes FileCheck output is trusted to have already summarised
    itself. With no markers, ``parse_filecheck_diagnostics`` owns the counts
    (``failed = total = len(diagnostics)``, overwritten rather than maxed) so an
    output contract reports one failure per assertion instead of one per process.
    Otherwise the generic summary scan applies.

    Dropped: shell exit-status-only scripts stay unstructured (the caller turns that
    into one synthetic failure), and ``set -e`` traces carry no verdicts.
    """
    from .filecheck import parse_filecheck_diagnostics

    text = _clean(text)
    outcome = ParseOutcome()
    markers = _apply_markers(outcome, text)
    if not markers:
        fc_diags = parse_filecheck_diagnostics(text)
        if fc_diags:
            outcome.failures.extend(fc_diags)
            for item in fc_diags:
                outcome.cases.append(
                    {
                        "name": item["case_name"],
                        "status": "FAIL",
                        "message": item["message"],
                        "file": item.get("file"),
                        "line": item.get("line"),
                    }
                )
            outcome.failed = len(fc_diags)
            outcome.total = len(fc_diags)
            first = fc_diags[0]["message"]
            outcome.summary_line = (
                first if len(fc_diags) == 1 else f"{first} (+{len(fc_diags) - 1} more)"
            )[:800]
            outcome.structured = True
            return outcome
    counts, line = _summary_counts(text)
    if counts:
        _apply_summary(outcome, counts, line)
    outcome.structured = outcome.structured or markers
    return outcome


def parse_re_project(text: str) -> ParseOutcome:
    """Parse marker lines plus an optional summary, with markers as ground truth.

    Adopts the summary's *total* only when it declares more cases than the markers
    showed; the per-case counts stay whatever the markers said. That asymmetry is
    the point: a harness that evaluates 40 assertions but prints a partial marker
    set must not have its ``failed`` overwritten by a total.
    """
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
    """Parse a ``sqlite_probe`` log: markers and an optional summary line only."""
    text = _clean(text)
    outcome = ParseOutcome()
    markers = _apply_markers(outcome, text)
    counts, line = _summary_counts(text)
    if counts:
        _apply_summary(outcome, counts, line)
    outcome.structured = outcome.structured or markers
    return outcome


def parse_pty_tmux(text: str) -> ParseOutcome:
    """Parse a ``pty_tmux`` screen dump: markers and an optional summary line.

    The captured text is the PTY transcript (ANSI already stripped by ``_clean``), so
    the same marker grammar applies; there is no attempt to reconstruct screen state.
    """
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
    """Parse a marker/summary log from an agent evaluation.

    The only parser that fills ``usage``: it takes the last
    ``TESTBENCH_MODEL_USAGE=<json object>`` line verbatim. A malformed payload is
    skipped rather than defaulted, so ``usage`` stays ``None`` and the caller has no
    number to bill against.
    """
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
    """Fallback for adapters with no dedicated parser: markers, summary, traceback.

    An unstructured ``Traceback (most recent call last)`` is promoted to one
    harness-error failure so a crashing tool is attributable rather than just a
    nonzero exit. This parser is also what ``execute_suite`` uses for an adapter it
    does not know (``ADAPTER_PARSERS.get(adapter, parse_generic)``), so a manifest
    cannot smuggle in a novel adapter name and get invented counts.
    """
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


MUTATION_BUCKETS = ("caught", "survived", "invalid", "timeout", "inconclusive")

# One canonical case status per bucket. ``caught`` proves the suite noticed the bug;
# ``survived`` is a real gap, everything else is "we could not measure this mutant".
_MUTATION_CASE_STATUS: Dict[str, str] = {
    "caught": "PASS",
    "survived": "FAIL",
    "invalid": "SKIP",
    "timeout": "SKIP",
    "inconclusive": "SKIP",
}

_MUTMUT_STATUS_BUCKET: Dict[str, str] = {
    "killed": "caught",
    "caught by type check": "caught",
    "survived": "survived",
    "timeout": "timeout",
    "no tests": "inconclusive",
    "not checked": "inconclusive",
    "skipped": "inconclusive",
    "suspicious": "inconclusive",
    "check was interrupted by user": "inconclusive",
    "segfault": "inconclusive",
}

_MUTMUT_EMOJI_BUCKET: Dict[str, str] = {
    "🎉": "caught",
    "🧙": "caught",
    "🙁": "survived",
    "⏰": "timeout",
    "🫥": "inconclusive",
    "🤔": "inconclusive",
    "🔇": "inconclusive",
    "🛑": "inconclusive",
    "💥": "inconclusive",
}

_CARGO_OUTCOME_BUCKET: Dict[str, str] = {
    "CaughtMutant": "caught",
    "MissedMutant": "survived",
    "Timeout": "timeout",
    "Unviable": "invalid",
    "UnviableMutant": "invalid",
    "Failure": "inconclusive",
    "Inconclusive": "inconclusive",
}

_CARGO_LINE_BUCKET: Dict[str, str] = {
    "CAUGHT": "caught",
    "MISSED": "survived",
    "TIMEOUT": "timeout",
    "UNVIABLE": "invalid",
    "FAILED": "inconclusive",
}

_CARGO_STATUS_LINE_RE = re.compile(r"^(CAUGHT|MISSED|TIMEOUT|UNVIABLE|FAILED)\s+(\S.*)$")
_CARGO_TALLY_RE = re.compile(
    r"(\d+) (caught|missed|timeout|timed out|unviable|failed)",
    re.IGNORECASE,
)
_MUTTALLY_BUCKET = {
    "caught": "caught",
    "missed": "survived",
    "timeout": "timeout",
    "timed out": "timeout",
    "unviable": "invalid",
    "failed": "inconclusive",
}
_MUTMUT_RESULT_LINE_RE = re.compile(r"^\s*(?P<name>[A-Za-z0-9_.\-]+):\s*(?P<status>[a-z][a-z \t]*[a-z])\s*$")
_MUTMUT_EMOJI_TALLY_RE = re.compile(r"(🎉|🧙|🙁|⏰|🫥|🤔|🔇|🛑|💥)\s*(\d+)")

_CARGO_BASELINE_FAILURE_MARKERS = (
    "cargo test failed in an unmutated tree",
    "no mutants were tested",
)
_MUTMUT_BASELINE_FAILURE_MARKERS = (
    "failed to collect stats",
    "failed to collect tests",
    "failed to run tests",
    "runner returned",
)


def _mutation_empty_buckets() -> Dict[str, int]:
    return {name: 0 for name in MUTATION_BUCKETS}


def _record_mutant(outcome: ParseOutcome, name: str, bucket: str, detail: str = "") -> None:
    outcome.buckets[bucket] = outcome.buckets.get(bucket, 0) + 1
    clean_name = (name or "<mutant>").strip()[:300]
    outcome.cases.append(
        {
            "name": clean_name,
            "status": _MUTATION_CASE_STATUS[bucket],
            "message": (detail or bucket).strip()[:300],
        }
    )


def _sync_mutation_counts(outcome: ParseOutcome) -> None:
    for name in MUTATION_BUCKETS:
        outcome.buckets.setdefault(name, 0)
    outcome.total = sum(outcome.buckets[name] for name in MUTATION_BUCKETS)
    outcome.passed = outcome.buckets["caught"]
    outcome.failed = outcome.buckets["survived"]
    outcome.skipped = (
        outcome.buckets["invalid"] + outcome.buckets["timeout"] + outcome.buckets["inconclusive"]
    )


def _mark_baseline_failed(outcome: ParseOutcome, reason: str) -> None:
    """A broken baseline means no mutant verdict is meaningful: count it as inconclusive."""
    outcome.baseline_failed = True
    outcome.buckets["inconclusive"] = outcome.buckets.get("inconclusive", 0) + 1
    outcome.cases.append(
        {"name": "<baseline>", "status": "SKIP", "message": reason.strip()[:300]}
    )


def parse_cargo_mutants(text: str, *, outcomes_json: str = "") -> ParseOutcome:
    """Classify a ``cargo-mutants`` run from its console log or ``mutants.out/outcomes.json``.

    Two sources, tried in that order:

    1. ``outcomes_json`` — expected to be an object with an ``outcomes`` list whose
       entries carry ``scenario`` (``"Baseline"`` or ``{"Mutant": {"name": ...}}``)
       and ``summary`` (``CaughtMutant`` and friends). Only a payload of that exact
       shape is used; anything else falls through to the console log. A non-``Success``
       baseline marks the whole run ``baseline_failed`` and contributes one
       ``inconclusive`` — no mutant verdict survives a broken baseline.
       UNVERIFIED: whether the ``cargo-mutants`` version on a given host writes that
       object shape or a bare JSON array of outcomes. If it writes an array, the
       precise per-mutant path is dead and the console fallback (below) is what
       classifies the run; ``tests/test_mutation_testing`` only exercises the object
       shape. Resolving it needs one real ``mutants.out/outcomes.json`` from the
       tool version in use.
    2. Console log — ``CAUGHT|MISSED|TIMEOUT|UNVIABLE|FAILED <mutant>`` lines, each
       one a case (the trailing ``in Ns build + Ns test`` timing is stripped from
       the name). Only when no such line exists is the final
       ``N mutants tested in Ns: N caught, N missed, ...`` tally consulted, so a log
       carrying both is classified per mutant and never double-counted.

    Unknown ``summary`` values and unknown keywords go to ``inconclusive``, never to
    ``caught``: an unrecognised verdict can only understate the score.
    """
    cleaned = _clean(text)
    outcome = ParseOutcome()
    outcome.buckets = _mutation_empty_buckets()

    if outcomes_json.strip():
        try:
            payload = json.loads(outcomes_json)
        except (json.JSONDecodeError, ValueError):
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("outcomes"), list):
            for entry in payload["outcomes"]:
                if not isinstance(entry, dict):
                    continue
                scenario = entry.get("scenario")
                summary = str(entry.get("summary", ""))
                if scenario == "Baseline":
                    if summary != "Success":
                        _mark_baseline_failed(outcome, f"unmutated baseline: {summary}")
                    continue
                mutant = scenario.get("Mutant") if isinstance(scenario, dict) else None
                if not isinstance(mutant, dict):
                    continue
                bucket = _CARGO_OUTCOME_BUCKET.get(summary, "inconclusive")
                name = str(mutant.get("name") or "<mutant>")
                _record_mutant(outcome, name, bucket, f"{bucket}: {summary or 'unknown'}")
            if outcome.cases or outcome.baseline_failed:
                outcome.structured = True
                counts = payload.get("total_mutants")
                outcome.summary_line = (
                    f"cargo-mutants: {outcome.buckets['caught']} caught, "
                    f"{outcome.buckets['survived']} survived, "
                    f"{outcome.buckets['invalid']} unviable, "
                    f"{outcome.buckets['timeout']} timeout, "
                    f"{outcome.buckets['inconclusive']} inconclusive"
                    + (f" ({counts} generated)" if isinstance(counts, int) else "")
                )
                _sync_mutation_counts(outcome)
                return outcome

    for line in cleaned.splitlines():
        stripped = line.strip()
        match = _CARGO_STATUS_LINE_RE.match(stripped)
        if match:
            keyword, detail = match.groups()
            if keyword == "FAILED" and "baseline" in detail.lower():
                _mark_baseline_failed(outcome, detail)
                continue
            # Strip the trailing timing so the case name stays the mutant description.
            name = re.sub(r"\s+in \d+(?:\.\d+)?s build.*$", "", detail).strip()
            _record_mutant(outcome, name, _CARGO_LINE_BUCKET[keyword], f"{keyword.lower()}: {name}")
            continue
        if not outcome.cases and _CARGO_TALLY_RE.search(stripped) and "mutant" in stripped.lower():
            for count, kind in _CARGO_TALLY_RE.findall(stripped):
                bucket = _MUTTALLY_BUCKET.get(kind.lower(), "inconclusive")
                outcome.buckets[bucket] = outcome.buckets.get(bucket, 0) + int(count)
            outcome.summary_line = stripped[:300]

    lowered = cleaned.lower()
    if any(marker in lowered for marker in _CARGO_BASELINE_FAILURE_MARKERS):
        # The console log reports the baseline broken without naming a mutant.
        if not outcome.baseline_failed:
            _mark_baseline_failed(outcome, "cargo test failed in an unmutated tree")

    if not outcome.summary_line and outcome.buckets["caught"]:
        outcome.summary_line = (
            f"cargo-mutants: {outcome.buckets['caught']} caught, "
            f"{outcome.buckets['survived']} survived"
        )
    outcome.structured = outcome.structured or bool(outcome.cases) or outcome.baseline_failed
    _sync_mutation_counts(outcome)
    return outcome


def parse_mutmut(text: str, *, results_text: str = "") -> ParseOutcome:
    """Classify a ``mutmut`` run from ``mutmut run`` progress and ``mutmut results`` output.

    ``results_text`` is the authoritative per-mutant source (``mutmut results --all true``
    prints ``<name>: <status>`` lines); when it is empty the run log is used, which is
    also the only source of the emoji tally. Per-mutant lines are matched first; the
    emoji tally is consulted only when *no* per-mutant line was recognised, so a run
    that printed both is never counted twice.

    Unrecognised statuses are dropped (``bucket is None: continue``) rather than guessed
    into a bucket — a mutmut vocabulary change understates a score instead of
    inflating it, and the suite falls to ``inconclusive``/harness error rather than
    a silent pass. A baseline marker (``failed to collect/run tests``) only counts as
    a baseline failure when no per-mutant verdict exists: a partial run that produced
    real verdicts still reports them.
    """
    cleaned = _clean(text)
    results = _clean(results_text) if results_text.strip() else cleaned
    outcome = ParseOutcome()
    outcome.buckets = _mutation_empty_buckets()

    for line in results.splitlines():
        match = _MUTMUT_RESULT_LINE_RE.match(line)
        if not match:
            continue
        name, raw_status = match.group("name"), match.group("status").strip().lower()
        bucket = _MUTMUT_STATUS_BUCKET.get(raw_status)
        if bucket is None:
            continue
        _record_mutant(outcome, name, bucket, f"{raw_status}: {name}")

    if not outcome.cases and cleaned:
        # Fall back to the final progress tally that ``mutmut run`` prints.
        last_tally: Dict[str, int] = {}
        for line in cleaned.splitlines():
            found = _MUTMUT_EMOJI_TALLY_RE.findall(line)
            if found:
                last_tally = {bucket: 0 for bucket in MUTATION_BUCKETS}
                for emoji, count in found:
                    bucket = _MUTMUT_EMOJI_BUCKET.get(emoji, "inconclusive")
                    last_tally[bucket] = last_tally.get(bucket, 0) + int(count)
        for bucket, count in last_tally.items():
            outcome.buckets[bucket] = outcome.buckets.get(bucket, 0) + count
        if last_tally:
            outcome.summary_line = (
                f"mutmut: {outcome.buckets['caught']} caught, "
                f"{outcome.buckets['survived']} survived, "
                f"{outcome.buckets['timeout']} timeout, "
                f"{outcome.buckets['inconclusive']} inconclusive"
            )

    lowered = cleaned.lower()
    if any(marker in lowered for marker in _MUTMUT_BASELINE_FAILURE_MARKERS):
        if not outcome.cases:
            _mark_baseline_failed(outcome, "mutmut could not collect a passing baseline")

    outcome.structured = outcome.structured or bool(outcome.cases) or outcome.baseline_failed
    _sync_mutation_counts(outcome)
    return outcome


_TB_TAPE_MARKER_RE = re.compile(r"^TB-TAPE\s+(\{.*\})\s*$")


def parse_terminal_tape(text: str) -> ParseOutcome:
    """Parse a ``testbench tape run`` marker line into per-check counts.

    The marker is authoritative: it carries the tape name, terminal status, and
    one entry per evaluated expectation. Without a marker the generic
    PASS/FAIL/SKIP marker and summary counters still apply.

    ``TB-TAPE {json}`` is emitted by the tape runner itself, not by the program under
    test, which is why it outranks everything else in the captured PTY text: a
    program that prints ``FAIL`` on screen must not add a failure the tape did not
    record. The last well-formed marker wins (a retry or relaunch prints a second
    one). ``status=blocked`` converts the run to one error and a
    ``BLOCKED_POLICY`` failure rather than a wall of assertion failures.
    """
    text = _clean(text)
    outcome = ParseOutcome()
    payload: Optional[Dict[str, Any]] = None
    for line in text.splitlines():
        match = _TB_TAPE_MARKER_RE.match(line.strip())
        if not match:
            continue
        try:
            candidate = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
    if payload is not None:
        checks = [c for c in (payload.get("checks") or []) if isinstance(c, dict)]
        name = str(payload.get("name") or "tape")
        status = str(payload.get("status") or "")
        failed = [c for c in checks if not c.get("ok")]
        outcome.total = len(checks) or 1
        outcome.passed = len(checks) - len(failed) if checks else 0
        outcome.failed = len(failed)
        if status == RunStatus.BLOCKED:
            outcome.passed = 0
            outcome.failed = 0
            outcome.errors = 1
        for check in failed:
            kind = FailureKind.ASSERTION
            if status == RunStatus.BLOCKED:
                kind = FailureKind.BLOCKED_POLICY
            outcome.failures.append(
                _failure(
                    f"{name}:{check.get('kind')}:{check.get('name') or check.get('mode') or 'check'}",
                    kind,
                    str(check.get("detail") or check.get("text") or "tape expectation failed"),
                )
            )
        if status == RunStatus.BLOCKED and not outcome.failures:
            outcome.failures.append(
                _failure(name, FailureKind.BLOCKED_POLICY, str(payload.get("message") or "blocked by policy"))
            )
        outcome.cases = [
            {
                "name": f"{name}:{c.get('kind')}:{c.get('name') or c.get('mode') or 'check'}",
                "status": RunStatus.PASS if c.get("ok") else RunStatus.FAIL,
                "message": str(c.get("detail") or "")[:300],
            }
            for c in checks
        ]
        outcome.summary_line = f"{name} {status}".strip()
        outcome.structured = True
        return outcome

    markers = _apply_markers(outcome, text)
    counts, line = _summary_counts(text)
    if counts:
        _apply_summary(outcome, counts, line)
    outcome.structured = outcome.structured or markers
    return outcome


# Adapter name -> parser. A name absent here falls back to ``parse_generic`` in
# ``execute_suite``; that is the only place an unknown adapter is tolerated, and it
# never produces invented counts. ``cargo_mutants``/``mutmut`` appear here so a direct
# call (tests, tooling) works, but ``execute_suite`` routes those two to
# ``run_mutation_suite`` before any parsing happens.
ADAPTER_PARSERS = {
    "agent_eval": parse_agent_eval,
    "c_sanitizer": parse_c_sanitizer,
    "cargo": parse_cargo,
    "cargo_mutants": lambda text: parse_cargo_mutants(text),
    "cobol": parse_cobol,
    "mutmut": lambda text: parse_mutmut(text),
    "pty_tmux": parse_pty_tmux,
    "python_pytest": parse_python_pytest,
    "python_unittest": parse_python_unittest,
    "re_project": parse_re_project,
    "shell": parse_shell,
    "shell_test": parse_shell,
    "sqlite_probe": parse_sqlite_probe,
    "terminal_tape": parse_terminal_tape,
}
