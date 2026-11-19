"""Structured report serializers and concise formatters (ADR-0004).

Everything in this module is **derived**. ``run.json``, ``junit.xml``,
``repro.sh`` and the terminal summary are all projections of one in-memory
``RunRecord``; none of them is evidence about the run that the database does
not already hold. Two consequences the owner should keep in mind:

* A JUnit ``testcase`` element is not proof that a case ran. Cases the adapter
  actually reported are emitted from ``suite.cases``/``suite.failures``; the
  rest of the element tree is *back-filled* from the counters
  (``total_count``/``pass_count``/``skip_count``/``error_count``) so the XML
  agrees with the numbers the run recorded. That is a deliberate trade
  (report consumers expect ``tests == sum(elements)``) and it means the XML
  can describe cases whose names are synthetic — ``suite:fail_1``,
  ``suite:pass_1`` — rather than real ones.
* ``repro.sh`` is a best-effort reconstruction from ``command``/``cwd``, not a
  captured invocation: environment, sandbox profile, trust state and the
  disposable ``HOME`` are not reproduced. It reproduces the *command*, not the
  *conditions*.

Sanitization happens at render time (``_clean_field``/``_deep_clean``), on top
of store-time redaction, so a record built in memory — a config error, an
adapter message — is still scrubbed on the way out. Output escaping is
ElementTree's job; this module never hand-builds XML, which is why a case name
containing ``&`` or ``<`` cannot produce malformed XML.

``parse_junit_xml_bounded`` is the one inbound path: untrusted XML with a size
cap, a DTD/ENTITY refusal and no external entity resolution.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, Dict, List
import xml.etree.ElementTree as ET

from .sanitize import redact_secrets, sanitize_terminal_text
from .schema import RunRecord, RunStatus, run_to_dict


def _clean_field(value: str) -> str:
    """Redact then strip: secrets first, escapes second.

    Order matters. ``redact_secrets`` pattern-matches on text, so it must see
    the escapes intact (a token split by an SGR sequence is a different token
    after stripping); ``sanitize_terminal_text`` then removes the control
    characters that would otherwise be persisted into XML/JSON.
    """
    return sanitize_terminal_text(redact_secrets(value or ""))


def _deep_clean(obj: Any) -> Any:
    """Apply ``_clean_field`` to every string *value* in a nested structure.

    Used for the JSON projection so there is no "unimportant" field: a secret
    can arrive in a command argument, a case name, an ``isolation_meta`` value
    or an environment snapshot, and a redaction pass that special-cases the
    obvious fields leaks on the others.

    Values are cleaned; keys are only stringified. So a secret used as a dict
    *key* survives into ``run.json``. Keys in practice are field names and
    environment variable names, never values, so this is a theoretical gap
    rather than a live leak — but it is a gap, and it is asymmetric with the
    treatment of values.
    """
    if isinstance(obj, str):
        return _clean_field(obj)
    if isinstance(obj, list):
        return [_deep_clean(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _deep_clean(v) for k, v in obj.items()}
    return obj


def render_junit_xml(run: RunRecord) -> str:
    """Render a JUnit document whose counters always agree with the run's numbers.

    Four emit passes per suite, in a fixed order, because each later pass
    depends on what the earlier ones produced:

    1. ``suite.cases`` — the adapter's real case results.
    2. ``suite.failures`` — failures that were never promoted to a case
       (parser-level, or an adapter that only reports failures). Skipped when
       a case with that name already exists, so one failure is never counted
       twice; classified by ``suite.status`` **and** ``fail.kind``, since a
       blocked suite's failure is a ``<skipped>`` and a timeout's is an
       ``<error>``.
    3. Back-fill for skips/errors/failures the counters claim but the emitted
       elements do not cover (``target_skip`` vs ``cur_s``, and so on). These
       are the synthetic ``suite:skip_1`` elements — see the module docstring:
       they exist to keep ``tests == len(testcase)``, not to claim a case name.
    4. Back-fill passes and plain cases up to ``total_count``.

    Classification of a "bad" case is deliberately conservative on the
    ``<error>`` side: TIMEOUT/ERROR/harness-level kinds become ``<error>`` so a
    CI consumer distinguishes "the tests failed" from "we could not run them".

    Every text field goes through ``_clean_field`` here (this function can be
    called on a record that never passed through the store, e.g. ``testbench
    results --format junit`` on a fresh run) and messages are truncated to 500
    chars in the ``message`` attribute while the full (still redacted) text
    goes in the element body.
    """
    root = ET.Element(
        "testsuites",
        {
            "id": _clean_field(run.run_id),
            "name": _clean_field(f"{run.project}:{run.profile}"),
            "tests": "0",
            "failures": "0",
            "errors": "0",
            "skipped": "0",
            "time": f"{(run.duration_ms or 0.0) / 1000.0:.3f}",
        },
    )
    tot_t = tot_f = tot_e = tot_s = 0
    for suite in run.suites:
        ts_el = ET.SubElement(
            root,
            "testsuite",
            {
                "name": _clean_field(suite.name),
                "tests": "0",
                "failures": "0",
                "errors": "0",
                "skipped": "0",
                "time": f"{suite.duration_ms / 1000.0:.3f}",
            },
        )
        emitted_names: set[str] = set()
        for case in suite.cases:
            emitted_names.add(case.name)
            tc_el = ET.SubElement(
                ts_el,
                "testcase",
                {
                    "classname": _clean_field(case.suite or suite.name),
                    "name": _clean_field(case.name),
                    "time": f"{case.duration_ms / 1000.0:.3f}",
                },
            )
            msg = _clean_field(case.message)
            if case.status in (RunStatus.FAIL, "FAILED"):
                f_el = ET.SubElement(tc_el, "failure", {"message": msg[:500]})
                f_el.text = msg
            elif case.status in (RunStatus.ERROR, RunStatus.TIMEOUT, RunStatus.CANCELLED):
                e_el = ET.SubElement(tc_el, "error", {"type": case.status, "message": msg[:500]})
                e_el.text = msg
            elif case.status in (RunStatus.SKIP, RunStatus.BLOCKED, "SKIPPED"):
                ET.SubElement(tc_el, "skipped", {"type": case.status, "message": msg[:500]})

        for fail in suite.failures:
            if fail.case_name in emitted_names:
                continue
            emitted_names.add(fail.case_name)
            tc_el = ET.SubElement(
                ts_el,
                "testcase",
                {"classname": _clean_field(suite.name), "name": _clean_field(fail.case_name)},
            )
            msg = _clean_field(fail.message or fail.excerpt)
            if suite.status in (RunStatus.BLOCKED, RunStatus.SKIP, RunStatus.CANCELLED) or fail.kind in (
                "blocked_policy",
                "untrusted_manifest",
            ):
                ET.SubElement(tc_el, "skipped", {"type": _clean_field(fail.kind or suite.status), "message": msg[:500]})
            elif suite.status in (RunStatus.TIMEOUT, RunStatus.ERROR) or fail.kind in (
                "timeout",
                "spawn_error",
                "harness_error",
                "sandbox_denied",
                "signal_crash",
                "abandoned_worker",
            ):
                e_el = ET.SubElement(tc_el, "error", {"type": _clean_field(fail.kind), "message": msg[:500]})
                e_el.text = _clean_field(fail.excerpt or fail.message)
            else:
                f_el = ET.SubElement(tc_el, "failure", {"type": _clean_field(fail.kind), "message": msg[:500]})
                f_el.text = _clean_field(fail.excerpt or fail.message)

        existing = ts_el.findall("testcase")
        cur_f = sum(1 for c in existing if c.find("failure") is not None)
        cur_e = sum(1 for c in existing if c.find("error") is not None)
        cur_s = sum(1 for c in existing if c.find("skipped") is not None)
        cur_p = len(existing) - cur_f - cur_e - cur_s

        target_skip = max(0, int(suite.skip_count or 0) + int(suite.blocked_count or 0))
        if target_skip == 0 and len(existing) == 0 and suite.status in (RunStatus.SKIP, RunStatus.BLOCKED, RunStatus.CANCELLED):
            target_skip = 1
        for idx in range(max(0, target_skip - cur_s)):
            name = suite.name if (target_skip == 1 and len(ts_el.findall("testcase")) == 0) else f"{suite.name}:skip_{idx + 1}"
            tc_el = ET.SubElement(
                ts_el,
                "testcase",
                {"classname": _clean_field(suite.name), "name": _clean_field(name)},
            )
            msg = _clean_field(suite.summary_excerpt or suite.status)
            ET.SubElement(tc_el, "skipped", {"type": _clean_field(suite.status), "message": msg[:500]})

        missing_e = max(0, int(suite.error_count or 0) - cur_e)
        for idx in range(missing_e):
            tc_el = ET.SubElement(
                ts_el,
                "testcase",
                {"classname": _clean_field(suite.name), "name": _clean_field(f"{suite.name}:error_{idx + 1}")},
            )
            msg = _clean_field(suite.summary_excerpt or "error")
            e_el = ET.SubElement(tc_el, "error", {"type": "error", "message": msg[:500]})
            e_el.text = msg

        total_bad = int(suite.fail_count or 0) + int(suite.error_count or 0)
        missing_f = max(0, total_bad - (cur_f + cur_e + missing_e))
        for idx in range(missing_f):
            tc_el = ET.SubElement(
                ts_el,
                "testcase",
                {"classname": _clean_field(suite.name), "name": _clean_field(f"{suite.name}:fail_{idx + 1}")},
            )
            msg = _clean_field(suite.summary_excerpt or suite.status)
            if suite.status in (RunStatus.TIMEOUT, RunStatus.ERROR):
                e_el = ET.SubElement(tc_el, "error", {"type": _clean_field(suite.status.lower()), "message": msg[:500]})
                e_el.text = msg
            else:
                f_el = ET.SubElement(tc_el, "failure", {"type": "assertion", "message": msg[:500]})
                f_el.text = msg

        missing_p = max(0, int(suite.pass_count or 0) - cur_p)
        for idx in range(missing_p):
            name = suite.name if (int(suite.pass_count or 0) == 1 and len(ts_el.findall("testcase")) == 0) else f"{suite.name}:pass_{idx + 1}"
            ET.SubElement(
                ts_el,
                "testcase",
                {"classname": _clean_field(suite.name), "name": _clean_field(name)},
            )

        now_len = len(ts_el.findall("testcase"))
        for idx in range(max(0, int(suite.total_count or 0) - now_len)):
            ET.SubElement(
                ts_el,
                "testcase",
                {"classname": _clean_field(suite.name), "name": _clean_field(f"{suite.name}:case_{now_len + idx + 1}")},
            )

        s_cases = ts_el.findall("testcase")
        s_f = sum(1 for c in s_cases if c.find("failure") is not None)
        s_e = sum(1 for c in s_cases if c.find("error") is not None)
        s_s = sum(1 for c in s_cases if c.find("skipped") is not None)
        s_t = len(s_cases)
        ts_el.attrib["tests"] = str(s_t)
        ts_el.attrib["failures"] = str(s_f)
        ts_el.attrib["errors"] = str(s_e)
        ts_el.attrib["skipped"] = str(s_s)
        tot_t += s_t
        tot_f += s_f
        tot_e += s_e
        tot_s += s_s

    root.attrib["tests"] = str(tot_t)
    root.attrib["failures"] = str(tot_f)
    root.attrib["errors"] = str(tot_e)
    root.attrib["skipped"] = str(tot_s)
    xml_body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + xml_body + "\n"


def parse_junit_xml_bounded(xml_bytes: bytes, max_bytes: int = 2_000_000) -> Dict[str, Any]:
    """Parse a JUnit document defensively; counters only, no case-level output.

    Three gates, in this order: size (refuse before touching the parser), a
    DTD/ENTITY scan of the first 4096 bytes, then ``ET.fromstring``. The scan
    is head-only because that is where a prolog must live — but "must" is the
    parser's rule, not this check's: bytes padded past 4096 can hide a
    ``<!DOCTYPE>`` that ElementTree will still accept and whose internal
    entities it will expand. The size cap is what bounds that exposure.

    Counters come from attributes, not from counting children, so a document
    with zero ``<testcase>`` elements but ``tests="3"`` reports 3. That is
    intentional for consumers that only want the summary, and it means this
    function cannot be used to validate a JUnit document's self-consistency.
    """
    if len(xml_bytes) > max_bytes:
        raise ValueError(f"JUnit XML exceeds max_bytes ({len(xml_bytes)} > {max_bytes})")
    upper_head = xml_bytes[:4096].upper()
    if b"<!DOCTYPE" in upper_head or b"<!ENTITY" in upper_head:
        raise ValueError("Forbidden DTD/ENTITY declaration in JUnit XML")
    try:
        root = ET.fromstring(xml_bytes.decode("utf-8", errors="replace"))
    except ET.ParseError as exc:
        raise ValueError(f"Invalid JUnit XML: {exc}") from exc

    suites_els = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = failures = errors = skipped = 0
    suites_out: List[Dict[str, Any]] = []
    for ts in suites_els:
        s_tests = int(ts.attrib.get("tests", "0") or 0)
        s_fail = int(ts.attrib.get("failures", "0") or 0)
        s_err = int(ts.attrib.get("errors", "0") or 0)
        s_skip = int(ts.attrib.get("skipped", "0") or 0)
        tests += s_tests
        failures += s_fail
        errors += s_err
        skipped += s_skip
        suites_out.append(
            {
                "name": ts.attrib.get("name", ""),
                "tests": s_tests,
                "failures": s_fail,
                "errors": s_err,
                "skipped": s_skip,
            }
        )
    return {
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "suites": suites_out,
    }


def render_run_json(run: RunRecord) -> str:
    """Canonical JSON projection: sorted keys, two-space indent, trailing newline.

    This exact string is what the store keeps in ``runs.payload_json`` and what
    ``testbench status --json`` prints, so stability matters beyond cosmetics:
    sorted keys make two runs byte-diffable, and the trailing newline keeps a
    shell redirect well-formed. Values are deep-cleaned (values only — keys are
    field names); ``model_usage`` stays ``null`` unless it was measured, since
    a fabricated zero would read as "this run cost nothing".
    """
    cleaned = _deep_clean(run_to_dict(run))
    return json.dumps(cleaned, indent=2, sort_keys=True) + "\n"


def format_terminal_summary(run: RunRecord) -> str:
    """One-screen, human summary: status, identity, counts, first failure, repro.

    Kept terse on purpose — this is what the CLI prints after every run and
    what the TUI detail sheet reuses. ``first_failure`` is shown capped at 220
    chars; ``run.json`` stores the 300-char form captured at finalize. The point
    is a pointer to evidence, not the evidence. An orphan reconciled to ``ERROR`` has no counts, so the
    tally reads all-zero by design (the reconciler invents no numbers) and the
    only substantive line is its ``first_failure``.
    """
    dur_s = f"{(run.duration_ms or 0.0) / 1000.0:.2f}s" if run.duration_ms is not None else "-"
    lines = [
        (
            f"{_clean_field(run.status):<9} {_clean_field(run.run_id)}  "
            f"{_clean_field(run.project)} [{_clean_field(run.profile)}]  "
            f"rev={_clean_field(run.revision or '-')}  "
            f"total={run.total_count} pass={run.pass_count} fail={run.fail_count} "
            f"skip={run.skip_count} blocked={run.blocked_count} err={run.error_count}  ({dur_s})"
        )
    ]
    if run.first_failure:
        lines.append(f"  failure: {_clean_field(run.first_failure)[:220]}")
    if run.repro_instructions:
        lines.append(f"  repro:   {_clean_field(run.repro_instructions)}")
    if run.evidence_dir:
        lines.append(f"  evidence:{_clean_field(run.evidence_dir)}")
    return "\n".join(lines)


def format_repro_script(run: RunRecord) -> str:
    """Build ``repro.sh``: shell-quoted commands, one suite after another.

    Three precedence rules, because a record's shape varies: per-suite commands
    (with their ``cwd``) when suites were recorded, else ``run.commands`` from
    the run root, else a comment carrying ``repro_instructions``. It is a
    ``/bin/sh`` script, not a copy of the harness: no environment is exported,
    no sandbox is applied, and ``set -eu`` aborts on the first failing command —
    which also means a suite whose ``command`` is empty is silently absent from
    the repro rather than visible as a gap.

    Relative ``cwd`` values *chain*: each ``cd`` applies to the directory left
    by the previous suite, so a run whose suites declare ``cwd = "sub"`` and
    then ``cwd = "."`` reproduces the second suite inside ``sub`` rather than at
    the project root. ``repro_instructions`` (the single-command case) does not
    have this problem; the multi-suite path does.

    Two honest limitations. The script is *not* redacted: command argv is
    interpolated through ``shlex.quote`` only, so a manifest that puts a token
    in ``command`` writes it verbatim to a ``0700`` file under the store, even
    though the same string is ``[REDACTED]`` in ``run.json``. And when suites
    exist the script never ``cd``s to ``project_root`` — it relies on each
    suite's declared ``cwd``, which for discovered suites is ``"."``, i.e. the
    invoker's working directory.
    """
    lines = [
        "#!/bin/sh",
        "# Generated by Hermes TestBench — deterministic reproduction script",
        "set -eu",
    ]
    if run.suites:
        for s in run.suites:
            if s.cwd:
                lines.append(f"cd {shlex.quote(s.cwd)}")
            if s.command:
                lines.append(" ".join(shlex.quote(arg) for arg in s.command))
    elif run.commands:
        if run.project_root:
            lines.append(f"cd {shlex.quote(run.project_root)}")
        for cmd in run.commands:
            if cmd:
                lines.append(" ".join(shlex.quote(arg) for arg in cmd))
    elif run.repro_instructions:
        lines.append(f"# Manual reproduction: {_clean_field(run.repro_instructions)}")
    return "\n".join(lines) + "\n"
