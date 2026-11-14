"""Structured report serializers and concise formatters (ADR-0004)."""

from __future__ import annotations

import json
import shlex
from typing import Any, Dict, List
import xml.etree.ElementTree as ET

from .sanitize import redact_secrets, sanitize_terminal_text
from .schema import RunRecord, RunStatus, run_to_dict


def _clean_field(value: str) -> str:
    return sanitize_terminal_text(redact_secrets(value or ""))


def _deep_clean(obj: Any) -> Any:
    if isinstance(obj, str):
        return _clean_field(obj)
    if isinstance(obj, list):
        return [_deep_clean(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _deep_clean(v) for k, v in obj.items()}
    return obj


def render_junit_xml(run: RunRecord) -> str:
    root = ET.Element(
        "testsuites",
        {
            "id": _clean_field(run.run_id),
            "name": _clean_field(f"{run.project}:{run.profile}"),
            "tests": str(run.total_count),
            "failures": str(run.fail_count),
            "errors": str(run.error_count),
            "skipped": str(run.skip_count),
            "time": f"{(run.duration_ms or 0.0) / 1000.0:.3f}",
        },
    )
    for suite in run.suites:
        ts_el = ET.SubElement(
            root,
            "testsuite",
            {
                "name": _clean_field(suite.name),
                "tests": str(suite.total_count),
                "failures": str(suite.fail_count),
                "errors": str(suite.error_count),
                "skipped": str(suite.skip_count),
                "time": f"{suite.duration_ms / 1000.0:.3f}",
            },
        )
        if suite.cases:
            for case in suite.cases:
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
                if case.status == RunStatus.FAIL:
                    f_el = ET.SubElement(tc_el, "failure", {"message": msg[:500]})
                    f_el.text = msg
                elif case.status in (RunStatus.ERROR, RunStatus.TIMEOUT, RunStatus.CANCELLED):
                    e_el = ET.SubElement(tc_el, "error", {"type": case.status, "message": msg[:500]})
                    e_el.text = msg
                elif case.status in (RunStatus.SKIP, RunStatus.BLOCKED):
                    ET.SubElement(tc_el, "skipped", {"message": msg[:500]})
        elif suite.failures:
            for fail in suite.failures:
                tc_el = ET.SubElement(
                    ts_el,
                    "testcase",
                    {"classname": _clean_field(suite.name), "name": _clean_field(fail.case_name)},
                )
                msg = _clean_field(fail.message or fail.excerpt)
                f_el = ET.SubElement(tc_el, "failure", {"type": _clean_field(fail.kind), "message": msg[:500]})
                f_el.text = _clean_field(fail.excerpt or fail.message)
    xml_body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + xml_body + "\n"


def parse_junit_xml_bounded(xml_bytes: bytes, max_bytes: int = 2_000_000) -> Dict[str, Any]:
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
    cleaned = _deep_clean(run_to_dict(run))
    return json.dumps(cleaned, indent=2, sort_keys=True) + "\n"


def format_terminal_summary(run: RunRecord) -> str:
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
