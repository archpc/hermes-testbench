"""TDD tests for JUnit XML, JSON, repro script, and terminal reporting."""

from __future__ import annotations

import json
import unittest

from hermes_testbench.reporting import (
    format_repro_script,
    format_terminal_summary,
    parse_junit_xml_bounded,
    render_junit_xml,
    render_run_json,
)
from hermes_testbench.schema import CaseResult, FailureRecord, RunRecord, RunStatus, SuiteResult


class ReportingTests(unittest.TestCase):
    def _sample_run(self) -> RunRecord:
        suite = SuiteResult(
            name="core",
            adapter="python_unittest",
            profile="unit",
            command=["python3.14", "-m", "unittest", "tests.test_x"],
            cwd="/tmp/repo",
            status=RunStatus.FAIL,
            exit_code=1,
            duration_ms=45.0,
            total_count=3,
            pass_count=1,
            fail_count=1,
            skip_count=1,
            repro_command="python3.14 -m unittest tests.test_x",
            failures=[FailureRecord(suite="core", case_name="test_bad", message="1 != 2 SECRET=xyz98765")],
            cases=[
                CaseResult(name="test_ok", suite="core", status=RunStatus.PASS, duration_ms=10.0),
                CaseResult(
                    name="test_bad",
                    suite="core",
                    status=RunStatus.FAIL,
                    duration_ms=25.0,
                    message="AssertionError: 1 != 2\x00\x1b[31m SECRET=xyz98765",
                ),
                CaseResult(name="test_skip", suite="core", status=RunStatus.SKIP, duration_ms=1.0, message="no blob"),
            ],
        )
        return RunRecord(
            run_id="tb_20261114_170500_abcd1234",
            project="demo",
            project_root="/tmp/repo",
            revision="deadbeef",
            profile="unit",
            status=RunStatus.FAIL,
            duration_ms=45.0,
            total_count=3,
            pass_count=1,
            fail_count=1,
            skip_count=1,
            first_failure="test_bad: 1 != 2 SECRET=xyz98765\x1b[31m",
            repro_instructions="python3.14 -m unittest tests.test_x",
            suites=[suite],
        )

    def test_junit_xml_roundtrip_sanitizes_secrets_and_rejects_xxe_or_oversize(self) -> None:
        run = self._sample_run()
        xml_text = render_junit_xml(run)
        self.assertNotIn("xyz98765", xml_text)
        self.assertNotIn("\x1b", xml_text)
        parsed = parse_junit_xml_bounded(xml_text.encode("utf-8"))
        self.assertEqual(parsed["tests"], 3)
        self.assertEqual(parsed["failures"], 1)
        self.assertEqual(parsed["skipped"], 1)

        with self.assertRaises(ValueError):
            parse_junit_xml_bounded(b'<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe "bad">]><testsuite/>')
        with self.assertRaises(ValueError):
            parse_junit_xml_bounded(b"<testsuite>" + (b"x" * 2048) + b"</testsuite>", max_bytes=512)

    def test_run_json_terminal_summary_and_repro_script_sanitize_secrets_and_escapes(self) -> None:
        run = self._sample_run()
        j = render_run_json(run)
        self.assertNotIn("xyz98765", j)
        obj = json.loads(j)
        self.assertEqual(obj["status"], "FAIL")

        term = format_terminal_summary(run)
        self.assertIn("FAIL", term)
        self.assertIn("tb_20261114_170500_abcd1234", term)
        self.assertNotIn("xyz98765", term)
        self.assertNotIn("\x1b", term)

        repro = format_repro_script(run)
        self.assertTrue(repro.startswith("#!/bin/sh\n"))
        self.assertIn("unittest", repro)


if __name__ == "__main__":
    unittest.main()
