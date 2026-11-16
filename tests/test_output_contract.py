"""Output-contract assertions: FileCheck-backed and literal fallback paths.

Every end-to-end control runs through TestBench's real ``execute_suite`` (shell
adapter, isolation, evidence capture) and through the reporting pipeline
(``render_run_json`` / ``render_junit_xml`` written by ``TestBenchStore.finalize_run``).
The FileCheck-path controls are skipped only when LLVM FileCheck is not installed;
the fallback controls always run so the contract never depends on optional tools.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hermes_testbench.adapters import filecheck as fc
from hermes_testbench.isolation import default_scratch_root
from hermes_testbench.reporting import parse_junit_xml_bounded
from hermes_testbench.schema import FailureKind, RunRecord, RunStatus
from hermes_testbench.store import TestBenchStore

FILECHECK = fc.find_filecheck()
requires_filecheck = unittest.skipUnless(FILECHECK, "LLVM FileCheck not installed")

SATISFIED = "alpha\nbeta\nfirst\nsecond\ntoken\ntoken\n"
FULL_CONTRACT = fc.OutputContract(
    required=("alpha", "beta"),
    ordered=("first", "second"),
    forbidden=("FORBIDDEN",),
    counts=(("token", 2), ("never", 0)),
)


class ContractValidationTests(unittest.TestCase):
    def test_empty_or_contradictory_contracts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fc.OutputContract()
        with self.assertRaises(ValueError):
            fc.OutputContract(required=("x",), forbidden=("x",))
        with self.assertRaises(ValueError):
            fc.OutputContract(counts=(("x", -1),))
        with self.assertRaises(ValueError):
            fc.OutputContract(counts=(("x", 1), ("x", 2)))
        with self.assertRaises(ValueError):
            fc.OutputContract(required=("",))
        with self.assertRaises(ValueError):
            fc.OutputContract(required=("two\nlines",))
        with self.assertRaises(ValueError):
            fc.OutputContract(forbidden=("bad\rpattern",))

    def test_build_check_steps_encodes_all_four_intents(self) -> None:
        steps = fc.build_check_steps(FULL_CONTRACT)
        self.assertEqual(len(steps), 5)
        self.assertIn("CHECK-DAG: alpha", steps[0].check_text)
        self.assertIn("CHECK-DAG: beta", steps[0].check_text)
        self.assertIn("CHECK: first", steps[1].check_text)
        self.assertIn("CHECK: second", steps[1].check_text)
        self.assertIn("CHECK-COUNT-2: token", steps[2].check_text)
        self.assertIn("CHECK-NOT: token", steps[2].check_text)
        # Zero-count assertions cannot use CHECK-COUNT-0 (FileCheck rejects it).
        self.assertEqual(steps[3].check_text.strip(), "CHECK-NOT: never")
        # Forbidden patterns are global command-line flags plus a zero-width anchor.
        self.assertEqual(steps[4].check_text.strip(), "CHECK: {{^}}")
        self.assertEqual(steps[4].extra_args, ("--implicit-check-not=FORBIDDEN",))

    def test_command_carries_step_flags_and_input_file(self) -> None:
        steps = fc.build_check_steps(FULL_CONTRACT)
        argv = fc.filecheck_command(
            steps[0], input_path=Path("in.txt"), check_file=Path("c.check"), executable="/bin/FileCheck"
        )
        self.assertEqual(argv, ["/bin/FileCheck", "--input-file=in.txt", "c.check"])
        forbidden = fc.filecheck_command(
            steps[4], input_path=Path("in.txt"), check_file=Path("f.check"), executable="/bin/FileCheck"
        )
        self.assertEqual(forbidden[1], "--implicit-check-not=FORBIDDEN")
        self.assertEqual(forbidden[-2:], ["--input-file=in.txt", "f.check"])


class DiagnosticParserTests(unittest.TestCase):
    def test_extracts_actionable_record_from_missing_check(self) -> None:
        text = (
            "demo.check:1:11: error: CHECK: expected string not found in input\n"
            "// CHECK: gamma\n"
            "          ^\n"
            "input.txt:1:1: note: scanning from here\n"
            "alpha\n"
            "^\n"
        )
        records = fc.parse_filecheck_diagnostics(text)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["case_name"], "CHECK: gamma")
        self.assertEqual(record["kind"], FailureKind.ASSERTION)
        self.assertEqual(record["file"], "demo.check")
        self.assertEqual(record["line"], 1)
        self.assertIn("expected string not found", record["message"])
        self.assertIn("// CHECK: gamma", record["excerpt"])

    def test_check_not_count_and_implicit_forms(self) -> None:
        not_text = (
            "demo.check:1:15: error: CHECK-NOT: excluded string found in input\n"
            "// CHECK-NOT: beta\n"
            "              ^\n"
            "input.txt:2:1: note: found here\n"
            "beta\n"
            "^~~~\n"
        )
        record = fc.parse_filecheck_diagnostics(not_text)[0]
        self.assertEqual(record["case_name"], "CHECK-NOT: beta")
        self.assertEqual(record["line"], 1)
        self.assertIn("beta", record["excerpt"])

        count_text = (
            "demo.check:1:19: error: CHECK-COUNT: expected string not found in input (2 out of 2)\n"
            "// CHECK-COUNT-2: alpha\n"
            "                  ^\n"
            "input.txt:1:6: note: scanning from here\n"
            "alpha\n"
            "     ^\n"
        )
        record = fc.parse_filecheck_diagnostics(count_text)[0]
        self.assertEqual(record["case_name"], "CHECK-COUNT-2: alpha")
        self.assertIn("2 out of 2", record["message"])

        implicit_text = (
            "command line:1:22: error: IMPLICIT-CHECK-NOT: excluded string found in input\n"
            "-implicit-check-not='beta'\n"
            "                     ^\n"
            "input.txt:2:1: note: found here\n"
            "beta\n"
            "^~~~\n"
        )
        record = fc.parse_filecheck_diagnostics(implicit_text)[0]
        self.assertEqual(record["case_name"], "IMPLICIT-CHECK-NOT: beta")
        self.assertIsNone(record["file"])  # "command line" is not a real source file
        self.assertIsNone(record["line"])

    def test_ignores_unrelated_text_and_dedupes_identical_errors(self) -> None:
        self.assertEqual(fc.parse_filecheck_diagnostics("all good\nnote: nothing\n"), [])
        bare = "error: no check strings found with prefix 'CHECK:'\n"
        record = fc.parse_filecheck_diagnostics(bare)[0]
        self.assertEqual(record["kind"], FailureKind.ASSERTION)
        self.assertIsNone(record["file"])
        self.assertIsNone(record["line"])
        self.assertIn("no check strings", record["message"])
        duplicated = bare + bare
        self.assertEqual(len(fc.parse_filecheck_diagnostics(duplicated)), 1)


class FileCheckEndToEndTests(unittest.TestCase):
    """Positive/negative controls through execute_suite and the shell adapter."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-fc-")
        self.root = Path(self.tmp.name)
        self.proj = self.root / "proj"
        self.proj.mkdir()
        self.evidence = self.root / "evidence"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, name: str, contract: fc.OutputContract, text: str, *, prefer: bool = True):
        return fc.execute_output_contract(
            name,
            contract=contract,
            project_root=self.proj,
            evidence_dir=self.evidence,
            run_id=f"tb_fc_{name}",
            input_text=text,
            prefer_filecheck=prefer,
        )

    @requires_filecheck
    def test_positive_control_passes_through_shell_adapter(self) -> None:
        result = self._run("positive", FULL_CONTRACT, SATISFIED)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual((result.total_count, result.pass_count, result.fail_count), (1, 1, 0))
        self.assertEqual(result.failures, [])
        self.assertEqual(result.metrics["output_contract_engine"], "filecheck")
        self.assertEqual(result.metrics["filecheck_binary"], fc.find_filecheck())
        self.assertEqual(result.metrics["output_contract_steps"], 5)
        check_dir = Path(result.metrics["check_dir"])
        self.assertIn("CHECK-DAG: alpha", (check_dir / "step0.check").read_text(encoding="utf-8"))
        self.assertTrue((check_dir / "run.sh").is_file())
        self.assertTrue(Path(result.stdout_path).is_file())

    @requires_filecheck
    def test_missing_required_pattern_reports_actionable_failure(self) -> None:
        result = self._run("missing", fc.OutputContract(required=("MISSING_MARKER",)), "hello\n")
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertGreaterEqual(result.total_count, 1)
        self.assertEqual(result.fail_count, 1)
        failure = result.failures[0]
        self.assertEqual(failure.kind, FailureKind.ASSERTION)
        self.assertEqual(failure.case_name, "CHECK-DAG: MISSING_MARKER")
        self.assertIn("expected string not found", failure.message)
        self.assertTrue(str(failure.file).endswith("step0.check"), failure.file)
        self.assertEqual(failure.line, 1)
        self.assertIn("MISSING_MARKER", failure.excerpt)

    @requires_filecheck
    def test_ordered_violation_reports_the_expected_next_pattern(self) -> None:
        result = self._run("ordered", fc.OutputContract(ordered=("first", "second")), "second\nfirst\n")
        self.assertEqual(result.status, RunStatus.FAIL)
        failure = result.failures[0]
        self.assertEqual(failure.case_name, "CHECK: second")
        self.assertIn("expected string not found", failure.message)

    @requires_filecheck
    def test_forbidden_pattern_fails_the_run(self) -> None:
        contract = fc.OutputContract(required=("alpha",), forbidden=("BANNED",))
        result = self._run("forbidden", contract, "alpha\nBANNED\n")
        self.assertEqual(result.status, RunStatus.FAIL)
        failure = result.failures[0]
        self.assertEqual(failure.case_name, "IMPLICIT-CHECK-NOT: BANNED")
        self.assertIsNone(failure.file)
        self.assertIsNone(failure.line)
        self.assertIn("BANNED", failure.excerpt)

    @requires_filecheck
    def test_multiple_independent_assertions_all_report_failures(self) -> None:
        contract = fc.OutputContract(
            required=("MISSING_MARKER",), forbidden=("BANNED",), counts=(("token", 1),)
        )
        result = self._run("multi_fail", contract, "alpha\nBANNED\ntoken\ntoken\n")
        self.assertEqual(result.status, RunStatus.FAIL)
        names = {failure.case_name for failure in result.failures}
        self.assertIn("CHECK-DAG: MISSING_MARKER", names)
        self.assertIn("IMPLICIT-CHECK-NOT: BANNED", names)
        self.assertIn("token", " ".join(names))
        self.assertEqual(result.fail_count, len(result.failures))
        self.assertGreaterEqual(result.fail_count, 3)

    @requires_filecheck
    def test_required_and_ordered_are_independent(self) -> None:
        contract = fc.OutputContract(required=("late",), ordered=("first", "second"))
        result = self._run("independent", contract, "first\nsecond\nlate\n")
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual(result.failures, [])

    @requires_filecheck
    def test_forbidden_only_contract(self) -> None:
        contract = fc.OutputContract(forbidden=("BANNED",))
        self.assertEqual(self._run("forb_ok", contract, "alpha\nbeta\n").status, RunStatus.PASS)
        result = self._run("forb_bad", contract, "BANNED\nalpha\n")
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertEqual(result.failures[0].case_name, "IMPLICIT-CHECK-NOT: BANNED")
        self.assertIn("BANNED", result.failures[0].excerpt)

    @requires_filecheck
    def test_exact_occurrence_counts_reject_extra_occurrences(self) -> None:
        contract = fc.OutputContract(counts=(("token", 2),))
        self.assertEqual(self._run("count_two", contract, "token\ntoken\n").status, RunStatus.PASS)
        extra = self._run("count_three", contract, "token\ntoken\ntoken\n")
        self.assertEqual(extra.status, RunStatus.FAIL)
        self.assertIn("token", extra.failures[0].case_name)
        self.assertIn("excluded string found", extra.failures[0].message)

        zero = fc.OutputContract(counts=(("never", 0),))
        self.assertEqual(self._run("absent_ok", zero, "alpha\n").status, RunStatus.PASS)
        present = self._run("absent_bad", zero, "alpha\nnever\n")
        self.assertEqual(present.status, RunStatus.FAIL)
        self.assertEqual(present.failures[0].case_name, "CHECK-NOT: never")


class LiteralFallbackTests(unittest.TestCase):
    """The fallback must keep contracts usable when FileCheck is absent."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-fc-")
        self.root = Path(self.tmp.name)
        self.proj = self.root / "proj"
        self.proj.mkdir()
        self.evidence = self.root / "evidence"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, name: str, contract: fc.OutputContract, text: str, **kwargs):
        return fc.execute_output_contract(
            name,
            contract=contract,
            project_root=self.proj,
            evidence_dir=self.evidence,
            run_id=f"tb_fc_{name}",
            input_text=text,
            **kwargs,
        )

    def test_fallback_positive_control(self) -> None:
        result = self._run("fallback_ok", FULL_CONTRACT, SATISFIED, prefer_filecheck=False)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual(result.metrics["output_contract_engine"], "literal-fallback")
        self.assertEqual(result.total_count, FULL_CONTRACT.assertion_count())
        self.assertEqual(result.pass_count, FULL_CONTRACT.assertion_count())
        self.assertEqual(result.fail_count, 0)
        self.assertEqual(len(result.cases), FULL_CONTRACT.assertion_count())
        self.assertTrue(all(case.status == RunStatus.PASS for case in result.cases))

    def test_fallback_negative_control_reports_every_violation(self) -> None:
        text = "beta\nalpha\nsecond\nfirst\ntoken\ntoken\ntoken\nFORBIDDEN\n"
        result = self._run("fallback_bad", FULL_CONTRACT, text, prefer_filecheck=False)
        self.assertEqual(result.status, RunStatus.FAIL)
        names = [failure.case_name for failure in result.failures]
        self.assertIn("CHECK: second", names)
        self.assertIn("IMPLICIT-CHECK-NOT: FORBIDDEN", names)
        self.assertIn("CHECK-COUNT-2: token", names)
        self.assertEqual(len(result.failures), 3)
        self.assertEqual(result.total_count, FULL_CONTRACT.assertion_count())
        self.assertEqual(result.pass_count, FULL_CONTRACT.assertion_count() - 3)
        self.assertTrue(all(f.kind == FailureKind.ASSERTION for f in result.failures))
        forbidden = next(f for f in result.failures if "FORBIDDEN" in f.case_name)
        self.assertEqual(forbidden.line, 8)
        self.assertTrue(str(forbidden.file).endswith(".input.txt"))

    def test_absent_tool_falls_back_instead_of_failing(self) -> None:
        with mock.patch.object(fc, "find_filecheck", return_value=None):
            result = self._run("tool_absent", FULL_CONTRACT, SATISFIED)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual(result.metrics["output_contract_engine"], "literal-fallback")
        self.assertFalse(result.metrics["filecheck_used"])

    def test_override_disables_tool_and_valid_override_is_used(self) -> None:
        old = os.environ.get("TESTBENCH_FILECHECK")
        try:
            os.environ["TESTBENCH_FILECHECK"] = "off"
            self.assertIsNone(fc.find_filecheck())
            fake = self.root / "FileCheck"
            fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake.chmod(0o755)
            os.environ["TESTBENCH_FILECHECK"] = str(fake)
            self.assertEqual(fc.find_filecheck(), str(fake))
        finally:
            if old is None:
                os.environ.pop("TESTBENCH_FILECHECK", None)
            else:
                os.environ["TESTBENCH_FILECHECK"] = old

    def test_check_text_is_pure_and_reports_lines(self) -> None:
        records = fc.check_text(fc.OutputContract(required=("late",)), "one\nlate\n", source_label="out.txt")
        self.assertEqual(records, [])
        records = fc.check_text(fc.OutputContract(required=("missing",)), "one\nlate\n", source_label="out.txt")
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0]["line"])
        records = fc.check_text(fc.OutputContract(forbidden=("late",)), "one\nlate\n", source_label="out.txt")
        self.assertEqual(records[0]["line"], 2)


class ReportingPipelineTests(unittest.TestCase):
    """run.json and junit.xml must carry the actionable assertion failures."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-fc-")
        self.root = Path(self.tmp.name)
        self.proj = self.root / "proj"
        self.proj.mkdir()
        self.evidence = self.root / "evidence"
        self.tb_home = self.root / "tb_home"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _export(self, result, run_id: str) -> RunRecord:
        run = RunRecord(
            run_id=run_id,
            project="output-contract",
            project_root=str(self.proj),
            revision="deadbeef",
            profile="unit",
            status=result.status,
            duration_ms=result.duration_ms,
            total_count=result.total_count,
            pass_count=result.pass_count,
            fail_count=result.fail_count,
            skip_count=result.skip_count,
            error_count=result.error_count,
            commands=[list(result.command)],
            suites=[result],
        )
        with TestBenchStore(self.tb_home) as store:
            store.finalize_run(run)
        return run

    def _run_json(self, run: RunRecord) -> dict:
        self.assertIsNotNone(run.run_json_path)
        return json.loads(Path(str(run.run_json_path)).read_text(encoding="utf-8"))

    def _junit_xml(self, run: RunRecord) -> str:
        self.assertIsNotNone(run.junit_xml_path)
        return Path(str(run.junit_xml_path)).read_text(encoding="utf-8")

    def test_fallback_failure_lands_in_run_json_and_junit_xml(self) -> None:
        contract = fc.OutputContract(required=("MISSING_MARKER",), forbidden=("BANNED",))
        result = fc.execute_output_contract(
            "reported",
            contract=contract,
            project_root=self.proj,
            evidence_dir=self.evidence,
            run_id="tb_fc_report_bad",
            input_text="alpha\nBANNED\n",
            prefer_filecheck=False,
        )
        run = self._export(result, "tb_fc_report_bad")

        data = self._run_json(run)
        suite = data["suites"][0]
        self.assertEqual(data["status"], RunStatus.FAIL)
        self.assertEqual(suite["fail_count"], result.fail_count)
        self.assertEqual(suite["total_count"], result.total_count)
        cases = {item["case_name"]: item for item in suite["failures"]}
        self.assertIn("CHECK-DAG: MISSING_MARKER", cases)
        self.assertIn("IMPLICIT-CHECK-NOT: BANNED", cases)
        self.assertEqual(cases["IMPLICIT-CHECK-NOT: BANNED"]["kind"], FailureKind.ASSERTION)
        self.assertEqual(cases["IMPLICIT-CHECK-NOT: BANNED"]["line"], 2)

        xml = self._junit_xml(run)
        parsed = parse_junit_xml_bounded(xml.encode("utf-8"))
        self.assertEqual(parsed["failures"], 2)
        self.assertEqual(parsed["errors"], 0)
        self.assertEqual(parsed["tests"], result.total_count)
        self.assertIn("<failure", xml)
        self.assertIn("MISSING_MARKER", xml)
        self.assertIn("BANNED", xml)

    @requires_filecheck
    def test_filecheck_failure_lands_in_run_json_and_junit_xml(self) -> None:
        contract = fc.OutputContract(required=("present",), counts=(("token", 2),))
        result = fc.execute_output_contract(
            "filecheck_reported",
            contract=contract,
            project_root=self.proj,
            evidence_dir=self.evidence,
            run_id="tb_fc_report_fc",
            input_text="present\ntoken\ntoken\ntoken\n",
        )
        self.assertEqual(result.status, RunStatus.FAIL)
        run = self._export(result, "tb_fc_report_fc")

        data = self._run_json(run)
        failure = data["suites"][0]["failures"][0]
        self.assertEqual(failure["kind"], FailureKind.ASSERTION)
        self.assertEqual(failure["line"], 2)
        self.assertIn("token", failure["case_name"])
        self.assertIn("excluded string found", failure["message"])

        xml = self._junit_xml(run)
        self.assertIn("type=\"assertion\"", xml)
        parsed = parse_junit_xml_bounded(xml.encode("utf-8"))
        self.assertEqual(parsed["failures"], 1)
        self.assertEqual(parsed["errors"], 0)

    def test_positive_control_exports_clean_reports(self) -> None:
        result = fc.execute_output_contract(
            "reported_ok",
            contract=fc.OutputContract(required=("alpha",)),
            project_root=self.proj,
            evidence_dir=self.evidence,
            run_id="tb_fc_report_ok",
            input_text="alpha\n",
            prefer_filecheck=False,
        )
        self.assertEqual(result.status, RunStatus.PASS)
        run = self._export(result, "tb_fc_report_ok")
        data = self._run_json(run)
        self.assertEqual(data["status"], RunStatus.PASS)
        self.assertEqual(data["suites"][0]["failures"], [])
        parsed = parse_junit_xml_bounded(self._junit_xml(run).encode("utf-8"))
        self.assertEqual((parsed["tests"], parsed["failures"], parsed["errors"]), (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
