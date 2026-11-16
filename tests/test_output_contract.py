"""Output-contract assertions through LLVM FileCheck and TestBench's shell adapter.

Every execution runs through TestBench's real ``execute_suite`` (``shell`` adapter,
isolation, Seatbelt confinement, evidence capture, and ``Gate 3`` ``SKIP`` when
``FileCheck`` is absent) and through ``TestBenchStore.finalize_run`` reporting
(``run.json`` and ``junit.xml``). Parser and contract-construction tests run
unconditionally even on hosts without LLVM installed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from hermes_testbench.adapters import execute_suite, filecheck as fc
from hermes_testbench.adapters.parsers import parse_shell
from hermes_testbench.isolation import default_scratch_root
from hermes_testbench.reporting import parse_junit_xml_bounded
from hermes_testbench.schema import FailureKind, RunRecord, RunStatus, SuiteSpec
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
        self.assertEqual(steps[3].check_text.strip(), "CHECK-NOT: never")
        self.assertEqual(steps[3].extra_args, ("--allow-empty",))
        self.assertEqual(steps[4].check_text.strip(), "CHECK: {{^}}")
        self.assertEqual(steps[4].extra_args, ("--allow-empty", "--implicit-check-not=FORBIDDEN"))

    def test_command_carries_step_flags_and_input_file(self) -> None:
        steps = fc.build_check_steps(FULL_CONTRACT)
        argv = fc.filecheck_command(
            steps[0], input_path=Path("in.txt"), check_file=Path("c.check"), executable="/bin/FileCheck"
        )
        self.assertEqual(argv, ["/bin/FileCheck", "--allow-empty", "--input-file=in.txt", "c.check"])
        forbidden = fc.filecheck_command(
            steps[4], input_path=Path("in.txt"), check_file=Path("f.check"), executable="/bin/FileCheck"
        )
        self.assertEqual(forbidden[1:3], ["--allow-empty", "--implicit-check-not=FORBIDDEN"])
        self.assertEqual(forbidden[-2:], ["--input-file=in.txt", "f.check"])


class DiagnosticParserAndShellIntegrationTests(unittest.TestCase):
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
        self.assertIsNone(record["file"])
        self.assertIsNone(record["line"])

    def test_ignores_benign_warnings_and_unrelated_compiler_errors(self) -> None:
        self.assertEqual(fc.parse_filecheck_diagnostics("warning: deprecated flag\nall good\n"), [])
        self.assertEqual(
            fc.parse_filecheck_diagnostics("foo.c:10:5: error: use of undeclared identifier 'x'\n"),
            [],
        )
        bare = "error: no check strings found with prefix 'CHECK:'\n"
        record = fc.parse_filecheck_diagnostics(bare)[0]
        self.assertEqual(record["kind"], FailureKind.ASSERTION)
        self.assertIsNone(record["file"])
        self.assertIn("no check strings", record["message"])
        self.assertEqual(len(fc.parse_filecheck_diagnostics(bare + bare)), 1)

    def test_parse_shell_handles_filecheck_input_dump_without_false_error_counts(self) -> None:
        raw_fc_dump = (
            "rules.check:1:8: error: CHECK: expected string not found in input\n"
            "CHECK: MUST_EXIST\n"
            "       ^\n"
            "in.txt:1:1: note: scanning from here\n"
            "actual line\n"
            "^\n"
            "Input was:\n"
            "<<<<<<\n"
            "           1: actual line\n"
            "check:1'0    {            } search range (exclusive bounds)\n"
            "check:1'1                   error: no match found in search range\n"
            ">>>>>>\n"
        )
        outcome = parse_shell(raw_fc_dump)
        self.assertTrue(outcome.structured)
        self.assertEqual(outcome.errors, 0)
        self.assertEqual(outcome.failed, 1)
        self.assertEqual(outcome.total, 1)
        self.assertEqual(len(outcome.failures), 1)
        self.assertEqual(outcome.failures[0]["case_name"], "CHECK: MUST_EXIST")
        self.assertEqual(outcome.failures[0]["file"], "rules.check")
        self.assertEqual(outcome.failures[0]["line"], 1)


class AbsentToolAndOverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-fc-skip-")
        self.root = Path(self.tmp.name)
        self.proj = self.root / "proj"
        self.proj.mkdir()
        self.evidence = self.root / "evidence"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_missing_filecheck_skips_cleanly_via_gate3_without_custom_fallback(self) -> None:
        with (
            mock.patch.object(fc, "find_filecheck", return_value=None),
            mock.patch("hermes_testbench.adapters._missing_binary", return_value="FileCheck"),
        ):
            result = fc.execute_output_contract(
                "absent_fc",
                contract=FULL_CONTRACT,
                project_root=self.proj,
                evidence_dir=self.evidence,
                run_id="tb_fc_skip",
                input_text=SATISFIED,
            )
        self.assertEqual(result.status, RunStatus.SKIP)
        self.assertEqual(result.skip_count, 1)
        self.assertIn("FileCheck", result.summary_excerpt)

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


@requires_filecheck
class FileCheckEndToEndTests(unittest.TestCase):
    """Positive and negative controls through execute_suite and reporting."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-fc-")
        self.root = Path(self.tmp.name)
        self.proj = self.root / "proj"
        self.proj.mkdir()
        self.evidence = self.root / "evidence"
        self.tb_home = self.root / "tb_home"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, name: str, contract: fc.OutputContract, text: Optional[str] = None, **kwargs):
        return fc.execute_output_contract(
            name,
            contract=contract,
            project_root=self.proj,
            evidence_dir=self.evidence,
            run_id=f"tb_fc_{name}",
            input_text=text,
            **kwargs,
        )

    def test_positive_control_passes_through_shell_adapter(self) -> None:
        result = self._run("positive", FULL_CONTRACT, SATISFIED)
        self.assertEqual(result.status, RunStatus.PASS)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual((result.total_count, result.pass_count, result.fail_count), (1, 1, 0))
        self.assertEqual(result.failures, [])
        self.assertEqual(result.metrics["output_contract_engine"], "filecheck")
        self.assertEqual(result.metrics["output_contract_steps"], 5)
        self.assertTrue(Path(result.stdout_path or "").is_file())
        self.assertTrue(Path(result.sanitized_path or "").is_file())

    def test_producer_command_runs_once_under_isolation_and_checks_output(self) -> None:
        marker = self.proj / "producer.py"
        marker.write_text(
            'print("header")\nprint("item")\nprint("item")\nprint("footer")\n',
            encoding="utf-8",
        )
        contract = fc.OutputContract(
            ordered=("header", "footer"),
            forbidden=("ERROR",),
            counts=(("item", 2),),
        )
        result = self._run("producer_ok", contract, producer_command=[sys.executable, str(marker)])
        self.assertEqual(result.status, RunStatus.PASS)

        bad_contract = fc.OutputContract(counts=(("item", 1),))
        bad_res = self._run("producer_bad", bad_contract, producer_command=[sys.executable, str(marker)])
        self.assertEqual(bad_res.status, RunStatus.FAIL)
        self.assertEqual(bad_res.failures[0].kind, FailureKind.ASSERTION)
        self.assertIn("item", bad_res.failures[0].case_name)

    def test_missing_required_pattern_reports_actionable_failure(self) -> None:
        result = self._run("missing", fc.OutputContract(required=("MISSING_MARKER",)), "hello\n")
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertEqual(result.fail_count, 1)
        failure = result.failures[0]
        self.assertEqual(failure.kind, FailureKind.ASSERTION)
        self.assertEqual(failure.case_name, "CHECK-DAG: MISSING_MARKER")
        self.assertIn("expected string not found", failure.message)
        self.assertTrue(str(failure.file).endswith("step0.check"), failure.file)
        self.assertEqual(failure.line, 1)
        self.assertIn("MISSING_MARKER", failure.excerpt)

    def test_ordered_violation_reports_the_expected_next_pattern(self) -> None:
        result = self._run("ordered", fc.OutputContract(ordered=("first", "second")), "second\nfirst\n")
        self.assertEqual(result.status, RunStatus.FAIL)
        failure = result.failures[0]
        self.assertEqual(failure.case_name, "CHECK: second")
        self.assertIn("expected string not found", failure.message)

    def test_forbidden_pattern_fails_the_run(self) -> None:
        contract = fc.OutputContract(required=("alpha",), forbidden=("BANNED",))
        result = self._run("forbidden", contract, "alpha\nBANNED\n")
        self.assertEqual(result.status, RunStatus.FAIL)
        failure = result.failures[0]
        self.assertEqual(failure.case_name, "IMPLICIT-CHECK-NOT: BANNED")
        self.assertIsNone(failure.file)
        self.assertIsNone(failure.line)
        self.assertIn("BANNED", failure.excerpt)

    def test_empty_output_passes_forbidden_and_reports_assertion_on_missing_required(self) -> None:
        ok_empty = self._run("forb_empty", fc.OutputContract(forbidden=("NOPE",)), "")
        self.assertEqual(ok_empty.status, RunStatus.PASS)
        self.assertEqual(ok_empty.failures, [])

        bad_empty = self._run("req_empty", fc.OutputContract(required=("MUST_PRINT",)), "")
        self.assertEqual(bad_empty.status, RunStatus.FAIL)
        self.assertEqual(bad_empty.error_count, 0)
        failure = bad_empty.failures[0]
        self.assertEqual(failure.kind, FailureKind.ASSERTION)
        self.assertEqual(failure.case_name, "CHECK-DAG: MUST_PRINT")
        self.assertTrue(str(failure.file).endswith("step0.check"), failure.file)
        self.assertEqual(failure.line, 1)

    def test_exact_occurrence_counts_reject_extra_and_fewer_occurrences(self) -> None:
        contract = fc.OutputContract(counts=(("token", 2),))
        self.assertEqual(self._run("count_two", contract, "token\ntoken\n").status, RunStatus.PASS)
        fewer = self._run("count_one", contract, "token\n")
        self.assertEqual(fewer.status, RunStatus.FAIL)
        extra = self._run("count_three", contract, "token\ntoken\ntoken\n")
        self.assertEqual(extra.status, RunStatus.FAIL)
        self.assertIn("token", extra.failures[0].case_name)
        self.assertIn("excluded string found", extra.failures[0].message)

        zero = fc.OutputContract(counts=(("never", 0),))
        self.assertEqual(self._run("absent_ok", zero, "alpha\n").status, RunStatus.PASS)
        self.assertEqual(self._run("empty_ok", zero, "").status, RunStatus.PASS)
        present = self._run("absent_bad", zero, "alpha\nnever\n")
        self.assertEqual(present.status, RunStatus.FAIL)
        self.assertEqual(present.failures[0].case_name, "CHECK-NOT: never")

    def test_multiple_independent_assertions_all_report_failures(self) -> None:
        contract = fc.OutputContract(
            required=("MISSING_MARKER",), forbidden=("BANNED",), counts=(("token", 1),)
        )
        result = self._run("multi_fail", contract, "alpha\nBANNED\ntoken\ntoken\n")
        self.assertEqual(result.status, RunStatus.FAIL)
        names = {failure.case_name for failure in result.failures}
        self.assertIn("CHECK-DAG: MISSING_MARKER", names)
        self.assertIn("IMPLICIT-CHECK-NOT: BANNED", names)
        self.assertEqual(result.fail_count, len(result.failures))
        self.assertGreaterEqual(result.fail_count, 3)

    def test_raw_filecheck_via_shell_adapter_surfaces_assertion_failure_and_junit(self) -> None:
        assert FILECHECK is not None
        chk = self.proj / "direct.check"
        chk.write_text("CHECK: EXPECTED_HEADER\n", encoding="utf-8")
        inp = self.proj / "out.txt"
        inp.write_text("wrong header\n", encoding="utf-8")
        suite = SuiteSpec(
            name="direct-filecheck",
            adapter="shell",
            profile="unit",
            command=[FILECHECK, f"--input-file={inp}", str(chk)],
        )
        result, _ = execute_suite(
            suite,
            project_root=self.proj,
            evidence_dir=self.evidence,
            run_id="tb_direct_fc",
        )
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(result.fail_count, 1)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].kind, FailureKind.ASSERTION)
        self.assertEqual(result.failures[0].case_name, "CHECK: EXPECTED_HEADER")
        self.assertEqual(result.failures[0].line, 1)

        run = RunRecord(
            run_id="tb_direct_fc",
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
        data = json.loads(Path(str(run.run_json_path)).read_text(encoding="utf-8"))
        self.assertEqual(data["suites"][0]["failures"][0]["kind"], FailureKind.ASSERTION)
        xml = Path(str(run.junit_xml_path)).read_text(encoding="utf-8")
        parsed = parse_junit_xml_bounded(xml.encode("utf-8"))
        self.assertEqual((parsed["tests"], parsed["failures"], parsed["errors"]), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
