"""Output-contract assertions through LLVM FileCheck and TestBench's shell adapter.

Every execution runs through TestBench's real ``execute_suite`` (``shell`` adapter,
isolation, Seatbelt confinement, evidence capture, and ``Gate 3`` ``SKIP`` when
``FileCheck`` is absent) and through ``TestBenchStore.finalize_run`` reporting
(``run.json`` and ``junit.xml``). Parser and contract-construction tests run
unconditionally even on hosts without LLVM installed.

Two layers, deliberately separated:

* pure functions — ``OutputContract`` validation, ``build_check_steps``,
  ``filecheck_command`` and ``parse_filecheck_diagnostics`` — are tested with no
  tool present, because they are what turns a contract into FileCheck text and
  turns FileCheck text back into attributable failures;
* ``FileCheckEndToEndTests`` is decorated with ``skipUnless`` on a FileCheck path
  resolved **at import time** (``FILECHECK = fc.find_filecheck()``). Installing
  FileCheck, or setting ``TESTBENCH_FILECHECK``, after this module is imported
  does not un-skip that class even though ``execute_output_contract`` resolves
  the binary per call. The class holds the positive *and* negative controls:
  a passing contract, and every failure shape (missing required, ordered
  violation, forbidden present, wrong counts, multiple independent failures)
  plus raw FileCheck run as an ordinary shell suite.

Assertion counts worth remembering when reading failures: ``required`` becomes
one ``CHECK-DAG`` invocation, ``ordered`` one ``CHECK`` invocation (sequential),
each ``counts`` entry one invocation, each ``forbidden`` pattern one
``--implicit-check-not`` invocation. ``FULL_CONTRACT`` therefore has five
FileCheck steps and seven declared assertions
(``output_contract_steps == 5``, ``output_contract_assertions == 7``).
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

# Resolved once at import: this is what decides whether FileCheckEndToEndTests runs.
FILECHECK = fc.find_filecheck()
requires_filecheck = unittest.skipUnless(FILECHECK, "LLVM FileCheck not installed")

# Satisfies FULL_CONTRACT: both DAG patterns, "first" before "second", exactly two
# "token" lines, and no "FORBIDDEN". Order of alpha/beta is irrelevant (DAG).
SATISFIED = "alpha\nbeta\nfirst\nsecond\ntoken\ntoken\n"
FULL_CONTRACT = fc.OutputContract(
    required=("alpha", "beta"),
    ordered=("first", "second"),
    forbidden=("FORBIDDEN",),
    counts=(("token", 2), ("never", 0)),
)


class ContractValidationTests(unittest.TestCase):
    def test_empty_or_contradictory_contracts_are_rejected(self) -> None:
        """Each constructor rejection maps to one specific misuse.

        ``OutputContract`` is frozen and validated in ``__post_init__``; the
        seven cases are, in order: no assertion at all; a pattern both asserted
        and forbidden (the contract would be unsatisfiable); a negative count; a
        duplicate count pattern (ambiguous, unlike duplicate ``required``
        patterns, which are silently deduped); an empty/blank pattern; an
        embedded newline; a bare CR. Patterns go to FileCheck verbatim, so a
        multi-line pattern would silently become two directives — hence the
        rejection rather than a rewrite.
        """
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
        """One invocation per intent, with the exact FileCheck text and flags.

        Splitting is required, not stylistic: FileCheck evaluates multiple
        ``--check-prefix`` groups as one interleaved sequence, and a trailing DAG
        group suppresses ``--implicit-check-not``. The assertions here fix the
        generated text, so a change in intent order or in the count encoding is
        visible: ``N > 0`` emits ``CHECK-COUNT-N`` *plus* a trailing ``CHECK-NOT``
        (rejecting an extra occurrence), ``N == 0`` emits a plain ``CHECK-NOT``,
        and every step carries ``--allow-empty`` so an empty input can still
        satisfy a "must not appear" assertion. The final step is the forbidden
        pattern expressed as an implicit check-not anchored on ``{{^}}``.
        """
        steps = fc.build_check_steps(FULL_CONTRACT)
        self.assertEqual(len(steps), 5)
        self.assertIn("CHECK-DAG: alpha", steps[0].check_text)
        self.assertIn("CHECK-DAG: beta", steps[0].check_text)
        self.assertIn("CHECK: first", steps[1].check_text)
        self.assertIn("CHECK: second", steps[1].check_text)
        self.assertIn("CHECK-COUNT-2: token", steps[2].check_text)
        # The trailing negative check is what makes the count *exact* rather
        # than "at least N".
        self.assertIn("CHECK-NOT: token", steps[2].check_text)
        self.assertEqual(steps[3].check_text.strip(), "CHECK-NOT: never")
        self.assertEqual(steps[3].extra_args, ("--allow-empty",))
        self.assertEqual(steps[4].check_text.strip(), "CHECK: {{^}}")
        self.assertEqual(steps[4].extra_args, ("--allow-empty", "--implicit-check-not=FORBIDDEN"))

    def test_command_carries_step_flags_and_input_file(self) -> None:
        """argv order: executable, step flags, input file, check file.

        The check file is the sole positional argument, so it must come last;
        asserting the whole list (not membership) pins that. The forbidden step's
        flags are asserted separately because ``--implicit-check-not`` must
        precede the positional file.
        """
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
        """A check-file diagnostic becomes a case named after the pattern.

        FileCheck's message ("expected string not found in input") does not name
        the string, so the parser recovers it from the snippet in the diagnostic
        window: ``// CHECK: gamma`` → ``case_name == "CHECK: gamma"``. ``file``
        and ``line`` are the *check file* location (where the directive lives),
        not the input location. Asserting all five fields is what separates
        "attributable failure" from a generic nonzero exit.
        """
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
        """The three non-plain directive spellings keep their identity.

        CHECK-NOT: case name is the directive + pattern; the excerpt includes the
        "found here" note (the offending input line), which is the actionable
        part. CHECK-COUNT: the count in the message ("2 out of 2") is preserved.
        IMPLICIT-CHECK-NOT: the pattern is recovered from the
        ``-implicit-check-not='beta'`` argument echo, and because the diagnostic
        source is the pseudo-file ``command line``, ``file``/``line`` become
        ``None`` rather than a path that does not exist.
        """
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
        """Only FileCheck-shaped ``error:`` lines are diagnostics.

        This parser runs on every shell suite's captured output, so false
        positives would invent failures. Dropped: ``warning:``; a compiler error
        (``foo.c:10:5: error:``) whose text is not FileCheck's; and
        ``conf.check:3:1: error: unexpected token`` — the location looks like a
        check file, but the text names no directive and matches no global
        FileCheck phrase, which is the discriminator. A bare
        ``error: no check strings found…`` IS accepted (a real FileCheck global
        error) with no file/line; repeating it must collapse to one record,
        because the dedupe key is (case name, file, line, message).
        """
        self.assertEqual(fc.parse_filecheck_diagnostics("warning: deprecated flag\nall good\n"), [])
        self.assertEqual(
            fc.parse_filecheck_diagnostics("foo.c:10:5: error: use of undeclared identifier 'x'\n"),
            [],
        )
        self.assertEqual(
            fc.parse_filecheck_diagnostics("conf.check:3:1: error: unexpected token\n"),
            [],
        )
        bare = "error: no check strings found with prefix 'CHECK:'\n"
        record = fc.parse_filecheck_diagnostics(bare)[0]
        self.assertEqual(record["kind"], FailureKind.ASSERTION)
        self.assertIsNone(record["file"])
        self.assertIn("no check strings", record["message"])
        self.assertEqual(len(fc.parse_filecheck_diagnostics(bare + bare)), 1)

    def test_parse_shell_handles_filecheck_input_dump_without_false_error_counts(self) -> None:
        """A FileCheck "Input was:" dump must not add cases or errors.

        The dump echoes the input, including a line containing ``error: no match
        found in search range``; that line is neither at line start in
        ``error: …`` form nor FileCheck-shaped, so it is not a diagnostic.
        With no TB-/PASS/FAIL markers present, ``parse_shell`` delegates counts
        to the diagnostic list and *overwrites* ``failed``/``total`` with
        ``len(diagnostics)`` — one assertion, one case, not one per process or
        per echoed line. ``errors == 0`` is the regression assertion: this output
        used to be read as a harness error because of the echoed word.
        """
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
        """Absent FileCheck is a Gate 3 SKIP, not a failure or a fallback engine.

        Both halves are mocked so the test is host-independent:
        ``fc.find_filecheck -> None`` makes the contract build a suite whose
        ``required_binaries`` name FileCheck, and ``_missing_binary -> "FileCheck"``
        guarantees the prerequisite gate reports it missing even where a binary
        exists. The assertions pin the *shape* of the skip: status SKIP, exactly
        one skipped case, and the tool named in the excerpt — so a user learns
        which prerequisite to install. There is no interpreter-free fallback;
        an unverifiable contract is never silently "passed".
        """
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
        """``TESTBENCH_FILECHECK`` is read per call and honoured both ways.

        ``off`` is one of the disable words, so discovery returns ``None`` even
        where a real FileCheck is installed; the disable-word path is what makes
        the override trustworthy. A path to an executable is returned verbatim
        (after ``expanduser``), without a search. The original environment value
        is restored in ``finally`` so this test cannot leak configuration into
        the rest of the suite — ordering independence matters because
        ``find_filecheck`` inspects the environment on every call.
        """
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
        # Each call is one generated /bin/sh script in one execute_suite run;
        # the name doubles as the evidence slug and the run id suffix, so runs
        # in this class cannot alias each other's artifacts.
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
        """A satisfied contract is one passing case with evidence on disk.

        The contract is modelled as *one suite containing one case*, not one per
        assertion: the generated script ANDs all steps and reports a single rc.
        Metrics prove the run really went through FileCheck (engine name and step
        count 5), and the stdout/sanitized files existing prove the isolation
        layer captured evidence rather than the caller synthesising a result.
        """
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
        """Producer mode judges what the program printed, not its exit status.

        The producer prints header/item/item/footer once and is captured
        (stdout+stderr) into ``$TMPDIR/fc_captured.txt`` inside the sandbox; the
        contract then checks that capture. The passing contract exercises order
        plus an exact count; the failing one declares ``item == 1`` against two
        printed occurrences, so it must FAIL with an ASSERTION failure naming
        the pattern — the negative control for "extra occurrence" without any
        change to the producer.

        Scope note: the generated script contains exactly one producer
        invocation, but these assertions only observe the captured output, so
        they would not detect a hypothetical double execution that produced the
        same text. The once-only property is structural, not asserted here.
        """
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
        # ASSERTION, not harness_error: the program ran fine, the contract is
        # what failed. This distinction is why a red output-contract run is
        # actionable rather than "tooling broke".
        self.assertEqual(bad_res.failures[0].kind, FailureKind.ASSERTION)
        self.assertIn("item", bad_res.failures[0].case_name)

    def test_missing_required_pattern_reports_actionable_failure(self) -> None:
        """A missing required pattern names the pattern and the check-file line.

        ``case_name`` is the directive plus pattern; ``file`` is the generated
        ``step0.check`` (the location FileCheck reports is the check file, *not*
        the input line) and ``line`` is that file's first line. The excerpt
        carries the check text, so a reader can see what was expected without
        opening the evidence dir.
        """
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
        """Ordering failure is attributed to the pattern that could not match.

        Input is ``second`` then ``first``. ``CHECK: first`` matches on line 2;
        ``CHECK: second`` then has nothing left to scan, so the failure names
        ``CHECK: second`` — the *unmatched* directive, not the one that matched.
        Asserting the case name (not just FAIL status) is what separates this
        from a generic "output wrong".
        """
        result = self._run("ordered", fc.OutputContract(ordered=("first", "second")), "second\nfirst\n")
        self.assertEqual(result.status, RunStatus.FAIL)
        failure = result.failures[0]
        self.assertEqual(failure.case_name, "CHECK: second")
        self.assertIn("expected string not found", failure.message)

    def test_forbidden_pattern_fails_the_run(self) -> None:
        """Forbidden patterns fail via implicit-check-not, with no file location.

        The diagnostic's source is the pseudo-file ``command line`` (the pattern
        was passed as an argument, not written in a check file), so ``file`` and
        ``line`` are ``None`` by design. The excerpt still names the offending
        text, which is the usable part of the record.
        """
        contract = fc.OutputContract(required=("alpha",), forbidden=("BANNED",))
        result = self._run("forbidden", contract, "alpha\nBANNED\n")
        self.assertEqual(result.status, RunStatus.FAIL)
        failure = result.failures[0]
        self.assertEqual(failure.case_name, "IMPLICIT-CHECK-NOT: BANNED")
        self.assertIsNone(failure.file)
        self.assertIsNone(failure.line)
        self.assertIn("BANNED", failure.excerpt)

    def test_empty_output_passes_forbidden_and_reports_assertion_on_missing_required(self) -> None:
        """Empty input is a legal PASS for negative assertions and a FAIL for positive ones.

        ``--allow-empty`` exists precisely so "must not appear" can be evaluated
        against zero bytes instead of FileCheck erroring on an empty file. The
        required-pattern case on empty input must be an ASSERTION with
        ``error_count == 0``: the harness worked, the assertion did not hold.
        """
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
        """``counts`` is exact in both directions, and 0 is a suppression.

        Fewer occurrences fail the ``CHECK-COUNT-N`` prefix; *more* fail the
        trailing ``CHECK-NOT``, whose message is "excluded string found" — the
        assertion that distinguishes the two error directions. A count of 0 is
        just ``CHECK-NOT``: absent and empty input both pass, present input fails
        with the ``CHECK-NOT: never`` case name.
        """
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
        """Independent intents fail independently and are all reported.

        The script sets ``rc=1`` per failing step but keeps running, so the whole
        evidence stream is collected instead of stopping at the first failure.
        The input violates three intents (missing required, forbidden present,
        wrong count), hence ``fail_count >= 3``; the names assertion proves both
        the DAG and implicit-check-not forms survive into one record set, and
        ``fail_count == len(failures)`` pins that no failure is deduped away when
        the run is aggregated.
        """
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
        """A hand-written FileCheck suite reports an assertion, not an error.

        This bypasses the contract layer: the suite command is FileCheck itself
        with a check file, going through the ordinary shell adapter. The point is
        that the adapter's diagnostic parser — not the contract builder — owns
        the attribution, so ``error_count == 0`` and ``fail_count == 1``.
        ``line == 1`` is the check file's line. After ``finalize_run`` the same
        classification must survive both projections: ``run.json`` keeps
        ``kind == assertion`` and ``junit.xml`` counts 1 test / 1 failure /
        0 errors. A JSON-only assertion would miss a renderer regression and
        vice versa, which is why both are read back.
        """
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
