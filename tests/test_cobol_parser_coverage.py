"""Mutation-driven coverage for parse_cobol (campaign phase-2 triage).

The declared mutation selection had 240 survivors in the cobol cluster
(parse_cobol 96, _summarize_markers 94, _apply_cobol_details 26,
_cobol_inline_details 24) while only one selected test touched parse_cobol
(bare TB-FAIL counting). These tests pin the documented contracts: unittest
delegation, success markers, contradiction summary, inline evidence folding,
compile-error attribution, and truncation. Each was verified to kill
non-equivalent survivors in a disposable Seatbelt-isolated stage.
"""

from __future__ import annotations

import unittest

from hermes_testbench.adapters.parsers import _cobol_inline_details, parse_cobol


class CobolUnittestDelegationTests(unittest.TestCase):
    def test_unittest_summary_delegates(self) -> None:
        out = parse_cobol("Ran 3 tests in 0.01s\n\nOK\n")
        self.assertEqual((out.total, out.passed, out.failed), (3, 3, 0))

    def test_delegation_with_leading_banner_needs_multiline(self) -> None:
        # The 'Ran N tests' line is rarely first; dropping re.M breaks this.
        out = parse_cobol("cobc (GnuCOBOL) 3.2\nRan 3 tests in 0.01s\n\nOK\n")
        self.assertEqual((out.total, out.passed, out.failed), (3, 3, 0))


class CobolSuccessMarkerTests(unittest.TestCase):
    def test_success_markers_count_one_passed(self) -> None:
        for marker in ("ALL TESTS PASSED", "TESTS PASSED", "SELFTEST OK"):
            with self.subTest(marker=marker):
                out = parse_cobol(marker + "\n")
                # Exact counts: a success-only run is ONE passing test, not
                # 'at least one' (weak >= assertions let max(...,2) survive).
                self.assertEqual(out.total, 1)
                self.assertEqual(out.passed, 1)
                self.assertEqual(out.failed, 0)
                self.assertTrue(out.structured)

    def test_summary_counts_line_is_applied(self) -> None:
        out = parse_cobol("SELFTEST OK\n3 passed, 0 failed\n")
        self.assertEqual((out.total, out.passed, out.failed), (3, 3, 0))
        self.assertEqual(out.summary_line, "3 passed, 0 failed")
        self.assertTrue(out.structured)

    def test_success_alongside_failures_is_flagged(self) -> None:
        out = parse_cobol("TB-FAIL case_x\nALL TESTS PASSED\n")
        self.assertEqual(out.failed, 1)
        self.assertEqual(
            out.summary_line, "cobol selftest reported success but also failures"
        )

    def test_plain_output_is_not_structured(self) -> None:
        out = parse_cobol("some unrelated output\n")
        self.assertFalse(out.structured)
        self.assertEqual(out.total, 0)


class CobolInlineEvidenceTests(unittest.TestCase):
    def test_indented_block_folds_into_its_marker(self) -> None:
        out = parse_cobol("TB-FAIL case_x\n  expected 42\n  actual 43\n")
        self.assertEqual(out.failed, 1)
        failure = out.failures[0]
        self.assertEqual(failure["case_name"], "case_x")
        self.assertEqual(failure["excerpt"], "expected 42\nactual 43")
        # The most specific (last) line is the actionable reason.
        self.assertEqual(failure["message"], "actual 43")

    def test_orphan_lines_and_indented_markers_do_not_create_details(self) -> None:
        details = _cobol_inline_details(
            "  orphan\nTB-FAIL x\n  ev1\n  TB-FAIL y\n  ev2\n"
        )
        self.assertEqual(details, {"x": ["ev1"]})

    def test_evidence_truncated_to_800(self) -> None:
        out = parse_cobol("TB-FAIL case_x\n  " + "z" * 900 + "\n")
        failure = out.failures[0]
        self.assertEqual(len(failure["excerpt"]), 800)
        self.assertEqual(len(failure["message"]), 800)


class CobolCompileErrorTests(unittest.TestCase):
    def test_bare_cobc_errors_become_attributed_failures(self) -> None:
        out = parse_cobol(
            "prog.cob:12: error: syntax error\nprog.cob:20: error: unknown word\n"
        )
        self.assertEqual(out.failed, 2)
        self.assertEqual(out.total, 2)
        self.assertEqual(out.failures[0]["case_name"], "cobol:prog.cob:12")
        self.assertIn("cobc error: syntax error", out.failures[0]["message"])
        self.assertTrue(out.summary_line.startswith("cobc compile failed:"))
        self.assertTrue(out.structured)

    def test_markers_suppress_compile_error_attribution(self) -> None:
        # Harness markers stay the single source of truth when present.
        out = parse_cobol("TB-FAIL case_x\nprog.cob:12: error: syntax error\n")
        self.assertEqual(out.failed, 1)
        self.assertEqual(out.failures[0]["case_name"], "case_x")


if __name__ == "__main__":
    unittest.main()