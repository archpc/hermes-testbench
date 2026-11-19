"""Mutation-driven contract tests for the shared marker scanner (phase-2).

_summarize_markers had 88 survivors under the declared mutation selection:
existing tests asserted counts but not the failure/case RECORDS (names, kinds,
messages, excerpts) or the block/bullet scanning rules. These pin the records
and the scan via parse_generic, the simplest shared consumer.
"""

from __future__ import annotations

import unittest

from hermes_testbench.adapters.parsers import parse_generic


class MarkerRecordContractTests(unittest.TestCase):
    def test_numbered_list_failure_records(self) -> None:
        out = parse_generic("FAILED:\n  1. first problem\n  2. second problem\n")
        self.assertEqual((out.total, out.failed), (2, 2))
        self.assertEqual(out.failures[0]["case_name"], "failure_1")
        self.assertEqual(out.failures[0]["kind"], "assertion")
        self.assertEqual(out.failures[0]["message"], "first problem")
        self.assertEqual(
            out.cases[1],
            {"name": "failure_2", "status": "FAIL", "message": "second problem"},
        )

    def test_excerpt_block_message_prefers_exception_line(self) -> None:
        out = parse_generic("TB-FAIL\n  context setup\n  AssertionError: boom\n")
        self.assertEqual(out.failed, 1)
        failure = out.failures[0]
        self.assertEqual(failure["case_name"], "case_1")
        self.assertEqual(failure["message"], "AssertionError: boom")
        self.assertEqual(failure["excerpt"], "context setup\nAssertionError: boom")

    def test_inline_and_bare_marker_names(self) -> None:
        out = parse_generic("TB-FAIL named_case\nFAIL\n")
        self.assertEqual(out.failed, 2)
        self.assertEqual(
            [f["case_name"] for f in out.failures], ["named_case", "case_2"]
        )

    def test_pass_and_skip_records(self) -> None:
        out = parse_generic("TB-PASS p1\nSKIP s1\n")
        self.assertEqual((out.passed, out.skipped, out.failed), (1, 1, 0))
        self.assertEqual(
            out.cases[0], {"name": "p1", "status": "PASS", "message": ""}
        )
        self.assertEqual(
            out.cases[1], {"name": "s1", "status": "SKIP", "message": "s1"}
        )

    def test_late_bullet_lines_are_excerpt_not_list_items(self) -> None:
        # A list only when the block OPENS with a bullet; diff-shaped lines
        # later in the block stay excerpt text (one failure, not three).
        out = parse_generic("TB-FAIL\n  real context\n  - removed line\n  + added line\n")
        self.assertEqual(out.failed, 1)
        self.assertIn("- removed line", out.failures[0]["excerpt"])
        self.assertIn("+ added line", out.failures[0]["excerpt"])

    def test_blank_lines_between_markers_do_not_skip_cases(self) -> None:
        out = parse_generic("TB-PASS a\n\nTB-FAIL b\n\nTB-PASS c\n")
        self.assertEqual((out.passed, out.failed), (2, 1))
        self.assertEqual([c["name"] for c in out.cases], ["a", "b", "c"])

    def test_excerpt_block_record_kind_and_case(self) -> None:
        out = parse_generic("TB-FAIL\n  context setup\n  AssertionError: boom\n")
        self.assertEqual(out.failures[0]["kind"], "assertion")
        self.assertEqual(
            out.cases[0],
            {"name": "case_1", "status": "FAIL", "message": "AssertionError: boom"},
        )

    def test_two_excerpt_blocks_increment_failure_count(self) -> None:
        out = parse_generic("TB-FAIL\n  first block\nFAIL\n  second block\n")
        self.assertEqual(out.failed, 2)
        self.assertEqual(
            [f["case_name"] for f in out.failures], ["case_1", "case_2"]
        )

    def test_blank_line_inside_block_ends_excerpt(self) -> None:
        out = parse_generic("TB-FAIL\n  line one\n\n  line two\n")
        self.assertEqual(out.failed, 1)
        self.assertEqual(out.failures[0]["excerpt"], "line one")

    def test_single_space_indent_is_excerpt(self) -> None:
        out = parse_generic("TB-FAIL\n line one\n line two\n")
        self.assertEqual(out.failed, 1)
        self.assertEqual(out.failures[0]["excerpt"], "line one\nline two")

    def test_markers_after_indented_marker_block_still_scanned(self) -> None:
        # An indented marker ends the excerpt block but must not end the scan:
        # break vs return differ only when more markers follow.
        out = parse_generic("FAIL\n  TB-PASS inner\nTB-PASS after\n")
        self.assertEqual((out.passed, out.failed), (2, 1))


if __name__ == "__main__":
    unittest.main()