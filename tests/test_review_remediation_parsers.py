"""Regression tests for review findings R1 (marker block over-counting) and C5.

R1: a bare FAIL/TB-FAIL/FAILED marker with no inline text used to count every
following indented non-empty line as its own failure. Only explicit bullet or
numbered items may split into separate failures; other indented lines are the
excerpt of the single failure the marker represents, and indented markers are
parsed as their own markers.
"""

from __future__ import annotations

import unittest

from hermes_testbench.adapters.parsers import (
    parse_cobol,
    parse_generic,
    parse_python_unittest,
    parse_shell,
    parse_sqlite_probe,
)


class BareFailMarkerTests(unittest.TestCase):
    def test_bare_fail_with_traceback_counts_one_failure(self) -> None:
        outcome = parse_shell(
            "FAIL\n"
            "  Traceback (most recent call last):\n"
            '    File "x.py", line 3, in <module>\n'
            "  AssertionError: boom\n"
        )
        self.assertEqual(outcome.failed, 1)
        self.assertEqual(outcome.total, 1)
        self.assertEqual(len(outcome.failures), 1)
        excerpt = outcome.failures[0]["excerpt"]
        self.assertIn("Traceback (most recent call last):", excerpt)
        self.assertIn('File "x.py", line 3, in <module>', excerpt)
        self.assertIn("AssertionError: boom", excerpt)
        self.assertEqual(outcome.failures[0]["message"], "AssertionError: boom")

    def test_indented_pass_markers_after_bare_fail_are_parsed(self) -> None:
        outcome = parse_shell("FAIL\n  PASS a\n  PASS b\n")
        self.assertEqual(outcome.passed, 2)
        self.assertEqual(outcome.failed, 1)
        self.assertEqual(outcome.total, 3)
        statuses = [case["status"] for case in outcome.cases]
        self.assertEqual(statuses.count("PASS"), 2)
        self.assertEqual(statuses.count("FAIL"), 1)

    def test_indented_fail_marker_is_not_consumed_as_excerpt(self) -> None:
        outcome = parse_shell("FAIL\n  FAIL: second\n")
        self.assertEqual(outcome.failed, 2)
        self.assertEqual(outcome.total, 2)

    def test_tb_fail_bare_marker_in_other_parsers_counts_once(self) -> None:
        for parser in (parse_cobol, parse_generic, parse_sqlite_probe):
            with self.subTest(parser=parser.__name__):
                outcome = parser("TB-FAIL\n  AssertionError: nope\n")
                self.assertEqual(outcome.failed, 1)
                self.assertEqual(outcome.total, 1)


class FailedHeaderListTests(unittest.TestCase):
    def test_failed_header_numbered_list_splits_per_item(self) -> None:
        outcome = parse_shell(
            "FAILED:\n"
            "  1. advisor profile missing: /tmp/fake/.hermes/profiles/opus\n"
            "  2. no profile carries ui_meta.hermes-bots\n"
        )
        self.assertEqual(outcome.failed, 2)
        self.assertEqual(outcome.total, 2)
        self.assertIn("advisor profile missing", outcome.failures[0]["message"])
        self.assertIn("ui_meta.hermes-bots", outcome.failures[1]["message"])

    def test_failed_header_mixed_numbered_and_bulleted_list_splits_per_item(self) -> None:
        outcome = parse_shell(
            "FAILED:\n"
            "  1) first item\n"
            "  - second item\n"
            "  * third item\n"
        )
        self.assertEqual(outcome.failed, 3)
        self.assertEqual(outcome.total, 3)
        messages = [f["message"] for f in outcome.failures]
        self.assertEqual(messages, ["first item", "second item", "third item"])

    def test_failed_header_plain_indented_block_counts_one(self) -> None:
        outcome = parse_shell(
            "FAILED:\n"
            "  Traceback (most recent call last):\n"
            "  AssertionError: boom\n"
        )
        self.assertEqual(outcome.failed, 1)
        self.assertEqual(outcome.total, 1)
        self.assertIn("Traceback", outcome.failures[0]["excerpt"])

    def test_numbered_items_outside_failed_header_still_split(self) -> None:
        outcome = parse_generic("TB-FAIL:\n  1. a\n  2. b\n")
        self.assertEqual(outcome.failed, 2)
        self.assertEqual(outcome.total, 2)


class BulletShapedExcerptTests(unittest.TestCase):
    """R1-residual (Fable 5.2 review): bullet-shaped lines inside a failure's
    excerpt (unittest diff ``- ``/``+ `` lines, numbered stack frames) must not
    split one bare marker into several failures. A block is a list only when it
    starts with a bullet item."""

    def test_unittest_diff_lines_count_one_failure_with_assertion_message(self) -> None:
        outcome = parse_shell(
            "FAIL\n"
            "  AssertionError: 'one\\ntwo\\nthree\\n' != 'one\\n2\\n3\\n'\n"
            "    one\n"
            "  - two\n"
            "  - three\n"
            "  + 2\n"
            "  + 3\n"
        )
        self.assertEqual((outcome.total, outcome.failed), (1, 1))
        self.assertTrue(outcome.failures[0]["message"].startswith("AssertionError:"))
        self.assertIn("- three", outcome.failures[0]["excerpt"])

    def test_traceback_then_list_diff_keeps_assertion_as_message(self) -> None:
        outcome = parse_shell(
            "FAIL\n"
            "  Traceback (most recent call last):\n"
            '    File "t.py", line 5, in test_x\n'
            "  AssertionError: Lists differ: ['a'] != ['x']\n"
            "  - ['a']\n"
            "  + ['x']\n"
        )
        self.assertEqual(outcome.failed, 1)
        self.assertTrue(outcome.failures[0]["message"].startswith("AssertionError: Lists differ"))
        self.assertIn("Traceback", outcome.failures[0]["excerpt"])

    def test_numbered_stack_frames_after_message_count_one(self) -> None:
        outcome = parse_generic("TB-FAIL\n  panic: boom\n  1) main.go:10\n  2) runtime.go:99\n")
        self.assertEqual(outcome.failed, 1)


class UnittestParseStillStructuredTests(unittest.TestCase):
    def test_plain_unittest_output_unaffected(self) -> None:
        outcome = parse_python_unittest(
            "test_ok (m.T.test_ok) ... ok\n"
            "test_bad (m.T.test_bad) ... FAIL\n"
            "\n"
            "Ran 2 tests in 0.001s\n"
            "FAILED (failures=1)\n"
        )
        self.assertEqual((outcome.total, outcome.passed, outcome.failed), (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
