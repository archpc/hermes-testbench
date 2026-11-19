"""Regression tests for review findings R1 (marker block over-counting) and C5.

R1: a bare FAIL/TB-FAIL/FAILED marker with no inline text used to count every
following indented non-empty line as its own failure. Only explicit bullet or
numbered items may split into separate failures; other indented lines are the
excerpt of the single failure the marker represents, and indented markers are
parsed as their own markers.

The mechanism under test is ``parsers._summarize_markers`` (shared by every
marker-based adapter) plus the parsers that merely delegate to it. The block
rules it must satisfy, in the order the code applies them:

* a bare marker owns the following indented, non-blank lines;
* an indented line that is itself a marker (PASS/FAIL/SKIP, any spelling)
  terminates the block and becomes its own case — never excerpt text;
* the block is a *list* (one failure per item) only when its **first** line is
  a bullet/numbered item;
* otherwise the whole block is one failure's excerpt, and the message is the
  most specific line: a reversal scan for ``*Error:``-shaped text, else the
  last line.

C5: ``parse_python_unittest`` output is unaffected by this scanner; it reports
from the runner's own ``Ran``/``FAILED`` header, which is why the last class
exists as a guard against "fixing" the marker grammar into the structured path.
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
        # The block opens with a traceback header, not a bullet, so it is one
        # failure. ``failed == total == 1`` is the count invariant; the excerpt
        # assertions prove the block was kept (not dropped) and the message
        # assertion proves the reversal scan picked AssertionError over the
        # trailing line. Before the fix this produced one failure per line.
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
        # An indented marker ends the block, so the bare FAIL has no excerpt and
        # the two PASS lines are their own cases. If indented markers were eaten
        # as excerpt text, passed would be 0 and total 1.
        outcome = parse_shell("FAIL\n  PASS a\n  PASS b\n")
        self.assertEqual(outcome.passed, 2)
        self.assertEqual(outcome.failed, 1)
        self.assertEqual(outcome.total, 3)
        statuses = [case["status"] for case in outcome.cases]
        self.assertEqual(statuses.count("PASS"), 2)
        self.assertEqual(statuses.count("FAIL"), 1)

    def test_indented_fail_marker_is_not_consumed_as_excerpt(self) -> None:
        # Two failures, not one failure with "FAIL: second" in its excerpt.
        # This is the sharpest form of the over/under-count boundary.
        outcome = parse_shell("FAIL\n  FAIL: second\n")
        self.assertEqual(outcome.failed, 2)
        self.assertEqual(outcome.total, 2)

    def test_tb_fail_bare_marker_in_other_parsers_counts_once(self) -> None:
        # The grammar lives in one shared scanner, so every marker-based parser
        # inherits the fix. subTest names the failing parser in the report; the
        # three chosen parsers differ in what they add (compile diagnostics,
        # banner detection, summary handling) but all start from markers.
        for parser in (parse_cobol, parse_generic, parse_sqlite_probe):
            with self.subTest(parser=parser.__name__):
                outcome = parser("TB-FAIL\n  AssertionError: nope\n")
                self.assertEqual(outcome.failed, 1)
                self.assertEqual(outcome.total, 1)


class FailedHeaderListTests(unittest.TestCase):
    def test_failed_header_numbered_list_splits_per_item(self) -> None:
        # A block that opens with "1." IS a list: one failure per numbered item,
        # each item's text used verbatim as its message.
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
        # All three list spellings the regex accepts: ``1)``, ``-``, ``*``.
        # Exact message list (not just a count) proves order and per-item text
        # survive the split and that no numbering/bullet prefix leaks into it.
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
        # Negative control for the list rule: identical shape, non-bullet first
        # line. Must stay 1 failure with the traceback in the excerpt.
        outcome = parse_shell(
            "FAILED:\n"
            "  Traceback (most recent call last):\n"
            "  AssertionError: boom\n"
        )
        self.assertEqual(outcome.failed, 1)
        self.assertEqual(outcome.total, 1)
        self.assertIn("Traceback", outcome.failures[0]["excerpt"])

    def test_numbered_items_outside_failed_header_still_split(self) -> None:
        # The list rule is a property of the block, not of the literal "FAILED:"
        # spelling: an inline TB-FAIL marker is the same grammar, reached through
        # parse_generic instead of parse_shell.
        outcome = parse_generic("TB-FAIL:\n  1. a\n  2. b\n")
        self.assertEqual(outcome.failed, 2)
        self.assertEqual(outcome.total, 2)


class BulletShapedExcerptTests(unittest.TestCase):
    """R1-residual (Fable 5.2 review): bullet-shaped lines inside a failure's
    excerpt (unittest diff ``- ``/``+ `` lines, numbered stack frames) must not
    split one bare marker into several failures. A block is a list only when it
    starts with a bullet item."""

    def test_unittest_diff_lines_count_one_failure_with_assertion_message(self) -> None:
        # The block begins with the AssertionError line, so "  - two" etc. are
        # excerpt. Asserting the *message* starts with AssertionError is what
        # distinguishes "kept as excerpt" from "split per diff line".
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
        # Traceback header first, list diff later: still one failure, message is
        # the AssertionError line (reversal scan), excerpt retains the traceback.
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
        # Go-style frames: "1) main.go:10" is bullet-shaped but not the block's
        # first line, so it stays excerpt. Note there is no ``AssertionError``
        # line here, so the message falls back to the last block line.
        outcome = parse_generic("TB-FAIL\n  panic: boom\n  1) main.go:10\n  2) runtime.go:99\n")
        self.assertEqual(outcome.failed, 1)


class UnittestParseStillStructuredTests(unittest.TestCase):
    def test_plain_unittest_output_unaffected(self) -> None:
        # parse_python_unittest counts from "Ran 2 tests" and the FAILED clause;
        # it does not use the marker scanner, so R1's grammar change must not
        # move these numbers. The verbose per-test "... FAIL" line is detail,
        # not a marker-owned block.
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
