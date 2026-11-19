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
    parse_mutmut,
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


class MutmutUnicodeMutantNameTests(unittest.TestCase):
    """mutmut >= 3 names method mutants with the Unicode class marker ``ǁ``
    (U+01C1), e.g. ``pkg.mod.xǁStyleǁ__call____mutmut_2``. The results-line
    parser must not silently drop them: dropping kills and survivors corrupts
    the score denominator and hides per-mutant evidence. Observed live on
    hermes-tmux-workspace run tb_20261119_190950_6c1b2120: 760 of 3178 verdicts
    (every class-method mutant) vanished from the receipt.
    """

    def test_class_scoped_unicode_names_are_counted_and_preserved(self) -> None:
        results_text = (
            "    pkg.mod.xǁStyleǁ__init____mutmut_1: killed\n"
            "    pkg.mod.xǁStyleǁ__call____mutmut_2: survived\n"
            "    pkg.mod.x_plain__mutmut_3: killed\n"
            "    pkg.mod.xǁStyleǁfg__mutmut_4: no tests\n"
        )
        outcome = parse_mutmut("", results_text=results_text)
        self.assertEqual(outcome.buckets["caught"], 2)
        self.assertEqual(outcome.buckets["survived"], 1)
        self.assertEqual(outcome.buckets["inconclusive"], 1)
        self.assertEqual(outcome.total, 4)
        self.assertEqual(
            {case["name"] for case in outcome.cases},
            {
                "pkg.mod.xǁStyleǁ__init____mutmut_1",
                "pkg.mod.xǁStyleǁ__call____mutmut_2",
                "pkg.mod.x_plain__mutmut_3",
                "pkg.mod.xǁStyleǁfg__mutmut_4",
            },
        )

    def test_unicode_survivors_change_the_score_denominator(self) -> None:
        # Two killed method mutants + one survived module mutant: the true
        # score is 2/3. Silently dropping the Unicode names yields 0/1.
        results_text = (
            "    pkg.mod.xǁKlassǁwork__mutmut_1: killed\n"
            "    pkg.mod.xǁKlassǁwork__mutmut_2: killed\n"
            "    pkg.mod.x_helper__mutmut_1: survived\n"
        )
        outcome = parse_mutmut("", results_text=results_text)
        self.assertEqual(outcome.passed, 2)
        self.assertEqual(outcome.failed, 1)
        self.assertEqual(outcome.total, 3)

    def test_hyphenated_module_names_still_parse(self) -> None:
        outcome = parse_mutmut(
            "", results_text="    my-module.x_f__mutmut_1: killed\n"
        )
        self.assertEqual(outcome.buckets["caught"], 1)
        self.assertEqual(outcome.cases[0]["name"], "my-module.x_f__mutmut_1")


class UnittestDocstringVerboseTests(unittest.TestCase):
    """unittest -v prints docstring-described tests on TWO lines: the test
    identity line without a state, then the docstring's first line carrying
    ``... ok`` / ``... skipped 'reason'``. The parser must attach the state to
    the pending identity instead of dropping the case. Observed live on
    core-unit run tb_20261119_191953_3f2f1afc: 10 of 224 case identities were
    lost and junit padded them with synthetic skip_N/pass_N placeholders.
    """

    def test_docstring_ok_and_skip_keep_their_identity(self) -> None:
        text = (
            "test_plain (tests.test_x.PlainTests.test_plain) ... ok\n"
            "test_doc (tests.test_x.DocTests.test_doc)\n"
            "First line of the docstring. ... ok\n"
            "test_doc_skip (tests.test_x.DocTests.test_doc_skip)\n"
            "Docstring skip. ... skipped 'capability unavailable'\n"
            "Ran 3 tests in 0.001s\n"
            "\n"
            "OK (skipped=1)\n"
        )
        outcome = parse_python_unittest(text)
        self.assertEqual((outcome.total, outcome.passed, outcome.skipped), (3, 2, 1))
        by_name = {c["name"]: c for c in outcome.cases}
        self.assertEqual(len(by_name), 3)
        self.assertEqual(by_name["test_doc (tests.test_x.DocTests.test_doc)"]["status"], "PASS")
        skip = by_name["test_doc_skip (tests.test_x.DocTests.test_doc_skip)"]
        self.assertEqual(skip["status"], "SKIP")
        self.assertEqual(skip["message"], "capability unavailable")

    def test_docstring_failure_keeps_identity_and_detail(self) -> None:
        text = (
            "test_doc_fail (tests.test_x.DocTests.test_doc_fail)\n"
            "Docstring fail. ... FAIL\n"
            "\n"
            "======================================================================\n"
            "FAIL: test_doc_fail (tests.test_x.DocTests.test_doc_fail)\n"
            "----------------------------------------------------------------------\n"
            "Traceback (most recent call last):\n"
            '  File "x.py", line 3, in test_doc_fail\n'
            "    self.assertEqual(1, 2)\n"
            "AssertionError: 1 != 2\n"
            "\n"
            "----------------------------------------------------------------------\n"
            "Ran 1 test in 0.001s\n"
            "\n"
            "FAILED (failures=1)\n"
        )
        outcome = parse_python_unittest(text)
        self.assertEqual((outcome.total, outcome.failed), (1, 1))
        self.assertEqual(len(outcome.cases), 1)
        case = outcome.cases[0]
        self.assertEqual(case["name"], "test_doc_fail (tests.test_x.DocTests.test_doc_fail)")
        self.assertEqual(case["status"], "FAIL")
        self.assertIn("1 != 2", case["message"])

    def test_docstring_line_with_ellipsis_inside_still_parses(self) -> None:
        text = (
            "test_dots (tests.test_x.DocTests.test_dots)\n"
            "Wait for it... then more... ... ok\n"
            "Ran 1 test in 0.001s\n"
            "\n"
            "OK\n"
        )
        outcome = parse_python_unittest(text)
        self.assertEqual(len(outcome.cases), 1)
        self.assertEqual(outcome.cases[0]["status"], "PASS")
        self.assertEqual(outcome.cases[0]["name"], "test_dots (tests.test_x.DocTests.test_dots)")


if __name__ == "__main__":
    unittest.main()
