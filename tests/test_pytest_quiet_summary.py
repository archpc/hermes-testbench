"""Regression: zero-config discovery runs ``pytest -q``, whose final summary has no ``=`` rule.

Fixture text is verbatim pytest 9.2.0 ``-q`` output captured on this host.
"""

from __future__ import annotations

import unittest

from hermes_testbench.adapters.parsers import parse_python_pytest

Q_MIX = (
    "..F.sxE\n"
    "==================================== ERRORS ====================================\n"
    "__________________________ ERROR at setup of test_err __________________________\n"
    "E   RuntimeError: fixture boom\n"
    "=================================== FAILURES ===================================\n"
    "__________________________________ test_fail ___________________________________\n"
    "E   assert 1 == 2\n"
    "tests/test_mix.py:4: AssertionError\n"
    "=========================== short test summary info ============================\n"
    "FAILED tests/test_mix.py::test_fail - assert 1 == 2\n"
    "ERROR tests/test_mix.py::test_err - RuntimeError: fixture boom\n"
    "1 failed, 2 passed, 1 skipped, 1 xfailed, 1 error in 0.01s\n"
)

Q_PASS = "..                                                                       [100%]\n2 passed in 0.00s\n"


class PytestQuietSummaryTests(unittest.TestCase):
    def test_quiet_mixed_summary_counts(self) -> None:
        o = parse_python_pytest(Q_MIX)
        self.assertTrue(o.structured)
        self.assertEqual((o.total, o.passed, o.failed, o.skipped, o.errors), (5, 2, 1, 1, 1))
        self.assertEqual(o.summary_line, "1 failed, 2 passed, 1 skipped, 1 xfailed, 1 error in 0.01s")
        self.assertEqual(
            sorted((c["name"], c["status"]) for c in o.cases),
            [("tests/test_mix.py::test_err", "ERROR"), ("tests/test_mix.py::test_fail", "FAIL")],
        )

    def test_quiet_all_pass_summary_counts(self) -> None:
        o = parse_python_pytest(Q_PASS)
        self.assertTrue(o.structured)
        self.assertEqual((o.total, o.passed, o.failed, o.skipped, o.errors), (2, 2, 0, 0, 0))

    def test_quiet_summary_with_warning_and_rerun_suffix(self) -> None:
        o = parse_python_pytest("3 passed, 1 warning in 1.20s (0:00:01)\n")
        self.assertEqual((o.total, o.passed), (3, 3))

    def test_delimited_summary_still_wins_and_is_unchanged(self) -> None:
        o = parse_python_pytest("===== 1 failed, 4 passed, 2 skipped in 0.05s =====\n")
        self.assertEqual((o.total, o.passed, o.failed, o.skipped), (7, 4, 1, 2))

    def test_prose_mentioning_counts_is_not_a_summary(self) -> None:
        o = parse_python_pytest("note: 3 passed in review, see log\nwe saw 2 failed in staging yesterday\n")
        self.assertFalse(o.structured)
        self.assertEqual(o.total, 0)


if __name__ == "__main__":
    unittest.main()
