"""Parser regression: zero-config discovery runs ``pytest -q``, whose final summary has no ``=`` rule.

Zero-config Python discovery drives pytest whenever the chosen interpreter can
import it, and the default invocation is ``pytest -q``. In quiet mode the run ends
with a bare line such as ``1 failed, 2 passed, 1 skipped, 1 xfailed, 1 error in
0.01s`` — no ``=====`` padding. A parser that only accepts the delimited rule falls
back to the exit status for every quiet run, which collapses real counts into one
synthetic pass/fail. These fixtures pin that shape so the fallback stays a genuine
last resort rather than the normal path.

The fixture text is verbatim pytest 9.2.0 ``-q`` output captured on one host, not
hand-written: the ``[100%]`` progress suffix, the padded progress line and the
``(0:00:01)`` duration suffix are exactly what ``_PYTEST_QUIET_SUMMARY_RE`` has to
tolerate. Nothing here spawns pytest, so these tests do not depend on pytest being
installed, or on which version is installed.

Counting detail that bites: ``xfailed``/``xpassed``/``warnings``/``rerun`` are not
buckets, so ``total`` is ``passed + failed + skipped + errors`` and can be smaller
than the number of comma-separated items on the summary line. The mixed fixture
carries an ``xfailed`` for precisely this reason.
"""

from __future__ import annotations

import unittest

from hermes_testbench.adapters.parsers import parse_python_pytest

# Mixed quiet run: every bucket appears at once, plus an ``xfailed`` that must be
# echoed in ``summary_line`` but never counted. The ``FAILED``/``ERROR`` short-summary
# lines are the only inputs that create failure records, so they are kept verbatim,
# with pytest's ``  - assert 1 == 2`` message tail intact.
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

# All-pass quiet run: the only verdict source is the bare summary. The padded
# progress line above it carries the trailing spaces and ``[100%]``, which is why
# the grammar is anchored on `` in <n>s$`` instead of on line shape.
Q_PASS = "..                                                                       [100%]\n2 passed in 0.00s\n"


class PytestQuietSummaryTests(unittest.TestCase):
    """One discriminating property per test; together they bracket the grammar."""

    def test_quiet_mixed_summary_counts(self) -> None:
        """Counts come from the quiet line, and only the short-summary lines make cases.

        Three separate claims, each failing for a different reason: ``structured``
        proves the run was not classified as unstructured (an exit-status fallback
        would leave counts at zero); the bucket tuple proves ``xfailed`` was not
        folded into ``skipped`` or ``total``; the case list proves failure records
        are named by ``path::test`` with ``FAIL``/``ERROR`` status, and that the
        traceback body contributes nothing. Sorted comparison makes the assertion
        independent of which pytest section (ERRORS before FAILURES) came first.
        """
        o = parse_python_pytest(Q_MIX)
        self.assertTrue(o.structured)
        self.assertEqual((o.total, o.passed, o.failed, o.skipped, o.errors), (5, 2, 1, 1, 1))
        self.assertEqual(o.summary_line, "1 failed, 2 passed, 1 skipped, 1 xfailed, 1 error in 0.01s")
        self.assertEqual(
            sorted((c["name"], c["status"]) for c in o.cases),
            [("tests/test_mix.py::test_err", "ERROR"), ("tests/test_mix.py::test_fail", "FAIL")],
        )

    def test_quiet_all_pass_summary_counts(self) -> None:
        """A clean quiet run must be structured from the summary alone.

        There are no marker lines to borrow counts from, so a green run is the case
        where a broken quiet-summary regex is most damaging: it would report
        unstructured and let the exit status stand in for the pass count.
        """
        o = parse_python_pytest(Q_PASS)
        self.assertTrue(o.structured)
        self.assertEqual((o.total, o.passed, o.failed, o.skipped, o.errors), (2, 2, 0, 0, 0))

    def test_quiet_summary_with_warning_and_rerun_suffix(self) -> None:
        """Singular ``warning`` and the ``(0:00:01)`` suffix are part of the grammar.

        pytest emits the parenthesised duration when the run crosses a second, and
        ``1 warning`` (not ``1 warnings``) for a single warning. Both variants are
        legal summary lines; anchoring on the plural or on a bare ``<n>s$`` would
        drop exactly the runs that most need their counts.
        """
        o = parse_python_pytest("3 passed, 1 warning in 1.20s (0:00:01)\n")
        self.assertEqual((o.total, o.passed), (3, 3))

    def test_delimited_summary_still_wins_and_is_unchanged(self) -> None:
        """Guard rail: the original ``=====`` summary path must not regress.

        Adding a second accepted shape is the obvious way to break the first, so this
        pins the delimited total (4+1+2=7) — a number that only appears if the rule
        was still recognised and demoted correctly.
        """
        o = parse_python_pytest("===== 1 failed, 4 passed, 2 skipped in 0.05s =====\n")
        self.assertEqual((o.total, o.passed, o.failed, o.skipped), (7, 4, 1, 2))

    def test_prose_mentioning_counts_is_not_a_summary(self) -> None:
        """Negative control: log text about past runs is not a verdict.

        Both lines contain ``N passed``/``N failed`` in prose. Without this test a
        loosened regex (search instead of full-line match) would satisfy every other
        case here while turning arbitrary log output into counts — a phantom PASS is
        worse than an unstructured run, because the exit status is at least honest.
        """
        o = parse_python_pytest("note: 3 passed in review, see log\nwe saw 2 failed in staging yesterday\n")
        self.assertFalse(o.structured)
        self.assertEqual(o.total, 0)


if __name__ == "__main__":
    unittest.main()
