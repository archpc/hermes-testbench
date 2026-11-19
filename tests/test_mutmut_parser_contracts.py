"""Mutation-driven contract tests for parse_mutmut (phase-2 triage).

54 survivors: the per-mutant detail message, scan continuation past junk and
unknown statuses, results_text precedence, and the emoji-tally fallback path
were unasserted. These pin the receipt-parsing contracts the mutation
adapter's evidence chain depends on.
"""

from __future__ import annotations

import unittest

from hermes_testbench.adapters.parsers import parse_mutmut


class MutmutResultsParsingTests(unittest.TestCase):
    def test_results_lines_build_cases_with_detail(self) -> None:
        out = parse_mutmut(
            "  pkg.mod.x_f__mutmut_1: killed\n  pkg.mod.x_f__mutmut_2: survived\n"
        )
        self.assertEqual(out.buckets["caught"], 1)
        self.assertEqual(out.buckets["survived"], 1)
        self.assertEqual(
            out.cases[0],
            {
                "name": "pkg.mod.x_f__mutmut_1",
                "status": "PASS",
                "message": "killed: pkg.mod.x_f__mutmut_1",
            },
        )
        self.assertEqual(out.cases[1]["status"], "FAIL")
        self.assertTrue(out.structured)

    def test_junk_and_unknown_status_lines_do_not_stop_scan(self) -> None:
        out = parse_mutmut("junk line\n  m1: weirdstatus\n  m2: killed\n")
        self.assertEqual(out.buckets["caught"], 1)
        self.assertEqual([c["name"] for c in out.cases], ["m2"])

    def test_default_results_text_reads_progress_lines(self) -> None:
        # With no results_text the progress text itself is scanned; a mutated
        # default or an always-true strip() guard loses these cases.
        out = parse_mutmut("  m1: killed\n")
        self.assertEqual([c["name"] for c in out.cases], ["m1"])
        self.assertEqual(out.buckets["caught"], 1)

    def test_results_text_overrides_progress_tally(self) -> None:
        out = parse_mutmut("🎉 5 🙁 3\n", results_text="  m1: killed\n")
        self.assertEqual([c["name"] for c in out.cases], ["m1"])
        # Per-mutant evidence wins; the fallback must not double-count.
        self.assertEqual(out.buckets["caught"], 1)
        self.assertEqual(out.buckets["survived"], 0)


class MutmutEmojiTallyFallbackTests(unittest.TestCase):
    def test_tally_fallback_buckets_and_summary(self) -> None:
        out = parse_mutmut("⠋ mutmut 3.8\n🎉 12 🙁 4 ⏰ 2 🫥 3\n")
        self.assertEqual(out.cases, [])
        self.assertEqual(out.buckets["caught"], 12)
        self.assertEqual(out.buckets["survived"], 4)
        self.assertEqual(out.buckets["timeout"], 2)
        self.assertEqual(out.buckets["inconclusive"], 3)
        self.assertEqual(
            out.summary_line,
            "mutmut: 12 caught, 4 survived, 2 timeout, 3 inconclusive",
        )
        # Note: tally-only fallback stays unstructured (no per-case evidence);
        # that is the existing contract, not a defect — buckets are authoritative.

    def test_last_tally_line_wins(self) -> None:
        out = parse_mutmut("🎉 1\n🎉 7 🙁 2\n")
        self.assertEqual(out.buckets["caught"], 7)
        self.assertEqual(out.buckets["survived"], 2)

    def test_unknown_emoji_counts_inconclusive(self) -> None:
        out = parse_mutmut("🤔 6\n")
        self.assertEqual(out.buckets["inconclusive"], 6)

    def test_same_bucket_emojis_accumulate(self) -> None:
        # 🎉 and 🧙 both map to caught; the tally must ADD, not overwrite.
        out = parse_mutmut("🎉 3 🧙 4\n")
        self.assertEqual(out.buckets["caught"], 7)

    def test_baseline_failure_marker(self) -> None:
        out = parse_mutmut("failed to collect tests\n")
        self.assertTrue(out.baseline_failed)
        self.assertTrue(out.structured)
        self.assertEqual(
            out.cases[0],
            {
                "name": "<baseline>",
                "status": "SKIP",
                "message": "mutmut could not collect a passing baseline",
            },
        )
        self.assertEqual(out.buckets["inconclusive"], 1)


if __name__ == "__main__":
    unittest.main()