"""TDD tests for Phase 1C: the Haunted Terminal invariant probe.

Two layers are verified:

1. The fixture layer: eight hostile corpora, a pure invariant checker, and a real
   PTY + disposable ``-L`` tmux transport with rapid ``SIGWINCH`` resizing and an
   interrupted code point.
2. The reporting layer: the same script run through ``execute_suite`` as a
   ``pty_tmux`` suite. The faithful run must PASS, and ``--fault-injection`` must
   FAIL with actionable per-case failures, proving the assertions detect defects
   instead of tautologically passing.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import unittest

from hermes_testbench.adapters import execute_suite
from hermes_testbench.adapters.parsers import parse_pty_tmux
from hermes_testbench.fixtures import (
    HAUNTED_RESIZE_SCHEDULE,
    HAUNTED_TERMINAL_AREAS,
    check_haunted_frame,
    check_haunted_text,
    decode_haunted_stream,
    haunted_combining_corpus,
    haunted_invariant_selftest,
    haunted_invisible_control_corpus,
    haunted_malformed_escape_corpus,
    haunted_narrow_sizes,
    haunted_osc52_corpus,
    haunted_partial_utf8_chunks,
    haunted_partial_utf8_expected,
    haunted_render_frame,
    haunted_render_naive,
    haunted_resize_schedule,
    haunted_wide_corpus,
    run_haunted_terminal_probe,
)
from hermes_testbench.isolation import default_scratch_root
from hermes_testbench.schema import RunStatus, SuiteSpec
from hermes_testbench.tui_render import display_width

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE = REPO_ROOT / "scripts" / "haunted_terminal_probe.py"
PY = sys.executable

_CACHE = {}


def _scratch() -> Path:
    return default_scratch_root()


def _positive_report():
    if "positive" not in _CACHE:
        _CACHE["positive"] = run_haunted_terminal_probe()
    return _CACHE["positive"]


def _fault_report():
    if "fault" not in _CACHE:
        _CACHE["fault"] = run_haunted_terminal_probe(fault_injection=True)
    return _CACHE["fault"]


def _run_suite(name: str, *extra_args: str):
    evidence = Path(tempfile.mkdtemp(dir=str(_scratch()), prefix="tb-hnev-"))
    spec = SuiteSpec(
        name=name,
        adapter="pty_tmux",
        profile="terminal",
        command=[PY, str(PROBE), *extra_args],
        timeout_s=180.0,
        needs_pty=True,
        needs_tmux=True,
    )
    started = time.monotonic()
    result, _usage = execute_suite(
        spec,
        project_root=REPO_ROOT,
        evidence_dir=evidence,
        run_id="tb_haunted_test_" + name.replace("-", "_"),
        scratch_root=_scratch(),
    )
    elapsed = time.monotonic() - started
    return result, evidence, elapsed


class HauntedCorpusTests(unittest.TestCase):
    def test_corpora_cover_the_eight_required_areas(self) -> None:
        self.assertEqual(len(HAUNTED_TERMINAL_AREAS), 8)
        self.assertEqual(len(set(HAUNTED_TERMINAL_AREAS)), 8)

        combining = haunted_combining_corpus()
        self.assertTrue(combining)
        self.assertTrue(all(any(ord(ch) > 0x7F for ch in e.text) for e in combining))

        wide = haunted_wide_corpus()
        self.assertTrue(wide)
        for entry in wide:
            wide_chars = [
                ch
                for ch in entry.text
                if unicodedata.east_asian_width(ch) in ("W", "F") and not unicodedata.combining(ch)
            ]
            self.assertTrue(wide_chars, entry.name)
            self.assertGreaterEqual(display_width(entry.text), 2 * len(wide_chars), entry.name)

    def test_bidi_corpus_carries_every_invisible_family(self) -> None:
        texts = "".join(e.text for e in haunted_invisible_control_corpus())
        for point in ("\u202e", "\u2066", "\u200b", "\u200e", "\ufeff", "\U000e0067"):
            self.assertIn(point, texts)

    def test_malformed_and_osc52_corpora_are_exact_byte_payloads(self) -> None:
        malformed = haunted_malformed_escape_corpus()
        self.assertTrue(any(e.text.endswith("\x1b[31;") for e in malformed))
        self.assertTrue(any("\x1b]52;c;ZXZpbA==\n" in e.text for e in malformed))

        osc = haunted_osc52_corpus()
        self.assertTrue(all("\x1b]52;" in e.text for e in osc))
        self.assertTrue(any("\x1b\\" in e.text for e in osc), "ST terminator form missing")

    def test_partial_utf8_chunks_decode_losslessly_when_reassembled(self) -> None:
        chunks = haunted_partial_utf8_chunks()
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(isinstance(c, bytes) for c in chunks))
        decoded = decode_haunted_stream(chunks)
        self.assertEqual(decoded["text"], haunted_partial_utf8_expected())
        self.assertEqual(decoded["replacements"], 0)
        with self.assertRaises(UnicodeDecodeError):
            chunks[0].decode("utf-8")  # every chunk boundary is mid-code-point
        with self.assertRaises(UnicodeDecodeError):
            chunks[1].decode("utf-8")

    def test_narrow_sizes_and_resize_schedule_include_degenerate_geometry(self) -> None:
        sizes = haunted_narrow_sizes()
        self.assertIn((1, 1), sizes)
        self.assertIn((2, 2), sizes)
        self.assertTrue(any(cols < 10 for cols, _rows in sizes))
        schedule = haunted_resize_schedule()
        self.assertEqual(len(schedule), len(HAUNTED_RESIZE_SCHEDULE))
        self.assertGreaterEqual(len(schedule), 8)
        self.assertTrue(any(cols <= 24 for cols, _rows in schedule))
        self.assertTrue(any(cols >= 120 for cols, _rows in schedule))

    def test_corpora_are_deterministic_and_json_safe(self) -> None:
        for builder in (
            haunted_combining_corpus,
            haunted_wide_corpus,
            haunted_invisible_control_corpus,
            haunted_malformed_escape_corpus,
            haunted_osc52_corpus,
        ):
            first = builder()
            second = builder()
            self.assertEqual([e.name for e in first], [e.name for e in second])
            self.assertEqual([e.payload() for e in first], [e.payload() for e in second])
            json.dumps([e.to_dict() for e in first], ensure_ascii=False)


class HauntedCheckerTests(unittest.TestCase):
    def test_invariant_checker_selftest_accepts_good_and_rejects_bad(self) -> None:
        report = haunted_invariant_selftest()
        self.assertTrue(report["ok"], report["checks"])
        self.assertTrue(all(report["checks"].values()), report["checks"])

    def test_defective_renderer_is_rejected_on_every_hazard(self) -> None:
        hazards = (
            "\u65e5\u672c\u8a9e",
            "abc\u202edef\u202c",
            "\x1b]52;c;ZXZpbA==\x07",
            "\x1b[2J\u001b]0;t\x07",
        )
        for text in hazards:
            bad = haunted_render_naive(text, 8, 4)
            violations = check_haunted_frame(bad, 8, height=4)
            self.assertTrue(violations, "checker accepted defective frame for %r" % text)

    def test_production_renderer_satisfies_the_same_invariants(self) -> None:
        for entry in haunted_invisible_control_corpus() + haunted_wide_corpus():
            for width in (10, 12, 24):
                frame = haunted_render_frame(entry.text, width, 6)
                plain_frame = haunted_render_frame("plain", width, 6)
                self.assertEqual(len(frame), 6)
                self.assertNotEqual(
                    frame,
                    plain_frame,
                    f"frame at width {width} ignored payload {entry.name}",
                )
                self.assertEqual(check_haunted_frame(frame, width, height=6), [], entry.name)

    def test_check_text_flags_controls_that_never_may_reach_a_terminal(self) -> None:
        self.assertEqual(check_haunted_text("plain ascii"), [])
        for hostile in ("\u202e", "\x07", "\x1b", "\u2066", "ZXZpbA=="):
            self.assertTrue(check_haunted_text("safe" + hostile), hostile)


class HauntedProbePositiveTests(unittest.TestCase):
    def test_faithful_probe_passes_every_area(self) -> None:
        report = _positive_report()
        self.assertFalse(report["fault_injection"])
        self.assertEqual(report["failed"], 0, [c for c in report["cases"] if c["status"] != "PASS"])
        self.assertTrue(report["all_passed"])
        self.assertGreaterEqual(report["passed"], 24)
        self.assertEqual(report["leaked_children"], 0)
        self.assertTrue(report["scratch_clean"])

    def test_every_area_has_at_least_one_passing_case(self) -> None:
        report = _positive_report()
        self.assertEqual(list(report["areas"].keys()), list(HAUNTED_TERMINAL_AREAS))
        for area, counts in report["areas"].items():
            self.assertGreaterEqual(counts["total"], 1, area)
            self.assertEqual(counts["fail"], 0, area)
            self.assertGreaterEqual(counts["pass"], 1, area)

    def test_pty_transport_observes_rapid_sigwinch_geometry_changes(self) -> None:
        report = _positive_report()
        self.assertTrue(report["pty"]["available"], report["pty"]["error"])
        self.assertGreaterEqual(report["pty"]["sigwinch_count"], 1)
        self.assertEqual(
            report["pty"]["applied_resizes"],
            [[cols, rows] for cols, rows in HAUNTED_RESIZE_SCHEDULE[1:]],
        )
        self.assertEqual(
            report["pty"]["final"], list(HAUNTED_RESIZE_SCHEDULE[-1]),
            "child must end at the last resized geometry",
        )

    def test_tmux_transport_is_isolated_and_cleaned_up(self) -> None:
        report = _positive_report()
        if not report["tmux"]["available"]:
            self.skipTest("tmux unavailable: " + report["tmux"]["error"])
        self.assertTrue(report["tmux"]["server_gone"])
        self.assertEqual(len(report["tmux"]["pane_size"]), 2)
        self.assertGreater(report["tmux"]["pane_size"][0], 0)

    def test_probe_scratch_directories_are_removed(self) -> None:
        scratch = Path(tempfile.mkdtemp(dir=str(_scratch()), prefix="tb-hnscratch-"))
        self.addCleanup(shutil.rmtree, scratch, True)
        report = run_haunted_terminal_probe(scratch_root=scratch, use_tmux=False)
        self.assertTrue(report["scratch_clean"])
        self.assertFalse(report["fault_injection"])
        self.assertEqual(report["failed"], 0)
        self.assertEqual(list(scratch.iterdir()), [], "probe left scratch entries behind")

    @unittest.skipUnless(shutil.which("tmux"), "tmux unavailable")
    def test_deeply_nested_scratch_root_still_runs_disposable_tmux(self) -> None:
        deep = Path(tempfile.mkdtemp(dir=str(_scratch()), prefix="tb-hndeep-")) / ("nested_level_" * 6) / "sub"
        deep.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, deep.parents[1], True)
        self.assertGreater(len(str(deep)), 115)
        report = run_haunted_terminal_probe(scratch_root=deep, use_tmux=True)
        self.assertTrue(report["tmux"]["available"], report["tmux"]["error"])
        self.assertTrue(report["tmux"]["server_gone"])
        self.assertTrue(report["scratch_clean"])
        self.assertEqual(report["failed"], 0)


class HauntedProbeNegativeControlTests(unittest.TestCase):
    def test_fault_injection_fails_with_actionable_messages(self) -> None:
        report = _fault_report()
        self.assertTrue(report["fault_injection"])
        self.assertGreaterEqual(report["failed"], 8, report["cases"])
        self.assertFalse(report["all_passed"])
        failing_areas = {c["area"] for c in report["cases"] if c["status"] == "FAIL"}
        for area in (
            "wide_display_cells",
            "bidi_and_invisible_controls",
            "malformed_ansi_osc",
            "osc52_clipboard",
            "narrow_dimensions",
            "rapid_sigwinch",
            "interrupted_output_partial_utf8",
        ):
            self.assertIn(area, failing_areas)
        messages = " ".join(c["message"] for c in report["cases"] if c["status"] == "FAIL")
        self.assertTrue(any(token in messages for token in ("cells", "forbidden code points", "geometry")))

    def test_fault_control_still_passes_untouched_shared_predicates(self) -> None:
        # The control is honest: cases that test sanitize/UTF-8 primitives still pass.
        report = _fault_report()
        passing = {c["name"] for c in report["cases"] if c["status"] == "PASS"}
        self.assertIn("combining_cells_count_once", passing)
        self.assertIn("eof_partial_utf8_uses_single_replacement", passing)


class HauntedSuiteIntegrationTests(unittest.TestCase):
    def test_faithful_suite_passes_through_execute_suite(self) -> None:
        result, evidence, elapsed = _run_suite("haunted-terminal")
        self.assertEqual(result.status, RunStatus.PASS, result.summary_excerpt)
        self.assertEqual(result.exit_code, 0)
        self.assertGreaterEqual(result.pass_count, 24)
        self.assertEqual(result.fail_count, 0)
        self.assertGreaterEqual(result.total_count, 24)
        self.assertEqual(result.isolation_meta.get("needs_pty"), True)
        self.assertEqual(result.isolation_meta.get("needs_tmux"), True)
        self.assertEqual(result.isolation_meta.get("leaked_children"), 0)
        self.assertLess(elapsed, 120.0)
        self.assertTrue(Path(result.stdout_path or "").exists())
        self.assertTrue(Path(result.sanitized_path or "").exists())
        self.addCleanup(shutil.rmtree, evidence, True)

    def test_negative_control_surfaces_failures_through_testbench(self) -> None:
        result, evidence, _elapsed = _run_suite(
            "haunted-terminal-negative-control", "--fault-injection"
        )
        self.addCleanup(shutil.rmtree, evidence, True)
        self.assertEqual(result.status, RunStatus.FAIL)
        self.assertNotEqual(result.exit_code, 0)
        self.assertGreaterEqual(result.fail_count, 8)
        self.assertGreaterEqual(result.pass_count, 5)
        self.assertTrue(result.failures)
        names = " ".join(f.case_name for f in result.failures)
        self.assertIn("wide", names)
        self.assertIn("osc52", names)
        self.assertTrue(
            all(f.kind == "assertion" for f in result.failures),
            [(f.case_name, f.kind) for f in result.failures],
        )
        self.assertTrue(any("cells exceed width" in f.message for f in result.failures))

    def test_probe_output_is_machine_readable_by_the_pty_tmux_parser(self) -> None:
        proc = subprocess.run(
            [PY, str(PROBE), "--no-tmux"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=180.0,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-400:])
        outcome = parse_pty_tmux(proc.stdout + proc.stderr)
        self.assertTrue(outcome.structured)
        self.assertEqual(outcome.failed, 0)
        self.assertGreaterEqual(outcome.passed, 24)
        self.assertGreaterEqual(outcome.total, 24)
        self.assertEqual(len(outcome.cases), outcome.total)
        self.assertIn("HAUNTED_SUMMARY", proc.stdout)

    def test_negative_control_exits_non_zero_and_reports_fail_markers(self) -> None:
        proc = subprocess.run(
            [PY, str(PROBE), "--fault-injection", "--no-tmux"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=180.0,
            check=False,
        )
        self.assertEqual(proc.returncode, 1)
        outcome = parse_pty_tmux(proc.stdout + proc.stderr)
        self.assertTrue(outcome.structured)
        self.assertGreaterEqual(outcome.failed, 8)
        self.assertGreaterEqual(outcome.passed, 5)

    def test_probe_source_has_no_machine_specific_paths(self) -> None:
        text = PROBE.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("TMUX=", text)


class HauntedSafetyTests(unittest.TestCase):
    def test_probe_never_connects_to_an_ambient_tmux_server(self) -> None:
        text = (REPO_ROOT / "hermes_testbench" / "fixtures.py").read_text(encoding="utf-8")
        segment = text[text.index("# ---------------------------------------------------------------- haunted terminal"):]
        self.assertNotIn('"TMUX"', segment)
        self.assertIn('"TMUX_TMPDIR"', segment)
        self.assertIn('"-L"', segment)
        self.assertIn("kill-server", segment)
        self.assertNotIn("Ghostty", segment)

    def test_no_haunted_child_processes_or_servers_survive(self) -> None:
        report = _positive_report()
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        ).stdout
        self.assertNotIn("haunted_child.py", out, "a haunted PTY child outlived the probe")
        socket = report["tmux"].get("socket") or ""
        if socket:
            probe = subprocess.run(
                ["tmux", "-L", socket, "list-sessions"],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
            self.assertNotEqual(probe.returncode, 0, "disposable tmux server still answers")


if __name__ == "__main__":
    unittest.main()
