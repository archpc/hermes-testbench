"""Tests for the Haunted Terminal invariant probe (Phase 1C).

Two layers are verified, and they fail for different reasons:

1. **Fixture layer** — five corpus builders (combining, wide, invisible, malformed,
   osc52) feeding eight areas, a pure invariant checker
   (``check_haunted_text`` / ``check_haunted_frame``), and a real PTY + disposable
   ``-L`` tmux transport with rapid ``SIGWINCH`` resizing and an interrupted
   multi-byte code point. These tests are cheap and deterministic; they assert that
   the inputs actually contain the hazards (a corpus that quietly lost its bidi
   characters would make every downstream assertion vacuous) and that the checker
   rejects a deliberately broken renderer.
2. **Reporting layer** — the same probe script run through ``execute_suite`` as a
   ``pty_tmux`` suite. The faithful run must PASS, and ``--fault-injection`` must
   FAIL with actionable per-case failures, proving the assertions detect defects
   instead of tautologically passing.

Why both directions: a checker that rejects everything would make the negative
control green and the faithful probe red; a checker that accepts everything is the
reverse. Only the pair distinguishes "the invariants hold" from "the invariants are
not being tested". A separate test also requires that cases exercising untouched
sanitize/UTF-8 primitives still PASS, so the control cannot be satisfied by making
the whole probe fail.

Cost and caching: a probe spawns a child on a fresh PTY, resizes it on a wall-clock
schedule, and (unless disabled) starts a throwaway tmux server, so each run costs
seconds. ``_CACHE`` memoizes one faithful and one fault report for the process; the
dicts are treated as read-only. A test that changes the environment (``PATH``,
``TMUX_TMPDIR``) after the first probe will not be reflected in the cached report.

Known flakiness, stated rather than papered over: ``test_no_haunted_child_processes_
or_servers_survive`` scans ``ps -A`` for the string ``haunted_child.py``. That is a
global scan, so a concurrently running probe — another test process, a parallel CI
job, or a developer's manual run — can make it fail even though *this* probe cleaned
up. The probe itself needs a PTY: the fault-injection area list assumes the PTY
transport ran, and the tmux-specific tests skip when the binary is missing.
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
    """Faithful probe report, computed once per process (see module docstring).

    Shared across test classes, so caching is what keeps the module's runtime
    proportional to the number of *distinct* probes rather than the number of tests.
    """
    if "positive" not in _CACHE:
        _CACHE["positive"] = run_haunted_terminal_probe()
    return _CACHE["positive"]


def _fault_report():
    """Fault-injected probe report, cached separately; never mutated by callers."""
    if "fault" not in _CACHE:
        _CACHE["fault"] = run_haunted_terminal_probe(fault_injection=True)
    return _CACHE["fault"]


def _run_suite(name: str, *extra_args: str):
    """Run the probe script through the real ``pty_tmux`` adapter path.

    This is not ``run_haunted_terminal_probe`` in-process: it builds a ``SuiteSpec``
    the way a manifest would (``needs_pty``/``needs_tmux``, 180 s timeout) and calls
    ``execute_suite``, so the test also covers the PTY/TMUX capability gate, the
    adapter parser and evidence-file writing. Returns ``(result, evidence_dir,
    elapsed)``; the caller removes ``evidence_dir`` with ``addCleanup`` so a failure
    leaves the logs for inspection until the test ends.
    """
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
    """The corpora must *contain* the hazards the probe claims to cover.

    Nothing here runs the probe. These are the inputs' own invariants: eight distinct
    areas, at least one non-ASCII payload per hostile class, a wide character set whose
    display width is at least two cells per wide character, every invisible-control
    family, exact escape bytes, and chunk boundaries that are genuinely mid-code-point.
    If this class were deleted, a corpus could silently degrade to ASCII and the probe
    would keep reporting PASS while testing nothing.
    """

    def test_corpora_cover_the_eight_required_areas(self) -> None:
        """Eight unique areas; combining text is non-ASCII; wide text is really wide.

        ``display_width >= 2 * len(wide_chars)`` is the discriminating assertion: a
        width function that counted East Asian characters as one cell would still pass
        a mere "contains CJK" check, but not this one. Combining marks are excluded
        from the count because they consume no cell of their own.
        """
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
        """RLO, LRI, ZWSP, LRM, BOM and a Plane-14 tag letter all appear verbatim.

        The assertion is on the concatenated text, so it does not care which entry
        carries which character — only that no family was dropped when the corpus was
        edited. Plane-14 tags matter because they are an invisible payload channel:
        they render as nothing but can encode text.
        """
        texts = "".join(e.text for e in haunted_invisible_control_corpus())
        for point in ("\u202e", "\u2066", "\u200b", "\u200e", "\ufeff", "\U000e0067"):
            self.assertIn(point, texts)

    def test_malformed_and_osc52_corpora_are_exact_byte_payloads(self) -> None:
        """Malformed sequences keep their exact bytes, including the unterminated forms.

        ``endswith("\\x1b[31;")`` pins a CSI cut off mid-parameter, and the OSC 52 entry
        terminated by a newline (rather than BEL/ST) pins a form a naive parser would
        treat as plain text. In the OSC 52 corpus, requiring an ST-terminated variant
        alongside BEL keeps both terminator grammars represented.
        """
        malformed = haunted_malformed_escape_corpus()
        self.assertTrue(any(e.text.endswith("\x1b[31;") for e in malformed))
        self.assertTrue(any("\x1b]52;c;ZXZpbA==\n" in e.text for e in malformed))

        osc = haunted_osc52_corpus()
        self.assertTrue(all("\x1b]52;" in e.text for e in osc))
        self.assertTrue(any("\x1b\\" in e.text for e in osc), "ST terminator form missing")

    def test_partial_utf8_chunks_decode_losslessly_when_reassembled(self) -> None:
        """Reassembly is lossless *and* every chunk boundary really is mid-code-point.

        The two ``assertRaises(UnicodeDecodeError)`` calls are the negative control: if
        a future edit aligned the chunks to code-point boundaries, incremental decoding
        would still pass while proving nothing about interruption handling. The decoded
        text must equal the expected string with zero replacement characters, i.e. no
        U+FFFD was substituted anywhere.
        """
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
        """1x1 and 2x2 are present, plus both narrow (``<= 24``) and wide (``>= 120``).

        Degenerate sizes are where width arithmetic divides by zero or forgets that one
        cell cannot hold a two-cell glyph. The length assertion against the public
        constant is tautological — it compares the builder to its own source — so only
        the ``>= 8`` floor below catches a shrunk schedule in ``fixtures``; the narrow
        and wide checks catch a schedule that lost its extremes.
        """
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
        """Two builds agree on names and bytes, and every entry survives ``json.dumps``.

        Determinism matters because the probe reports case names in a stream parsed by
        regex; a generated name would break the marker grammar. ``ensure_ascii=False``
        is used deliberately: the payload must be serializable as-is, not only after
        escaping, since it is also persisted and compared.
        """
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
    """The checker is proven to discriminate, not merely to return a list.

    Three directions: its own selftest (good accepted, bad rejected, overflow and
    orphan-combining detected), a deliberately naive renderer that must be rejected on
    every hazard class, and the production renderer that must satisfy the same
    invariants *and* differ from a frame rendered for plain text. That last comparison
    is the one that catches a renderer that "passes" by ignoring its input.
    """

    def test_invariant_checker_selftest_accepts_good_and_rejects_bad(self) -> None:
        """Selftest is green, and every named sub-check is independently true.

        ``assertTrue(report["ok"])`` alone would hide a selftest whose ``all()`` was
        computed over an empty dict; asserting every value catches a silently disabled
        check. The report is a plain dict so it can be printed in a failure message.
        """
        report = haunted_invariant_selftest()
        self.assertTrue(report["ok"], report["checks"])
        self.assertTrue(all(report["checks"].values()), report["checks"])

    def test_defective_renderer_is_rejected_on_every_hazard(self) -> None:
        """The naive renderer is rejected for wide glyphs, bidi, OSC 52 and clearing.

        Each hazard is a different failure mode: cell overflow, an invisible control
        reaching the frame, a clipboard write, and a screen-clear sequence. A checker
        that only measured line width would accept the last three, so requiring a
        non-empty violation list per input is the discriminating part.
        """
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
        """Frames have the exact height, contain the payload, and pass the checker.

        ``frame != plain_frame`` is the anti-vacuity guard: an implementation that
        blanked every hostile payload would trivially satisfy ``check_haunted_frame``
        and lose the text. Widths 10/12/24 stay below the 120-cell split threshold, so
        the single-column path is what is exercised.
        """
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
        """Clean ASCII yields no violations; each forbidden form yields at least one.

        ``ZXZpbA==`` is flagged as a clipboard payload even without its OSC wrapper,
        which is what makes the checker useful on already-sanitized text: the base64
        body alone is evidence of an attempted hijack.
        """
        self.assertEqual(check_haunted_text("plain ascii"), [])
        for hostile in ("\u202e", "\x07", "\x1b", "\u2066", "ZXZpbA=="):
            self.assertTrue(check_haunted_text("safe" + hostile), hostile)


class HauntedProbePositiveTests(unittest.TestCase):
    def test_faithful_probe_passes_every_area(self) -> None:
        """Faithful probe: zero failures, >= 24 passing cases, no leaks, clean scratch.

        The case-count floor is a coverage guard, not a contract: cases are added as
        areas grow, so the assertion only fails when the probe shrinks. ``scratch_clean``
        is the only teardown observation in this assertion — computed in the probe's
        ``finally`` from what is still on disk — so this asserts the probe cleaned up
        its own PTY/tmux work dirs, an observation about this run, not a claim about
        arbitrary suites. ``leaked_children`` is a fixed schema placeholder ``0``, not
        a survivor count; the real leak guarantee is the raise-from-``finally`` in the
        PTY and tmux transport helpers.
        """
        report = _positive_report()
        self.assertFalse(report["fault_injection"])
        self.assertEqual(report["failed"], 0, [c for c in report["cases"] if c["status"] != "PASS"])
        self.assertTrue(report["all_passed"])
        self.assertGreaterEqual(report["passed"], 24)
        self.assertEqual(report["leaked_children"], 0)
        self.assertTrue(report["scratch_clean"])

    def test_every_area_has_at_least_one_passing_case(self) -> None:
        """Every one of the eight areas contributes a pass; area order is the constant's.

        Order matters because the report's area table is seeded from
        ``HAUNTED_TERMINAL_AREAS`` and printed/parsed in that order; comparing the key
        list, not just the set, catches a probe that reordered its buckets. A zero-fail
        plus nonzero-pass requirement means an area cannot be "covered" by a SKIP.
        """
        report = _positive_report()
        self.assertEqual(list(report["areas"].keys()), list(HAUNTED_TERMINAL_AREAS))
        for area, counts in report["areas"].items():
            self.assertGreaterEqual(counts["total"], 1, area)
            self.assertEqual(counts["fail"], 0, area)
            self.assertGreaterEqual(counts["pass"], 1, area)

    def test_pty_transport_observes_rapid_sigwinch_geometry_changes(self) -> None:
        """The child saw every scheduled resize and ended at the final geometry.

        ``applied_resizes`` is the parent's record of what it actually issued, compared
        exactly to ``schedule[1:]`` (the first entry is the initial size, not a resize).
        That makes the test sensitive to a dropped or duplicated ``SIGWINCH``. The
        final-geometry assertion is the one that fails if the child caches the first
        ``TIOCGWINSZ`` result — the fault-injected PTY child does exactly that, which is
        why the negative control can rely on this area failing.
        """
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
        """The disposable tmux server answered, reported a pane size, and is gone.

        Skips rather than fails when the cached probe could not start tmux (binary
        missing or socket path too long); that is an environment limitation, and the
        skip message carries the probe's own reason. A pane width of zero would mean
        the geometry never reached the pane, so both dimensions are checked.
        """
        report = _positive_report()
        if not report["tmux"]["available"]:
            self.skipTest("tmux unavailable: " + report["tmux"]["error"])
        self.assertTrue(report["tmux"]["server_gone"])
        self.assertEqual(len(report["tmux"]["pane_size"]), 2)
        self.assertGreater(report["tmux"]["pane_size"][0], 0)

    def test_probe_scratch_directories_are_removed(self) -> None:
        """A probe given its own scratch root leaves that root empty.

        Unlike the cached happy-path assertions, this runs a fresh probe with
        ``use_tmux=False`` so the root's emptiness is caused only by the PTY/intermediate
        work dirs. ``scratch_clean`` is the probe's own claim; ``list(scratch.iterdir())``
        is the independent check of it — the probe could be wrong about its own cleanup.
        """
        scratch = Path(tempfile.mkdtemp(dir=str(_scratch()), prefix="tb-hnscratch-"))
        self.addCleanup(shutil.rmtree, scratch, True)
        report = run_haunted_terminal_probe(scratch_root=scratch, use_tmux=False)
        self.assertTrue(report["scratch_clean"])
        self.assertFalse(report["fault_injection"])
        self.assertEqual(report["failed"], 0)
        self.assertEqual(list(scratch.iterdir()), [], "probe left scratch entries behind")

    @unittest.skipUnless(shutil.which("tmux"), "tmux unavailable")
    def test_deeply_nested_scratch_root_still_runs_disposable_tmux(self) -> None:
        """A >115-char scratch path still gets a working tmux, via a short socket base.

        ``sun_path`` is 104 bytes on macOS, so a tmux socket under a deep scratch tree
        cannot be created. The fixture therefore asks ``short_writable_tmux_base`` for a
        shorter parent while still cleaning up the scratch tree it was given. The
        length assertion documents the condition being exercised; if the platform
        changes the limit, this test is the one that should be revisited. Skipped when
        the tmux binary is absent.
        """
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
        """Fault injection fails >= 8 cases across seven areas, with useful messages.

        Requiring the *set of areas* to cover seven named areas is what makes this a
        real control: a single aggregate failure could come from a crash rather than
        from each invariant. The message check is deliberately loose (any of three
        tokens) because exact wording is a presentation detail; what must hold is that
        a human reading the log learns *why* it failed.
        """
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
        """The control is honest: cases on untouched primitives still PASS.

        Without this, a probe whose fault mode crashed everything would satisfy the
        "fails with actionable messages" test while proving nothing about the checker.
        ``combining_cells_count_once`` and ``eof_partial_utf8_uses_single_replacement``
        are intentionally cases the fault injection does not break.
        """
        # The control is honest: cases that test sanitize/UTF-8 primitives still pass.
        report = _fault_report()
        passing = {c["name"] for c in report["cases"] if c["status"] == "PASS"}
        self.assertIn("combining_cells_count_once", passing)
        self.assertIn("eof_partial_utf8_uses_single_replacement", passing)


class HauntedSuiteIntegrationTests(unittest.TestCase):
    """The probe exercised as a real ``pty_tmux`` suite, through the adapter path.

    These tests are slower than the in-process ones because each creates a fresh
    evidence dir and spawns the probe script. They cover what the fixture-level tests
    cannot: capability gating (``needs_pty``/``needs_tmux`` recorded in
    ``isolation_meta``), the parser's count agreement with the summary line, and the
    exit-code contract (0 faithful, 1 fault-injected).
    """

    def test_faithful_suite_passes_through_execute_suite(self) -> None:
        """``execute_suite`` reports PASS, >= 24 cases, PTY+tmux recorded, logs written.

        ``elapsed < 120`` is a wall-clock ceiling against the 180 s suite timeout: it
        fails on a hang long before the adapter's timeout turns the result into
        TIMEOUT, which makes the failure easier to attribute. ``stdout_path`` and
        ``sanitized_path`` are checked because a PASS with no evidence would break
        ``testbench results`` for the run.
        """
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
        """The fault run is FAIL with per-case assertion failures, not one blunt error.

        ``pass_count >= 5`` is the paired guard: failures must coexist with passes, so a
        probe that died on startup (all errors, no passes) cannot satisfy this test.
        Every failure's ``kind`` must be ``assertion`` — a ``spawn_error`` or
        ``harness_error`` here would mean the negative control broke rather than bit.
        Case names are matched loosely (``wide``, ``osc52``) to keep the assertion about
        coverage rather than about exact naming.
        """
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
        """The probe's raw stream parses to one case per marker, zero failures.

        ``len(outcome.cases) == outcome.total`` is the count-agreement invariant between
        markers and the ``HAUNTED_SUMMARY`` line; without it a parser could report a
        total that no individual case backs. ``--no-tmux`` keeps this test independent
        of a tmux binary so the parser contract is always exercised.
        """
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
        """Fault mode exits exactly 1 (not a crash code) and the stream counts >= 8 fails.

        Requiring ``returncode == 1`` distinguishes "the probe decided its invariants
        failed" from "the interpreter or a signal killed it" — a traceback would exit 1
        too, which is why the structured-count assertions follow rather than stand
        alone.
        """
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
        """The probe stays portable: no absolute home paths and no inherited ``TMUX``.

        A ``/Users/`` or ``/home/`` literal would make the suite pass only on its
        author's machine, and ``TMUX=`` in the source is read as an attempt to join the
        ambient tmux server. This is a text scan over the shipped script — cheap, and
        it fails at the point the mistake is introduced rather than at runtime on
        another host.
        """
        text = PROBE.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("TMUX=", text)


class HauntedSafetyTests(unittest.TestCase):
    """The probe must never touch the developer's terminal or ambient tmux server.

    One test reads the ``fixtures`` source to pin the *mechanism* (isolated
    ``TMUX_TMPDIR``, ``-L`` socket, ``kill-server``, no ambient ``TMUX`` inheritance),
    the other observes the machine. They are deliberately different evidence: the
    source scan cannot prove the server died, and the process scan cannot prove the
    probe avoided a live Ghostty.
    """

    def test_probe_never_connects_to_an_ambient_tmux_server(self) -> None:
        """Source of the haunted section: no ``"TMUX"`` env key, but ``TMUX_TMPDIR``/-L.

        The slice starts at the section banner, so this checks only the haunted code
        path and not the rest of ``fixtures``. ``"TMUX"`` as a dictionary key would
        mean the child inherits the caller's server; ``TMUX_TMPDIR`` plus ``-L`` is the
        isolation mechanism. ``kill-server`` must appear so teardown is not merely
        implied, and ``Ghostty`` is banned because driving a real terminal emulator
        would make the test destructive.
        """
        text = (REPO_ROOT / "hermes_testbench" / "fixtures.py").read_text(encoding="utf-8")
        segment = text[text.index("# ---------------------------------------------------------------- haunted terminal"):]
        self.assertNotIn('"TMUX"', segment)
        self.assertIn('"TMUX_TMPDIR"', segment)
        self.assertIn('"-L"', segment)
        self.assertIn("kill-server", segment)
        self.assertNotIn("Ghostty", segment)

    def test_no_haunted_child_processes_or_servers_survive(self) -> None:
        """No ``haunted_child.py`` anywhere on the machine, and the socket is dead.

        Known flaky, by construction: ``ps -A`` is machine-global, so a *concurrent*
        probe (another test process, a parallel CI job, a manual run) makes this fail
        even when this probe cleaned up. It is kept because the alternative — scanning
        only this probe's recorded pids — would miss the very leak class it targets
        (a child reparented away from the process tree). Treat a failure here as a
        lead to investigate, not a verdict, unless the suite is run alone.

        The tmux half is narrower and reliable: the recorded socket name is queried
        with ``list-sessions`` and a non-zero exit means no server answers. If the
        probe reported no socket, that half is skipped rather than assumed.
        """
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
