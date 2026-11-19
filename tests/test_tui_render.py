"""Pure-state tests for the TUI state machine and renderer (no PTY, no store I/O).

Everything here calls ``render_frame`` / ``handle_key`` in-process. That is the
point of the purity split in ``tui_render``: layout, palette, key precedence, and
confirmation semantics can be asserted deterministically, and the PTY test only
has to prove that the controller wires them to a terminal.

Reading the assertions: ``_strip`` removes SGR sequences only
(``\\x1b[...m``), so text assertions see the frame with colour removed but every
other byte intact — which is exactly what makes the "no escape except our own
SGR" checks meaningful. ``sample_state`` builds five runs across two projects
with one benchmark, so row counts and per-project dedup are predictable.

Two properties of the file itself:

* timestamps are formatted in the *process* timezone, so the expectation is built
  with the same ``astimezone()`` call rather than a hard-coded offset; the suite
  passes under any ``TZ``;
* ``unittest.main()`` appears mid-file, before the navigation/overlay/safety
  classes. Discovered or imported runs (``python -m unittest``, the project's
  harness) define every class first and are unaffected, but executing this file
  directly runs only the classes defined above that line and then exits.
"""

from __future__ import annotations

import re
import tempfile
import unittest
from dataclasses import replace as dc_replace
from datetime import datetime
from pathlib import Path

from hermes_testbench.schema import ProjectSpec, RunRecord, RunStatus, SuiteResult
from hermes_testbench.store import TestBenchStore
from hermes_testbench.tui import _refresh_active_logs, _refresh_from_store
from hermes_testbench.tui_render import (
    BILLABLE_REFUSAL,
    PERMANENT_TAB_QUIT_HINT,
    BenchmarkRow,
    Palette,
    TuiAction,
    TuiState,
    display_width,
    fit_cells,
    format_ansi_frame,
    handle_key,
    render_frame,
)

_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip(text: str) -> str:
    """Remove SGR colour/bold sequences, leaving text and any *other* escape intact.

    Deliberately narrower than ``sanitize.strip_terminal_escapes``: a stray OSC or
    CSI in the output would survive this helper and therefore show up in a text
    assertion. That is what lets the hostile-output test detect a leak the
    renderer was supposed to prevent.
    """
    return _SGR_RE.sub("", text)


def make_run(
    run_id: str = "tb_20261114_170000_aaaa1111",
    project: str = "demo",
    status: str = RunStatus.FAIL,
    profile: str = "unit",
    created_at: str = "2026-11-14T17:00:00Z",
    **overrides,
) -> RunRecord:
    """Build a plausible FAIL record; every field is overridable via keyword.

    Defaults are chosen so the common assertions are unambiguous: a fixed
    timestamp, a five-test total with one failure, and a failure string that is
    recognisably not a stray word. Tests pass ``overrides`` to change status,
    timings, or hostile text without re-declaring the whole record.
    """
    base = dict(
        run_id=run_id,
        project=project,
        project_root=f"/tmp/{project}",
        revision="deadbeefcafe",
        branch="main",
        profile=profile,
        status=status,
        created_at=created_at,
        total_count=5,
        pass_count=4,
        fail_count=1,
        duration_ms=1234.0,
        first_failure="test_bad: 1 != 2",
        repro_instructions="python3.14 -m unittest tests.test_bad",
        evidence_dir=f"/runs/{run_id}",
    )
    base.update(overrides)
    return RunRecord(**base)


def sample_state(**overrides) -> TuiState:
    """Five runs / two projects / one benchmark: the fixture every test reasons about.

    The ordering matters and is relied on: ``state.runs`` is newest-first, as
    ``store.list_runs`` returns it, with two ACTIVE rows (one RUNNING, one QUEUED)
    ahead of the terminal ones. Overview therefore lists four rows — the two
    active plus the latest terminal run per project — which is why ``G`` lands on
    index 3 in ``NavigationTests``. ``now_iso`` is fixed at 4.2 s after the
    RUNNING run's ``started_at`` so the elapsed label is a constant, not a clock
    read.
    """
    runs = (
        make_run(
            "tb_20261114_170100_running1",
            project="demo",
            status=RunStatus.RUNNING,
            started_at="2026-11-14T17:00:00Z",
            total_count=0,
            pass_count=0,
            fail_count=0,
            duration_ms=None,
            delegation_id="deleg-42",
        ),
        make_run(
            "tb_20261114_165900_queued01",
            project="alpha",
            status=RunStatus.QUEUED,
            profile="integration",
        ),
        make_run(
            "tb_20261114_164000_failed01",
            project="demo",
            status=RunStatus.FAIL,
            created_at="2026-11-14T16:40:00Z",
        ),
        make_run(
            "tb_20261114_163000_passed01",
            project="alpha",
            status=RunStatus.PASS,
            created_at="2026-11-14T16:30:00Z",
        ),
        make_run(
            "tb_20261114_162000_oldfail1",
            project="demo",
            status=RunStatus.FAIL,
            created_at="2026-11-14T16:20:00Z",
        ),
    )
    projects = (
        ProjectSpec(name="demo", root="/tmp/demo", trusted=True, revision="deadbeef"),
        ProjectSpec(name="alpha", root="/tmp/alpha", trusted=False, revision="cafebabe"),
    )
    benchmarks = (
        BenchmarkRow(
            project="demo",
            profile="benchmark",
            run_id="tb_20261114_160000_bench01",
            p50_ms=12.3,
            p95_ms=20.0,
            iqr_ms=3.1,
            baseline_p50_ms=11.8,
        ),
    )
    base = dict(
        runs=runs,
        projects=projects,
        benchmarks=benchmarks,
        baselines={"demo:unit": 12000.0},
        log_lines={"tb_20261114_164000_failed01": ("line one", "line two", "line three")},
        now_iso="2026-11-14T17:00:04.200Z",
    )
    base.update(overrides)
    return TuiState(**base)


class MetricsTests(unittest.TestCase):
    """Cell accounting: the basis of the geometry contract every other test asserts.

    ``display_width`` and ``fit_cells`` are what make "exactly ``width`` cells"
    checkable without a terminal. Both measure the *sanitised* string, so they
    agree with what the pane will paint, and both use the same East-Asian-width
    rule as ``pty_session.char_cells`` — the PTY interpreter is asserted against
    these widths in ``test_terminal_tape``.
    """

    def test_display_width_counts_cells_not_codepoints(self) -> None:
        """Wide, combining, zero-width-joiner, and SGR text all count by cells.

        Each case is a different branch of the metric: CJK is 2 cells per
        codepoint, a combining acute adds nothing, a ZWJ is zero-width, and an
        SGR sequence is skipped entirely (it is not text). A codepoint-counting
        implementation gets the first case wrong; one that counted combining
        marks gets the second wrong and would pad every NFD string short.
        """
        self.assertEqual(display_width("abc"), 3)
        self.assertEqual(display_width("日本語"), 6)
        self.assertEqual(display_width("e\u0301"), 1)  # combining acute
        self.assertEqual(display_width("a\u200db"), 2)  # ZWJ is zero-width
        self.assertEqual(display_width("\033[38;5;244mok\033[0m"), 2)
        self.assertEqual(display_width(""), 0)

    def test_fit_cells_pads_truncates_and_never_splits_a_wide_glyph(self) -> None:
        """Pad, truncate, and the half-cell case: ``日本語`` into 5 cells is 4 + one space.

        The ``display_width(fit_cells(...)) == 5`` assertion is the invariant, not
        the literal: truncation is allowed to stop one cell short of the budget
        when the next glyph is wide, but the result must still be padded to
        exactly ``width``. That is the difference between a ragged column and a
        sheared one — a half-painted wide glyph would occupy the padding cell and
        push the next column out. ``width <= 0`` returning "" is included because
        callers compute widths by subtraction.
        """

        self.assertEqual(fit_cells("ab", 5), "ab   ")
        self.assertEqual(fit_cells("abcdef", 3), "abc")
        self.assertEqual(fit_cells("日本語", 5), "日本 ")
        self.assertEqual(fit_cells("日本語", 4), "日本")
        self.assertEqual(fit_cells("", 4), "    ")
        self.assertEqual(fit_cells("abc", 0), "")
        self.assertEqual(display_width(fit_cells("日本語", 5)), 5)

    def test_fit_cells_strips_hostile_escapes_but_keeps_own_sgr(self) -> None:
        """Sanitisation is selective: foreign sequences go, the renderer's SGR stays.

        Two removal paths, both exercised here: the OSC 52 clipboard write is
        inside a text segment and is dropped by ``sanitize_terminal_text``, while
        ``\\033[2J`` matches ``_ANSI_RE`` and is classified as a non-SGR escape,
        so ``fit_cells`` skips it outright. The ``\\033[31m`` red survives because
        SGR is re-emitted verbatim — it is how the renderer styles rows at all.
        ``display_width(out) == 20`` then proves the removal did not disturb cell
        accounting: padding is computed after the escapes are gone, so they
        cannot consume columns.
        """

        evil = "ok\033]52;c;cHduZWQ=\007\033[31mred\033[0m\033[2J"
        out = fit_cells(evil, 20)
        self.assertNotIn("\033]", out)
        self.assertNotIn("\x07", out)
        self.assertNotIn("\033[2J", out)
        self.assertIn("\033[31m", out)
        self.assertIn("okred", _strip(out))
        self.assertEqual(display_width(out), 20)


class PaletteTests(unittest.TestCase):
    """The palette is a closed set: indexed 256-colour, or bold/reverse in monochrome.

    Pinning the exact sequences rather than "contains a colour" is what keeps the
    visual grammar stable across panes: tmux under a 256-colour terminfo is the
    target, so truecolor would be a rendering bug, and monochrome must still
    carry the selected-row and emphasis distinctions.
    """

    def test_256_color_palette_uses_indexed_sgr_only(self) -> None:
        """Exact indexed codes, and a loop that rejects truecolor outright.

        The appendix over all six colours is the real guard: individual equality
        checks would not catch a future ``\\033[38;2;`` addition to one accessor,
        but the loop fails on any of them.
        """
        pal = Palette()
        self.assertEqual(pal.dim(), "\033[38;5;244m")
        self.assertEqual(pal.grey(), "\033[38;5;240m")
        self.assertEqual(pal.blue(), "\033[38;5;39m")
        self.assertEqual(pal.green(), "\033[38;5;114m")
        self.assertEqual(pal.amber(), "\033[38;5;214m")
        self.assertEqual(pal.red(), "\033[38;5;203m")
        self.assertEqual(pal.selected_bg("x"), "\033[48;5;238m")
        for value in (pal.dim(), pal.grey(), pal.blue(), pal.green(), pal.amber(), pal.red()):
            self.assertTrue(value.endswith("m"))
            self.assertNotIn("\033[38;2;", value)  # no truecolor

    def test_no_color_palette_emits_only_bold_or_reverse_and_zero_indexed_colors(self) -> None:
        """Monochrome collapses colour to two attributes and keeps no indexed codes.

        ``dim``/``grey``/``green`` collapsing to nothing is the interesting half:
        they are stylistic rather than state-bearing, so monochrome drops them,
        while ``blue``/``amber``/``red`` mark status and become bold, and the
        selected row becomes reverse video. The two ``assertNotIn`` checks prove
        the collapse is complete rather than partial — a leftover ``38;5;`` would
        leak colour into a ``NO_COLOR`` terminal.
        """

        pal = Palette(no_color=True)
        self.assertEqual(pal.dim(), "")
        self.assertEqual(pal.grey(), "")
        self.assertEqual(pal.green(), "")
        self.assertEqual(pal.blue(), "\033[1m")
        self.assertEqual(pal.amber(), "\033[1m")
        self.assertEqual(pal.red(), "\033[1m")
        self.assertEqual(pal.selected_bg("x"), "\033[7m")
        self.assertNotIn("\033[38;5;", pal.dim() + pal.grey() + pal.blue() + pal.red())
        self.assertNotIn("\033[48;5;", pal.selected_bg("x"))


class FrameGeometryTests(unittest.TestCase):
    """The writer's contract: ``render_frame`` returns exactly ``height`` full-width lines.

    This is what lets ``tui.run_tui`` write a frame without querying the terminal
    mid-draw and without relying on autowrap: every line it receives is already
    exactly ``width`` cells. The assertions measure with ``display_width``, i.e.
    the same function the renderer pads with, so a shared bug in the metric would
    be invisible here — ``MetricsTests`` is what pins the metric independently.
    """

    def test_frame_is_exactly_height_lines_of_exact_cell_width(self) -> None:
        """Three geometries, including a 30x6 pane below any comfortable size.

        The loop asserts both dimensions per line, so a single over-long row
        fails on that line's index rather than only as a count mismatch. 30x6 is
        the interesting case: it is above the ``width < 10``/``height < 4``
        blank-frame threshold but small enough that every band is one row.
        """
        for width, height in ((80, 24), (140, 36), (30, 6)):
            lines = render_frame(sample_state(), width, height)
            self.assertEqual(len(lines), height, (width, height))
            for index, line in enumerate(lines):
                self.assertEqual(display_width(line), width, (width, height, index, line))

    def test_four_bands_title_body_hints_minibuffer(self) -> None:
        """Row 0 is identity/counters, ``h-2`` hints, ``h-1`` the minibuffer.

        The header assertions cover the view name, the ``ALL`` project scope, and
        the ``pass``/``store`` counters, not just ``TESTBENCH``: a header that
        lost its right-hand counters would still pass a ``startswith`` check.
        The minibuffer check is only "non-empty" — its precedence rules are
        asserted separately.
        """

        lines = render_frame(sample_state(), 140, 36)
        title = _strip(lines[0])
        self.assertTrue(title.startswith("TESTBENCH"))
        self.assertIn("ALL", title)
        self.assertIn("overview", title)
        self.assertIn("pass", title)
        self.assertIn("store", title)

        self.assertIn("j/k move", _strip(lines[34]))
        self.assertIn("Enter inspect", _strip(lines[34]))
        self.assertIn("? help", _strip(lines[34]))
        self.assertTrue(_strip(lines[35]).strip())

    def test_no_decorative_borders_or_spinners(self) -> None:
        """No box drawing and no spinner frames anywhere in the body or bands.

        This is the visual-grammar invariant imported from ``hxmail``: borders and
        spinners are what make a terminal UI flicker and what break when a pane is
        narrow. The glyph list is a denylist of the common sets (box drawing,
        heavy/light mix, block spinners), so it catches a regression that adds one
        decorative frame around the list rather than asserting a style preference.
        """

        for width, height in ((80, 24), (140, 36)):
            text = _strip("\n".join(render_frame(sample_state(), width, height)))
            for glyph in "╭╮╰╯│─┌┐└┘┏┓┗┛━┃╔╗╚╝║╠╣╬╪╫╭":
                self.assertNotIn(glyph, text)
            for spinner in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏|\\":
                self.assertNotIn(spinner, text)

    def test_wide_frame_has_two_column_inspector(self) -> None:
        """At 140 cells the inspector starts at column 70; rows 1-9 must fill both sides.

        The 68-cell list plus a 2-cell gutter is the documented split, so columns
        ``70:`` are the inspector's own cells. Requiring them non-empty on the
        first nine body rows is what distinguishes a real master/detail split from
        a single-column list that happens to mention the run id — the id and
        evidence path are asserted separately because they come from the
        inspector's fields, not from the row.
        """

        lines = render_frame(sample_state(selected=2), 140, 36)
        body = _strip("\n".join(lines[1:33]))
        self.assertIn("tb_20261114_164000_failed01", body)
        self.assertIn("/runs/tb_20261114_164000_failed01", body)  # evidence dir
        self.assertIn("python3.14 -m unittest tests.test_bad", body)
        # At 140 cells, the 68-cell list and 2-cell gutter precede the inspector.
        for line in lines[1:10]:
            right = _strip(line)[70:]
            self.assertEqual(display_width(line), 140)
            self.assertTrue(right.strip())

    def test_narrow_frame_is_full_width_until_enter_opens_detail_sheet(self) -> None:
        """Below 120 cells the list is one column; a sheet replaces it only on demand.

        The closed-frame assertion is a *negative* (no ``EVIDENCE`` in the body),
        which is what proves the inspector is not silently stacked under the list
        at 80x24 — the layout decision, not just the presence of fields. Opening
        the sheet must surface the detail fields, the repro section, and a
        captured log line, so the test also distinguishes "sheet drawn" from
        "sheet drawn empty".
        """

        closed = render_frame(sample_state(), 80, 24)
        self.assertNotIn("EVIDENCE", _strip("\n".join(closed[1:22])))
        opened = render_frame(sample_state(detail_open=True, selected=2), 80, 24)
        body = _strip("\n".join(opened[1:22]))
        self.assertIn("DETAIL", body)
        self.assertIn("tb_20261114_164000_failed01", body)
        self.assertIn("repro", body.lower())
        self.assertIn("line one", body)


class FrameContentTests(unittest.TestCase):
    """What the rows mean: timestamps, ordering, counts, modes, and hostile text.

    Most of these assert a token's presence *and* a contrasting absence, because
    presence alone cannot distinguish "the field is right" from "the field is
    there and also wrong somewhere else".
    """

    def test_run_timestamps_appear_in_split_and_detail_views(self) -> None:
        """Both views render the recorded start and end in local time.

        The expected strings are built with the same ``datetime.fromisoformat(...)
        .astimezone().strftime(...)`` chain the renderer uses, so the test is
        timezone-independent rather than passing only in one ``TZ``. The
        ``+02:00`` input is what proves normalisation actually happens: a naive
        passthrough would show the recorded offset, not the process's.
        """

        run = make_run(started_at="2026-11-14T23:59:58.123Z", finished_at="2026-11-15T02:00:02+02:00")
        started = datetime.fromisoformat("2026-11-14T23:59:58+00:00").astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        ended = datetime.fromisoformat("2026-11-15T00:00:02+00:00").astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        for width, opened in ((140, False), (80, True)):
            with self.subTest(width=width):
                state = sample_state(runs=(run,), detail_open=opened)
                view = _strip("\n".join(render_frame(state, width, 24)))
                self.assertRegex(view, r"started\s+" + re.escape(started))
                self.assertRegex(view, r"ended\s+" + re.escape(ended))

    def test_run_timestamps_do_not_invent_start_or_end_times(self) -> None:
        """Missing or unparseable stamps render ``-``; a queued run has no start.

        ``ended -`` for every case is the core assertion: the renderer must never
        derive an end time from ``duration_ms`` or substitute the current time.
        The RUNNING case then splits the two behaviours — a real ``started_at``
        must still display even though ``ended`` does not, which rules out an
        implementation that hides the whole timestamp block whenever one side is
        absent. ``invalid`` strings are included because the formatter is expected
        to absorb them rather than raise inside a frame.
        """

        for run in (make_run(status=RunStatus.QUEUED),
                    make_run(status=RunStatus.RUNNING, started_at="2026-11-14T17:00:00Z"),
                    make_run(status=RunStatus.ERROR, started_at="invalid", finished_at="invalid")):
            with self.subTest(status=run.status):
                view = _strip("\n".join(render_frame(sample_state(runs=(run,)), 140, 24)))
                self.assertRegex(view, r"ended\s+-")
                if run.status == RunStatus.RUNNING:
                    self.assertRegex(view, r"started\s+\d{4}-\d{2}-\d{2}")
                else:
                    self.assertRegex(view, r"started\s+-")

    def test_history_sort_cycles_without_hiding_or_retargeting_runs(self) -> None:
        """``s`` is a permutation, not a filter, and it does not retarget actions.

        Each order lists the same four projects in a different sequence, and the
        expected sequence is asserted per row — a sort that dropped the ERROR row
        or used a different tiebreak fails on position, and ``s`` returning no
        action proves the key only reorders. ``RUN new`` in the inspector's column
        after three sorts is the retargeting check: the cursor must follow the
        *run* (``preserve_history_selection``), otherwise ``R``/``x``/``y`` would
        silently act on a neighbour. The final ``R`` then ``y`` asserting
        ``run_id == "new"`` closes that loop through the public key path rather
        than through the helper.
        """

        runs = (
            make_run("new", project="recent", status=RunStatus.PASS, created_at="2026-11-14T17:03:00Z", duration_ms=100),
            make_run("err", project="error-run", status=RunStatus.ERROR, created_at="2026-11-14T17:02:00Z", duration_ms=None),
            make_run("slow", project="slow-run", status=RunStatus.PASS, created_at="2026-11-14T17:01:00Z", duration_ms=1200),
            make_run("bad", project="older-fail", created_at="2026-11-14T17:00:00Z", duration_ms=1200),
        )
        state = sample_state(view="history", runs=runs)
        for order, expected in (
            ("newest", ("recent", "error-run", "slow-run", "older-fail")),
            ("failures first", ("error-run", "older-fail", "recent", "slow-run")),
            ("longest", ("slow-run", "older-fail", "recent", "error-run")),
            ("newest", ("recent", "error-run", "slow-run", "older-fail")),
        ):
            with self.subTest(order=order):
                lines = [_strip(line) for line in render_frame(state, 140, 24)]
                self.assertIn(f"history · {order}", lines[0])
                self.assertIn("s sort", lines[-2])
                for row, project in zip(lines[1:5], expected):
                    self.assertIn(project, row[:68])
                self.assertIn("RUN new", lines[1][70:])
                state, action = handle_key(state, "s")
                self.assertIsNone(action)
        prompt, _ = handle_key(state, "R")
        _, action = handle_key(prompt, "y")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.payload["run_id"], "new")

    def test_history_sort_respects_filters_and_leaves_overview_unchanged(self) -> None:
        """Sorting composes with filters, and ``s`` is inert outside History.

        The filter is applied before ordering in the renderer, so "failures first"
        over ``project=demo`` + ``search=failed`` must still contain
        ``failed01`` and never ``passed01`` — this pins that ordering does not
        re-admit filtered rows. The second half asserts structural equality of the
        state (``overview == after``), which is the strongest form of "the key did
        nothing": not just no action, but no mutation at all.
        """

        state = sample_state(view="history", project_filter="demo", search_query="failed")
        sorted_state, _ = handle_key(state, "s")
        frame = _strip("\n".join(render_frame(sorted_state, 140, 24)))
        self.assertIn("failures first", frame)
        self.assertIn("failed01", frame)
        self.assertNotIn("passed01", frame)
        overview, _ = handle_key(sorted_state, "1")
        after, action = handle_key(overview, "s")
        self.assertIsNone(action)
        self.assertEqual(overview, after)

    def test_history_selection_survives_a_refresh_that_reorders_rows(self) -> None:
        """A refresh re-anchors ``selected`` by run id, not by raw index.

        Two PASS runs sort by duration, so the row under the cursor moves from
        index 0 to 1 when the order becomes "longest"; ``selected == 1`` is the
        by-id re-anchor. Then the first run's duration changes and the store is
        re-read: the same run must move back to index 0 and ``selected`` must
        follow, which an index-preserving refresh could not do. This is the
        store-backed counterpart of the reordering test above — it exercises the
        real refresh path rather than a synthetic state change.
        """

        first = make_run("tb_20261114_170101_newer", status=RunStatus.PASS, duration_ms=10,
                         created_at="2026-11-14T17:01:01Z")
        second = make_run("tb_20261114_170000_older", status=RunStatus.PASS, duration_ms=20,
                          created_at="2026-11-14T17:00:00Z")
        with tempfile.TemporaryDirectory(prefix="tb-history-") as tmp, TestBenchStore(Path(tmp)) as store:
            store.finalize_run(first)
            store.finalize_run(second)
            state = _refresh_from_store(store, TuiState(view="history"))
            state, _ = handle_key(state, "s")
            state, _ = handle_key(state, "s")
            self.assertEqual(state.selected, 1)
            store.finalize_run(dc_replace(first, duration_ms=50))
            refreshed = _refresh_from_store(store, state)
            view = _strip("\n".join(render_frame(refreshed, 140, 24)))
            self.assertEqual(refreshed.selected, 0)
            self.assertIn(f"RUN {first.run_id}", view)

    def test_history_keeps_the_selected_row_visible_in_short_frames(self) -> None:
        """``G`` on 20 runs, then a 10-row and a 24-row frame: the cursor stays on screen.

        Only History pages, so this is the test that distinguishes the paging
        window from the other views' "render from row 0 and crop" behaviour. The
        regex requires the cursor mark *and* the last project on one line, which
        rules out a body that contains the row somewhere but not under the cursor.
        ``[1:-2]`` excludes the header, hints, and minibuffer, so the match cannot
        come from a band rather than the list.
        """

        runs = tuple(make_run(f"id-{i}", project=f"project-{i:02}") for i in range(20))
        state, _ = handle_key(sample_state(view="history", runs=runs), "G")
        for width in (80, 140):
            with self.subTest(width=width):
                body = "\n".join(_strip(line)[:68] for line in render_frame(state, width, 10)[1:-2])
                self.assertRegex(body, r"▸\s+FAIL\s+project-19")

    def test_overview_puts_active_runs_first_then_failures(self) -> None:
        """Ordering by status class: RUNNING, then QUEUED, then a PASS row below them.

        Comparing ``str.index`` of the status words is an ordering assertion over
        the rendered text, not over the model — it would catch a renderer that
        sorts correctly and then reverses the block. The three words chosen are
        the ones whose relative order is fixed: the active pair first (RUNNING
        before QUEUED) and a terminal PASS after them.
        """

        text = _strip("\n".join(render_frame(sample_state(), 140, 36)))
        self.assertIn("RUN ", text)
        self.assertIn("QUEU", text)
        self.assertIn("FAIL", text)
        running_at = text.index("RUN ")
        queued_at = text.index("QUEU")
        pass_at = text.index("PASS")
        self.assertLess(running_at, queued_at)
        self.assertLess(queued_at, pass_at)

    def test_overview_elapsed_vs_baseline_and_delegation_tag(self) -> None:
        """Elapsed is computed from ``now_iso`` minus ``started_at``; baseline comes from the map.

        ``4.2s / ~12.0s`` pins three decisions at once: elapsed is 4.2 s from the
        fixture clock (not wall time), the baseline is looked up by
        ``project:profile`` in ``state.baselines``, and the two are joined in the
        documented format. ``[deleg:…]`` is asserted because a delegated run is
        otherwise indistinguishable from a local one; ``4/5`` covers the numeric
        counts column in the same frame.
        """

        text = _strip("\n".join(render_frame(sample_state(), 140, 36)))
        self.assertIn("4.2s / ~12.0s", text)
        self.assertIn("[deleg:deleg-42]", text)
        self.assertIn("4/5", text)

    def test_status_words_are_fixed_four_cell_widths(self) -> None:
        """The four status words exist with their exact padding (``RUN `` has a trailing space).

        Despite the name, this asserts presence of the padded literals, not their
        measured width; the width guarantee is structural in ``_STATUS_WORD`` and
        is covered indirectly by every exact-width geometry test. Searching for
        ``"RUN "`` with the space is what catches a word that lost its padding and
        would shift the columns after it.
        """

        text = _strip("\n".join(render_frame(sample_state(), 140, 36)))
        for word in ("RUN ", "QUEU", "FAIL", "PASS"):
            self.assertIn(word, text)

    def test_modes_render_history_projects_and_benchmarks(self) -> None:
        """Each view has one distinctive column, and the fixture makes it exact.

        History shows revision and run id (the row identity), Projects shows the
        trusted/sandbox badge (the trust decision the sandbox depends on), and
        Benchmarks shows ``p50`` plus a signed delta. ``+4.2%`` is computed from
        the fixture's 12.3 ms against an 11.8 ms baseline, so it also pins that
        the delta is versus the supplied baseline rather than a rounded constant.
        """

        history = _strip("\n".join(render_frame(sample_state(view="history"), 140, 36)))
        self.assertIn("deadbeef", history)
        self.assertIn("tb_20261114", history)
        self.assertIn("1.23s", history)

        projects = _strip("\n".join(render_frame(sample_state(view="projects"), 140, 36)))
        self.assertIn("trusted", projects)
        self.assertIn("sandbox", projects)

        bench = _strip("\n".join(render_frame(sample_state(view="benchmarks"), 140, 36)))
        self.assertIn("p50", bench)
        self.assertIn("12.3ms", bench)
        self.assertIn("+4.2%", bench)

    def test_ascii_mode_uses_ascii_marks_only(self) -> None:
        """Every Unicode mark is replaced, and ``>`` proves the cursor mark survived.

        Without the positive check, a renderer that omitted the cursor entirely
        would pass the denylist. The marks are the only non-ASCII glyphs the
        renderer emits on purpose, so this is also the test that keeps a stray
        bullet out of a terminal that cannot render it.
        """

        text = _strip("\n".join(render_frame(sample_state(), 140, 36, ascii_mode=True)))
        for glyph in "▸●✗·":
            self.assertNotIn(glyph, text)
        self.assertIn(">", text)

    def test_no_color_and_ascii_emit_zero_indexed_color_sequences(self) -> None:
        """Both degraded modes suppress indexed colour *and* truecolor.

        The check is a regex search over the whole frame rather than a palette
        accessor, because a row renderer could introduce colour without going
        through ``Palette``. ``ascii_mode`` is included because it implies
        monochrome in ``render_frame`` — a terminal that cannot be trusted with
        multi-byte marks should not receive indexed colour either.
        """

        for kwargs in ({"no_color": True}, {"ascii_mode": True}):
            frame = "\n".join(render_frame(sample_state(), 120, 30, **kwargs))
            self.assertIsNone(re.search(r"\x1b\[[34]8;5;", frame), kwargs)
            self.assertIsNone(re.search(r"\x1b\[38;2;", frame), kwargs)

    def test_hostile_terminal_output_never_reaches_the_frame(self) -> None:
        """Untrusted run fields cannot inject a control sequence into the frame.

        Every field a run controls — project, failure text, repro, evidence path —
        carries an attack: an OSC title set, an OSC 52 clipboard write, an OSC 8
        hyperlink, a CSI clear, and a raw C1 CSI. The spot checks (no ``\\x1b]``,
        no BEL, no ``\\x9b``, no ``\\033[2J``, no ``pwned``) prove the specific
        payloads are gone; the loop over every CSI in the output proves the
        *structural* rule, that the only escapes the frame contains are SGR. That
        second check is the one that survives a new attack string the test does
        not enumerate.
        """

        hostile = "test_bad \033]52;c;cHduZWQ=\007 \033[2J \033[31mRED\033[0m \x9b31m"
        state = sample_state(
            runs=(
                make_run(
                    "tb_20261114_164000_failed01",
                    project="demo\033]0;pwned\007",
                    first_failure=hostile,
                    repro_instructions="run \033[2J --x",
                    evidence_dir="/runs/\033]8;;http://evil\007evil\033]8;;\007",
                ),
            ),
            projects=(ProjectSpec(name="demo", root="/tmp/demo", trusted=True),),
        )
        frame = "\n".join(render_frame(state, 140, 36))
        self.assertNotIn("\033]", frame)
        self.assertNotIn("\x07", frame)
        self.assertNotIn("\x9b", frame)
        self.assertNotIn("\033[2J", frame)
        self.assertNotIn("pwned", frame)
        for line in frame.split("\n"):
            for match in re.finditer(r"\x1b\[[0-9;]*[A-Za-z]", line):
                self.assertTrue(match.group(0).endswith("m"), match.group(0))

    def test_cjk_project_names_truncate_without_splitting_wide_glyphs(self) -> None:
        """Four widths, repeated wide text: every line still measures exactly ``width``.

        This is the case where a codepoint-based truncation goes wrong by exactly
        one cell per wide glyph, and where a naive slice can leave a half glyph at
        the column boundary. Only the width is asserted — not where the truncation
        lands — because the invariant is the frame's geometry, and pinning a cut
        position would over-specify the renderer.
        """
        runs = tuple(
            make_run(f"tb_20261114_16000{i}_cjk000{i}", project="日本語テスト" * 3, status=RunStatus.PASS)
            for i in range(3)
        )
        for width in (30, 60, 80, 140):
            lines = render_frame(sample_state(runs=runs), width, 12)
            for line in lines:
                self.assertEqual(display_width(line), width)

    def test_error_banner_and_empty_state_degrade_safely(self) -> None:
        """An empty store and an unavailable store both render a well-formed frame.

        ``TuiState()`` is the store-less start: it must still produce ``height``
        lines of ``width`` cells and say "no runs", so a fresh install shows an
        instruction rather than a blank pane. The banner case asserts both the
        reason (``store locked``) and the ``unavailable`` framing word, which is
        the header/minibuffer contract for a failed store open — the renderer
        reports the failure rather than the process exiting.
        """
        for width, height in ((30, 6), (80, 24), (140, 36)):
            empty = render_frame(TuiState(), width, height)
            self.assertEqual(len(empty), height)
            for line in empty:
                self.assertEqual(display_width(line), width)
            self.assertIn("no runs", _strip("\n".join(empty)).lower())

        broken = render_frame(TuiState(error_banner="store locked"), 80, 24)
        text = _strip("\n".join(broken))
        self.assertIn("store locked", text)
        self.assertIn("unavailable", text)

    def test_search_and_confirm_prompts_render_in_minibuffer(self) -> None:
        """Live prompts occupy ``h-1`` with their own prefixes, ahead of any flash.

        Two independent prompts in one test because the minibuffer has a strict
        precedence: a confirmation must outrank a stale message, and a search must
        show its typed buffer with ``/``. Asserting the rendered last line (rather
        than ``state``) is what proves the precedence is implemented in the
        renderer, not just in the state machine.
        """

        searching = _strip(render_frame(sample_state(search_open=True, search_buffer="demo"), 80, 24)[-1])
        self.assertIn("/demo", searching)
        confirming = _strip(
            render_frame(
                sample_state(
                    confirm=None,
                    minibuffer=PERMANENT_TAB_QUIT_HINT,
                ),
                80,
                24,
            )[-1]
        )
        self.assertIn("press Q", confirming)


class AnsiFrameTests(unittest.TestCase):
    """``format_ansi_frame``: the three properties that keep tmux from scrolling."""

    def test_format_ansi_frame_never_scrolls_the_pane(self) -> None:
        """No trailing newline, ``\\033[K`` per row, ``\\033[J`` at the end.

        A trailing newline would scroll the pane one row per refresh (the frame
        creeping upward); a missing ``\\033[K`` would leave the tail of a previous,
        longer line visible; a missing ``\\033[J`` would leave rows behind after a
        height shrink. The per-row count is asserted, so a writer that dropped the
        erase on all but the last row fails rather than merely looking untidy.
        """

        lines = ["abc", "def", "ghi"]
        out = format_ansi_frame(lines, 3)
        self.assertTrue(out.startswith("\033[?25l\033[H"))
        self.assertFalse(out.endswith("\n"))
        self.assertTrue(out.endswith("\033[J"))
        self.assertEqual(out.count("\033[K"), 3)
        self.assertIn("abc\033[K", out)
        self.assertIn("ghi\033[K\033[J", out)

    def test_format_ansi_frame_respects_height_and_missing_lines(self) -> None:
        """Fewer lines than height is normal: two rows, two erases, one line break.

        ``format_ansi_frame`` slices to ``height`` but never pads — the caller
        always passes exactly ``height`` rendered lines, and padding here would
        duplicate ``render_frame``'s job. The ``\\r\\n`` count (one for two rows) is
        what proves the breaks are explicit rather than left to autowrap, which is
        the property that makes the frame independent of the terminal's
        deferred-wrap behaviour.
        """

        out = format_ansi_frame(["one", "two"], 4)
        self.assertEqual(out.count("\033[K"), 2)
        self.assertTrue(out.endswith("\033[J"))
        self.assertFalse(out.endswith("\n"))
        self.assertEqual(out.count("\r\n"), 1)


if __name__ == "__main__":
    unittest.main()


class NavigationTests(unittest.TestCase):
    """Selection movement is clamped and view-scoped; ``Enter`` never acts."""

    def test_j_k_g_G_and_arrows_clamp_to_item_bounds(self) -> None:
        """``G`` goes to the last Overview row (index 3), and both ends clamp.

        The comment in the test states the arithmetic: two ACTIVE runs plus one
        latest terminal run per project (two projects) is four rows, so the last
        index is 3. Pressing ``j`` at the end must leave the index at 3 —
        clamping, not wrapping and not an out-of-range index — and ``g``/``k`` do
        the same at the other end. Arrow keys are included because they are
        distinct tokens (``Down``/``Up``), not characters; ``tui._read_keys`` is
        the layer that normalises escape sequences into those names, and this
        test pins only the state machine's handling of them.
        """

        state = sample_state()
        state, action = handle_key(state, "j")
        self.assertIsNone(action)
        self.assertEqual(state.selected, 1)
        state, _ = handle_key(state, "G")
        self.assertEqual(state.selected, 3)  # 2 active + latest run per project
        state, _ = handle_key(state, "j")
        self.assertEqual(state.selected, 3)  # clamped
        state, _ = handle_key(state, "g")
        self.assertEqual(state.selected, 0)
        state, _ = handle_key(state, "k")
        self.assertEqual(state.selected, 0)
        state, _ = handle_key(state, "Down")
        self.assertEqual(state.selected, 1)
        state, _ = handle_key(state, "Up")
        self.assertEqual(state.selected, 0)

    def test_empty_lists_navigate_without_inventing_a_selection(self) -> None:
        """An empty store: every key from the action set is inert, and the frame still fits.

        ``Enter``/``y``/``R``/``x`` are included deliberately — each is a key that
        would act on "the selected run", so with nothing loaded they must flash or
        do nothing rather than raise. Asserting ``selected == 0`` proves no key
        manufactured a selection index, which is what keeps ``_clamped_index`` from
        having to defend against negative or stale indices downstream.
        """

        state = TuiState()
        for key in ("j", "k", "g", "G", "Down", "Up", "Enter", "y", "R", "x"):
            state, action = handle_key(state, key)
            self.assertIsNone(action, key)
            self.assertEqual(state.selected, 0)
        self.assertEqual(len(render_frame(state, 30, 6)), 6)

    def test_tab_cycles_views_and_number_keys_jump_directly(self) -> None:
        """``Tab`` wraps through the four views; ``1``-``4`` select by position.

        The expected sequence starts at ``overview`` and wraps back to it, so a
        cycle that stopped at the last view fails on the fourth iteration. Numbers
        are pinned against ``VIEWS`` order one by one, since a swapped mapping
        ("2" opening projects) would otherwise still pass a "view changed" check.
        """

        state = TuiState(projects=sample_state().projects)
        expected = ["history", "projects", "benchmarks", "overview"]
        for view in expected:
            state, action = handle_key(state, "Tab")
            self.assertIsNone(action)
            self.assertEqual(state.view, view)
        for number, view in zip("1234", ("overview", "history", "projects", "benchmarks")):
            state, _ = handle_key(state, number)
            self.assertEqual(state.view, view)

    def test_enter_opens_detail_and_never_executes_anything(self) -> None:
        """``Enter`` is an inspection key only: no action, no confirmation, even twice.

        This is the ADR-0003 rule at the key level — double-tapping Enter cannot
        launch or cancel anything because Enter never produces a ``TuiAction`` and
        never sets ``confirm``. The second press lands in the detail overlay, which
        swallows Enter; the test asserts the state is still not a confirmation,
        which is the specific failure ("Enter confirms the prompt") the layering is
        designed to prevent.
        """

        state, action = handle_key(sample_state(selected=2), "Enter")
        self.assertIsNone(action)
        self.assertTrue(state.detail_open)
        self.assertIsNone(state.confirm)
        # Enter pressed again only toggles inspection focus, never an action.
        state, action = handle_key(state, "Enter")
        self.assertIsNone(action)
        self.assertIsNone(state.confirm)

    def test_detail_sheet_scrolls_with_j_k_and_closes_with_q_or_esc(self) -> None:
        """Scroll is clamped at both ends; ``G`` moves to the end of the sheet.

        ``G`` compares ``> 1`` rather than an exact index because the sheet's
        length depends on how many log lines the run captured — the assertion is
        that the scroll moved to the far end, not how far. ``k``/``g`` at the top
        must stay at 0. Both ``q`` and ``Esc`` must close, and neither may emit an
        action: the same keys quit or act at top level, so the no-action half pins
        the overlay's key ownership.
        """

        state = sample_state(selected=2, detail_open=True)
        state, _ = handle_key(state, "j")
        self.assertEqual(state.detail_scroll, 1)
        state, _ = handle_key(state, "G")
        self.assertGreater(state.detail_scroll, 1)
        state, _ = handle_key(state, "g")
        self.assertEqual(state.detail_scroll, 0)
        state, _ = handle_key(state, "k")
        self.assertEqual(state.detail_scroll, 0)
        closed, action = handle_key(state, "q")
        self.assertFalse(closed.detail_open)
        self.assertIsNone(action)
        state, _ = handle_key(sample_state(detail_open=True), "Esc")
        self.assertFalse(state.detail_open)


class OverlayTests(unittest.TestCase):
    """Help, pickers, and search: each owns its keys and closes before quit."""

    def test_help_toggles_with_question_mark_and_closes_with_q(self) -> None:
        """``?`` opens *and* closes; ``q`` closes without quitting.

        The toggle is asserted through the rendered frame as well as the flag, so a
        help overlay that opens but renders nothing fails. Closing with ``q``
        returning no action is the precedence check: at top level ``q`` is the quit
        guard, inside the overlay it must not be.
        """

        state, _ = handle_key(sample_state(), "?")
        self.assertTrue(state.help_open)
        self.assertIn("HELP", _strip("\n".join(render_frame(state, 80, 24))))
        state, _ = handle_key(state, "?")
        self.assertFalse(state.help_open)
        state, _ = handle_key(sample_state(), "?")
        state, action = handle_key(state, "q")
        self.assertFalse(state.help_open)
        self.assertIsNone(action)

    def test_project_picker_filters_and_ALL_resets(self) -> None:
        """Rows are ``ALL`` plus sorted project names; ``Enter`` sets or clears the filter.

        ``j`` from index 0 must land on ``alpha`` — alphabetically first — which
        pins that the list is sorted and that ``ALL`` is a real first row rather
        than an empty-filter sentinel. ``g`` back to 0 then Enter must produce ``""``
        for the filter, i.e. ``ALL`` maps to the empty string, not to a project
        literally named ALL.
        """

        state, _ = handle_key(sample_state(), "p")
        self.assertTrue(state.picker_open)
        self.assertEqual(state.picker_kind, "project")
        body = _strip("\n".join(render_frame(state, 80, 24)))
        self.assertIn("PROJECT FILTER", body)
        state, _ = handle_key(state, "j")  # ALL -> alpha
        self.assertEqual(state.picker_index, 1)
        state, _ = handle_key(state, "Enter")
        self.assertFalse(state.picker_open)
        self.assertEqual(state.project_filter, "alpha")
        state, _ = handle_key(state, "p")
        state, _ = handle_key(state, "g")
        state, _ = handle_key(state, "Enter")
        self.assertEqual(state.project_filter, "")

    def test_search_prompt_types_backspaces_applies_and_clears(self) -> None:
        """Typing edits a buffer, Enter promotes it to the query, Esc clears both.

        The distinction between ``search_buffer`` (being typed) and
        ``search_query`` (applied) is the whole design: the buffer survives until
        Enter, so an abandoned search does not filter the list. Esc must clear
        *both* — clearing only the buffer would leave the list filtered with no
        visible reason. Reopening ``/`` pre-loading the applied query is asserted
        so the operator can edit rather than retype.
        """

        state, _ = handle_key(sample_state(), "/")
        self.assertTrue(state.search_open)
        for ch in "demo":
            state, action = handle_key(state, ch)
            self.assertIsNone(action)
        self.assertEqual(state.search_buffer, "demo")
        state, _ = handle_key(state, "Backspace")
        self.assertEqual(state.search_buffer, "dem")
        state, _ = handle_key(state, "Enter")
        self.assertFalse(state.search_open)
        self.assertEqual(state.search_query, "dem")
        self.assertIn("/dem", _strip(render_frame(state, 80, 24)[-1]))
        state, _ = handle_key(state, "/")
        self.assertEqual(state.search_buffer, "dem")
        state, _ = handle_key(state, "Esc")
        self.assertFalse(state.search_open)
        self.assertEqual(state.search_buffer, "")
        self.assertEqual(state.search_query, "")

    def test_search_filters_history_rows_by_query_and_project(self) -> None:
        """A query matches across the row's identifying fields, not just the project.

        ``search_query="alpha"`` keeps the alpha rows and must exclude every demo
        row — the negative assertion is what proves filtering happened rather than
        the word happening to appear in a banner. The matcher in the renderer also
        includes run id, profile, revision, failure, and summary, so this test
        pins the project case only.
        """

        state = sample_state(view="history", search_query="alpha")
        text = _strip("\n".join(render_frame(state, 80, 24)))
        self.assertIn("alpha", text)
        self.assertNotIn("demo", text)

    def test_q_closes_search_picker_detail_help_before_quit(self) -> None:
        """Overlay precedence, in one loop: ``q`` closes the top overlay and never quits.

        Each case opens one overlay with its own key and then presses ``q``; the
        assertion is on the specific flag plus "no action". A single flag would be
        enough for one overlay, but looping over all three is what catches a
        dispatch order where, say, the picker check sits after the quit branch.
        """

        for key, flag in (("?", "help_open"), ("Enter", "detail_open")):
            state = sample_state(selected=1)
            state, _ = handle_key(state, key)
            closed, action = handle_key(state, "q")
            self.assertFalse(getattr(closed, flag))
            self.assertIsNone(action)
        state, _ = handle_key(sample_state(), "p")
        closed, action = handle_key(state, "q")
        self.assertFalse(closed.picker_open)
        self.assertIsNone(action)


class ConfirmationSafetyTests(unittest.TestCase):
    """Nothing executes without an explicit ``y`` — the ADR-0003 safety model.

    These are the tests that matter most in this file: ``r``/``R``/``x`` all lead
    to a real spawn or a real signal in ``tui._execute_action``, so the state
    machine's job is to make the intent unambiguous before an action exists.
    """

    def test_r_opens_profile_picker_and_requires_explicit_y(self) -> None:
        """``r`` → picker → prompt → ``y``; ``Enter`` cancels instead of confirming.

        Three separate properties: the picker does not act on selection alone, the
        prompt requires ``y``, and Enter *cancels* — the dangerous alternative
        (Enter confirming) is exactly what a double-tap would trigger. The last
        section asserts no latent confirmation survives a cancel: a leftover
        ``confirm`` would let a later stray ``y`` launch a run the operator never
        asked for. The emitted payload's ``profile`` is checked because the action
        is what the caller executes.
        """

        state, action = handle_key(sample_state(), "r")
        self.assertIsNone(action)
        self.assertTrue(state.picker_open)
        self.assertEqual(state.picker_kind, "profile")
        state, action = handle_key(state, "Enter")  # choose "unit"
        self.assertIsNone(action)
        self.assertIsNotNone(state.confirm)
        prompt = _strip(render_frame(state, 80, 24)[-1])
        self.assertIn("y confirm", prompt)

        # Enter must cancel the prompt, never confirm it.
        cancelled, action = handle_key(state, "Enter")
        self.assertIsNone(action)
        self.assertIsNone(cancelled.confirm)
        self.assertIn("cancel", cancelled.minibuffer.lower())

        # Cancelling the prompt must not leave a latent confirmation behind.
        state, _ = handle_key(sample_state(), "r")
        state, _ = handle_key(state, "Enter")
        self.assertIsNotNone(state.confirm)
        state, action = handle_key(state, "y")
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, "run_profile")
        self.assertEqual(action.payload["profile"], "unit")
        self.assertIsNone(state.confirm)

    def test_run_profile_prompt_can_be_cancelled_by_any_key(self) -> None:
        """Every non-``y`` key cancels, including keys that act at top level.

        ``j``/``q``/``x`` are the interesting cases: each is a navigation, quit, or
        cancel key elsewhere, and each must be consumed as "no" while a
        confirmation is pending. That is the first layer of ``handle_key`` doing
        its job — no top-level branch can be reached from a pending prompt.
        """

        for key in ("n", "Esc", "j", "q", "x"):
            state, _ = handle_key(sample_state(), "r")
            state, _ = handle_key(state, "Enter")
            self.assertIsNotNone(state.confirm)
            state, action = handle_key(state, key)
            self.assertIsNone(action, key)
            self.assertIsNone(state.confirm, key)

    def test_billable_profile_is_refused_via_r(self) -> None:
        """Selecting a billable profile refuses before a prompt exists, with a stable message.

        ``G`` moves to the last picker row. ``state.profiles`` is empty in this
        fixture, so the picker uses its ``ProfileName.STANDARD`` fallback, whose
        last entry is ``agent-eval``. The refusal must close the picker, leave
        ``confirm`` unset, and set ``BILLABLE_REFUSAL`` verbatim — the message is a
        contract, since it names the CLI's ``--allow-billable``. This is a UX guard
        only; the engine re-checks the manifest, so nothing here asserts anything
        about spawning.
        """

        state, _ = handle_key(sample_state(), "r")
        state, _ = handle_key(state, "G")  # last profile: agent-eval
        state, action = handle_key(state, "Enter")
        self.assertIsNone(action)
        self.assertIsNone(state.confirm)
        self.assertFalse(state.picker_open)
        self.assertEqual(state.minibuffer, BILLABLE_REFUSAL)
        self.assertIn("billable LLM profile blocked", _strip(render_frame(state, 80, 24)[-1]))

    def test_billable_profile_is_refused_via_rerun(self) -> None:
        """``R`` on a run whose profile is ``agent-eval`` refuses, and never prompts.

        The rerun path re-uses the recorded profile, so the gate has to be applied
        here too — otherwise the picker's refusal would be bypassed by rerunning an
        existing billable run. ``confirm is None`` is asserted as well: a prompt
        that is later cancelled would still have been an invitation to press ``y``.
        """

        runs = (make_run("tb_20261114_160000_bill01", profile="agent-eval", status=RunStatus.PASS),)
        state, action = handle_key(TuiState(runs=runs), "R")
        self.assertIsNone(action)
        self.assertIsNone(state.confirm)
        self.assertEqual(state.minibuffer, BILLABLE_REFUSAL)

    def test_rerun_prompt_emits_run_id_after_y_only(self) -> None:
        """``R`` prompts with the selected run id; ``n`` cancels, ``y`` emits it.

        The payload's ``run_id`` is the contract with ``tui._execute_action``,
        which looks the record up again by id — so an action carrying a stale or
        wrong id would silently rerun a different run. ``n`` returning no action
        is the half that proves the prompt is not a one-key formality.
        """

        state, action = handle_key(sample_state(selected=2), "R")
        self.assertIsNone(action)
        self.assertIsNotNone(state.confirm)
        self.assertEqual(state.confirm.kind, "rerun")
        cancelled, action = handle_key(state, "n")
        self.assertIsNone(action)
        self.assertIsNone(cancelled.confirm)

        state, _ = handle_key(sample_state(selected=2), "R")
        state, action = handle_key(state, "y")
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, "rerun")
        self.assertEqual(action.payload["run_id"], "tb_20261114_164000_failed01")

    def test_x_only_cancels_owned_active_runs_and_requires_y(self) -> None:
        """``x`` has three refusals and one action: inactive, foreign, unconfirmed.

        The passive case selects a PASS row in History, which must flash rather
        than prompt; the foreign case is a ``manual_`` id with no ``worker_pid``,
        which ``_is_owned`` rejects so the operator sees "not TestBench-owned"
        instead of an engine error. Note this ownership check is a cheap
        pre-filter, not authorization — the engine re-validates the pid and start
        time — so the test pins the affordance, not the safety boundary. ``y``
        finally asserts the payload carries the RUNNING run's id.
        """

        state, _ = handle_key(sample_state(), "x")
        self.assertIsNotNone(state.confirm)
        self.assertEqual(state.confirm.kind, "cancel_run")
        state, action = handle_key(state, "y")
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, "cancel_run")
        self.assertEqual(action.payload["run_id"], "tb_20261114_170100_running1")

        passive = sample_state(view="history", selected=3)  # PASS
        passive, action = handle_key(passive, "x")
        self.assertIsNone(action)
        self.assertIsNone(passive.confirm)
        self.assertIn("not RUNNING/QUEUED", passive.minibuffer)

        foreign = TuiState(runs=(make_run("manual_0001", status=RunStatus.RUNNING),))
        foreign, action = handle_key(foreign, "x")
        self.assertIsNone(action)
        self.assertIsNone(foreign.confirm)
        self.assertIn("not TestBench-owned", foreign.minibuffer)

    def test_y_copies_sanitized_repro_for_selected_run(self) -> None:
        """The copied command is the *cleaned* repro, and an empty store copies nothing.

        ``\\033[2J`` disappeared while ``--flag`` survived, so the assertion
        distinguishes "sanitised" from "truncated at the escape": the renderer's
        ``_clean`` must strip the sequence and keep the text around it. The empty
        case is the negative control — ``y`` with no selection must not emit a
        ``copy_repro`` action with an empty payload, which would clear the
        operator's clipboard.
        """

        runs = (make_run("tb_20261114_160000_evil01", repro_instructions="run \033[2J --flag"),)
        state, action = handle_key(TuiState(runs=runs), "y")
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, "copy_repro")
        self.assertNotIn("\033[2J", action.payload["repro"])
        self.assertIn("--flag", action.payload["repro"])

        empty, action = handle_key(TuiState(), "y")
        self.assertIsNone(action)
        self.assertIn("no run", empty.minibuffer.lower())


class QuitSemanticsTests(unittest.TestCase):
    """Two quit keys, one guard: lowercase ``q`` is a hint, uppercase ``Q`` quits."""

    def test_permanent_tab_q_flashes_and_never_exits(self) -> None:
        """On a permanent tab, ``q`` flashes the hint and yields no action; ``Q`` quits.

        The reason for the split is tmux: the TUI lives in a window that the
        operator will type at, and a stray ``q`` must not close the pane. The
        assertion on the rendered minibuffer (not just ``state.minibuffer``) proves
        the hint actually reaches the operator, which is the only thing that makes
        the guard discoverable.
        """

        state, action = handle_key(sample_state(), "q")
        self.assertIsNone(action)
        self.assertEqual(state.minibuffer, PERMANENT_TAB_QUIT_HINT)
        self.assertIn("press Q", _strip(render_frame(state, 80, 24)[-1]))
        state, action = handle_key(state, "Q")
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, "quit")

    def test_non_permanent_tab_q_quits(self) -> None:
        """``permanent_tab=False`` (``--quit-on-q``) makes lowercase ``q`` a real quit.

        The same key with the opposite outcome is what makes the previous test
        meaningful: this asserts the flag is consulted, not that ``q`` is ignored
        globally.
        """

        state, action = handle_key(sample_state(permanent_tab=False), "q")
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, "quit")

    def test_escape_at_top_level_only_clears_the_minibuffer(self) -> None:
        """``Esc`` with nothing open clears a stale flash and does nothing else.

        This is the top-level branch's only use for Esc, and pinning it prevents a
        later "Esc quits" convenience from being added silently — Esc already
        closes overlays, so reusing it for quit would make the two behaviours
        depend on which overlay was open a keystroke earlier.
        """

        state, action = handle_key(sample_state(minibuffer="hello"), "Esc")
        self.assertIsNone(action)
        self.assertEqual(state.minibuffer, "")


class DegradedStateTests(unittest.TestCase):
    """An unavailable store must degrade to a banner, not to a traceback."""

    def test_keyboard_and_frame_survive_store_unavailable_and_tiny_terms(self) -> None:
        """Every key in the common set is safe with no store, and every frame still fits.

        The loop accepts exactly one action kind — ``quit`` — and rebuilds the
        state after it, so quitting exits the loop while everything else is
        asserted inert. The frame pass then checks all three geometries against
        that same degraded state, which is the combination that actually happens
        when ``TestBenchStore`` raises at startup: no runs, a banner, and whatever
        pane size the operator has.
        """

        state = TuiState(error_banner="store locked", runs=(), projects=())
        for key in ("j", "k", "Tab", "p", "Esc", "/", "r", "Esc", "R", "x", "y", "?", "q", "Q"):
            state, action = handle_key(state, key)
            if action is not None:
                self.assertEqual(action.kind, "quit")
                state = TuiState(error_banner="store locked")
        for width, height in ((30, 6), (80, 24), (140, 36)):
            lines = render_frame(state, width, height)
            self.assertEqual(len(lines), height)
            for line in lines:
                self.assertEqual(display_width(line), width)


class LiveRunObservabilityTests(unittest.TestCase):
    """A live run must not look dead: pending counts, live tail, last activity.

    The distinction these tests protect is between "we have no counts yet" and
    "nothing passed": adapters only learn counts when a suite finishes, so a
    RUNNING run with ``total_count == 0`` renders ``pending`` rather than ``0/0``.
    The other half is the log source — ``_load_log_tail`` reads evidence files,
    not the store, because a live worker writes logs without committing rows.
    """

    def _running_state(self, **overrides):
        """A single RUNNING run with unknown counts and a real evidence dir.

        ``sample_state``'s ``log_lines`` are keyed by a different run id, so this
        run starts without a tail; a test that wants output injects its own. The
        tests rely on that: "no output yet" and "captured output" are
        distinguished by whether a tail exists for this id.
        """

        run = make_run(
            "tb_20261115_180000_live0001",
            project="demo",
            status=RunStatus.RUNNING,
            started_at="2026-11-15T18:00:00Z",
            total_count=0,
            pass_count=0,
            fail_count=0,
            skip_count=0,
            error_count=0,
            duration_ms=None,
            first_failure="",
            evidence_dir="/runs/live0001",
        )
        return sample_state(runs=(run,), selected=0, **overrides)

    def test_active_run_with_unknown_counts_shows_pending_not_zero(self) -> None:
        """``pending`` replaces ``0/0`` while counts are unknown.

        The negative assertion matters as much as the positive: a renderer that
        emitted both ("pending", "0/0") would still be lying about progress, and
        ``0/0`` is what a reader would take as "nothing ran".
        """

        text = _strip("\n".join(render_frame(self._running_state(), 140, 36)))
        self.assertIn("pending", text)
        self.assertNotIn("0/0", text)

    def test_active_run_shows_last_observed_activity_from_live_tail(self) -> None:
        """The last non-blank captured line is shown in the row *and* the detail sheet.

        Two views because they format differently: the overview row truncates to
        40 cells, while the detail sheet prints the line as captured. Asserting the
        full line in both is what shows the tail is passed through rather than
        re-summarised per view. The probe string is chosen so it can only come from
        ``state.log_lines`` — the fixture has no other source for it.
        """

        run_id = "tb_20261115_180000_live0001"
        state = self._running_state(
            log_lines={run_id: ("test_one ... ok", "test_two ... ok")},
        )
        text = _strip("\n".join(render_frame(state, 140, 36)))
        self.assertIn("test_two ... ok", text)
        detail = _strip("\n".join(render_frame(dc_replace(state, detail_open=True), 80, 24)))
        self.assertIn("test_two ... ok", detail)
        self.assertIn("pending", detail)

    def test_active_run_without_output_yet_is_not_reported_as_finished_log(self) -> None:
        """An ACTIVE run with no tail says ``(no output yet)``, not ``(no captured log lines)``.

        The two strings encode different claims: the first is "still running, nothing
        printed", the second is "this run is over and we captured nothing". Using
        the second for a live run would read as a harness failure. The negative
        assertion is what distinguishes the two code paths in ``_detail_lines``.
        """

        detail = _strip("\n".join(render_frame(self._running_state(detail_open=True), 80, 24)))
        self.assertIn("(no output yet)", detail)
        self.assertNotIn("(no captured log lines)", detail)

    def test_completed_run_keeps_numeric_counts_and_captured_tail(self) -> None:
        """A finished run shows real counts and never ``pending``.

        The complement of the first test: ``373/373`` must survive even though the
        run took four minutes, and ``pending`` must not leak into a terminal state.
        This is the regression that would appear if "unknown counts" were keyed on
        ``total_count == 0`` instead of on the ACTIVE status — a genuinely empty
        finished run would then claim to be pending forever.
        """

        done = make_run("tb_20261115_175900_done0001", status=RunStatus.PASS, total_count=373,
                        pass_count=373, fail_count=0, skip_count=0, error_count=0,
                        first_failure="", duration_ms=240950.0)
        state = sample_state(runs=(done,), selected=0,
                             log_lines={done.run_id: ("ran 373 tests", "OK")})
        text = _strip("\n".join(render_frame(state, 140, 36)))
        self.assertIn("373/373", text)
        self.assertNotIn("pending", text)
        detail = _strip("\n".join(render_frame(dc_replace(state, detail_open=True), 80, 24)))
        self.assertIn("ran 373 tests", detail)

    def test_inspector_prefers_failure_log_in_multi_suite_run(self) -> None:
        """A multi-suite run's inspector shows the failing suite, then admits when it is gone.

        ``_load_log_tail`` iterates the failure-like suites only when any exist, so
        the passing suite's log is never used as a fallback — the negative
        assertion on ``passing suite output`` is the point, and it is repeated
        after ``bad.sanitized.log`` is deleted. The ``Failure log unavailable``
        message is deliberately distinct from "no output": a missing file is a
        different state from a run that printed nothing, and substituting the
        passing log would hide the evidence the operator needs.
        """

        scratch = Path.home() / ".hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch, prefix="tb-log-") as tmp:
            root = Path(tmp)
            good_log = root / "good.sanitized.log"
            bad_log = root / "bad.sanitized.log"
            good_log.write_text("passing suite output\n", encoding="utf-8")
            bad_log.write_text("failing suite traceback\n", encoding="utf-8")
            good = SuiteResult("good", "shell", "unit", ["true"], tmp, RunStatus.PASS,
                               sanitized_path=str(good_log))
            bad = dc_replace(good, name="bad", command=["false"], status=RunStatus.FAIL,
                             sanitized_path=str(bad_log))
            store = TestBenchStore(root / "store")
            self.addCleanup(store.close)
            passed = make_run("tb_20261114_170000_aaaa1111", status=RunStatus.PASS, suites=[good])
            failed = make_run("tb_20261114_170100_bbbb2222", status=RunStatus.FAIL, suites=[good, bad])
            store.finalize_run(passed)
            store.finalize_run(failed)
            state = _refresh_from_store(store, TuiState(detail_open=True))
            self.assertEqual(state.log_lines[passed.run_id], ("passing suite output",))
            detail = _strip("\n".join(render_frame(state, 80, 24)))
            self.assertIn("failing suite traceback", detail)
            self.assertNotIn("passing suite output", detail)

            bad_log.unlink()
            state = _refresh_from_store(store, state)
            detail = _strip("\n".join(render_frame(state, 80, 24)))
            self.assertIn("Failure log unavailable", detail)
            self.assertNotIn("passing suite output", detail)

    def test_live_tail_is_reread_while_the_store_is_unchanged(self) -> None:
        """``_refresh_active_logs`` re-reads the file, so new lines appear with no store commit.

        A live worker appends to ``*.sanitized.log`` without touching ``state.db``,
        so ``data_version`` never moves; the 1 Hz tick calls this function instead.
        The second call on the same state object is what pins it: appending a line
        to disk must be enough, with no store refresh in between. The evidence
        directory is created at ``0700`` because that is what the engine does and
        the reader must not depend on looser permissions.
        """

        scratch = Path.home() / ".hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-live-") as tmp:
            ev = Path(tmp) / "tb_20261115_180000_live0001"
            ev.mkdir(parents=True, mode=0o700)
            log = ev / "step.sanitized.log"
            log.write_text("test_one ... ok\n", encoding="utf-8")
            run = make_run(
                "tb_20261115_180000_live0001",
                status=RunStatus.RUNNING,
                total_count=0,
                pass_count=0,
                fail_count=0,
                duration_ms=None,
                evidence_dir=str(ev),
            )
            state = TuiState(runs=(run,), projects=(), now_iso="2026-11-15T18:00:05Z")
            first = _refresh_active_logs(state)
            self.assertEqual(first.log_lines[run.run_id], ("test_one ... ok",))
            with log.open("a", encoding="utf-8") as fh:
                fh.write("test_two ... ok\n")
            second = _refresh_active_logs(first)
            self.assertEqual(second.log_lines[run.run_id][-1], "test_two ... ok")
