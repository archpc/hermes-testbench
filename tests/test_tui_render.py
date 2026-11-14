"""TDD tests for the pure hx:testbench TUI state machine and renderer."""

from __future__ import annotations

import re
import unittest

from hermes_testbench.schema import ProjectSpec, RunRecord, RunStatus
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
    return _SGR_RE.sub("", text)


def make_run(
    run_id: str = "tb_20261114_170000_aaaa1111",
    project: str = "demo",
    status: str = RunStatus.FAIL,
    profile: str = "unit",
    created_at: str = "2026-11-14T17:00:00Z",
    **overrides,
) -> RunRecord:
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
    def test_display_width_counts_cells_not_codepoints(self) -> None:
        self.assertEqual(display_width("abc"), 3)
        self.assertEqual(display_width("日本語"), 6)
        self.assertEqual(display_width("e\u0301"), 1)  # combining acute
        self.assertEqual(display_width("a\u200db"), 2)  # ZWJ is zero-width
        self.assertEqual(display_width("\033[38;5;244mok\033[0m"), 2)
        self.assertEqual(display_width(""), 0)

    def test_fit_cells_pads_truncates_and_never_splits_a_wide_glyph(self) -> None:
        self.assertEqual(fit_cells("ab", 5), "ab   ")
        self.assertEqual(fit_cells("abcdef", 3), "abc")
        self.assertEqual(fit_cells("日本語", 5), "日本 ")
        self.assertEqual(fit_cells("日本語", 4), "日本")
        self.assertEqual(fit_cells("", 4), "    ")
        self.assertEqual(fit_cells("abc", 0), "")
        self.assertEqual(display_width(fit_cells("日本語", 5)), 5)

    def test_fit_cells_strips_hostile_escapes_but_keeps_own_sgr(self) -> None:
        evil = "ok\033]52;c;cHduZWQ=\007\033[31mred\033[0m\033[2J"
        out = fit_cells(evil, 20)
        self.assertNotIn("\033]", out)
        self.assertNotIn("\x07", out)
        self.assertNotIn("\033[2J", out)
        self.assertIn("\033[31m", out)
        self.assertIn("okred", _strip(out))
        self.assertEqual(display_width(out), 20)


class PaletteTests(unittest.TestCase):
    def test_256_color_palette_uses_indexed_sgr_only(self) -> None:
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
    def test_frame_is_exactly_height_lines_of_exact_cell_width(self) -> None:
        for width, height in ((80, 24), (140, 36), (30, 6)):
            lines = render_frame(sample_state(), width, height)
            self.assertEqual(len(lines), height, (width, height))
            for index, line in enumerate(lines):
                self.assertEqual(display_width(line), width, (width, height, index, line))

    def test_four_bands_title_body_hints_minibuffer(self) -> None:
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
        for width, height in ((80, 24), (140, 36)):
            text = _strip("\n".join(render_frame(sample_state(), width, height)))
            for glyph in "╭╮╰╯│─┌┐└┘┏┓┗┛━┃╔╗╚╝║╠╣╬╪╫╭":
                self.assertNotIn(glyph, text)
            for spinner in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏|\\":
                self.assertNotIn(spinner, text)

    def test_wide_frame_has_two_column_inspector(self) -> None:
        lines = render_frame(sample_state(selected=2), 140, 36)
        body = _strip("\n".join(lines[1:33]))
        self.assertIn("tb_20261114_164000_failed01", body)
        self.assertIn("/runs/tb_20261114_164000_failed01", body)  # evidence dir
        self.assertIn("python3.14 -m unittest tests.test_bad", body)
        # The inspector occupies the right half of the split.
        for line in lines[1:10]:
            right = _strip(line)[80:]
            self.assertEqual(display_width(line), 140)
            self.assertTrue(right.strip())

    def test_narrow_frame_is_full_width_until_enter_opens_detail_sheet(self) -> None:
        closed = render_frame(sample_state(), 80, 24)
        self.assertNotIn("EVIDENCE", _strip("\n".join(closed[1:22])))
        opened = render_frame(sample_state(detail_open=True, selected=2), 80, 24)
        body = _strip("\n".join(opened[1:22]))
        self.assertIn("DETAIL", body)
        self.assertIn("tb_20261114_164000_failed01", body)
        self.assertIn("repro", body.lower())
        self.assertIn("line one", body)


class FrameContentTests(unittest.TestCase):
    def test_overview_puts_active_runs_first_then_failures(self) -> None:
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
        text = _strip("\n".join(render_frame(sample_state(), 140, 36)))
        self.assertIn("4.2s / ~12.0s", text)
        self.assertIn("[deleg:deleg-42]", text)
        self.assertIn("4/5", text)

    def test_status_words_are_fixed_four_cell_widths(self) -> None:
        text = _strip("\n".join(render_frame(sample_state(), 140, 36)))
        for word in ("RUN ", "QUEU", "FAIL", "PASS"):
            self.assertIn(word, text)

    def test_modes_render_history_projects_and_benchmarks(self) -> None:
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
        text = _strip("\n".join(render_frame(sample_state(), 140, 36, ascii_mode=True)))
        for glyph in "▸●✗·":
            self.assertNotIn(glyph, text)
        self.assertIn(">", text)

    def test_no_color_and_ascii_emit_zero_indexed_color_sequences(self) -> None:
        for kwargs in ({"no_color": True}, {"ascii_mode": True}):
            frame = "\n".join(render_frame(sample_state(), 120, 30, **kwargs))
            self.assertIsNone(re.search(r"\x1b\[[34]8;5;", frame), kwargs)
            self.assertIsNone(re.search(r"\x1b\[38;2;", frame), kwargs)

    def test_hostile_terminal_output_never_reaches_the_frame(self) -> None:
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
        runs = tuple(
            make_run(f"tb_20261114_16000{i}_cjk000{i}", project="日本語テスト" * 3, status=RunStatus.PASS)
            for i in range(3)
        )
        for width in (30, 60, 80, 140):
            lines = render_frame(sample_state(runs=runs), width, 12)
            for line in lines:
                self.assertEqual(display_width(line), width)

    def test_error_banner_and_empty_state_degrade_safely(self) -> None:
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
    def test_format_ansi_frame_never_scrolls_the_pane(self) -> None:
        lines = ["abc", "def", "ghi"]
        out = format_ansi_frame(lines, 3)
        self.assertTrue(out.startswith("\033[?25l\033[H"))
        self.assertFalse(out.endswith("\n"))
        self.assertTrue(out.endswith("\033[J"))
        self.assertEqual(out.count("\033[K"), 3)
        self.assertIn("abc\033[K", out)
        self.assertIn("ghi\033[K\033[J", out)

    def test_format_ansi_frame_respects_height_and_missing_lines(self) -> None:
        out = format_ansi_frame(["one", "two"], 4)
        self.assertEqual(out.count("\033[K"), 2)
        self.assertTrue(out.endswith("\033[J"))
        self.assertFalse(out.endswith("\n"))
        self.assertEqual(out.count("\r\n"), 1)


if __name__ == "__main__":
    unittest.main()


class NavigationTests(unittest.TestCase):
    def test_j_k_g_G_and_arrows_clamp_to_item_bounds(self) -> None:
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
        state = TuiState()
        for key in ("j", "k", "g", "G", "Down", "Up", "Enter", "y", "R", "x"):
            state, action = handle_key(state, key)
            self.assertIsNone(action, key)
            self.assertEqual(state.selected, 0)
        self.assertEqual(len(render_frame(state, 30, 6)), 6)

    def test_tab_cycles_views_and_number_keys_jump_directly(self) -> None:
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
        state, action = handle_key(sample_state(selected=2), "Enter")
        self.assertIsNone(action)
        self.assertTrue(state.detail_open)
        self.assertIsNone(state.confirm)
        # Enter pressed again only toggles inspection focus, never an action.
        state, action = handle_key(state, "Enter")
        self.assertIsNone(action)
        self.assertIsNone(state.confirm)

    def test_detail_sheet_scrolls_with_j_k_and_closes_with_q_or_esc(self) -> None:
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
    def test_help_toggles_with_question_mark_and_closes_with_q(self) -> None:
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
        state = sample_state(view="history", search_query="alpha")
        text = _strip("\n".join(render_frame(state, 80, 24)))
        self.assertIn("alpha", text)
        self.assertNotIn("demo", text)

    def test_q_closes_search_picker_detail_help_before_quit(self) -> None:
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
    def test_r_opens_profile_picker_and_requires_explicit_y(self) -> None:
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
        for key in ("n", "Esc", "j", "q", "x"):
            state, _ = handle_key(sample_state(), "r")
            state, _ = handle_key(state, "Enter")
            self.assertIsNotNone(state.confirm)
            state, action = handle_key(state, key)
            self.assertIsNone(action, key)
            self.assertIsNone(state.confirm, key)

    def test_billable_profile_is_refused_via_r(self) -> None:
        state, _ = handle_key(sample_state(), "r")
        state, _ = handle_key(state, "G")  # last profile: agent-eval
        state, action = handle_key(state, "Enter")
        self.assertIsNone(action)
        self.assertIsNone(state.confirm)
        self.assertFalse(state.picker_open)
        self.assertEqual(state.minibuffer, BILLABLE_REFUSAL)
        self.assertIn("billable LLM profile blocked", _strip(render_frame(state, 80, 24)[-1]))

    def test_billable_profile_is_refused_via_rerun(self) -> None:
        runs = (make_run("tb_20261114_160000_bill01", profile="agent-eval", status=RunStatus.PASS),)
        state, action = handle_key(TuiState(runs=runs), "R")
        self.assertIsNone(action)
        self.assertIsNone(state.confirm)
        self.assertEqual(state.minibuffer, BILLABLE_REFUSAL)

    def test_rerun_prompt_emits_run_id_after_y_only(self) -> None:
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
    def test_permanent_tab_q_flashes_and_never_exits(self) -> None:
        state, action = handle_key(sample_state(), "q")
        self.assertIsNone(action)
        self.assertEqual(state.minibuffer, PERMANENT_TAB_QUIT_HINT)
        self.assertIn("press Q", _strip(render_frame(state, 80, 24)[-1]))
        state, action = handle_key(state, "Q")
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, "quit")

    def test_non_permanent_tab_q_quits(self) -> None:
        state, action = handle_key(sample_state(permanent_tab=False), "q")
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, "quit")

    def test_escape_at_top_level_only_clears_the_minibuffer(self) -> None:
        state, action = handle_key(sample_state(minibuffer="hello"), "Esc")
        self.assertIsNone(action)
        self.assertEqual(state.minibuffer, "")


class DegradedStateTests(unittest.TestCase):
    def test_keyboard_and_frame_survive_store_unavailable_and_tiny_terms(self) -> None:
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
