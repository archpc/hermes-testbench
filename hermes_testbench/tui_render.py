"""Pure deterministic state machine and borderless 4-band renderer for ``hx:testbench``.

No I/O, no clocks, no curses: everything here is a pure function over
:class:`TuiState` so the TUI can be exercised in-process (ADR-0003).
Visual grammar mirrors ``hxmail`` ``render.rs``: zero borders, zero spinners,
256-colour indexed palette, fixed-width status words, and ASCII fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import unicodedata

from .schema import ProfileName, RunRecord, RunStatus
from .sanitize import sanitize_terminal_text

#: Colour used for a value the tester has not supplied.
UNSET = "-"

HINTS_LINE = (
    "j/k move  Enter inspect  Tab view  p project  / filter  r run  "
    "R rerun  x cancel  y copy-repro  ? help"
)

PERMANENT_TAB_QUIT_HINT = "press Q (shift-q) to quit TUI, or switch tmux windows"
BILLABLE_REFUSAL = "billable LLM profile blocked in TUI; use CLI --allow-billable"

VIEWS = ("overview", "history", "projects", "benchmarks")


@dataclass(frozen=True)
class TuiAction:
    """A side effect the caller (``tui.py``) is allowed to perform."""

    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileOption:
    name: str
    billable: bool = False
    description: str = ""


@dataclass(frozen=True)
class BenchmarkRow:
    project: str
    profile: str
    run_id: str = ""
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    iqr_ms: float = 0.0
    baseline_p50_ms: Optional[float] = None
    status: str = RunStatus.PASS
    created_at: str = ""


@dataclass(frozen=True)
class ConfirmPrompt:
    """Explicit-``y`` confirmation. ``Enter`` never confirms (ADR-0003)."""

    kind: str
    label: str
    profile: str = ""
    run_id: str = ""
    project: str = ""


@dataclass(frozen=True)
class TuiState:
    runs: Tuple[RunRecord, ...] = ()
    projects: Tuple[Any, ...] = ()
    benchmarks: Tuple[BenchmarkRow, ...] = ()
    profiles: Tuple[ProfileOption, ...] = ()
    baselines: Dict[str, float] = field(default_factory=dict)
    log_lines: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    view: str = "overview"
    selected: int = 0
    project_filter: str = ""
    search_query: str = ""

    detail_open: bool = False
    detail_scroll: int = 0
    help_open: bool = False
    picker_open: bool = False
    picker_kind: str = "project"  # "project" | "profile"
    picker_index: int = 0
    search_open: bool = False
    search_buffer: str = ""
    confirm: Optional[ConfirmPrompt] = None

    minibuffer: str = ""
    error_banner: str = ""
    store_status: str = "ok"
    now_iso: Optional[str] = None
    permanent_tab: bool = True


class Palette:
    """256-colour indexed palette with a strict monochrome mode."""

    def __init__(self, no_color: bool = False) -> None:
        self.no_color = bool(no_color)

    def primary(self) -> str:
        return ""

    def bold(self) -> str:
        return "\033[1m"

    def dim(self) -> str:
        return "" if self.no_color else "\033[38;5;244m"

    def grey(self) -> str:
        return "" if self.no_color else "\033[38;5;240m"

    def blue(self) -> str:
        return "\033[1m" if self.no_color else "\033[38;5;39m"

    def green(self) -> str:
        return "" if self.no_color else "\033[38;5;114m"

    def amber(self) -> str:
        return "\033[1m" if self.no_color else "\033[38;5;214m"

    def red(self) -> str:
        return "\033[1m" if self.no_color else "\033[38;5;203m"

    def selected_bg(self, base: str = "") -> str:
        return "\033[7m" if self.no_color else "\033[48;5;238m"

    def reset(self) -> str:
        return "\033[0m"


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_SGR_RE = re.compile(r"\x1b\[[0-9;]*m\Z")


def _char_cells(ch: str) -> int:
    cp = ord(ch)
    if cp < 0x20 or cp == 0x7F:
        return 0
    if 0x80 <= cp <= 0x9F:
        return 0
    if cp in (0x200B, 0x200C, 0x200D, 0xFEFF):
        return 0
    if unicodedata.combining(ch):
        return 0
    if unicodedata.category(ch) in ("Mn", "Me", "Cf"):
        return 0
    if 0x1160 <= cp <= 0x11FF:
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _segments(text: str) -> List[Tuple[str, str]]:
    """Split into ``("text"|"sgr"|"esc", chunk)`` tokens, preserving order."""
    segments: List[Tuple[str, str]] = []
    pos = 0
    for match in _ANSI_RE.finditer(text):
        if match.start() > pos:
            segments.append(("text", text[pos : match.start()]))
        seq = match.group(0)
        segments.append(("sgr" if _SGR_RE.match(seq) else "esc", seq))
        pos = match.end()
    if pos < len(text):
        segments.append(("text", text[pos:]))
    return segments


def _flatten(text: str) -> str:
    return text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")


def display_width(text: str) -> int:
    width = 0
    for kind, chunk in _segments(text or ""):
        if kind == "text":
            width += sum(_char_cells(ch) for ch in sanitize_terminal_text(chunk))
    return width


def fit_cells(text: str, width: int) -> str:
    if width <= 0:
        return ""
    out: List[str] = []
    used = 0
    has_sgr = False
    truncated = False
    for kind, chunk in _segments(text or ""):
        if kind == "esc":
            continue
        if kind == "sgr":
            out.append(chunk)
            has_sgr = True
            continue
        for ch in _flatten(sanitize_terminal_text(chunk)):
            cells = _char_cells(ch)
            if cells == 0:
                if used > 0:
                    out.append(ch)
                continue
            if used + cells > width:
                truncated = True
                break
            out.append(ch)
            used += cells
        if truncated:
            break
    if truncated and has_sgr:
        out.append("\033[0m")
    out.append(" " * (width - used))
    return "".join(out)


_UNICODE_MARKS = {"cursor": "▸", "dot": "●", "fail": "✗", "sep": "·"}
_ASCII_MARKS = {"cursor": ">", "dot": "*", "fail": "!", "sep": "-"}

_STATUS_WORD = {
    RunStatus.RUNNING: "RUN ",
    RunStatus.QUEUED: "QUEU",
    RunStatus.PASS: "PASS",
    RunStatus.FAIL: "FAIL",
    RunStatus.SKIP: "SKIP",
    RunStatus.BLOCKED: "BLKD",
    RunStatus.TIMEOUT: "TOUT",
    RunStatus.CANCELLED: "CNCL",
    RunStatus.ERROR: "ERR ",
}
_FAILURE_STATUSES = frozenset({RunStatus.FAIL, RunStatus.ERROR, RunStatus.TIMEOUT})
_STATUS_RANK = {
    RunStatus.FAIL: 0,
    RunStatus.ERROR: 1,
    RunStatus.TIMEOUT: 2,
    RunStatus.RUNNING: 3,
    RunStatus.QUEUED: 4,
    RunStatus.SKIP: 5,
    RunStatus.BLOCKED: 6,
    RunStatus.CANCELLED: 7,
    RunStatus.PASS: 8,
}


def _marks(ascii_mode: bool) -> Dict[str, str]:
    return _ASCII_MARKS if ascii_mode else _UNICODE_MARKS


def _clean(value: Any) -> str:
    return _flatten(sanitize_terminal_text("" if value is None else str(value)))


def _paint(pal: Palette, style: str, text: str) -> str:
    return f"{style}{text}" if style else text


def _status_style(pal: Palette, status: str) -> str:
    if status in (RunStatus.FAIL, RunStatus.ERROR):
        return pal.red()
    if status == RunStatus.TIMEOUT:
        return pal.amber()
    if status in (RunStatus.RUNNING, RunStatus.QUEUED):
        return pal.blue()
    if status == RunStatus.PASS:
        return pal.green()
    return pal.dim()


def _attr(obj: Any, name: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _fmt_duration(ms: Optional[float]) -> str:
    if ms is None:
        return UNSET
    return f"{ms / 1000.0:.2f}s"


def _parse_iso(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _elapsed_label(state: TuiState, run: RunRecord) -> str:
    started = _parse_iso(run.started_at)
    now = _parse_iso(state.now_iso)
    if started is None or now is None:
        return UNSET
    elapsed = max(0.0, now - started)
    baseline_ms = None
    env = run.environment or {}
    if isinstance(env.get("baseline_duration_ms"), (int, float)):
        baseline_ms = float(env["baseline_duration_ms"])
    elif f"{run.project}:{run.profile}" in state.baselines:
        baseline_ms = float(state.baselines[f"{run.project}:{run.profile}"])
    if baseline_ms and baseline_ms > 0:
        return f"{elapsed:.1f}s / ~{baseline_ms / 1000.0:.1f}s"
    return f"{elapsed:.1f}s"


def _project_ok(state: TuiState, project: str) -> bool:
    return not state.project_filter or project == state.project_filter


def _query_ok(state: TuiState, run: RunRecord) -> bool:
    if not state.search_query:
        return True
    needle = state.search_query.lower()
    haystack = " ".join(
        (run.run_id, run.project, run.profile, run.revision, run.first_failure, run.summary)
    ).lower()
    return needle in haystack


def _run_sort_ts(run: RunRecord) -> float:
    parsed = _parse_iso(run.created_at)
    return parsed if parsed is not None else 0.0


def _overview_runs(state: TuiState) -> List[RunRecord]:
    indexed = [(idx, r) for idx, r in enumerate(state.runs) if _project_ok(state, r.project) and _query_ok(state, r)]
    active = [
        r
        for _idx, r in sorted(
            ((idx, r) for idx, r in indexed if r.status in RunStatus.ACTIVE),
            key=lambda item: (0 if item[1].status == RunStatus.RUNNING else 1, -_run_sort_ts(item[1]), item[0]),
        )
    ]
    terminal = [
        (idx, r) for idx, r in indexed if r.status not in RunStatus.ACTIVE and r.profile != ProfileName.BENCHMARK
    ]
    latest: Dict[str, RunRecord] = {}
    # Ascending chronological order (ties broken so smaller index = newer overwrites larger index = older)
    for _idx, run in sorted(terminal, key=lambda item: (_run_sort_ts(item[1]), -item[0])):
        latest[run.project] = run
    ranked = sorted(latest.values(), key=lambda r: (_STATUS_RANK.get(r.status, 9), r.project))
    return list(active) + ranked


def _history_runs(state: TuiState) -> List[RunRecord]:
    indexed = [(idx, r) for idx, r in enumerate(state.runs) if _project_ok(state, r.project) and _query_ok(state, r)]
    return [r for _idx, r in sorted(indexed, key=lambda item: (_run_sort_ts(item[1]), -item[0]), reverse=True)]


def _project_list(state: TuiState) -> List[Any]:
    return [p for p in state.projects if _project_ok(state, _clean(_attr(p, "name")))]


def _benchmark_rows(state: TuiState) -> List[BenchmarkRow]:
    return [b for b in state.benchmarks if _project_ok(state, b.project)]


def _view_items(state: TuiState) -> List[Any]:
    if state.view == "history":
        return _history_runs(state)
    if state.view == "projects":
        return _project_list(state)
    if state.view == "benchmarks":
        return _benchmark_rows(state)
    return _overview_runs(state)


def _clamped_index(state: TuiState, items: Sequence[Any]) -> int:
    if not items:
        return 0
    return max(0, min(state.selected, len(items) - 1))


def _latest_run_for(state: TuiState, project: str) -> Optional[RunRecord]:
    candidates = [
        (idx, r) for idx, r in enumerate(state.runs) if r.project == project and r.status not in RunStatus.ACTIVE
    ]
    if not candidates:
        candidates = [(idx, r) for idx, r in enumerate(state.runs) if r.project == project]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (_run_sort_ts(item[1]), -item[0]))[1]


def _selected_run(state: TuiState) -> Optional[RunRecord]:
    items = _view_items(state)
    if not items:
        return None
    index = _clamped_index(state, items)
    item = items[index]
    if isinstance(item, RunRecord):
        return item
    if state.view == "projects":
        return _latest_run_for(state, _clean(_attr(item, "name")))
    if state.view == "benchmarks":
        run_id = _clean(item.run_id)
        for run in state.runs:
            if run.run_id == run_id:
                return run
        return _latest_run_for(state, item.project)
    return None


def _repro_for(run: RunRecord) -> str:
    if run.repro_instructions:
        return _clean(run.repro_instructions)
    if run.suites:
        command = run.suites[0].command
        if command:
            return " ".join(_clean(part) for part in command)
    if run.commands:
        return " ".join(_clean(part) for part in run.commands[0])
    return ""


def _counts(run: RunRecord) -> str:
    total = run.total_count or 0
    passed = run.pass_count or 0
    skipped = run.skip_count or 0
    if (run.fail_count or 0) == 0 and (run.error_count or 0) == 0 and skipped > 0 and passed > 0:
        return f"{passed}+{skipped}s"
    return f"{passed}/{total}"


def _overview_row(state: TuiState, pal: Palette, run: RunRecord, selected: bool, marks: Dict[str, str], width: int) -> str:
    cursor = marks["cursor"] if selected else " "
    status = _STATUS_WORD.get(run.status, _clean(run.status)[:4])
    parts = [
        cursor,
        " ",
        _paint(pal, _status_style(pal, run.status), status),
        " ",
        fit_cells(_clean(run.project), 14),
        " ",
        fit_cells(_clean(run.profile), 11),
        " ",
        _counts(run).rjust(6),
        " ",
        _fmt_duration(run.duration_ms).rjust(7),
    ]
    if run.status in RunStatus.ACTIVE:
        parts.extend([" ", _paint(pal, pal.blue(), _elapsed_label(state, run))])
    if run.delegation_id:
        parts.extend([" ", _paint(pal, pal.dim(), f"[deleg:{_clean(run.delegation_id)}]")])
    if run.status in _FAILURE_STATUSES and run.first_failure:
        parts.extend(["  ", _paint(pal, pal.red(), f"{marks['fail']} {_clean(run.first_failure)}")])
    content = "".join(parts)
    if run.status in RunStatus.ACTIVE and display_width(content) > width:
        # Narrow split: keep the live elapsed/baseline and delegation tag visible.
        compact = [
            cursor,
            " ",
            _paint(pal, _status_style(pal, run.status), status),
            " ",
            fit_cells(_clean(run.project), 12),
            " ",
            fit_cells(_clean(run.profile), 9),
            " ",
            _paint(pal, pal.blue(), _elapsed_label(state, run)),
        ]
        if run.delegation_id:
            compact.extend([" ", _paint(pal, pal.dim(), f"[deleg:{_clean(run.delegation_id)}]")])
        content = "".join(compact)
    if selected:
        content = pal.selected_bg() + content
    return fit_cells(content, width) + pal.reset()


def _history_row(pal: Palette, run: RunRecord, selected: bool, marks: Dict[str, str], width: int) -> str:
    cursor = marks["cursor"] if selected else " "
    status = _STATUS_WORD.get(run.status, _clean(run.status)[:4])
    content = "".join(
        [
            cursor,
            " ",
            _paint(pal, _status_style(pal, run.status), status),
            " ",
            fit_cells(_clean(run.project), 14),
            " ",
            fit_cells(_clean(run.profile), 11),
            " ",
            _counts(run).rjust(6),
            " ",
            _fmt_duration(run.duration_ms).rjust(8),
            " ",
            fit_cells(_clean(run.revision or UNSET), 10),
            " ",
            _paint(pal, pal.dim(), _clean(run.run_id)),
        ]
    )
    if selected:
        content = pal.selected_bg() + content
    return fit_cells(content, width) + pal.reset()


def _project_row(state: TuiState, pal: Palette, project: Any, selected: bool, marks: Dict[str, str], width: int) -> str:
    name = _clean(_attr(project, "name", UNSET))
    profiles = sorted({_clean(_attr(s, "profile", "")) for s in (_attr(project, "suites", ()) or ())})
    profile_text = ",".join(p for p in profiles if p) or UNSET
    trusted = bool(_attr(project, "trusted", False))
    badge = _paint(pal, pal.green() if trusted else pal.amber(), "trusted" if trusted else "sandbox")
    latest = _latest_run_for(state, name)
    last = _STATUS_WORD.get(latest.status, UNSET) if latest else UNSET
    cursor = marks["cursor"] if selected else " "
    content = "".join(
        [
            cursor,
            " ",
            fit_cells(name, 18),
            " ",
            fit_cells(profile_text, 22),
            " ",
            badge,
            " ",
            _paint(pal, _status_style(pal, latest.status) if latest else pal.dim(), last),
        ]
    )
    if selected:
        content = pal.selected_bg() + content
    return fit_cells(content, width) + pal.reset()


def _bench_row(pal: Palette, bench: BenchmarkRow, selected: bool, marks: Dict[str, str], width: int) -> str:
    delta = UNSET
    if bench.baseline_p50_ms and bench.baseline_p50_ms > 0:
        delta = f"{(bench.p50_ms - bench.baseline_p50_ms) / bench.baseline_p50_ms * 100.0:+.1f}%"
    status = _STATUS_WORD.get(bench.status, _clean(bench.status)[:4])
    cursor = marks["cursor"] if selected else " "
    content = "".join(
        [
            cursor,
            " ",
            fit_cells(_clean(bench.project), 12),
            " ",
            fit_cells(_clean(bench.profile), 9),
            " ",
            f"p50 {bench.p50_ms:.1f}ms",
            " ",
            f"p95 {bench.p95_ms:.1f}ms",
            " ",
            f"iqr {bench.iqr_ms:.1f}ms",
            " ",
            _paint(pal, pal.amber() if delta.startswith("+") else pal.green(), f"{delta}".rjust(6)),
            " ",
            _paint(pal, _status_style(pal, bench.status), status),
        ]
    )
    if selected:
        content = pal.selected_bg() + content
    return fit_cells(content, width) + pal.reset()


def _list_rows(state: TuiState, pal: Palette, width: int, marks: Dict[str, str]) -> List[str]:
    items = _view_items(state)
    index = _clamped_index(state, items)
    rows: List[str] = []
    for position, item in enumerate(items):
        selected = position == index
        if state.view == "projects":
            rows.append(_project_row(state, pal, item, selected, marks, width))
        elif state.view == "benchmarks":
            rows.append(_bench_row(pal, item, selected, marks, width))
        elif state.view == "history":
            rows.append(_history_row(pal, item, selected, marks, width))
        else:
            rows.append(_overview_row(state, pal, item, selected, marks, width))
    return rows


def _empty_message(state: TuiState, marks: Dict[str, str]) -> str:
    project = _clean(state.project_filter) or "ALL"
    if state.view == "history":
        return f" no run history for {project}"
    if state.view == "projects":
        return " no projects discovered - run testbench discover"
    if state.view == "benchmarks":
        return " no benchmark runs recorded"
    return " no runs recorded - press r to run a profile"


def _wrap_field(pal: Palette, label: str, value: str, max_width: int) -> List[str]:
    prefix_plain = f"{label:<9}"
    avail = max(16, max_width - len(prefix_plain))
    cleaned = _clean(value or UNSET)
    if len(cleaned) <= avail:
        return [_paint(pal, pal.dim(), prefix_plain) + cleaned]
    chunks: List[str] = []
    pos = 0
    while pos < len(cleaned):
        piece = cleaned[pos : pos + avail]
        if pos == 0:
            chunks.append(_paint(pal, pal.dim(), prefix_plain) + piece)
        else:
            chunks.append(" " * len(prefix_plain) + piece)
        pos += avail
    return chunks


def _prioritized_logs(run: RunRecord, raw_logs: Sequence[str], limit: int = 4) -> List[str]:
    if not raw_logs:
        return []
    if run.status in _FAILURE_STATUSES:
        fail_lines = [
            ln for ln in raw_logs
            if any(tok in ln for tok in ("[FAIL]", "FAIL:", "ERROR:", "FAILED", "AssertionError", "Traceback"))
        ]
        other_tail = [ln for ln in raw_logs[-limit:] if ln not in fail_lines]
        return (fail_lines + other_tail)[:limit]
    return list(raw_logs[-limit:])


def _inspector_lines(state: TuiState, pal: Palette, run: Optional[RunRecord], max_width: int = 64) -> List[str]:
    if run is None:
        return [_paint(pal, pal.dim(), "no run selected")]
    tb_repro = f"testbench reproduce {_clean(run.run_id)}"
    raw_repro = _clean(_repro_for(run))
    lines = [
        _paint(pal, pal.bold(), "RUN ") + _clean(run.run_id),
        _paint(pal, pal.dim(), "project  ") + _clean(run.project),
        _paint(pal, pal.dim(), "profile  ") + _clean(run.profile),
        _paint(pal, pal.dim(), "revision ") + _clean(run.revision or UNSET),
        _paint(pal, pal.dim(), "status   ")
        + _paint(pal, _status_style(pal, run.status), _STATUS_WORD.get(run.status, _clean(run.status)[:4])),
        _paint(pal, pal.dim(), "counts   ")
        + f"{_counts(run)}  fail={run.fail_count}"
        + (f" err={run.error_count}" if run.error_count else "")
        + f" skip={run.skip_count}",
        _paint(pal, pal.dim(), "duration ") + _fmt_duration(run.duration_ms),
    ]
    lines.extend(_wrap_field(pal, "failure", run.first_failure or UNSET, max_width))
    lines.extend(_wrap_field(pal, "repro", tb_repro, max_width))
    if raw_repro:
        lines.extend(_wrap_field(pal, "cmd", raw_repro, max_width))
    lines.extend(_wrap_field(pal, "evidence", run.evidence_dir or UNSET, max_width))
    if run.delegation_id:
        lines.append(_paint(pal, pal.dim(), "deleg    ") + f"[deleg:{_clean(run.delegation_id)}]")
    raw_logs = list(state.log_lines.get(run.run_id, ()))
    lines.append(_paint(pal, pal.dim(), "log      ") + f"{len(raw_logs)} lines")
    for log in _prioritized_logs(run, raw_logs, limit=4):
        lines.append("  " + _clean(log))
    return lines


def _detail_lines(state: TuiState, pal: Palette) -> List[str]:
    run = _selected_run(state)
    if run is None:
        return ["DETAIL", "", "no run selected"]
    lines = [
        _paint(pal, pal.bold(), "DETAIL ") + _paint(pal, pal.dim(), "· ") + _clean(run.run_id),
        _paint(pal, pal.dim(), "project  ") + _clean(run.project),
        _paint(pal, pal.dim(), "profile  ") + _clean(run.profile),
        _paint(pal, pal.dim(), "revision ") + _clean(run.revision or UNSET),
        _paint(pal, pal.dim(), "status   ")
        + _paint(pal, _status_style(pal, run.status), _STATUS_WORD.get(run.status, run.status)),
        _paint(pal, pal.dim(), "counts   ")
        + f"pass {run.pass_count}/{run.total_count}  fail={run.fail_count}"
        + (f"  err={run.error_count}" if run.error_count else "")
        + f"  skip={run.skip_count}",
        _paint(pal, pal.dim(), "duration ") + _fmt_duration(run.duration_ms),
        _paint(pal, pal.dim(), "failure  ") + _clean(run.first_failure or UNSET),
        _paint(pal, pal.dim(), "repro    ") + _clean(_repro_for(run) or UNSET),
        _paint(pal, pal.dim(), "evidence ") + _clean(run.evidence_dir or UNSET),
        _paint(pal, pal.dim(), "deleg    ") + (_clean(run.delegation_id) or UNSET),
        "",
        _paint(pal, pal.bold(), "LOG"),
    ]
    log_lines = list(state.log_lines.get(run.run_id, ()))
    if not log_lines:
        lines.append(_paint(pal, pal.dim(), " (no captured log lines)"))
    for number, log in enumerate(log_lines, start=1):
        lines.append(f" {number:03d} {_clean(log)}")
    lines.append("")
    lines.append(_paint(pal, pal.dim(), "q close  j/k scroll"))
    return lines


def _help_lines(pal: Palette) -> List[str]:
    entries = [
        ("j / k", "move selection"),
        ("g / G", "first / last"),
        ("Enter", "inspect selected run (never executes)"),
        ("Tab / 1-4", "overview, history, projects, benchmarks"),
        ("p", "project filter picker"),
        ("/", "search and filter runs"),
        ("r", "run a profile (y confirms)"),
        ("R", "rerun selected run (y confirms)"),
        ("x", "cancel owned active job (y confirms)"),
        ("y", "copy repro command"),
        ("q / Esc", "close overlay; Q quits the TUI"),
        ("?", "close this help"),
    ]
    lines = [_paint(pal, pal.bold(), "HELP"), ""]
    for key, label in entries:
        lines.append("  " + _paint(pal, pal.bold(), fit_cells(key, 10)) + _paint(pal, pal.dim(), label))
    return lines


def _picker_options(state: TuiState) -> List[Tuple[str, str, bool]]:
    if state.picker_kind == "profile":
        profiles = state.profiles or tuple(
            ProfileOption(name, billable=(name == ProfileName.AGENT_EVAL)) for name in ProfileName.STANDARD
        )
        return [(p.name, p.description, p.billable) for p in profiles]
    names = {_clean(_attr(p, "name")) for p in state.projects}
    names.update(_clean(r.project) for r in state.runs)
    return [("ALL", "", False)] + [(name, "", False) for name in sorted(names) if name]


def _picker_lines(state: TuiState, pal: Palette, marks: Dict[str, str]) -> List[str]:
    options = _picker_options(state)
    title = "PROFILE" if state.picker_kind == "profile" else "PROJECT FILTER"
    lines = [_paint(pal, pal.bold(), title), ""]
    index = max(0, min(state.picker_index, len(options) - 1)) if options else 0
    for position, (name, description, billable) in enumerate(options):
        cursor = marks["cursor"] if position == index else " "
        label = _clean(name) if not description else f"{_clean(name)}  {_clean(description)}"
        if billable:
            label += "  (billable)"
        lines.append(f"{cursor} {label}")
    lines.append("")
    lines.append(_paint(pal, pal.dim(), "Enter select  j/k move  q close"))
    return lines


def _minibuffer_line(state: TuiState, pal: Palette, width: int) -> str:
    if state.confirm is not None:
        text = " " + _clean(state.confirm.label) + "  (y confirm · any other key cancels)"
        return fit_cells(_paint(pal, pal.amber(), text), width) + pal.reset()
    if state.search_open:
        text = " /" + _clean(state.search_buffer) + "  (Enter apply · Esc clear)"
        return fit_cells(_paint(pal, pal.blue(), text), width) + pal.reset()
    if state.minibuffer:
        return fit_cells(" " + _clean(state.minibuffer), width) + pal.reset()
    if state.error_banner:
        return fit_cells(_paint(pal, pal.red(), " store unavailable: " + _clean(state.error_banner)), width) + pal.reset()
    return fit_cells(_paint(pal, pal.dim(), " ready"), width) + pal.reset()


def _header_line(state: TuiState, pal: Palette, width: int, marks: Dict[str, str]) -> str:
    sep = f" {marks['sep']} "
    project = _clean(state.project_filter).upper() or "ALL"
    left = (
        _paint(pal, pal.bold(), "TESTBENCH")
        + _paint(pal, pal.dim(), sep)
        + _paint(pal, pal.bold(), project)
        + _paint(pal, pal.dim(), sep)
        + _paint(pal, pal.blue(), _clean(state.view))
    )
    scope_runs = _overview_runs(state) if state.view == "overview" else _history_runs(state)
    running = sum(1 for r in state.runs if r.status == RunStatus.RUNNING)
    queued = sum(1 for r in state.runs if r.status == RunStatus.QUEUED)
    passed = sum(1 for r in scope_runs if r.status == RunStatus.PASS)
    failed = sum(1 for r in scope_runs if r.status in _FAILURE_STATUSES)
    store = "unavailable" if state.error_banner else _clean(state.store_status or "ok")
    right_bits = [
        _paint(pal, pal.blue() if running or queued else pal.dim(), f"{running} running"),
        f"{queued} queued",
        _paint(pal, pal.green() if passed else pal.dim(), f"{passed} pass"),
        _paint(pal, pal.red() if failed else pal.dim(), f"{failed} fail"),
        _paint(pal, pal.dim(), f"store {store}"),
    ]
    right = _paint(pal, pal.dim(), sep).join(right_bits)
    gap = width - display_width(left) - display_width(right)
    if gap >= 1:
        content = left + " " * gap + right
    else:
        content = left
    return fit_cells(content, width) + pal.reset()


def _split_line(left: str, right: str, pal: Palette, left_w: int, right_w: int) -> str:
    return fit_cells(left, left_w) + pal.reset() + "  " + fit_cells(right, right_w) + pal.reset()


def _fit_body(lines: Sequence[str], body_h: int, width: int) -> List[str]:
    rows = [line for line in lines]
    if len(rows) > body_h:
        rows = rows[:body_h]
    while len(rows) < body_h:
        rows.append("")
    return [fit_cells(row, width) + "\033[0m" for row in rows]


_ESC_KEYS = frozenset({"\x1b", "Esc", "Escape"})
_DOWN_KEYS = frozenset({"j", "Down"})
_UP_KEYS = frozenset({"k", "Up"})
_ENTER_KEYS = frozenset({"Enter", "\r", "\n"})
_TAB_KEYS = frozenset({"Tab", "\t"})
_BACKSPACE_KEYS = frozenset({"Backspace", "\x7f", "\b"})


def _view_len(state: TuiState) -> int:
    return len(_view_items(state))


def _is_billable(state: TuiState, profile: str) -> bool:
    if profile == ProfileName.AGENT_EVAL:
        return True
    return any(option.name == profile and option.billable for option in state.profiles)


def _is_owned(run: RunRecord) -> bool:
    return run.worker_pid is not None or run.run_id.startswith("tb_")


def _prompt_label(kind: str, profile: str = "", run_id: str = "", project: str = "") -> str:
    where = project or "ALL"
    if kind == "run_profile":
        return f"run profile {profile} on {where}?"
    if kind == "rerun":
        return f"rerun {run_id}?"
    return f"cancel {run_id}? (owned job only)"


def _confirm_action(prompt: ConfirmPrompt) -> TuiAction:
    if prompt.kind == "run_profile":
        return TuiAction("run_profile", {"profile": prompt.profile, "project": prompt.project})
    if prompt.kind == "rerun":
        return TuiAction(
            "rerun",
            {"run_id": prompt.run_id, "project": prompt.project, "profile": prompt.profile},
        )
    return TuiAction("cancel_run", {"run_id": prompt.run_id, "project": prompt.project})


def handle_key(state: TuiState, key: str) -> Tuple[TuiState, Optional[TuiAction]]:
    # 1. Pending confirmation: only an explicit "y" may execute.
    if state.confirm is not None:
        prompt = state.confirm
        if key in ("y", "Y"):
            action = _confirm_action(prompt)
            label = prompt.profile if prompt.kind == "run_profile" else prompt.run_id
            done = replace(state, confirm=None, minibuffer=f"{prompt.kind}: {_clean(label)}")
            return done, action
        return replace(state, confirm=None, minibuffer="cancelled"), None

    # 2. Overlays, each closing with q/Esc.
    if state.help_open:
        if key in _ESC_KEYS or key in ("q", "?"):
            return replace(state, help_open=False, minibuffer=""), None
        return state, None

    if state.detail_open:
        if key in _ESC_KEYS or key == "q":
            return replace(state, detail_open=False, detail_scroll=0, minibuffer=""), None
        if key in _DOWN_KEYS:
            return replace(state, detail_scroll=state.detail_scroll + 1), None
        if key in _UP_KEYS:
            return replace(state, detail_scroll=max(0, state.detail_scroll - 1)), None
        if key == "g":
            return replace(state, detail_scroll=0), None
        if key == "G":
            total = len(_detail_lines(state, Palette(no_color=True)))
            return replace(state, detail_scroll=max(0, total - 1)), None
        if key == "y":
            run = _selected_run(state)
            if run is None:
                return replace(state, minibuffer="no run selected"), None
            return state, TuiAction("copy_repro", {"repro": _repro_for(run)})
        return state, None

    if state.picker_open:
        options = _picker_options(state)
        last = max(0, len(options) - 1)
        if key in _ESC_KEYS or key in ("q", "p"):
            return replace(state, picker_open=False, minibuffer=""), None
        if key in _DOWN_KEYS:
            return replace(state, picker_index=min(last, state.picker_index + 1)), None
        if key in _UP_KEYS:
            return replace(state, picker_index=max(0, state.picker_index - 1)), None
        if key == "g":
            return replace(state, picker_index=0), None
        if key == "G":
            return replace(state, picker_index=last), None
        if key in _ENTER_KEYS and options:
            choice, _description, billable = options[state.picker_index]
            if state.picker_kind == "profile":
                if billable or _is_billable(state, choice):
                    return replace(
                        state, picker_open=False, confirm=None, minibuffer=BILLABLE_REFUSAL
                    ), None
                prompt = ConfirmPrompt(
                    kind="run_profile",
                    label=_prompt_label("run_profile", profile=choice, project=state.project_filter),
                    profile=choice,
                    project=state.project_filter,
                )
                return replace(state, picker_open=False, confirm=prompt, minibuffer=""), None
            chosen = "" if choice == "ALL" else choice
            return replace(
                state,
                picker_open=False,
                project_filter=chosen,
                selected=0,
                minibuffer=f"project filter: {chosen or 'ALL'}",
            ), None
        return state, None

    if state.search_open:
        if key in _ESC_KEYS:
            return replace(state, search_open=False, search_buffer="", search_query=""), None
        if key in _ENTER_KEYS:
            query = state.search_buffer
            return replace(
                state,
                search_open=False,
                search_query=query,
                selected=0,
                minibuffer=(f"/{query} applied" if query else "filter cleared"),
            ), None
        if key in _BACKSPACE_KEYS:
            return replace(state, search_buffer=state.search_buffer[:-1]), None
        if len(key) == 1 and key.isprintable():
            return replace(state, search_buffer=state.search_buffer + key), None
        return state, None

    # 3. Top level. Every fresh keypress clears a stale flash first.
    base = replace(state, minibuffer="")
    items = _view_len(state)
    last = max(0, items - 1)

    if key == "Q":
        return base, TuiAction("quit")
    if key == "q":
        if state.permanent_tab:
            return replace(base, minibuffer=PERMANENT_TAB_QUIT_HINT), None
        return base, TuiAction("quit")
    if key in _ESC_KEYS:
        return base, None
    if key in _TAB_KEYS:
        return replace(base, view=_next_view(state.view), selected=0, detail_scroll=0), None
    if key in ("1", "2", "3", "4"):
        return replace(base, view=VIEWS[int(key) - 1], selected=0, detail_scroll=0), None
    if key in _DOWN_KEYS:
        return replace(base, selected=min(last, state.selected + 1)), None
    if key in _UP_KEYS:
        return replace(base, selected=max(0, state.selected - 1)), None
    if key == "g":
        return replace(base, selected=0), None
    if key == "G":
        return replace(base, selected=last), None
    if key in ("?",):
        return replace(base, help_open=True), None
    if key == "p":
        options = _picker_options(replace(base, picker_kind="project"))
        current = state.project_filter or "ALL"
        index = next((i for i, opt in enumerate(options) if opt[0] == current), 0)
        return replace(
            base, picker_open=True, picker_kind="project", picker_index=index
        ), None
    if key == "/":
        return replace(base, search_open=True, search_buffer=state.search_query), None
    if key == "r":
        return replace(base, picker_open=True, picker_kind="profile", picker_index=0), None
    if key in _ENTER_KEYS:
        return replace(base, detail_open=True, detail_scroll=0), None

    run = _selected_run(state)
    if key == "R":
        if run is None:
            return replace(base, minibuffer="no run selected"), None
        if _is_billable(state, run.profile):
            return replace(base, minibuffer=BILLABLE_REFUSAL, confirm=None), None
        prompt = ConfirmPrompt(
            kind="rerun",
            label=_prompt_label("rerun", run_id=run.run_id),
            profile=run.profile,
            run_id=run.run_id,
            project=run.project,
        )
        return replace(base, confirm=prompt), None
    if key == "x":
        if run is None:
            return replace(base, minibuffer="no run selected"), None
        if run.status not in RunStatus.ACTIVE:
            return replace(base, minibuffer="cancel: selected run is not RUNNING/QUEUED"), None
        if not _is_owned(run):
            return replace(base, minibuffer="cancel: run is not TestBench-owned"), None
        prompt = ConfirmPrompt(
            kind="cancel_run",
            label=_prompt_label("cancel_run", run_id=run.run_id),
            profile=run.profile,
            run_id=run.run_id,
            project=run.project,
        )
        return replace(base, confirm=prompt), None
    if key == "y":
        if run is None:
            return replace(base, minibuffer="no run selected"), None
        return base, TuiAction("copy_repro", {"repro": _repro_for(run)})
    return base, None


def _next_view(view: str) -> str:
    try:
        index = VIEWS.index(view)
    except ValueError:
        return VIEWS[0]
    return VIEWS[(index + 1) % len(VIEWS)]


def render_frame(
    state: TuiState,
    width: int,
    height: int,
    *,
    no_color: bool = False,
    ascii_mode: bool = False,
) -> List[str]:
    if width <= 0 or height <= 0:
        return []
    monochrome = bool(no_color or ascii_mode)
    pal = Palette(no_color=monochrome)
    marks = _marks(ascii_mode)
    blank = " " * width
    if height < 4 or width < 10:
        return [blank for _ in range(height)]

    body_h = height - 3
    lines = [_header_line(state, pal, width, marks)]

    body: List[str] = []
    if state.error_banner:
        body.append(
            fit_cells(
                _paint(pal, pal.red(), "! TestBench store unavailable: " + _clean(state.error_banner)),
                width,
            )
            + pal.reset()
        )
    remaining = body_h - len(body)

    if state.help_open:
        body.extend(_fit_body(_help_lines(pal), remaining, width))
    elif state.detail_open:
        detail = _detail_lines(state, pal)
        scroll = max(0, min(state.detail_scroll, max(0, len(detail) - 1)))
        body.extend(_fit_body(detail[scroll : scroll + remaining], remaining, width))
    elif state.picker_open:
        body.extend(_fit_body(_picker_lines(state, pal, marks), remaining, width))
    elif width >= 120:
        left_w = max(30, min(68, width // 2 - 2))
        right_w = max(10, width - left_w - 2)
        left_rows = _list_rows(state, pal, left_w, marks)
        if not left_rows:
            left_rows = [_paint(pal, pal.dim(), _empty_message(state, marks))]
        right_rows = _inspector_lines(state, pal, _selected_run(state), max_width=right_w)
        split: List[str] = []
        for index in range(remaining):
            left = left_rows[index] if index < len(left_rows) else ""
            right = right_rows[index] if index < len(right_rows) else ""
            split.append(_split_line(left, right, pal, left_w, right_w))
        body.extend(split)
    else:
        rows = _list_rows(state, pal, width, marks)
        if not rows:
            rows = [_paint(pal, pal.dim(), _empty_message(state, marks))]
        sel_run = _selected_run(state)
        if sel_run is not None and remaining >= len(rows) + 5:
            rows = list(rows) + [""] + _inspector_lines(state, pal, sel_run, max_width=width)[: max(4, remaining - len(rows) - 1)]
        body.extend(_fit_body(rows, remaining, width))

    if len(body) > body_h:
        body = body[:body_h]
    while len(body) < body_h:
        body.append(blank)
    lines.extend(body)

    compact_hints = "j/k move  Enter inspect  Tab view  p proj  r run  R rerun  x cancel  ? help"
    hints_str = HINTS_LINE if width >= len(HINTS_LINE) + 2 else compact_hints
    lines.append(fit_cells(" " + hints_str, width) + "\033[0m")
    lines.append(_minibuffer_line(state, pal, width))
    return lines


def format_ansi_frame(lines: Sequence[str], height: int) -> str:
    if height <= 0:
        return "\033[?25l\033[H\033[J"
    rows = list(lines)[:height]
    chunks: List[str] = ["\033[?25l\033[H"]
    for index, line in enumerate(rows):
        chunks.append(line)
        chunks.append("\033[K")
        if index != len(rows) - 1:
            chunks.append("\r\n")
    chunks.append("\033[J")
    return "".join(chunks)
