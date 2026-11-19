"""Interactive terminal controller for hx:testbench (ADR-0003).

Drives pure ``tui_render`` over a real TTY with ``select.kqueue`` + SQLite
``PRAGMA data_version`` refresh, immediate ``SIGWINCH`` resize, and safe action
dispatch through :class:`TestBenchEngine`. Never auto-runs suites on startup.
"""

from __future__ import annotations

from dataclasses import replace
import errno
import fcntl
import os
from pathlib import Path
import select
import shutil
import signal
import struct
import subprocess
import sys
import termios
import time
import tty
from typing import Dict, List, Optional, Tuple

from .engine import TestBenchEngine
from .schema import ProfileName, RunRecord, RunStatus, utc_now_iso
from .store import SchemaMismatchError, TestBenchStore, default_testbench_home
from .tui_render import (
    BenchmarkRow,
    ProfileOption,
    TuiAction,
    TuiState,
    format_ansi_frame,
    handle_key,
    preserve_history_selection,
    render_frame,
)


def _write_ansi(fd: int, text: str) -> None:
    buf = memoryview(text.encode("utf-8"))
    sent = 0
    total = len(buf)
    while sent < total:
        try:
            n = os.write(fd, buf[sent:])
            sent += n
        except BlockingIOError:
            select.select([], [fd], [], 0.2)
        except OSError:
            break


def _term_size(fd: int) -> Tuple[int, int]:
    try:
        packed = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\x00" * 8)
        rows, cols, _, _ = struct.unpack("HHHH", packed)
        if cols > 0 and rows > 0:
            return int(cols), int(rows)
    except OSError:
        pass
    sz = shutil.get_terminal_size((120, 36))
    return int(sz.columns), int(sz.lines)


def _read_keys(fd: int) -> List[str]:
    try:
        raw = os.read(fd, 128)
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return []
        return ["Q"]
    if not raw:
        return ["Q"]

    text = raw.decode("utf-8", errors="replace")
    keys: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\x03":  # Ctrl-C exits cleanly
            keys.append("Q")
            i += 1
        elif ch == "\x1b":
            if i + 2 < n and text[i + 1] == "[":
                code = text[i + 2]
                if code == "A":
                    keys.append("Up")
                    i += 3
                    continue
                if code == "B":
                    keys.append("Down")
                    i += 3
                    continue
                if code == "Z":
                    keys.append("Tab")
                    i += 3
                    continue
            keys.append("Esc")
            i += 1
        elif ch in ("\r", "\n"):
            keys.append("Enter")
            i += 1
        elif ch == "\t":
            keys.append("Tab")
            i += 1
        elif ch in ("\x7f", "\x08"):
            keys.append("Backspace")
            i += 1
        else:
            keys.append(ch)
            i += 1
    return keys


def _load_log_tail(run: RunRecord, max_lines: int = 80) -> Tuple[str, ...]:
    # The inspector shows one tail. Prefer failure evidence over passing output.
    failure_statuses = {RunStatus.FAIL, RunStatus.ERROR, RunStatus.TIMEOUT, RunStatus.CANCELLED, RunStatus.BLOCKED}
    failures = [s for s in run.suites if s.status in failure_statuses]
    for s in failures or run.suites:
        if s.sanitized_path and Path(s.sanitized_path).is_file():
            try:
                lines = Path(s.sanitized_path).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = []
            if lines:
                return tuple(lines[-max_lines:])
    if failures:
        return ("Failure log unavailable",)
    ev = Path(run.evidence_dir) if run.evidence_dir else None
    if ev and ev.is_dir():
        for cand in sorted(ev.glob("*.sanitized.log")):
            try:
                lines = cand.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            if lines:
                return tuple(lines[-max_lines:])
    return ()


def _refresh_active_logs(state: TuiState) -> TuiState:
    """Re-read sanitized tails for ACTIVE runs.

    A live run writes its log but not the store, so no data_version change
    signals the TUI; without this the panel shows the pre-run tail until the
    suite finishes.
    """
    merged = dict(state.log_lines)
    for r in state.runs:
        if r.status in RunStatus.ACTIVE:
            tail = _load_log_tail(r)
            if tail:
                merged[r.run_id] = tail
    return replace(state, log_lines=merged)


def _refresh_from_store(store: TestBenchStore, prev: TuiState) -> TuiState:
    store.reconcile_orphans()
    runs = tuple(store.list_runs(limit=80))
    projects = tuple(store.list_projects())
    bench_rows_raw = store.list_benchmarks(limit=40)

    baselines: Dict[str, float] = {}
    for r in reversed(runs):
        if r.status == RunStatus.PASS and r.duration_ms is not None:
            baselines[f"{r.project}:{r.profile}"] = float(r.duration_ms)

    bench_rows: List[BenchmarkRow] = []
    seen_workloads: Dict[str, float] = {}
    for b in reversed(bench_rows_raw):
        key = f"{b['project']}:{b['workload']}"
        p = b.get("payload") or {}
        stats = p.get("stats") or {}
        p50 = float(stats.get("p50_ms") or b.get("median_ms") or 0.0)
        p95 = float(stats.get("p95_ms") or 0.0)
        iqr = float(stats.get("iqr_ms") or 0.0)
        base_p50 = seen_workloads.get(key)
        seen_workloads.setdefault(key, p50)
        bench_rows.append(
            BenchmarkRow(
                project=str(b["project"]),
                profile=str(b["workload"]),
                run_id=str(b["run_id"]),
                p50_ms=p50,
                p95_ms=p95,
                iqr_ms=iqr,
                baseline_p50_ms=base_p50,
                created_at=str(b.get("created_at") or ""),
            )
        )
    bench_rows.reverse()

    log_map: Dict[str, Tuple[str, ...]] = {}
    for r in runs[:15]:
        tail = _load_log_tail(r)
        if tail:
            log_map[r.run_id] = tail

    default_profiles = (
        ProfileOption(ProfileName.UNIT, False, "fast isolated unit tests"),
        ProfileOption(ProfileName.INTEGRATION, False, "cross-module & SQLite integration"),
        ProfileOption(ProfileName.TERMINAL, False, "PTY & tmux lifecycle checks"),
        ProfileOption(ProfileName.SECURITY, False, "escape, sanitizer & confinement checks"),
        ProfileOption(ProfileName.REGRESSION, False, "full regression verification"),
        ProfileOption(ProfileName.BENCHMARK, False, "repeated workload measurements"),
        ProfileOption(ProfileName.AGENT_EVAL, True, "opt-in bounded LLM behavior probe"),
    )

    updated = replace(
        prev,
        runs=runs,
        projects=projects,
        benchmarks=tuple(bench_rows),
        profiles=default_profiles,
        baselines=baselines,
        log_lines=log_map,
        error_banner="",
        store_status="ok",
        now_iso=utc_now_iso(),
    )
    return preserve_history_selection(prev, updated)


def _copy_text(text: str) -> str:
    if not text:
        return "nothing to copy"
    tmux_bin = shutil.which("tmux")
    if os.environ.get("TMUX") and tmux_bin:
        rc = subprocess.run([tmux_bin, "set-buffer", "--", text], capture_output=True, timeout=2.0, check=False).returncode
        if rc == 0:
            return "copied repro command to tmux buffer"
    pbcopy = shutil.which("pbcopy")
    if pbcopy:
        rc = subprocess.run([pbcopy], input=text, text=True, capture_output=True, timeout=2.0, check=False).returncode
        if rc == 0:
            return "copied repro command to clipboard"
    return f"repro: {text[:80]}"


def _execute_action(eng: TestBenchEngine, state: TuiState, action: TuiAction) -> TuiState:
    if action.kind == "copy_repro":
        msg = _copy_text(str(action.payload.get("repro") or ""))
        return replace(state, minibuffer=msg)

    if action.kind == "cancel_run":
        rid = str(action.payload.get("run_id") or "")
        try:
            eng.cancel(rid)
            return replace(_refresh_from_store(eng.store, state), minibuffer=f"cancelled {rid}")
        except Exception as exc:
            return replace(state, minibuffer=f"cancel failed: {exc}")

    if action.kind == "rerun":
        rid = str(action.payload.get("run_id") or "")
        orig = eng.store.get_run(rid)
        if orig is None:
            return replace(state, minibuffer=f"unknown run {rid}")
        try:
            rec = eng.spawn_async(Path(orig.project_root), profile=orig.profile, wait_up_to_s=0.15)
            return replace(_refresh_from_store(eng.store, state), minibuffer=f"launched {rec.run_id} ({rec.status})")
        except Exception as exc:
            return replace(state, minibuffer=f"rerun failed: {exc}")

    if action.kind == "run_profile":
        profile = str(action.payload.get("profile") or "unit")
        proj_name = str(action.payload.get("project") or state.project_filter or "")
        target_path: Optional[Path] = None
        for r in state.runs:
            if not proj_name or r.project == proj_name:
                target_path = Path(r.project_root)
                break
        if target_path is None:
            for p in state.projects:
                pname = p.get("name") if isinstance(p, dict) else getattr(p, "name", "")
                proot = p.get("project_root") if isinstance(p, dict) else getattr(p, "root", "")
                if proot and (not proj_name or pname == proj_name):
                    target_path = Path(str(proot))
                    break
        if target_path is None:
            target_path = Path.cwd()
        try:
            rec = eng.spawn_async(target_path, profile=profile, wait_up_to_s=0.15)
            return replace(_refresh_from_store(eng.store, state), minibuffer=f"launched {rec.run_id} ({rec.status})")
        except Exception as exc:
            return replace(state, minibuffer=f"run failed: {exc}")

    return state


def run_tui(
    *,
    home: Optional[Path] = None,
    permanent_tab: bool = True,
    initial_project: Optional[str] = None,
) -> int:
    in_fd = sys.stdin.fileno()
    out_fd = sys.stdout.fileno()
    is_tty = os.isatty(in_fd) and os.isatty(out_fd)

    no_color = bool(os.environ.get("NO_COLOR")) or os.environ.get("TERM") == "dumb"
    ascii_mode = os.environ.get("HERMES_TMUX_ASCII") == "1"

    state = TuiState(
        project_filter=initial_project or "",
        permanent_tab=permanent_tab,
        now_iso=utc_now_iso(),
    )

    store: Optional[TestBenchStore] = None
    eng: Optional[TestBenchEngine] = None
    tb_home = (home or default_testbench_home()).expanduser()
    try:
        store = TestBenchStore(tb_home)
        eng = TestBenchEngine(store)
        state = _refresh_from_store(store, state)
    except (OSError, SchemaMismatchError) as exc:
        state = replace(state, error_banner=str(exc), store_status="error")

    if not is_tty:
        cols, rows = _term_size(out_fd)
        for ln in render_frame(state, cols, rows, no_color=no_color, ascii_mode=ascii_mode):
            sys.stdout.write(ln + "\n")
        sys.stdout.flush()
        if store is not None:
            store.close()
        return 0

    orig_attrs = termios.tcgetattr(in_fd)
    wake_r, wake_w = os.pipe()
    os.set_blocking(wake_r, False)
    os.set_blocking(wake_w, False)

    resized = False
    stop_requested = False

    def _on_winch(_s: int, _f: object) -> None:
        nonlocal resized
        resized = True
        try:
            os.write(wake_w, b"w")
        except OSError:
            pass

    def _on_stop(_s: int, _f: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        try:
            os.write(wake_w, b"q")
        except OSError:
            pass

    old_winch = signal.signal(signal.SIGWINCH, _on_winch)
    old_term = signal.signal(signal.SIGTERM, _on_stop)
    old_int = signal.signal(signal.SIGINT, _on_stop)

    kq = select.kqueue() if hasattr(select, "kqueue") else None
    watched_dir_fd = -1
    if kq is not None and tb_home.is_dir():
        try:
            watched_dir_fd = os.open(str(tb_home), os.O_RDONLY)
            kq.control(
                [
                    select.kevent(
                        watched_dir_fd,
                        filter=select.KQ_FILTER_VNODE,
                        flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                        fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_EXTEND | select.KQ_NOTE_RENAME,
                    )
                ],
                0,
                0,
            )
        except OSError:
            pass

    last_dv = store.data_version() if store is not None else -1
    last_frame_sig: Optional[Tuple[str, int, int]] = None
    last_tick = time.monotonic()

    try:
        tty.setraw(in_fd, when=termios.TCSANOW)
        _write_ansi(out_fd, "\033[?1049h\033[?25l")

        while not stop_requested:
            cols, rows = _term_size(out_fd)
            has_active = any(r.status in RunStatus.ACTIVE for r in state.runs)
            now_mono = time.monotonic()
            if has_active and (now_mono - last_tick) >= 1.0:
                state = replace(state, now_iso=utc_now_iso())
                state = _refresh_active_logs(state)
                last_tick = now_mono

            lines = render_frame(state, cols, rows, no_color=no_color, ascii_mode=ascii_mode)
            frame_str = format_ansi_frame(lines, rows)
            sig = (frame_str, cols, rows)
            if sig != last_frame_sig or resized:
                resized = False
                _write_ansi(out_fd, frame_str)
                last_frame_sig = sig

            wait_timeout = 0.25 if has_active else 0.35
            rlist, _, _ = select.select([in_fd, wake_r], [], [], wait_timeout)

            if wake_r in rlist:
                try:
                    os.read(wake_r, 64)
                except OSError:
                    pass

            if in_fd in rlist:
                for key in _read_keys(in_fd):
                    state, act = handle_key(state, key)
                    if act is not None:
                        if act.kind == "quit":
                            stop_requested = True
                            break
                        if eng is not None:
                            state = _execute_action(eng, state, act)
                            last_dv = store.data_version() if store is not None else -1

            if stop_requested:
                break

            kq_fired = False
            if kq is not None:
                try:
                    evs = kq.control(None, 4, 0)
                    kq_fired = bool(evs)
                except OSError:
                    pass

            if store is not None:
                cur_dv = store.data_version()
                if kq_fired or cur_dv != last_dv or has_active:
                    if cur_dv != last_dv or kq_fired:
                        state = _refresh_from_store(store, state)
                        last_dv = cur_dv
        return 0
    finally:
        try:
            termios.tcsetattr(in_fd, termios.TCSANOW, orig_attrs)
        except termios.error:
            pass
        try:
            _write_ansi(out_fd, "\033[0m\033[?25h\033[?1049l")
        except OSError:
            pass
        signal.signal(signal.SIGWINCH, old_winch)
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        for fd in (wake_r, wake_w, watched_dir_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if kq is not None:
            try:
                kq.close()
            except OSError:
                pass
        if store is not None:
            store.close()
