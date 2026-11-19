"""Interactive terminal controller for hx:testbench (ADR-0003).

Two entry shapes, one state type:

* real TTY — alternate screen, raw mode, frame diffing, signal-driven resize,
  key dispatch, background refresh (the ``run_tui`` loop);
* anything else (pipe, CI, ``testbench tui > file``) — render one frame at the
  current size and exit. No alternate screen, no keys, no store writes beyond
  the refresh that produced the frame.

Division of labour: ``tui_render`` decides *what a frame looks like* and is
pure; this module decides *when* to render and *which side effects are allowed*.
Actions leave the state machine as :class:`TuiAction` values and are executed
here, so a key that would run or cancel a suite cannot execute anything until
``handle_key`` has seen an explicit ``y``.

Refresh has four independent inputs, because a live run is invisible to any one
of them:

1. ``select.kqueue`` on ``$TESTBENCH_HOME`` (VNODE write/extend/rename) — fires
   on the SQLite WAL and on newly created run directories;
2. ``PRAGMA data_version`` — a per-connection counter that changes when
   *another* connection commits; immune to WAL checkpointing, unlike a file
   watch, and the only mechanism on platforms without kqueue;
3. a 1 Hz tick, armed only while a run is QUEUED/RUNNING — advances ``now_iso``
   for elapsed labels and re-reads active log tails, which a worker writes
   without touching the store;
4. keys, which mutate state, may execute an action, and then force a store read.

Every hint is advisory: state is only ever *rebuilt* from the store (or from a
log file), never patched in place from a signal. A missed hint costs a
sub-second staleness, not a lost update visible to the operator.

Invariants: nothing here runs a suite in the foreground (suites are detached
through the engine, so a slow suite cannot freeze the pane); nothing auto-runs
on startup; and terminal state is restored in a ``finally``, because an
exception escaping the loop must not leave a raw-mode pane behind.
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
    """Write one whole frame, tolerating a short or temporarily blocked write.

    A terminal write can accept fewer bytes than offered, so the remainder is
    retried after a bounded wait. A hard ``OSError`` (EIO on a pane that is
    going away) ends the write silently: nothing here can repair a broken tty,
    and raising would escape through the loop's teardown.

    The silent-stop case has a visible consequence worth knowing when debugging
    a half-drawn pane: the caller still records the frame signature, so that
    partially written frame is not retried until the state or geometry changes.
    """
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
    """Current ``(cols, rows)`` for ``fd``.

    ioctl on the tty is authoritative and is re-read every loop iteration rather
    than cached, so a resize is correct on the very next frame even if the
    SIGWINCH handler and the geometry read are separated by other work. The
    ``shutil`` fallback covers a non-tty stdout (the one-shot render path) and an
    ioctl that fails while the pane is being torn down; 120x36 is the same
    default the renderer is exercised against in tests.
    """
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
    """Decode one ``os.read`` chunk into key tokens.

    The mapping is deliberately tiny, because the renderer's state machine only
    knows these tokens: printable characters pass through as themselves, Enter
    normalises CR/LF, Backspace normalises DEL/BS, ``ESC [ A|B|Z`` becomes
    Up/Down/Tab, and Ctrl-C is mapped onto the same clean-quit path as ``Q``.

    A lone ESC is Esc — which closes whatever overlay is open. There is no
    partial-sequence buffer here, so an arrow key split across two reads
    degrades to "Esc, then ``[`` and ``A`` as printable keys". That is a known
    limitation, not a design statement; it matters because a stray Esc closes
    the detail sheet.

    End of file or a non-EAGAIN read error returns ``["Q"]``: a raw-mode tty
    reports EOF only when the pane or its writer is gone, and exiting is then
    the honest action. EAGAIN means "no key waiting" and returns ``[]`` — an
    empty read is never itself evidence that the pane disappeared.
    """
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
    """Sanitized tail the inspector should show for ``run`` (up to ``max_lines``).

    Precedence, highest first:

    1. the first suite whose status is failure-like *and* whose sanitized log
       file still exists and is non-empty;
    2. the first suite with any readable log (so a passing multi-suite run is
       not blank);
    3. if a failure-like suite existed but its log is gone, the single line
       ``"Failure log unavailable"`` — a missing file must not read as "no
       output";
    4. otherwise the newest ``*.sanitized.log`` under ``evidence_dir``;
    5. otherwise ``()``.

    Failure-like here is five statuses (FAIL, ERROR, TIMEOUT, CANCELLED,
    BLOCKED), wider than the three used to rank History; CANCELLED and BLOCKED
    runs still need an explanation even though they are not regressions.
    """
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
    """Rebuild the store-owned half of ``TuiState`` from one fresh read.

    Replaced: ``runs`` (newest 80), ``projects``, ``benchmarks`` (newest 40),
    ``profiles``, ``baselines``, ``log_lines`` (tails for the newest 15 runs),
    ``now_iso``; ``error_banner`` is cleared and ``store_status`` set to ``ok``.
    Preserved: every interaction field (view, filters, selection, overlays,
    minibuffer, history order) and ``permanent_tab``. History selection is
    re-anchored to the same run id afterwards, because a refresh can reorder or
    re-page rows between two keystrokes.

    Side effect worth remembering when debugging unexpected ERROR rows:
    ``reconcile_orphans`` runs first. The TUI therefore reaps QUEUED/RUNNING
    rows whose ``run.lock`` is unheld (QUEUED only once older than 10s) into
    ERROR/``abandoned_worker``. The store is not a passive input here.

    Baselines: ``runs`` is newest-first, so walking it reversed and overwriting
    leaves the *newest* PASS duration per ``project:profile`` — the figure the
    elapsed label compares a live run against. Benchmarks instead take the
    *oldest* recorded p50 per ``project:workload`` (``seen_workloads.setdefault``
    is consulted before append), so the displayed delta is "versus first
    observation", not "versus the previous run".

    The 80/40 limits make History a window on recent work, not an archive: a
    project filter can only show rows that survived the limit.
    """
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
        # Hand-maintained, deliberately ordered by cost: this tuple — not
        # ProfileName.STANDARD — is what the ``r`` picker offers, so a profile
        # added to the schema is invisible in the TUI until it is added here
        # (mutation is currently absent from the picker).
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
    """Best-effort repro copy; returns the minibuffer message for the outcome.

    Order: tmux buffer (only when ``$TMUX`` is set and the binary resolves),
    then ``pbcopy``, then a truncated echo of the command so the operator can
    still retype it. Both subprocess calls are bounded at 2s against a wedged
    tmux server, and neither is wrapped here — ``subprocess.TimeoutExpired``
    propagates to ``_execute_action``, which does not catch it.
    """
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
    """Execute one post-confirmation side effect and return the resulting state.

    This is the only place in the TUI that mutates the engine or the store.
    ``handle_key`` produced ``action`` only after the explicit ``y``; by the time
    control lands here the intent is settled, so the branches stay short.

    Kinds: ``copy_repro``, ``cancel_run``, ``rerun``, ``run_profile``; anything
    else is a no-op (a defensive default, not a path a current keybinding takes).

    ``rerun`` replays the recorded ``profile`` verbatim, which is wrong for runs
    whose profile is a replay/verification label (``repro:unit``,
    ``verification``): no suite matches, so the engine finalises an empty SKIP
    run and the minibuffer still says "launched". ``run_profile`` resolves its
    cwd from the newest run of the filtered project, then the project registry,
    then the process cwd — in an empty store that last fallback is whatever
    directory the pane happens to be in.

    Success paths re-read the store so the next frame shows the new row without
    waiting for a kqueue/data_version hint. Each branch owns its message, which
    has one sharp edge: the ``try`` around ``cancel`` also covers the refresh, so
    a store read that fails *after* a successful cancel is reported as
    ``cancel failed``.
    """
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
    """Run the TUI until quit/SIGTERM/SIGINT and return a process exit code.

    ``home`` overrides the store location; ``None`` defers to
    ``default_testbench_home`` (``$TESTBENCH_HOME``, then ``$XDG_DATA_HOME``,
    then ``~/.local/share/testbench``). ``permanent_tab`` is the ``q`` guard:
    true means lowercase ``q`` only flashes a hint, false (``--quit-on-q``)
    means it quits. ``initial_project`` seeds the project filter only.

    Non-tty path: render one frame at the detected size, flush, close the store,
    return 0. A store that failed to open is still 0 — the failure is rendered
    into the banner, because a one-shot render is not a health check.

    TTY lifecycle, in order: remember termios, create the self-pipe that lets a
    signal handler wake ``select``, install SIGWINCH/SIGTERM/SIGINT, arm the
    kqueue watch on the store directory (absent on non-BSD platforms; then the
    data_version poll carries the entire refresh load), enter the alternate
    screen, hide the cursor, loop. Teardown is all ``finally``: restore termios,
    show the cursor, leave the alternate screen, restore handlers, close the
    pipe/fds and the store. A raised exception must not leave a raw-mode pane.

    Loop contract per iteration: re-read geometry (never cached across a signal),
    advance the 1 Hz tick while something is active, render, write only when the
    frame string or geometry changed, wait at most 0.25s (active) / 0.35s (idle)
    on stdin or the wake pipe, dispatch keys, then check kqueue and
    data_version. The bounded wait is what makes a missed wakeup cost
    sub-second staleness instead of a hang, and why no busy-poll is needed.

    Stop conditions: ``Q``, ``q`` when not a permanent tab, EOF on stdin, or a
    signal. No path in this loop waits on a child process.
    """
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

    # Sampled on the TUI's own connection: this counter moves only when *another*
    # connection commits, which is exactly what the TUI needs to notice.
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
            # Byte-exact diffing: identical state must not cost a terminal write,
            # and every differing byte (elapsed label, new log line, store row)
            # does. ``resized`` forces the write even when the render is unchanged.
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
                            # Sampled after the refresh, for the next iteration.
                            # Window: a commit landing between that read and this
                            # sample is folded into last_dv without being in
                            # ``state``, so that row stays invisible until the
                            # next commit.
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
                # has_active alone never re-reads: the 1 Hz tick above already
                # advances now_iso and re-reads live tails. A store row only moves
                # on another connection's commit (data_version) or a vnode event.
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
