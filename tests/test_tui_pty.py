"""End-to-end PTY test for the real TUI: startup, live refresh, resize, keys, quit.

Unlike ``test_tui_render`` (pure functions, in-process), everything here runs the
actual ``python -m hermes_testbench tui`` as a child of a real pseudo-terminal,
so the properties under test are the ones only the controller can break: the
alternate screen is entered, a store commit made by *another* process is noticed
without a keystroke, a resize forces a redraw, lowercase ``q`` is guarded, and
teardown restores the terminal.

How the output is read: ``PtyTuiHarness`` accumulates decoded bytes in a single
string and each assertion is a substring search over it. There is no VT
interpreter and no frame cleaning here — the buffer is the raw stream, so it
contains every escape sequence and every frame ever written. Two consequences
worth remembering when adding assertions:

* a match proves the text was emitted at *some* point since the buffer was last
  reset, not that it is on screen now. The test resets ``buffer`` at each phase
  boundary precisely to bound that window;
* control-sequence assertions (``\\033[?1049h``, ``\\033[?1049l``, ``\\033[?25h``)
  are checks on the controller's lifecycle writes in ``tui.run_tui``, not on the
  renderer.

Timing: waits poll with ``select`` every 50 ms against a 4 s default deadline;
the TUI's own loop waits at most 0.35 s idle / 0.25 s while a run is active, so a
frame always arrives well inside the deadline. ``close`` is bounded too — 2 s of
draining, then SIGTERM to the group, then ``wait(timeout=2)`` and a final drain
to EOF, which is what lets the exit sequence be observed at all.
"""

from __future__ import annotations

import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
import unittest
from pathlib import Path

from hermes_testbench.schema import RunRecord, RunStatus
from hermes_testbench.store import TestBenchStore

PY = sys.executable
REPO_ROOT = Path(__file__).resolve().parent.parent


class PtyTuiHarness:
    """One PTY-attached child plus the accumulated bytes it has written.

    The child is put in its own session (``start_new_session=True``), so its pid
    is also its process-group id and every signal here — ``SIGWINCH``,
    ``SIGTERM`` — reaches the whole group rather than just the leader. The parent
    closes its copy of the slave fd immediately: with both ends held open, EOF
    would never arrive and ``close``'s final drain would block instead of
    returning on EIO.
    """

    def __init__(self, argv: list[str], env: dict[str, str], cols: int = 80, rows: int = 24) -> None:
        self.master, slave = pty.openpty()
        self.set_size(cols, rows, slave)
        os.set_blocking(self.master, False)
        self.proc = subprocess.Popen(
            argv,
            cwd=str(REPO_ROOT),
            env=env,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave)
        self.buffer = ""

    @staticmethod
    def set_size(cols: int, rows: int, fd: int) -> None:
        """Set the pty pair's window size; ``ncols``/``nrows`` order is ioctl-specific.

        ``TIOCSWINSZ`` takes ``struct winsize`` (rows first, then cols, then two
        pixel fields left at zero). Setting it on either end updates the shared
        pair, which is why ``__init__`` can size the slave before ``exec`` — the
        child's first ``TIOCGWINSZ`` already sees the final geometry and there is
        no first-paint-at-80x24 race.
        """
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def resize(self, cols: int, rows: int) -> None:
        """Resize the pair, then raise ``SIGWINCH`` explicitly.

        macOS does not deliver ``SIGWINCH`` from ``TIOCSWINSZ`` for this case, so
        the signal is sent to the child's process group; ``ESRCH`` is swallowed
        because a child that already exited is reported by ``close``, not here.
        The assertion this enables is "the TUI redrew", not "the TUI redrew at
        exactly N columns": the buffer carries no geometry, so the wide layout
        itself is a renderer property checked in ``test_tui_render``.
        """
        self.set_size(cols, rows, self.master)
        try:
            os.killpg(self.proc.pid, signal.SIGWINCH)
        except OSError:
            pass

    def send(self, data: str) -> None:
        """Write keystrokes to the master; the child's tty is in raw mode by then."""
        os.write(self.master, data.encode("utf-8"))

    def wait_for_text(self, substring: str, timeout: float = 4.0) -> str:
        """Read until ``substring`` appears in the accumulated stream, or fail with a tail.

        The failure message keeps the last 600 characters: when this trips it is
        almost always because the TUI never painted the expected band, and the
        tail is the only surviving evidence of what it painted instead. ``OSError``
        on read ends the loop rather than retrying — it means the slave side is
        gone, so no further bytes can arrive and waiting out the deadline would
        only delay the same failure.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r, _, _ = select.select([self.master], [], [], 0.05)
            if r:
                try:
                    chunk = os.read(self.master, 16384)
                    if chunk:
                        self.buffer += chunk.decode("utf-8", errors="replace")
                except OSError:
                    break
            if substring in self.buffer:
                return self.buffer
        raise AssertionError(f"Timed out waiting for {substring!r} in PTY output; tail={self.buffer[-600:]!r}")

    def close(self) -> int:
        """Drain, escalate to ``SIGTERM`` if needed, then drain once more for the exit sequence.

        The first drain waits up to 2 s for a self-exit; only then is the group
        signalled, so a TUI that quits on ``Q`` is not killed mid-write. The
        second drain runs *after* ``wait`` on purpose: the alternate-screen exit
        and cursor-show bytes are written by ``run_tui``'s ``finally``, i.e. after
        the leader has begun exiting, and a single drain before ``wait`` would
        miss them — which would make the alt-screen-restore assertions below
        fail for the wrong reason. Reading stops on ``OSError``/EOF, both of which
        mean the master side has no writer left.
        """
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and self.proc.poll() is None:
            r, _, _ = select.select([self.master], [], [], 0.05)
            if r:
                try:
                    chunk = os.read(self.master, 16384)
                    if chunk:
                        self.buffer += chunk.decode("utf-8", errors="replace")
                except OSError:
                    break
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except OSError:
                pass
        rc = self.proc.wait(timeout=2.0)
        # Drain remaining bytes (e.g. alt-screen exit sequence)
        while True:
            try:
                chunk = os.read(self.master, 16384)
                if not chunk:
                    break
                self.buffer += chunk.decode("utf-8", errors="replace")
            except OSError:
                break
        os.close(self.master)
        return rc


class TuiPtyTests(unittest.TestCase):
    """One long scenario, because each phase establishes the precondition for the next."""

    def setUp(self) -> None:
        scratch = Path.home() / ".hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-tui-pty-")
        self.root = Path(self.tmp.name)
        self.tb_home = self.root / "tb_home"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _env(self) -> dict[str, str]:
        """Child environment: private store, source tree importable, colour forced on.

        ``TESTBENCH_HOME`` is what makes phase 2 meaningful — the store starts
        empty, so any row that appears must have come from the write the test
        performs in this process. ``NO_COLOR`` is removed because an inherited
        one would switch the palette to monochrome and change which escape
        sequences the buffer contains; ``TERM`` is pinned away from ``dumb`` for
        the same reason.
        """
        env = dict(os.environ)
        env["TESTBENCH_HOME"] = str(self.tb_home)
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["TERM"] = "xterm-256color"
        env.pop("NO_COLOR", None)
        return env

    def test_pty_empty_start_live_store_refresh_resize_permanent_q_guard_and_alt_screen_restore(self) -> None:
        """Six phases, each asserting what only a real controller can get wrong.

        1. **Startup.** The store is empty, so the first frame can only be the
           header/empty state; ``TESTBENCH`` proves a frame was painted and
           ``\\033[?1049h`` proves it was painted on the alternate screen. This
           does not by itself prove no run was auto-spawned — that would need a
           store assertion — so treat the "no auto-run" reading as inferred from
           the empty store fixture rather than asserted here.
        2. **Live refresh.** A failing record is written by *this* process after
           the TUI has already rendered. No key is sent, so the appearance of
           ``hxmail`` and ``FAIL`` can only come from the controller noticing
           another connection's commit (kqueue on ``$TESTBENCH_HOME`` or the
           ``PRAGMA data_version`` change). The test proves the refresh happened
           within the 4 s deadline, not which of the two mechanisms fired.
        3. **Resize.** The buffer is cleared first so the match cannot be a
           pre-resize frame, then the pty is resized and ``SIGWINCH`` sent. The
           run id appearing afterwards proves a frame was written at the new
           geometry (``run_tui`` forces a write when ``resized`` is set even if
           the rendered bytes are identical). It does *not* measure columns — the
           wide-split layout is a renderer property covered in
           ``test_tui_render``.
        4. **Sort.** ``2`` switches to History and ``s`` cycles to "failures
           first". Both strings are view- and order-specific: the order only
           appears in the header while ``view == "history"``, and ``s sort`` only
           in the History hint variant. A second ``s`` reaching "longest" proves
           the cycle advances rather than re-rendering the same order. Row
           ordering itself is asserted in the renderer tests, not here.
        5. **Quit guard.** Lowercase ``q`` on a permanent tab must flash
           ``press Q`` *and* leave the process alive; the liveness check is the
           negative control against a TUI that prints the hint and exits anyway.
        6. **Clean exit.** ``Q`` ends the loop, and the teardown writes restore
           the screen (``\\033[?1049l``) and the cursor (``\\033[?25h``). Those
           two sequences come from ``run_tui``'s ``finally``, not from rendering,
           which is why they can only be observed through ``close``'s
           post-``wait`` drain.

        The ``except`` re-raises after closing the harness: a failing assertion
        mid-scenario must not leave a raw-mode child holding the pty.
        """

        # 1. Start TUI against an empty store at 80x24: must NOT auto-run tests, must render header
        pty_h = PtyTuiHarness([PY, "-m", "hermes_testbench", "tui"], self._env(), cols=80, rows=24)
        try:
            pty_h.wait_for_text("TESTBENCH")
            self.assertIn("\033[?1049h", pty_h.buffer)

            # 2. External process writes a failing run to state.db -> TUI refreshes automatically
            with TestBenchStore(self.tb_home) as store:
                rec = RunRecord(
                    run_id="tb_20261114_174000_pty00001",
                    project="hxmail",
                    project_root="/Users/sein/Code/hermes-terminal-env",
                    revision="8234029",
                    profile="unit",
                    status=RunStatus.FAIL,
                    created_at="2026-11-14T22:40:00Z",
                    finished_at="2026-11-14T22:40:02Z",
                    duration_ms=1230.0,
                    total_count=75,
                    pass_count=74,
                    fail_count=1,
                    first_failure="hostile_escape: unexpected token",
                    repro_instructions="cargo test --offline",
                )
                store.create_run(rec)
                store.finalize_run(rec)

            pty_h.wait_for_text("hxmail")
            pty_h.wait_for_text("FAIL")

            # 3. Resize 80x24 -> 136x36 triggers immediate SIGWINCH redraw with wide inspector
            pty_h.buffer = ""
            pty_h.resize(136, 36)
            pty_h.wait_for_text("tb_20261114_174000_pty00001")

            # Sort stays interactive after a resize and is shown in the title.
            pty_h.buffer = ""
            pty_h.send("2s")
            pty_h.wait_for_text("failures first")
            pty_h.wait_for_text("s sort")
            pty_h.send("s")
            pty_h.wait_for_text("longest")

            # 4. Lowercase 'q' in permanent tab mode does NOT exit; shows hint
            pty_h.buffer = ""
            pty_h.send("q")
            pty_h.wait_for_text("press Q")
            self.assertIsNone(pty_h.proc.poll())

            # 5. Uppercase 'Q' exits cleanly and restores normal screen (\033[?1049l) and cursor (\033[?25h)
            pty_h.send("Q")
            rc = pty_h.close()
            self.assertEqual(rc, 0)
            self.assertIn("\033[?1049l", pty_h.buffer)
            self.assertIn("\033[?25h", pty_h.buffer)
        except Exception:
            pty_h.close()
            raise


if __name__ == "__main__":
    unittest.main()
