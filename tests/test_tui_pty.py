"""Real PTY integration tests for TestBench TUI: rendering, resize, live refresh, and safe quit."""

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
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def resize(self, cols: int, rows: int) -> None:
        self.set_size(cols, rows, self.master)
        try:
            os.killpg(self.proc.pid, signal.SIGWINCH)
        except OSError:
            pass

    def send(self, data: str) -> None:
        os.write(self.master, data.encode("utf-8"))

    def wait_for_text(self, substring: str, timeout: float = 4.0) -> str:
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
    def setUp(self) -> None:
        scratch = Path.home() / ".hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-tui-pty-")
        self.root = Path(self.tmp.name)
        self.tb_home = self.root / "tb_home"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["TESTBENCH_HOME"] = str(self.tb_home)
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["TERM"] = "xterm-256color"
        env.pop("NO_COLOR", None)
        return env

    def test_pty_empty_start_live_store_refresh_resize_permanent_q_guard_and_alt_screen_restore(self) -> None:
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
