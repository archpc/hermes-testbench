"""Terminal tape: real-PTY scripted replay, deterministic screens, golden gate.

Everything here runs offline: one isolated ``pty.openpty()`` program, no browser,
no VHS, no ambient tmux, and no writes outside the scratch directory.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from hermes_testbench.adapters import TAPE_ADAPTER, execute_suite
from hermes_testbench.adapters.parsers import ADAPTER_PARSERS, parse_terminal_tape
from hermes_testbench.discovery import KNOWN_ADAPTERS, discover_project
from hermes_testbench.isolation import can_apply_seatbelt, default_scratch_root
from hermes_testbench.pty_session import PtySession, VTScreen, char_cells, key_sequence
from hermes_testbench.schema import FailureKind, ProfileName, RunStatus, SuiteSpec
from hermes_testbench.store import TestBenchStore
from hermes_testbench.tape import (
    VHS_DEFERRAL_MESSAGE,
    GoldenApprovalRefused,
    TapeParseError,
    TapePolicyError,
    approve_golden,
    check_golden,
    golden_paths,
    load_tape,
    normalized_screen,
    replay_tape,
    run_tape,
    screen_sha256,
)
from hermes_testbench.tui_render import display_width

PY = sys.executable
REPO_ROOT = Path(__file__).resolve().parent.parent


FIXTURE_SRC = r'''
"""Deterministic interactive full-screen fixture used by terminal-tape tests."""
import fcntl
import os
import select
import signal
import struct
import sys
import termios
import tty


def _size(fd):
    rows, cols, _, _ = struct.unpack("HHHH", fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8))
    return cols, rows


def main():
    fd = 0
    attrs = termios.tcgetattr(fd)
    tty.setraw(fd)
    cols, rows = _size(1)
    state = {"buffer": "", "value": "", "cols": cols, "rows": rows}
    resized = {"flag": False}
    signal.signal(signal.SIGWINCH, lambda *a: resized.__setitem__("flag", True))

    def render():
        note = "-"
        if os.path.exists("note.txt"):
            with open("note.txt", encoding="utf-8") as handle:
                note = handle.read().strip() or "-"
        lines = [
            "FIXTURE %dx%d" % (state["cols"], state["rows"]),
            "READY",
            "buffer: " + state["buffer"],
            "value: " + (state["value"] or "-"),
            "note: " + note,
            "keys: Enter submits, q quits",
        ]
        chunks = []
        for index, line in enumerate(lines, start=1):
            chunks.append("\x1b[%d;1H%s\x1b[K" % (index, line))
        chunks.append("\x1b[%d;1H\x1b[J" % (len(lines) + 1))
        os.write(1, "".join(chunks).encode())

    render()
    while True:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if resized["flag"]:
            resized["flag"] = False
            state["cols"], state["rows"] = _size(1)
            render()
        if not ready:
            continue
        data = os.read(fd, 64)
        if not data:
            break
        for ch in data.decode("utf-8", "replace"):
            if ch == "q":
                termios.tcsetattr(fd, termios.TCSANOW, attrs)
                return 0
            if ch in "\r\n":
                state["value"] = state["buffer"]
                state["buffer"] = ""
                render()
            elif ch in "\x7f\x08":
                state["buffer"] = state["buffer"][:-1]
                render()
            elif ch.isprintable():
                state["buffer"] += ch
                render()
    return 0


sys.exit(main())
'''


TAPE_BODY = """# deterministic fixture drive
Size 60 12
Command {python} fixture.py
WaitFor "READY" 4000
Type "42"
Key Enter
WaitFor "value: 42" 4000
Resize 72 14
WaitFor "FIXTURE 72x14" 4000
WaitQuiet 150
ExpectScreen contains "value: 42"
ExpectScreen excludes "value: 99"
ExpectRaw contains "READY"
"""


class _FakeTTY:
    """Stand-in stdin that claims to be an interactive terminal."""

    def isatty(self) -> bool:  # noqa: D401 - test double
        return True

    def read(self, size: int = -1) -> str:
        return ""

    def fileno(self) -> int:
        return 0

    def readline(self, size: int = -1) -> str:
        return ""


class _NoTTY:
    def isatty(self) -> bool:
        return False

    def read(self, size: int = -1) -> str:
        return ""

    def fileno(self) -> int:
        return 0


@contextlib.contextmanager
def stdin_replaced(replacement):  # type: ignore[no-untyped-def]
    original = sys.stdin
    sys.stdin = replacement
    try:
        yield
    finally:
        sys.stdin = original


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dir_snapshot(path: Path) -> dict:
    if not path.is_dir():
        return {}
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.iterdir())
        if p.is_file()
    }


class TapeTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-tape-test-")
        self.root = Path(self._tmp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "fixture.py").write_text(FIXTURE_SRC, encoding="utf-8")
        self.golden_dir = self.project / "golden"
        self.tb_home = self.root / "tbhome"
        # These tests exercise golden approval, which is a deliberate *host* action. Running
        # the suite from inside TestBench sets TESTBENCH_SANDBOX=1 and would make every
        # approval refusal test assert the sandbox guard instead of the guard under test.
        self._sandbox_env = os.environ.pop("TESTBENCH_SANDBOX", None)

    def tearDown(self) -> None:
        saved = getattr(self, "_sandbox_env", None)
        if saved is not None:
            os.environ["TESTBENCH_SANDBOX"] = saved
        self._tmp.cleanup()

    # -- helpers -------------------------------------------------------------
    def write_tape(self, body: str, name: str = "smoke") -> Path:
        path = self.project / f"{name}.tape"
        path.write_text(body, encoding="utf-8")
        return path

    def replay(self, tape_path: Path, *, evidence: str = "evidence", **kwargs):  # type: ignore[no-untyped-def]
        return replay_tape(
            load_tape(tape_path),
            project_root=self.project,
            evidence_dir=self.root / evidence,
            golden_dir=self.golden_dir,
            **kwargs,
        )

    def approve(self, evidence_dir: Path, name: str, *, assume_yes: bool = True) -> dict:
        with stdin_replaced(_FakeTTY()):
            return approve_golden(
                evidence_dir,
                golden_name=name,
                golden_dir=self.golden_dir,
                assume_yes=assume_yes,
            )


# --------------------------------------------------------------------- VTScreen


class VTScreenTests(unittest.TestCase):
    def test_carriage_return_linefeed_backspace_and_tab(self) -> None:
        screen = VTScreen(20, 4)
        screen.feed("hello\r\nworld\tX\bY")
        self.assertEqual(screen.to_lines()[0], "hello")
        self.assertEqual(screen.to_lines()[1], "world   Y")

    def test_cursor_position_and_relative_moves(self) -> None:
        screen = VTScreen(20, 6)
        screen.feed("\x1b[3;5Habc")
        self.assertEqual(screen.cursor.row, 2)
        self.assertEqual(screen.cursor.col, 7)
        self.assertEqual(screen.to_lines()[2], "    abc")
        screen.feed("\x1b[2A\x1b[2DX")
        self.assertEqual((screen.cursor.row, screen.cursor.col), (0, 6))
        self.assertEqual(screen.to_lines()[0], "     X")
        screen.feed("\x1b[10GZ")
        self.assertEqual(screen.to_lines()[0], "     X   Z")
        screen.feed("\x1b[6;1HQ")
        self.assertEqual(screen.to_lines()[5], "Q")
        screen.feed("\x1b[fR")
        self.assertEqual(screen.to_lines()[0], "R    X   Z")

    def test_erase_line_and_display(self) -> None:
        screen = VTScreen(10, 3)
        screen.feed("abcdefghij\r\nklm")
        screen.feed("\x1b[1;4H\x1b[K")
        self.assertEqual(screen.to_lines()[0], "abc")
        screen.feed("\x1b[2;1H\x1b[1K")
        self.assertEqual(screen.to_lines()[1], " lm")
        screen.feed("\x1b[2J")
        self.assertEqual(screen.to_lines(), ["", "", ""])

    def test_alt_screen_saves_and_restores(self) -> None:
        screen = VTScreen(12, 3)
        screen.feed("main")
        screen.feed("\x1b[?1049h")
        self.assertEqual(screen.to_lines(), ["", "", ""])
        screen.feed("\x1b[Halt")
        self.assertEqual(screen.to_lines()[0], "alt")
        screen.feed("\x1b[?1049l")
        self.assertEqual(screen.to_lines()[0], "main")

    def test_cursor_visibility_modes(self) -> None:
        screen = VTScreen(10, 2)
        screen.feed("\x1b[?25l")
        self.assertFalse(screen.cursor.visible)
        screen.feed("\x1b[?25h")
        self.assertTrue(screen.cursor.visible)

    def test_sgr_osc_and_dcs_never_reach_text_cells(self) -> None:
        screen = VTScreen(40, 3)
        screen.feed("\x1b[1;31;40mred\x1b[0m")
        screen.feed("\x1b]0;window title\x07")
        screen.feed("\x1b]8;;https://example.invalid\x1b\\link\x1b]8;;\x1b\\")
        screen.feed("\x1bPq#0;2;0;0;0\x1b\\")
        screen.feed("\x1b(Bplain")
        text = screen.to_text()
        self.assertIn("red", text)
        self.assertIn("link", text)
        self.assertIn("plain", text)
        self.assertNotIn("\x1b", text)
        self.assertNotIn("[1;31", text)
        self.assertNotIn("window title", text)
        self.assertNotIn("#0;2;0;0;0", text)

    def test_wide_characters_match_display_width(self) -> None:
        samples = ["日本語", "mixed 日本 x", "ｆｕｌｌ", "e\u0301clair", "a\tb"]
        for sample in samples:
            screen = VTScreen(40, 2)
            screen.feed(sample.replace("\t", " " * 8))
            flat = sample.replace("\t", " " * 8)
            expected = sum(char_cells(ch) for ch in flat)
            self.assertEqual(display_width(flat), expected)
            self.assertEqual(display_width(screen.to_lines()[0]), expected)
        screen = VTScreen(10, 2)
        screen.feed("日本語x")
        self.assertEqual(screen.to_lines()[0], "日本語x")
        self.assertEqual(display_width(screen.to_lines()[0]), 7)

    def test_chunked_feed_matches_wholesale_feed(self) -> None:
        payload = (
            "\x1b[?1049h\x1b[H\x1b[2;2Hこんにちは\x1b[K\r\nline2\r\n"
            "\x1b[31mred\x1b[0m\x1b]8;;http://x\x07link\x1b]8;;\x07\x1b[?1049l"
        )
        whole = VTScreen(30, 6)
        whole.feed(payload)
        chunked = VTScreen(30, 6)
        for ch in payload:
            chunked.feed(ch)
        self.assertEqual(whole.to_text(), chunked.to_text())
        self.assertNotIn("\x1b", whole.to_text())

    def test_resize_preserves_content_and_geometry(self) -> None:
        screen = VTScreen(10, 3)
        screen.feed("\x1b[2;3Habc")
        screen.resize(20, 6)
        self.assertEqual(screen.cols, 20)
        self.assertEqual(screen.rows, 6)
        self.assertEqual(screen.to_lines()[1], "  abc")
        screen.feed("\x1b[6;20HZ")
        self.assertEqual(screen.to_lines()[5], "                   Z")
        screen.resize(5, 2)
        self.assertEqual(screen.cols, 5)
        self.assertEqual(screen.rows, 2)
        self.assertEqual(screen.cursor.row, 1)
        self.assertEqual(len(screen.to_lines()[1]), 5)

    def test_to_text_is_normalized(self) -> None:
        screen = VTScreen(12, 3)
        screen.feed("abc   \r\ndef")
        self.assertEqual(screen.to_text(), "abc\ndef")
        self.assertEqual(screen.to_lines(), ["abc", "def", ""])
        self.assertEqual(normalized_screen(screen.to_text()), "abc\ndef\n")

    def test_full_screen_redraw_is_stable(self) -> None:
        frame = "\x1b[?25l\x1b[H\x1b[2Jrow one\x1b[K\r\nrow two\x1b[K\x1b[J"
        screens = []
        for _ in range(3):
            screen = VTScreen(40, 6)
            screen.feed(frame)
            screens.append(screen.to_text())
        self.assertEqual(len(set(screens)), 1)


# -------------------------------------------------------------------- PtySession


class PtySessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-pty-sess-")
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_key_sequences(self) -> None:
        self.assertEqual(key_sequence("Enter"), "\r")
        self.assertEqual(key_sequence("enter"), "\r")
        self.assertEqual(key_sequence("Up"), "\x1b[A")
        self.assertEqual(key_sequence("q"), "q")
        with self.assertRaises(ValueError):
            key_sequence("Nope")

    def test_session_runs_program_and_tears_down(self) -> None:
        session = PtySession(
            [PY, "-c", "import sys,time; print('hello pty', flush=True); time.sleep(30)"],
            cwd=self.root,
        )
        try:
            session.wait_for("hello pty", 5000)
            self.assertTrue(session.is_running())
        finally:
            rc = session.close()
        self.assertIsNotNone(rc)
        self.assertFalse(session.is_running())
        self.assertFalse(session.sandbox_dir.exists())

    def test_raw_buffer_is_bounded(self) -> None:
        session = PtySession(
            [PY, "-c", "import sys; sys.stdout.write('x' * 200000); sys.stdout.flush()"],
            cwd=self.root,
            max_raw_bytes=4096,
        )
        try:
            session.wait_quiet(300, timeout_ms=8000)
        finally:
            session.close()
        self.assertLessEqual(len(session.raw_bytes), 4096)
        self.assertTrue(session.truncated)

    def test_env_is_isolated_from_ambient_home_and_tmux(self) -> None:
        snippet = (
            "import json,os;"
            "print(json.dumps({'home': os.environ.get('HOME'),"
            " 'tmp': os.environ.get('TMPDIR'),"
            " 'tmux': os.environ.get('TMUX'),"
            " 'sandbox': os.environ.get('TESTBENCH_SANDBOX')}))"
        )
        session = PtySession([PY, "-c", snippet], cwd=self.root)
        try:
            session.wait_quiet(300, timeout_ms=8000)
            payload = json.loads([ln for ln in session.text.splitlines() if ln.strip()][-1])
        finally:
            session.close()
        self.assertEqual(payload["tmux"], None)
        self.assertEqual(payload["sandbox"], "1")
        self.assertNotEqual(Path(payload["home"]), Path.home())
        self.assertNotEqual(Path(payload["tmp"]), Path(str(self.root)))

    def test_resize_updates_child_window_size(self) -> None:
        session = PtySession(
            [PY, "-c", "import time,sys; print('sized', flush=True); time.sleep(30)"],
            cwd=self.root,
            cols=60,
            rows=12,
        )
        try:
            session.wait_for("sized", 5000)
            session.resize(100, 30)
            self.assertEqual((session.cols, session.rows), (100, 30))
            self.assertEqual((session.screen.cols, session.screen.rows), (100, 30))
        finally:
            session.close()


# ------------------------------------------------------------------- tape parsing


class TapeParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-tape-parse-")
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_dsl_parses_every_action(self) -> None:
        body = (
            "# comment\n"
            "Size 90 30\n"
            'Command {python} -m demo --flag "a b"\n'
            'Type "hi\\n"\n'
            "Key Enter\n"
            "Resize 100 32\n"
            'WaitFor "ready" 1234\n'
            "WaitQuiet 99\n"
            'ExpectScreen contains "screen"\n'
            'ExpectRaw excludes "\\u001b[?1049h"\n'
            'Golden "gold"\n'
        )
        path = self.root / "full.tape"
        path.write_text(body, encoding="utf-8")
        script = load_tape(path)
        self.assertEqual((script.cols, script.rows), (90, 30))
        self.assertEqual(script.command[0], PY)
        self.assertEqual(script.command[-1], "a b")
        actions = [s.action for s in script.steps]
        self.assertEqual(
            actions,
            ["type", "key", "resize", "waitfor", "waitquiet", "expectscreen", "expectraw", "golden"],
        )
        self.assertEqual(script.steps[0].text, "hi\n")
        self.assertEqual(script.steps[4].quiet_ms, 99)
        self.assertEqual(script.steps[6].text, "\x1b[?1049h")
        self.assertEqual(script.steps[7].name, "gold")
        self.assertEqual(script.sha256, hashlib.sha256(body.encode()).hexdigest())

    def test_vhs_backend_requests_are_refused(self) -> None:
        for body in (
            "Size 80 24\nBackend vhs\nCommand {python} x.py\n",
            'Size 80 24\nVhs "demo.gif"\nCommand {python} x.py\n',
            "Size 80 24\nScreenshot out.png\nCommand {python} x.py\n",
        ):
            path = self.root / "vhs.tape"
            path.write_text(body, encoding="utf-8")
            with self.assertRaises(TapePolicyError) as ctx:
                load_tape(path)
            self.assertEqual(getattr(ctx.exception, "failure_kind", ""), FailureKind.BLOCKED_POLICY)
            self.assertIn(VHS_DEFERRAL_MESSAGE, str(ctx.exception))

    def test_json_spec_parses_and_refuses_browser_backend(self) -> None:
        spec = {
            "name": "json-tape",
            "cols": 70,
            "rows": 20,
            "command": ["{python}", "fixture.py"],
            "steps": [
                {"action": "waitfor", "text": "READY", "timeout_ms": 2000},
                {"action": "expectscreen", "mode": "contains", "text": "READY"},
                {"action": "golden", "name": "json-gold"},
            ],
        }
        path = self.root / "spec.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        script = load_tape(path)
        self.assertEqual(script.source_format, "json")
        self.assertEqual(script.command[0], PY)
        self.assertEqual([s.action for s in script.steps], ["waitfor", "expectscreen", "golden"])
        spec["backend"] = "chrome"
        path.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaises(TapePolicyError):
            load_tape(path)

    def test_parse_errors_are_explicit(self) -> None:
        cases = [
            ("Size 80 24\nFrobnicate 1\n", "unknown tape action"),
            ("Size 80 24\nType x\n", "no Command"),
            ('Size 80 24\nCommand {python} x\nGolden "a"\nGolden "b"\n', "only one Golden"),
            ("Size 0 24\nCommand {python} x\n", "Size must be positive"),
            ("Size 80 24\nCommand {python} x\nExpectScreen maybe \"y\"\n", "contains or excludes"),
            ("Size 80 24\nCommand {python} x\nResize 10\n", "Resize requires"),
        ]
        for body, needle in cases:
            path = self.root / "bad.tape"
            path.write_text(body, encoding="utf-8")
            with self.assertRaises(TapeParseError) as ctx:
                load_tape(path)
            self.assertIn(needle, str(ctx.exception))


# ------------------------------------------------------------------ replay flows


class TapeReplayTests(TapeTestBase):
    def test_scripted_input_passes_and_is_deterministic(self) -> None:
        tape = self.write_tape(TAPE_BODY)
        first = self.replay(tape, evidence="ev1")
        second = self.replay(tape, evidence="ev2")
        self.assertEqual(first.status, RunStatus.PASS, first.message)
        self.assertEqual(first.failure_kind, None)
        self.assertEqual(first.screen_text, second.screen_text)
        self.assertEqual(screen_sha256(first.screen_text), screen_sha256(second.screen_text))
        self.assertIn("FIXTURE 72x14", first.screen_text)
        self.assertIn("value: 42", first.screen_text)
        self.assertIn("READY", first.raw_text)
        self.assertTrue(all(check["ok"] for check in first.checks))
        self.assertEqual(first.passed, 3)
        self.assertTrue(Path(first.screen_path).is_file())
        meta = json.loads((Path(first.evidence_dir) / "tape.meta.json").read_text())
        self.assertEqual(meta["status"], RunStatus.PASS)
        self.assertEqual(meta["checks"][0]["kind"], "expectscreen")

    def test_failing_expectation_reports_diff_and_screen(self) -> None:
        body = TAPE_BODY + 'ExpectScreen contains "value: 99"\n'
        tape = self.write_tape(body, name="failing")
        outcome = self.replay(tape)
        self.assertEqual(outcome.status, RunStatus.FAIL)
        self.assertEqual(outcome.failure_kind, FailureKind.ASSERTION)
        self.assertIn("value: 99", outcome.message)
        evidence = Path(outcome.evidence_dir)
        actual = evidence / "failing.actual.screen.txt"
        self.assertTrue(actual.is_file())
        self.assertIn("value: 42", actual.read_text(encoding="utf-8"))
        diff = evidence / "failing.expect.diff"
        self.assertTrue(diff.is_file())
        diff_text = diff.read_text(encoding="utf-8")
        self.assertIn("expectation failed", diff_text)
        self.assertIn("--- expected", diff_text)
        self.assertIn("+++ actual", diff_text)
        self.assertIn("value: 99", diff_text)
        self.assertEqual(actual.stat().st_mode & 0o777, 0o600)

    def test_waitfor_timeout_is_reported_as_timeout(self) -> None:
        body = 'Size 60 12\nCommand {python} fixture.py\nWaitFor "never-appears" 250\n'
        tape = self.write_tape(body, name="timeout")
        outcome = self.replay(tape)
        self.assertEqual(outcome.status, RunStatus.TIMEOUT)
        self.assertEqual(outcome.failure_kind, FailureKind.TIMEOUT)
        self.assertIn("never-appears", outcome.message)

    def test_resize_is_observable_in_the_program(self) -> None:
        body = (
            "Size 60 12\n"
            "Command {python} fixture.py\n"
            'WaitFor "READY" 4000\n'
            'WaitFor "FIXTURE 60x12" 4000\n'
            "Resize 90 20\n"
            'WaitFor "FIXTURE 90x20" 4000\n'
            'ExpectScreen contains "FIXTURE 90x20"\n'
        )
        tape = self.write_tape(body, name="resize")
        outcome = self.replay(tape)
        self.assertEqual(outcome.status, RunStatus.PASS, outcome.message)
        self.assertEqual((outcome.cols, outcome.rows), (90, 20))
        self.assertEqual(outcome.passed, 1)

    @unittest.skipUnless(can_apply_seatbelt(), "Seatbelt (sandbox-exec) unavailable on this host")
    def test_seatbelt_still_denies_writes_outside_the_sandbox(self) -> None:
        target = self.project / "escape.txt"
        probe = self.project / "probe.py"
        probe.write_text(
            "import pathlib, time\n"
            "try:\n"
            "    pathlib.Path(r'%s').write_text('escaped')\n"
            "    print('WROTE', flush=True)\n"
            "except OSError as exc:\n"
            "    print('DENIED', flush=True)\n"
            "time.sleep(20)\n" % target,
            encoding="utf-8",
        )
        tape = self.write_tape('Size 60 12\nCommand {python} probe.py\nWaitQuiet 400\n', name="seatbelt")
        outcome = self.replay(tape)
        self.assertEqual(outcome.seatbelt, "active")
        self.assertIn("DENIED", outcome.screen_text)
        self.assertNotIn("WROTE", outcome.screen_text)
        self.assertFalse(target.exists())

    def test_real_testbench_tui_is_driven_over_a_pty(self) -> None:
        body = (
            "Size 80 24\n"
            "Command {python} -m hermes_testbench tui\n"
            'WaitFor "TESTBENCH" 15000\n'
            "WaitQuiet 200\n"
            'ExpectScreen contains "TESTBENCH"\n'
            'ExpectRaw contains "\\u001b[?1049h"\n'
            'Type "q"\n'
            'WaitFor "press Q" 5000\n'
            "Resize 100 30\n"
            'WaitFor "TESTBENCH" 5000\n'
            "WaitQuiet 150\n"
            'ExpectScreen contains "overview"\n'
            'ExpectScreen contains "TESTBENCH"\n'
            "Key Q\n"
        )
        tape = self.write_tape(body, name="tui")
        # A source-tree checkout is not importable from the temporary project cwd; the
        # installed console entry point would be. Export the repo root explicitly so this
        # test does not silently depend on an ambient PYTHONPATH.
        with mock.patch.dict(os.environ, {"PYTHONPATH": str(REPO_ROOT)}):
            outcome = self.replay(tape, timeout_s=45.0)
        self.assertEqual(outcome.status, RunStatus.PASS, outcome.message)
        self.assertGreaterEqual(outcome.passed, 3)


# ------------------------------------------------------------------- golden gate


class GoldenGateTests(TapeTestBase):
    def _tape_with_golden(self, name: str = "smoke") -> Path:
        return self.write_tape(TAPE_BODY + f'Golden "{name}"\n', name="golden-tape")

    def test_missing_golden_blocks_and_never_writes(self) -> None:
        tape = self._tape_with_golden("smoke")
        outcome = self.replay(tape)
        self.assertEqual(outcome.status, RunStatus.BLOCKED)
        self.assertEqual(outcome.failure_kind, FailureKind.BLOCKED_POLICY)
        self.assertIn("golden missing", outcome.message)
        self.assertFalse(self.golden_dir.exists())

    def test_unapproved_golden_manifest_blocks(self) -> None:
        tape = self._tape_with_golden("smoke")
        first = self.replay(tape)
        self.approve(Path(first.evidence_dir), "smoke")
        screen_path, manifest_path = golden_paths(self.golden_dir, "smoke")
        manifest = json.loads(manifest_path.read_text())
        manifest["approved_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        before = _dir_snapshot(self.golden_dir)
        blocked = self.replay(tape, evidence="ev-unapproved")
        self.assertEqual(blocked.status, RunStatus.BLOCKED)
        self.assertEqual(blocked.failure_kind, FailureKind.BLOCKED_POLICY)
        self.assertIn("not approved", blocked.message)
        self.assertEqual(_dir_snapshot(self.golden_dir), before)
        self.assertEqual(screen_sha256(screen_path.read_text(encoding="utf-8")), manifest["screen_sha256"])

    def test_tampered_golden_file_blocks(self) -> None:
        tape = self._tape_with_golden("smoke")
        first = self.replay(tape)
        self.approve(Path(first.evidence_dir), "smoke")
        screen_path, _manifest = golden_paths(self.golden_dir, "smoke")
        screen_path.write_text(screen_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
        outcome = self.replay(tape, evidence="ev-tampered")
        self.assertEqual(outcome.status, RunStatus.BLOCKED)
        self.assertIn("changed after approval", outcome.message)

    def test_stale_tape_blocks_without_rewriting(self) -> None:
        tape = self._tape_with_golden("smoke")
        first = self.replay(tape)
        self.approve(Path(first.evidence_dir), "smoke")
        before = _dir_snapshot(self.golden_dir)
        changed = self.write_tape("Size 61 12\n" + TAPE_BODY.split("Size 60 12\n", 1)[1] + 'Golden "smoke"\n')
        outcome = self.replay(changed, evidence="ev-stale")
        self.assertEqual(outcome.status, RunStatus.BLOCKED)
        self.assertEqual(outcome.failure_kind, FailureKind.BLOCKED_POLICY)
        self.assertIn("tape changed since approval", outcome.message)
        self.assertEqual(_dir_snapshot(self.golden_dir), before)

    def test_geometry_change_blocks_as_stale(self) -> None:
        tape = self._tape_with_golden("smoke")
        first = self.replay(tape)
        self.approve(Path(first.evidence_dir), "smoke")
        screen_path, manifest_path = golden_paths(self.golden_dir, "smoke")
        manifest = json.loads(manifest_path.read_text())
        manifest["cols"] = 5
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        outcome = self.replay(tape, evidence="ev-geom")
        self.assertEqual(outcome.status, RunStatus.BLOCKED)
        self.assertIn("geometry changed", outcome.message)
        self.assertTrue(screen_path.is_file())

    def test_mismatching_golden_fails_with_diff_and_keeps_repo_file(self) -> None:
        tape = self._tape_with_golden("smoke")
        first = self.replay(tape)
        self.approve(Path(first.evidence_dir), "smoke")
        screen_path, manifest_path = golden_paths(self.golden_dir, "smoke")
        before = _dir_snapshot(self.golden_dir)
        # Same tape bytes, different program output: the golden must now difference.
        (self.project / "note.txt").write_text("changed-world\n", encoding="utf-8")
        outcome = self.replay(tape, evidence="ev-diff")
        self.assertEqual(outcome.status, RunStatus.FAIL)
        self.assertEqual(outcome.failure_kind, FailureKind.ASSERTION)
        self.assertIn("differs from approved golden", outcome.message)
        self.assertEqual(_dir_snapshot(self.golden_dir), before)
        self.assertNotIn("changed-world", screen_path.read_text(encoding="utf-8"))
        diff_path = Path(outcome.evidence_dir) / "golden-tape.golden.diff"
        self.assertTrue(diff_path.is_file())
        diff_text = diff_path.read_text(encoding="utf-8")
        self.assertIn("+note: changed-world", diff_text)
        self.assertIn("-note: -", diff_text)
        self.assertTrue((Path(outcome.evidence_dir) / "golden-tape.actual.screen.txt").is_file())

    def test_approved_golden_passes_on_replay(self) -> None:
        tape = self._tape_with_golden("smoke")
        first = self.replay(tape)
        self.assertEqual(first.status, RunStatus.BLOCKED)
        self.approve(Path(first.evidence_dir), "smoke")
        screen_path, manifest_path = golden_paths(self.golden_dir, "smoke")
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["screen_sha256"], manifest["approved_sha256"])
        self.assertEqual(manifest["cols"], 72)
        self.assertEqual(manifest["rows"], 14)
        self.assertEqual(screen_path.stat().st_mode & 0o777, 0o644)
        second = self.replay(tape, evidence="ev-approved")
        self.assertEqual(second.status, RunStatus.PASS, second.message)
        self.assertEqual(second.golden["status"], "pass")
        self.assertIn("golden", [c["kind"] for c in second.checks])

    def test_check_golden_is_pure(self) -> None:
        self.golden_dir.mkdir()
        result = check_golden("abc\n", golden_dir=self.golden_dir, name="nope", tape_sha256="t", cols=10, rows=2)
        self.assertEqual(result.status, "missing")
        self.assertEqual(list(self.golden_dir.iterdir()), [])


# --------------------------------------------------------------- approval gate


class GoldenApprovalTests(TapeTestBase):
    def _evidence_with_capture(self) -> Path:
        tape = self.write_tape(TAPE_BODY + 'Golden "smoke"\n', name="golden-tape")
        outcome = self.replay(tape)
        self.assertEqual(outcome.status, RunStatus.BLOCKED)
        return Path(outcome.evidence_dir)

    def test_refused_without_a_tty(self) -> None:
        evidence = self._evidence_with_capture()
        with stdin_replaced(_NoTTY()):
            with self.assertRaises(GoldenApprovalRefused) as ctx:
                approve_golden(evidence, golden_name="smoke", golden_dir=self.golden_dir, assume_yes=True)
        self.assertIn("interactive TTY", str(ctx.exception))
        self.assertFalse(self.golden_dir.exists())

    def test_refused_inside_a_sandbox_even_with_a_tty(self) -> None:
        evidence = self._evidence_with_capture()
        with mock.patch.dict(os.environ, {"TESTBENCH_SANDBOX": "1"}):
            with stdin_replaced(_FakeTTY()):
                with self.assertRaises(GoldenApprovalRefused) as ctx:
                    approve_golden(evidence, golden_name="smoke", golden_dir=self.golden_dir, assume_yes=True)
        self.assertIn("sandbox", str(ctx.exception))
        self.assertFalse(self.golden_dir.exists())

    def test_declined_prompt_writes_nothing(self) -> None:
        evidence = self._evidence_with_capture()
        with stdin_replaced(_FakeTTY()):
            with mock.patch("builtins.input", return_value="n"):
                with self.assertRaises(GoldenApprovalRefused):
                    approve_golden(evidence, golden_name="smoke", golden_dir=self.golden_dir)
        self.assertFalse(self.golden_dir.exists())

    def test_confirmed_prompt_approves(self) -> None:
        evidence = self._evidence_with_capture()
        with stdin_replaced(_FakeTTY()):
            with mock.patch("builtins.input", return_value="y"):
                info = approve_golden(evidence, golden_name="smoke", golden_dir=self.golden_dir)
        self.assertTrue(info["approved"])
        self.assertTrue(Path(info["screen_path"]).is_file())
        self.assertTrue(Path(info["manifest_path"]).is_file())

    def test_missing_capture_is_rejected(self) -> None:
        evidence = self.root / "empty-evidence"
        evidence.mkdir()
        with stdin_replaced(_FakeTTY()):
            with self.assertRaises(ValueError):
                approve_golden(evidence, golden_name="smoke", golden_dir=self.golden_dir, assume_yes=True)


# ---------------------------------------------------------------------- run record


class TapeRunRecordTests(TapeTestBase):
    def test_run_tape_persists_record_and_artifacts(self) -> None:
        tape = self.write_tape(TAPE_BODY, name="record")
        with TestBenchStore(self.tb_home) as store:
            rec = run_tape(store, tape, project_root=self.project, golden_dir=self.golden_dir)
            stored = store.get_run(rec.run_id)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.status, RunStatus.PASS)
        self.assertEqual(stored.profile, ProfileName.TERMINAL)
        self.assertEqual(stored.total_count, 3)
        self.assertEqual(stored.fail_count, 0)
        run_dir = Path(stored.evidence_dir)
        for name in ("run.json", "junit.xml", "repro.sh"):
            self.assertTrue((run_dir / name).is_file(), name)
        payload = json.loads((run_dir / "run.json").read_text())
        self.assertEqual(payload["status"], RunStatus.PASS)
        self.assertEqual(payload["suites"][0]["adapter"], TAPE_ADAPTER)
        self.assertIn("tape run", (run_dir / "repro.sh").read_text())

    def test_approve_by_run_id_reaches_the_evidence_dir(self) -> None:
        tape = self.write_tape(TAPE_BODY + 'Golden "smoke"\n', name="record-golden")
        with TestBenchStore(self.tb_home) as store:
            rec = run_tape(store, tape, project_root=self.project, golden_dir=self.golden_dir)
            self.assertEqual(rec.status, RunStatus.BLOCKED)
            with stdin_replaced(_FakeTTY()):
                info = approve_golden(
                    rec.run_id, golden_name="smoke", golden_dir=self.golden_dir, store=store, assume_yes=True
                )
            self.assertTrue(info["approved"])
            self.assertEqual(Path(info["evidence_dir"]), Path(rec.evidence_dir))
            replay = run_tape(store, tape, project_root=self.project, golden_dir=self.golden_dir)
            self.assertEqual(replay.status, RunStatus.PASS)

    def test_run_tape_persists_failure_kind(self) -> None:
        tape = self.write_tape(TAPE_BODY + 'ExpectScreen contains "nope"\n', name="record-fail")
        with TestBenchStore(self.tb_home) as store:
            rec = run_tape(store, tape, project_root=self.project, golden_dir=self.golden_dir)
            stored = store.get_run(rec.run_id)
        assert stored is not None
        self.assertEqual(stored.status, RunStatus.FAIL)
        self.assertEqual(stored.failure_kind, FailureKind.ASSERTION)
        self.assertEqual(stored.fail_count, 1)
        self.assertIn("nope", stored.first_failure)


# ------------------------------------------------------------------------ the CLI


class TapeCliTests(TapeTestBase):
    def _env(self) -> dict:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["TMPDIR"] = str(self.root)
        env.pop("TESTBENCH_SANDBOX", None)
        return env

    def _cli(self, *args: str, **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(
            [PY, "-m", "hermes_testbench", "--home", str(self.tb_home), *args],
            cwd=str(self.project),
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=kwargs.get("timeout", 180),
            stdin=kwargs.get("stdin", subprocess.DEVNULL),
        )

    def test_cli_run_persists_and_emits_adapter_marker(self) -> None:
        tape = self.write_tape(TAPE_BODY, name="cli")
        proc = self._cli("tape", "run", str(tape), "--project-root", str(self.project), "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout[-800:])
        marker_line = proc.stdout.strip().splitlines()[-1]
        self.assertTrue(marker_line.startswith("TB-TAPE "))
        payload = json.loads(marker_line[len("TB-TAPE ") :])
        self.assertEqual(payload["status"], RunStatus.PASS)
        self.assertEqual(payload["total"], 3)
        run_json = json.loads(proc.stdout[: proc.stdout.rindex(marker_line)])
        self.assertEqual(run_json["status"], RunStatus.PASS)
        run_dir = Path(run_json["evidence_dir"])
        for name in ("run.json", "junit.xml", "repro.sh"):
            self.assertTrue((run_dir / name).is_file(), name)

    def test_cli_failing_tape_exits_nonzero(self) -> None:
        tape = self.write_tape(TAPE_BODY + 'ExpectScreen contains "nope"\n', name="clifail")
        proc = self._cli("tape", "run", str(tape), "--project-root", str(self.project), "--json")
        self.assertEqual(proc.returncode, 1)
        marker = json.loads(proc.stdout.strip().splitlines()[-1][len("TB-TAPE ") :])
        self.assertEqual(marker["status"], RunStatus.FAIL)
        self.assertEqual(marker["failure_kind"], FailureKind.ASSERTION)

    def test_cli_refuses_vhs_without_spawning_anything(self) -> None:
        canary = self.root / "SPAWNED"
        tape = self.write_tape(
            f'Size 80 24\nVhs "demo.gif"\nCommand /bin/sh -c "touch {canary}"\n',
            name="vhs",
        )
        proc = self._cli("tape", "run", str(tape), "--project-root", str(self.project))
        self.assertEqual(proc.returncode, 1)
        self.assertIn(VHS_DEFERRAL_MESSAGE, proc.stderr)
        self.assertFalse(canary.exists())

    def test_cli_approve_requires_a_tty(self) -> None:
        tape = self.write_tape(TAPE_BODY + 'Golden "smoke"\n', name="golden-tape")
        run = self._cli("tape", "run", str(tape), "--project-root", str(self.project), "--json")
        run_json = json.loads(run.stdout[: run.stdout.rindex("TB-TAPE ")])
        evidence = Path(run_json["evidence_dir"])
        proc = self._cli(
            "tape",
            "approve",
            str(evidence),
            "--golden",
            "smoke",
            "--golden-dir",
            str(self.golden_dir),
            "--yes",
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("interactive TTY", proc.stderr)
        self.assertFalse(self.golden_dir.exists())

    def test_cli_unknown_tape_action(self) -> None:
        proc = self._cli("tape")
        self.assertEqual(proc.returncode, 2)


# ---------------------------------------------------------------- adapter wiring


class TerminalTapeAdapterTests(TapeTestBase):
    def test_adapter_is_registered(self) -> None:
        self.assertIn(TAPE_ADAPTER, KNOWN_ADAPTERS)
        self.assertIn(TAPE_ADAPTER, ADAPTER_PARSERS)
        self.assertIs(ADAPTER_PARSERS[TAPE_ADAPTER], parse_terminal_tape)

    def test_parser_reads_marker_payload(self) -> None:
        text = (
            "human summary line\n"
            'TB-TAPE {"adapter": "terminal_tape", "name": "smoke", "status": "PASS", '
            '"failure_kind": null, "checks": [{"kind": "expectscreen", "ok": true}, '
            '{"kind": "golden", "ok": true}], "total": 2, "passed": 2, "failed": 0, '
            '"evidence_dir": "/tmp/ev"}\n'
        )
        outcome = parse_terminal_tape(text)
        self.assertTrue(outcome.structured)
        self.assertEqual((outcome.total, outcome.passed, outcome.failed), (2, 2, 0))
        self.assertEqual(outcome.summary_line, "smoke PASS")
        self.assertEqual(len(outcome.cases), 2)

    def test_parser_maps_failures_and_blocked(self) -> None:
        fail_text = (
            'TB-TAPE {"name": "smoke", "status": "FAIL", "checks": '
            '[{"kind": "expectscreen", "mode": "contains", "text": "x", "ok": false, '
            '"detail": "expectscreen contains x failed"}], "total": 1, "passed": 0, "failed": 1}\n'
        )
        outcome = parse_terminal_tape(fail_text)
        self.assertEqual(outcome.failed, 1)
        self.assertEqual(outcome.failures[0]["kind"], FailureKind.ASSERTION)
        blocked_text = (
            'TB-TAPE {"name": "smoke", "status": "BLOCKED", "message": "golden missing", '
            '"checks": [{"kind": "golden", "ok": false, "status": "missing", '
            '"detail": "golden missing"}], "total": 1, "passed": 0, "failed": 0}\n'
        )
        blocked = parse_terminal_tape(blocked_text)
        self.assertEqual(blocked.errors, 1)
        self.assertEqual(blocked.failures[0]["kind"], FailureKind.BLOCKED_POLICY)
        self.assertFalse(parse_terminal_tape("nothing here").structured)

    def test_discovery_finds_tape_files(self) -> None:
        self.write_tape(TAPE_BODY, name="suite")
        spec = discover_project(self.project, store=None)
        suites = [s for s in spec.suites if s.adapter == TAPE_ADAPTER]
        self.assertEqual(len(suites), 1)
        suite = suites[0]
        self.assertEqual(suite.name, "tape:suite")
        self.assertEqual(suite.profile, ProfileName.TERMINAL)
        self.assertIn("tape", suite.command)
        self.assertEqual(suite.command[-1], "--json")
        self.assertFalse(suite.allow_network)

    def test_tape_suite_cannot_request_network(self) -> None:
        suite = SuiteSpec(
            name="tape:demo",
            adapter=TAPE_ADAPTER,
            profile=ProfileName.TERMINAL,
            command=[PY, "-c", "print('should not run')"],
            allow_network=True,
        )
        evidence = self.root / "evidence"
        result, usage = execute_suite(
            suite,
            project_root=self.project,
            evidence_dir=evidence,
            run_id="tb_test_network_gate",
        )
        self.assertEqual(result.status, RunStatus.BLOCKED)
        self.assertEqual(result.failures[0].kind, FailureKind.BLOCKED_POLICY)
        self.assertIsNone(usage)
        self.assertEqual(list(evidence.glob("*.log")), [])


if __name__ == "__main__":
    unittest.main()
