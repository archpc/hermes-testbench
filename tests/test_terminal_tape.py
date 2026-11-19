"""Terminal tape: real-PTY scripted replay, deterministic screens, golden gate.

Everything here runs offline: one isolated ``pty.openpty()`` program, no browser,
no VHS, no ambient tmux, and no writes outside the scratch directory.

Coverage order follows the layering, because a lower layer's bug would otherwise
be reported as the next layer's failure:

* ``VTScreenTests`` — the interpreter is a pure function of the byte stream. A
  wrong grid makes every later assertion meaningless, so it is pinned first and
  without a PTY.
* ``PtySessionTests`` — one child, isolated env, bounded buffers, real resize.
* ``TapeParseTests`` — both source formats, and every refusal, before a process
  exists.
* ``TapeReplayTests`` / ``GoldenGateTests`` / ``GoldenApprovalTests`` — the three
  states a tape can be in at the golden gate (capture, compare, promote), each
  with a negative control.
* ``TapeRunRecordTests`` / ``TapeCliTests`` / ``TerminalTapeAdapterTests`` — the
  path an operator and the engine actually take: store row, exit codes, marker.

Two properties are asserted rather than assumed, and most of the file leans on
them. Determinism: the same tape replayed twice yields byte-identical normalized
screen text (``test_scripted_input_passes_and_is_deterministic``), which is the
only reason a hash-keyed golden means anything at all. Read-only goldens: no
replay, in any state, writes into the golden directory — the blocking tests
snapshot that directory's hashes before and after, so "read-only" is verified
instead of trusted.

Sandbox and PTY assumptions. Fixtures live under
``isolation.default_scratch_root()`` rather than the system temp dir, so a host
that only permits writes under the TestBench scratch tree still runs them.
``TapeTestBase`` removes ``TESTBENCH_SANDBOX`` for the duration of a test: running
this suite from inside TestBench sets it to ``1`` in every child, and then the
approval-refusal tests would pass for the wrong reason (asserting the sandbox
gate against its own inherited marker while the TTY gate they are meant to
isolate goes unexercised). The Seatbelt test is skipped unless
``can_apply_seatbelt()`` — that probe runs ``sandbox-exec`` against a trivial
profile — because a binary that is present but unusable is indistinguishable from
an outer sandbox here.

What these tests do not establish: that the VT model matches any particular real
terminal. ``VTScreen`` has no reflow, no scrollback and no SGR, and
resize-while-alternate-screen is a known defect (see
``pty_session.VTScreen.resize``). A passing replay means the program's text
stream, interpreted by this model at this geometry, satisfied the script — not
that a human sees the same frame.
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

# The child must be the same interpreter as the runner: a version skew between
# the two would change both the VT bytes and the tracebacks under test.
PY = sys.executable
REPO_ROOT = Path(__file__).resolve().parent.parent


# The fixture is embedded, not a file on disk, for two reasons: the child's source
# stays auditable in this file, and there is no second copy to drift out of sync
# with the tape that drives it. It is deliberately dumb so every assertion below
# has an obvious cause: fixed CUP+EL full-frame writes, a buffer/value pair that
# only changes on input, one line of `note.txt` for content changes that do not
# touch the tape bytes, and a SIGWINCH handler that re-reads its own winsize and
# repaints it (the only way to prove a resize reached the child).
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


# The canonical drive. Three expectations (contains / excludes / raw) is the
# denominator the pass-path and CLI tests assert on, and the trailing Resize is
# what makes the golden's recorded geometry 72x14 rather than 60x12. Every wait
# names a string the fixture actually paints, so no step depends on a sleep.
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
    """Stand-in stdin that claims to be an interactive terminal.

    ``approve_golden``'s gate reads ``sys.stdin.isatty()``, not the controlling
    terminal, so substituting this object exercises the real guard and then lets
    the prompt run against a patched ``builtins.input``.
    """

    def isatty(self) -> bool:  # noqa: D401 - test double
        return True

    def read(self, size: int = -1) -> str:
        return ""

    def fileno(self) -> int:
        return 0

    def readline(self, size: int = -1) -> str:
        return ""


class _NoTTY:
    """Stand-in stdin that reports "not a terminal"; the negative control for the gate."""

    def isatty(self) -> bool:
        return False

    def read(self, size: int = -1) -> str:
        return ""

    def fileno(self) -> int:
        return 0


@contextlib.contextmanager
def stdin_replaced(replacement):  # type: ignore[no-untyped-def]
    """Swap ``sys.stdin`` for the duration of a block and restore it on any exit.

    Restoration is in ``finally`` because a refusal raised inside the block is the
    expected outcome of most of these tests; a leaked fake stdin would silently
    make every later test think it has a TTY.
    """
    original = sys.stdin
    sys.stdin = replacement
    try:
        yield
    finally:
        sys.stdin = original


def _sha(text: str) -> str:
    """Plain SHA-256 of UTF-8 text.

    Kept for ad-hoc comparisons only. Every assertion that matters hashes through
    ``tape.screen_sha256``, which normalizes first — asserting on raw bytes would
    pass while the normalization the golden identity depends on was wrong.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dir_snapshot(path: Path) -> dict:
    """Map ``filename -> sha256`` for every regular file in ``path``.

    A missing directory yields ``{}`` so "the replay must not create the golden
    directory" and "the replay must not rewrite existing goldens" can be the same
    comparison. Directory entries are sorted by name for a stable diff on failure.
    """
    if not path.is_dir():
        return {}
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.iterdir())
        if p.is_file()
    }


class TapeTestBase(unittest.TestCase):
    """Shared fixture: one scratch root, one project with ``fixture.py``, one store home.

    The temp directory is created *under* ``default_scratch_root()`` rather than in
    the system temp dir: on a host where only the TestBench scratch tree is
    writable, a fixture in ``/var/folders`` would fail for reasons unrelated to the
    code under test. ``self.project`` is what the tape's ``Command`` runs in, so
    every child's cwd is bounded by the same tree.
    """

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
        # Restore only if it was actually set, so the parent process's environment is
        # left byte-identical; then drop the tree, including any sandbox a child leaked.
        saved = getattr(self, "_sandbox_env", None)
        if saved is not None:
            os.environ["TESTBENCH_SANDBOX"] = saved
        self._tmp.cleanup()

    # -- helpers -------------------------------------------------------------
    def write_tape(self, body: str, name: str = "smoke") -> Path:
        """Write a tape body to ``project/<name>.tape`` and return its path.

        The filename is the tape's identity (``TapeScript.name`` defaults to the
        stem), which is why several tests deliberately use a file called
        ``golden-tape`` while the golden itself is named ``smoke``: evidence files
        key off the tape slug, the golden off its own name, and the approval
        candidate search has to bridge the two.
        """
        path = self.project / f"{name}.tape"
        path.write_text(body, encoding="utf-8")
        return path

    def replay(self, tape_path: Path, *, evidence: str = "evidence", **kwargs):  # type: ignore[no-untyped-def]
        """Parse and replay a tape with this fixture's project, evidence and golden dirs.

        ``evidence`` is a *name* under ``self.root`` on purpose: the determinism test
        replays into two different directories so identical screens cannot be an
        artifact of reusing one file.
        """
        return replay_tape(
            load_tape(tape_path),
            project_root=self.project,
            evidence_dir=self.root / evidence,
            golden_dir=self.golden_dir,
            **kwargs,
        )

    def approve(self, evidence_dir: Path, name: str, *, assume_yes: bool = True) -> dict:
        """Approve a capture with a fake TTY installed; returns ``approve_golden``'s info dict."""
        with stdin_replaced(_FakeTTY()):
            return approve_golden(
                evidence_dir,
                golden_name=name,
                golden_dir=self.golden_dir,
                assume_yes=assume_yes,
            )


# --------------------------------------------------------------------- VTScreen


class VTScreenTests(unittest.TestCase):
    """Grid invariants of the VT interpreter: pure text in, text out, no PTY.

    Each test pins one property the tape layer depends on — tab stops and
    backspace semantics, cursor addressing, erase modes, alt-screen fidelity,
    escape consumption, width agreement with the renderer, chunk-boundary
    invisibility, truncating resize, and normalization. Nothing here spawns a
    process, so a failure points at the interpreter rather than at timing.
    """

    def test_carriage_return_linefeed_backspace_and_tab(self) -> None:
        """C0 controls move the cursor; they never become cells.

        TAB advances to the next multiple of 8 and clamps at the last column, and
        BS moves left so the following ``Y`` *overwrites* ``X`` — the expected
        ``"world   Y"`` fails if BS were implemented as a delete instead of a
        cursor move, which is the distinction a tape's editing assertions rely on.
        """
        screen = VTScreen(20, 4)
        screen.feed("hello\r\nworld\tX\bY")
        self.assertEqual(screen.to_lines()[0], "hello")
        self.assertEqual(screen.to_lines()[1], "world   Y")

    def test_cursor_position_and_relative_moves(self) -> None:
        """CUP is 1-based; relative moves clamp; ``H`` and ``f`` share one handler.

        ``\\x1b[3;5H`` plus three printable cells leaves the cursor at (2, 7) —
        row 0-based but *after* the glyphs, the "pending wrap" column. The last
        step uses the empty-parameter form ``\\x1b[f``, so the expected ``R`` at
        the origin proves both the ``f`` alias and the default of 1 for a missing
        parameter.
        """
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
        """EL's three modes differ where it counts: whether the cursor cell goes too.

        ``\\x1b[K`` erases *from* the cursor (so ``abc`` survives), ``\\x1b[1K``
        erases *through* the cursor inclusive — the expected ``" lm"`` keeps the
        leading space because the cell under the cursor was blanked. ``ED 2``
        clears the whole grid, trailing rows included.
        """
        screen = VTScreen(10, 3)
        screen.feed("abcdefghij\r\nklm")
        screen.feed("\x1b[1;4H\x1b[K")
        self.assertEqual(screen.to_lines()[0], "abc")
        screen.feed("\x1b[2;1H\x1b[1K")
        self.assertEqual(screen.to_lines()[1], " lm")
        screen.feed("\x1b[2J")
        self.assertEqual(screen.to_lines(), ["", "", ""])

    def test_alt_screen_saves_and_restores(self) -> None:
        """``1049``/``1047`` save the *grid*, not a flag: text comes back on exit.

        The restored ``"main"`` is the assertion that matters — a mode-only
        implementation would pass the blank-alt-screen check and lose the primary
        buffer, which is exactly the bug that would make every post-teardown
        evidence artifact empty.
        """
        screen = VTScreen(12, 3)
        screen.feed("main")
        screen.feed("\x1b[?1049h")
        self.assertEqual(screen.to_lines(), ["", "", ""])
        screen.feed("\x1b[Halt")
        self.assertEqual(screen.to_lines()[0], "alt")
        screen.feed("\x1b[?1049l")
        self.assertEqual(screen.to_lines()[0], "main")

    def test_cursor_visibility_modes(self) -> None:
        """DEC mode 25 is tracked independently of the text grid."""
        screen = VTScreen(10, 2)
        screen.feed("\x1b[?25l")
        self.assertFalse(screen.cursor.visible)
        screen.feed("\x1b[?25h")
        self.assertTrue(screen.cursor.visible)

    def test_sgr_osc_and_dcs_never_reach_text_cells(self) -> None:
        """Whole sequences are consumed, not just the ESC byte.

        The negative assertions are the point: a leaked ``[1;31`` body, an OSC
        window title, or the sixel header ``#0;2;0;0;0`` would appear in every
        golden and in every ``ExpectScreen`` comparison. ``\\x1b(B`` is a
        two-byte charset designator — it must disappear while ``plain`` survives,
        which rules out "strip everything after ESC to end of line".
        """
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
        """The renderer's width table and the interpreter's must agree, cell for cell.

        ``pty_session.char_cells`` decides placement; ``tui_render.display_width``
        decides padding and truncation. If they diverge, ``fit_cells`` pads to a
        different width than the screen paints and goldens become
        geometry-dependent. The comparison is done both ways — sum of per-char
        cells against ``display_width`` and ``display_width`` against the rendered
        row — so an error in either implementation surfaces. The combining sample
        pins the "attach to the previous cell" rule: ``e\\u0301clair`` is 6 cells
        and keeps its mark in ``to_lines()``.
        """
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
        """Chunk boundaries are invisible: the ``_pending`` buffer is the whole story.

        Feeding one character at a time splits every escape sequence in the
        payload at every possible offset (alt-screen enter/exit, a wide glyph, SGR,
        an OSC hyperlink). Equality with the wholesale feed is what allows
        ``PtySession`` to hand the interpreter arbitrary ``os.read`` chunk sizes.
        """
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
        """Resize truncates or pads; it does not reflow.

        Growing keeps the top-left content and leaves new cells blank, so a
        cursor placed at the far edge of the wider grid paints where expected.
        Shrinking clamps the *cursor* (row 1, not 5) and the assertion on
        ``len(...) == 5`` proves the row itself was cut to the new width rather
        than left at its old length — the invariant every later ``EL``/``ED``
        relies on when it indexes up to ``self.cols``.
        """
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
        """The two normalizers split the contract, and this test pins the seam.

        ``VTScreen.to_text()`` strips per-line padding and trailing blank rows and
        ends with *no* newline; ``tape.normalized_screen`` adds exactly one. Both
        sides are asserted here so a change to either cannot silently shift every
        golden hash by a byte.
        """
        screen = VTScreen(12, 3)
        screen.feed("abc   \r\ndef")
        self.assertEqual(screen.to_text(), "abc\ndef")
        self.assertEqual(screen.to_lines(), ["abc", "def", ""])
        self.assertEqual(normalized_screen(screen.to_text()), "abc\ndef\n")

    def test_full_screen_redraw_is_stable(self) -> None:
        """Repainting an identical frame three times must yield one distinct string.

        This is the minimal determinism check the golden path depends on: if the
        erase/CUP ordering left residue, ``WaitQuiet`` + ``Golden`` would produce a
        hash that changes between runs with no program change.
        """
        frame = "\x1b[?25l\x1b[H\x1b[2Jrow one\x1b[K\r\nrow two\x1b[K\x1b[J"
        screens = []
        for _ in range(3):
            screen = VTScreen(40, 6)
            screen.feed(frame)
            screens.append(screen.to_text())
        self.assertEqual(len(set(screens)), 1)


# -------------------------------------------------------------------- PtySession


class PtySessionTests(unittest.TestCase):
    """One real PTY child: key map, lifecycle, buffer caps, env isolation, resize.

    These tests pay for an actual ``openpty()`` spawn because the engine's
    guarantees are about a *process*, not a string: the sandbox directory must be
    gone after ``close()``, the child must not inherit the developer's ``TMUX`` or
    ``HOME``, and the request/observed split for geometry has to be visible in the
    object. Children are ``python -c`` one-liners so there is no fixture file to
    keep in sync; several deliberately sleep after printing, noted at the call
    site, so teardown runs against a live process rather than an exited one.
    """

    def setUp(self) -> None:
        """Temp cwd under the scratch root so the child's sandbox lands there too."""
        self._tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-pty-sess-")
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        """Drop the tree; a leaked sandbox would surface as a non-empty scratch root."""
        self._tmp.cleanup()

    def test_key_sequences(self) -> None:
        """Named keys resolve case-insensitively; single characters pass through.

        ``"q"`` is not in ``KEY_SEQUENCES`` — it is accepted by the length-1
        branch — while an unknown multi-character name raises ``ValueError`` so a
        typo in a tape fails instead of writing nothing and letting the next
        ``WaitFor`` time out with a misleading message.
        """
        self.assertEqual(key_sequence("Enter"), "\r")
        self.assertEqual(key_sequence("enter"), "\r")
        self.assertEqual(key_sequence("Up"), "\x1b[A")
        self.assertEqual(key_sequence("q"), "q")
        with self.assertRaises(ValueError):
            key_sequence("Nope")

    def test_session_runs_program_and_tears_down(self) -> None:
        """Output arrives, ``close()`` returns a real status, and the sandbox is deleted.

        The 30 s sleep keeps the child alive when ``close`` is called, so teardown
        takes the SIGTERM path instead of the already-exited fast path. The final
        assertion is the leak check: every session owns exactly one ``0700``
        scratch directory and a clean exit must remove it, or a long test run
        fills the scratch root.
        """
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
        """``raw_bytes`` is front-truncated to exactly ``max_raw_bytes`` and latches.

        A 200 KB burst against a 4 KB cap proves both halves: the retained window
        never exceeds the cap, and ``truncated`` reports that bytes were dropped so
        ``ExpectRaw`` can be read as "somewhere in the last N bytes" rather than
        "somewhere in the session". The cap is on *bytes*; the text window is
        counted in characters, which is a looser bound for UTF-8-heavy output.
        """
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
        """The child sees a scrubbed environment, not the developer's.

        Four properties, each meaningful on its own: no ``TMUX`` (a suite must not
        attach to the live tmux server), ``TESTBENCH_SANDBOX=1`` (a nested
        ``tape approve`` is therefore refused structurally), ``HOME`` redirected
        away from the real home, and ``TMPDIR`` no longer the ambient one. The
        payload is parsed from the last non-blank line so a partial trailing line
        from the PTY cannot break the parse — the snippet prints nothing else.
        """
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
        """Resize updates the session's declared geometry and the interpreter's grid.

        Deliberately *not* evidence about the child: geometry here is declared via
        ``TIOCSWINSZ`` and never observed, and this child never reads its winsize.
        The observable-in-child claim is made by
        ``test_resize_is_observable_in_the_program``, where the fixture repaints its
        own size.
        """
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
    """Both tape formats, and every refusal, without spawning a process.

    Parsing is where a tape's policy lives (blocked backends) and where a typo must
    fail loudly rather than quietly dropping a check. Only files are touched, so
    these tests are fast and host-independent.
    """

    def setUp(self) -> None:
        """Only a directory is needed: these tests never spawn a process."""
        self._tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-tape-parse-")
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        """Remove the tape files written by each case."""
        self._tmp.cleanup()

    def test_dsl_parses_every_action(self) -> None:
        """Each directive lands in one step, with quoting, defaults and hash pinned.

        Four things are checked that a round-trip alone would miss: ``{python}``
        is substituted with the running interpreter (so the child cannot be a
        different Python), ``shlex`` keeps ``"a b"`` as one argv element, quoted
        literals are JSON-decoded (``\\\\n`` becomes a newline, ``\\\\u001b`` an ESC
        byte) so control characters stay readable in the source, and ``sha256``
        covers the raw bytes *including the comment line* — that hash is the golden
        staleness key. ``WaitQuiet 99`` also proves an explicit value survives
        instead of being replaced by the default at parse time.
        """
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
        """Three spellings of a screen-recording backend, one policy refusal.

        The DSL has to catch the action form (``Vhs``), the config form
        (``Backend vhs``) and the ``Screenshot`` alias, because a tape author can
        reach any of them. All three must raise ``TapePolicyError`` with
        ``failure_kind`` BLOCKED_POLICY and the shared deferral message — the
        refusals happen during parsing, so no process is ever created (the CLI test
        proves that with a canary file).
        """
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
        """The JSON form is a real second path: geometry, argv substitution, aliases.

        ``source_format`` is asserted because it lands in the run meta and tells a
        reader which parser ran; ``{python}`` substitution is repeated here so the
        two parsers cannot drift on it. Rewriting the same file with
        ``"backend": "chrome"`` must be refused at parse time, not at replay.
        """
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
        """Six independent malformations, six distinguishable messages.

        Each needle identifies the branch that fired (unknown action, missing
        ``Command``, a second ``Golden``, non-positive ``Size``, an invalid
        ``ExpectScreen`` mode, a short ``Resize``). Asserting the message rather
        than just the exception class is what stops a newly added directive from
        starting to satisfy a neighbouring error and turning a real syntax error
        into a silently weaker tape.
        """
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
    """Replay of the scripted fixture: pass path, failure evidence, timeouts, resize.

    The fixture reads stdin only after ``tty.setraw``, so the kernel's erase/kill
    processing and the terminal's echo cannot insert bytes the frame did not
    contain — that echo would otherwise show up in the captured screen and make
    ``Type`` assertions nondeterministic. Every wait names a string the fixture
    paints, so no step is a sleep.
    """

    def test_scripted_input_passes_and_is_deterministic(self) -> None:
        """Two replays, identical screen text and hash; then the evidence contract.

        Determinism is asserted twice (raw text equality and hash equality) because
        it is the premise a hash-keyed golden rests on. ``passed == 3`` is the
        number of declared expectations, so a tape that silently lost one is
        visible. ``FIXTURE 72x14`` in the *screen* is the resize round-trip. The
        ``tape.meta.json`` assertions matter beyond this test: that file is what
        ``tape approve`` reads, so its status and check shape are an approval-path
        contract, not decoration.
        """
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
        """A failed expectation keeps the real screen, a literal-vs-screen diff, and 0600.

        The appended ``value: 99`` is guaranteed absent (the fixture only ever
        paints 42), so the FAIL is deterministic. ``actual.screen.txt`` must still
        show ``value: 42``: the artifact is what lets a reviewer tell "the program
        was wrong" from "the expectation was wrong". ``expect.diff`` is a unified
        diff of the literal against the normalized screen (not the golden), which
        is why its header lines are ``--- expected`` / ``+++ actual``. Evidence is
        ``0600`` because program output may contain redacted-but-still-sensitive
        text.
        """
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
        """A missed wait is TIMEOUT, not FAIL, and the message names the needle.

        The 250 ms budget is far below any render of a needle the fixture never
        prints, so the test cannot race the program. The class mapping matters
        downstream: ``_aggregate`` turns TIMEOUT into ``FailureKind.TIMEOUT`` and
        the record keeps ``timeout`` rather than collapsing to ``assertion``.
        """
        body = 'Size 60 12\nCommand {python} fixture.py\nWaitFor "never-appears" 250\n'
        tape = self.write_tape(body, name="timeout")
        outcome = self.replay(tape)
        self.assertEqual(outcome.status, RunStatus.TIMEOUT)
        self.assertEqual(outcome.failure_kind, FailureKind.TIMEOUT)
        self.assertIn("never-appears", outcome.message)

    def test_resize_is_observable_in_the_program(self) -> None:
        """The child re-reads its winsize and repaints it — the resize really arrived.

        Waiting on ``FIXTURE 60x12`` before resizing removes the race between the
        first paint and ``TIOCSWINSZ``. The later ``FIXTURE 90x20`` can only be
        produced by the fixture's SIGWINCH handler calling ``TIOCGWINSZ``, which is
        the one observation ``PtySession.resize`` cannot make on its own.
        ``outcome.cols/rows`` are the tape's *requested* final geometry, not a
        measurement, and the single expectation is the denominator 1.
        """
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
        """A tape child is confined even though its cwd is the project.

        The probe writes into the project directory, which is *not* in the
        profile's write allow-list (``allow_repo_write`` defaults False), so the
        denial is a real policy decision and not a missing parent directory. Three
        assertions separate "denied" from "never ran": ``DENIED`` on screen (the
        child executed and caught ``OSError``), no ``WROTE``, and no file on disk.
        Skipped when ``can_apply_seatbelt()`` fails its probe, because that result
        also means "already inside an outer Seatbelt" and the assertion would be
        measuring the wrong layer.
        """
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
        """The shipped TUI, not a fixture: alt-screen, guard prompt, resize, exit.

        ``ExpectRaw`` on the alt-screen sequence checks the controller's lifecycle
        write, which no screen assertion can see. Lowercase ``q`` must *not* quit:
        the tape waits for the "press Q" guard prompt, then uppercase ``Q`` exits —
        the positive and negative control for the quit binding in one script.
        ``PYTHONPATH`` is patched because the temporary project cwd cannot import a
        source checkout, and relying on an ambient value would make the test
        environment-dependent; ``timeout_s=45`` covers interpreter start plus the
        15 s first-paint budget.
        """
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
        # Three expectations are declared above; the floor also catches a script that
        # grew assertions without this line being revisited.
        self.assertGreaterEqual(outcome.passed, 3)


# ------------------------------------------------------------------- golden gate


class GoldenGateTests(TapeTestBase):
    """The read-only half of the golden contract: every blocking state, with proofs.

    A golden is a hash of (normalized screen, tape bytes, geometry) plus an
    approval marker. Each test walks one rung of ``check_golden``'s ladder and
    asserts the distinguishing message, because ``missing``, ``unapproved`` and
    ``stale`` all end BLOCKED — the wording is the only thing telling an operator
    what to fix. Where a test touches the golden directory it also snapshots the
    directory hashes, so "replay never writes the baseline" is verified rather
    than trusted.
    """

    def _tape_with_golden(self, name: str = "smoke") -> Path:
        """The canonical drive plus a trailing ``Golden``, written as ``golden-tape.tape``.

        The file stem and the golden name differ on purpose: evidence files key off
        the tape slug while the baseline keys off the golden name, and the approval
        candidate search has to bridge them.
        """
        return self.write_tape(TAPE_BODY + f'Golden "{name}"\n', name="golden-tape")

    def test_missing_golden_blocks_and_never_writes(self) -> None:
        """No baseline is a policy block, and the golden directory is never created.

        ``check_golden`` is pure and ``replay_tape`` has no mkdir on that path, so
        the absence of the directory is the strongest available statement that a
        run cannot bootstrap its own baseline.
        """
        tape = self._tape_with_golden("smoke")
        outcome = self.replay(tape)
        self.assertEqual(outcome.status, RunStatus.BLOCKED)
        self.assertEqual(outcome.failure_kind, FailureKind.BLOCKED_POLICY)
        self.assertIn("golden missing", outcome.message)
        self.assertFalse(self.golden_dir.exists())

    def test_unapproved_golden_manifest_blocks(self) -> None:
        """A manifest whose approval marker does not match its screen hash is "unapproved".

        Only ``approved_sha256`` is corrupted, and the last assertion confirms the
        screen file still hashes to the manifest's ``screen_sha256`` — so the block
        provably comes from the approval field rather than from a tampered
        baseline. Directory hashes are unchanged: the rejection did not repair or
        rewrite anything.
        """
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
        """Editing the baseline after approval is caught by the file hash, not the manifest.

        ``"changed after approval"`` exists only on that rung of the ladder, which
        separates "someone edited the golden" from "this golden was never approved"
        — the two states have the same BLOCKED status and need different fixes.
        """
        tape = self._tape_with_golden("smoke")
        first = self.replay(tape)
        self.approve(Path(first.evidence_dir), "smoke")
        screen_path, _manifest = golden_paths(self.golden_dir, "smoke")
        screen_path.write_text(screen_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
        outcome = self.replay(tape, evidence="ev-tampered")
        self.assertEqual(outcome.status, RunStatus.BLOCKED)
        self.assertIn("changed after approval", outcome.message)

    def test_stale_tape_blocks_without_rewriting(self) -> None:
        """Any edit to the tape file — here one geometry digit — invalidates the golden.

        The staleness key is the hash of the raw bytes, comments included, so the
        changed ``Size`` is enough. Splitting on ``Size 60 12`` keeps the rest of
        the script byte-identical, which isolates the tape hash as the only
        difference; the unchanged directory snapshot proves the replay reported
        staleness instead of quietly refreshing the baseline.
        """
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
        """Geometry is a separate staleness dimension from the tape hash.

        Same tape bytes, one manifest field edited to ``cols: 5``, and the run is
        stale rather than a diff. Without this rung, replaying the same tape at a
        different size would present a screen diff that is really a configuration
        change — and the screen file is asserted to still exist, so the block
        cannot be confused with a missing baseline.
        """
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
        """A real screen difference is a FAIL with a diff, never a rewrite.

        Changing ``note.txt`` changes the program's output while leaving the tape
        bytes untouched — the only way to produce a legitimate diff without also
        tripping staleness. The diff direction is asserted explicitly (``+`` is
        actual, ``-`` is the golden) because inverting it would send a reviewer to
        the wrong side of the comparison. The repository's approved file must not
        contain the new text: capture is evidence, approval is a human act.
        """
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
        """The happy path, plus the manifest invariants that make the ladder meaningful.

        ``approved_sha256 == screen_sha256`` is what a completed approval means.
        Geometry is 72x14 because the ``Golden`` step runs after the tape's
        ``Resize 72 14`` and the comparison records the session's geometry at that
        moment — asserting it here pins when the geometry is sampled. The baseline
        file is ``0644`` (a repository artifact) while evidence is ``0600``, and the
        second replay must report ``pass`` in both ``outcome.golden`` and the check
        list.
        """
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
        """Purity is a contract, not an accident: asking must not create anything.

        ``replay_tape`` is the only caller that translates these statuses into run
        statuses, so a caller may ask "would this pass?" without side effects. The
        empty directory check is the assertion that enforces it.
        """
        self.golden_dir.mkdir()
        result = check_golden("abc\n", golden_dir=self.golden_dir, name="nope", tape_sha256="t", cols=10, rows=2)
        self.assertEqual(result.status, "missing")
        self.assertEqual(list(self.golden_dir.iterdir()), [])


# --------------------------------------------------------------- approval gate


class GoldenApprovalTests(TapeTestBase):
    """The write half: approval is human-gated, so every refusal is exercised.

    The guards are structural (``TESTBENCH_SANDBOX=1`` and ``sys.stdin.isatty()``)
    rather than prompt-only, because a suite, a cron job or an MCP caller can reach
    this function. ``_FakeTTY``/``_NoTTY`` swap ``sys.stdin`` instead of allocating
    a terminal — the guard reads that attribute, so the double is faithful — and
    the confirmation prompt itself is patched at ``builtins.input``, which only
    runs after both gates have passed.
    """

    def _evidence_with_capture(self) -> Path:
        """Run a golden tape once and hand back its evidence directory.

        The run is BLOCKED by design (no baseline yet); the capture written to
        ``NAME.screen.txt`` is what approval promotes.
        """
        tape = self.write_tape(TAPE_BODY + 'Golden "smoke"\n', name="golden-tape")
        outcome = self.replay(tape)
        self.assertEqual(outcome.status, RunStatus.BLOCKED)
        return Path(outcome.evidence_dir)

    def test_refused_without_a_tty(self) -> None:
        """``assume_yes`` skips the prompt, never the TTY check.

        With ``_NoTTY`` installed and ``assume_yes=True`` the call must still fail
        on the "interactive TTY" branch and write no directory — that is the
        property cron jobs, CI and piped shell scripts depend on.
        """
        evidence = self._evidence_with_capture()
        with stdin_replaced(_NoTTY()):
            with self.assertRaises(GoldenApprovalRefused) as ctx:
                approve_golden(evidence, golden_name="smoke", golden_dir=self.golden_dir, assume_yes=True)
        self.assertIn("interactive TTY", str(ctx.exception))
        self.assertFalse(self.golden_dir.exists())

    def test_refused_inside_a_sandbox_even_with_a_tty(self) -> None:
        """A TTY is not enough: the sandbox marker wins, which is what stops a suite.

        This is precisely why ``TapeTestBase`` pops ``TESTBENCH_SANDBOX`` — with the
        marker inherited from an outer TestBench run, this test would still pass,
        but for the wrong guard, and the TTY test above would stop distinguishing
        anything.
        """
        evidence = self._evidence_with_capture()
        with mock.patch.dict(os.environ, {"TESTBENCH_SANDBOX": "1"}):
            with stdin_replaced(_FakeTTY()):
                with self.assertRaises(GoldenApprovalRefused) as ctx:
                    approve_golden(evidence, golden_name="smoke", golden_dir=self.golden_dir, assume_yes=True)
        self.assertIn("sandbox", str(ctx.exception))
        self.assertFalse(self.golden_dir.exists())

    def test_declined_prompt_writes_nothing(self) -> None:
        """Anything short of ``y``/``yes`` is a decline, and a decline leaves no trace.

        The prompt is the last line of defence for a human who is looking at a
        diff they did not intend to bake in; ``n`` must produce neither the screen
        file nor the manifest.
        """
        evidence = self._evidence_with_capture()
        with stdin_replaced(_FakeTTY()):
            with mock.patch("builtins.input", return_value="n"):
                with self.assertRaises(GoldenApprovalRefused):
                    approve_golden(evidence, golden_name="smoke", golden_dir=self.golden_dir)
        self.assertFalse(self.golden_dir.exists())

    def test_confirmed_prompt_approves(self) -> None:
        """``y`` promotes the capture and reports both paths it wrote."""
        evidence = self._evidence_with_capture()
        with stdin_replaced(_FakeTTY()):
            with mock.patch("builtins.input", return_value="y"):
                info = approve_golden(evidence, golden_name="smoke", golden_dir=self.golden_dir)
        self.assertTrue(info["approved"])
        self.assertTrue(Path(info["screen_path"]).is_file())
        self.assertTrue(Path(info["manifest_path"]).is_file())

    def test_missing_capture_is_rejected(self) -> None:
        """An evidence directory without ``tape.meta.json`` fails before the prompt.

        A TTY is installed deliberately, so the failure under test is the missing
        capture rather than the gate; the raised type is ``TapeError``, a
        ``ValueError`` subclass, which is what keeps the CLI's handler from
        printing a traceback.
        """
        evidence = self.root / "empty-evidence"
        evidence.mkdir()
        with stdin_replaced(_FakeTTY()):
            with self.assertRaises(ValueError):
                approve_golden(evidence, golden_name="smoke", golden_dir=self.golden_dir, assume_yes=True)


# ---------------------------------------------------------------------- run record


class TapeRunRecordTests(TapeTestBase):
    """``replay_tape``'s outcome folded into a real ``RunRecord`` through a real store.

    A tape can pass while the *record* is wrong (counts, profile, artifacts,
    failure kind), and the record is what the TUI, JUnit and the adapter read.
    Each test uses an on-disk ``TestBenchStore`` under the test root, so nothing
    touches the developer's ``--home``.
    """

    def test_run_tape_persists_record_and_artifacts(self) -> None:
        """A pass is stored with terminal profile, 3/3 counts, and all three artifacts.

        The counts come from the evaluated checks, not from what the tape
        declared, so a tape that died early would show a smaller denominator.
        ``repro.sh`` containing "tape run" is asserted because a repro that does not
        name the command is not reproducible, and ``run.json``'s suite adapter is
        what routes the record to ``parse_terminal_tape`` later.
        """
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
        """A run id resolves through the store to that run's evidence; the loop closes.

        ``approve_golden`` accepts either a path or a run id, and the returned
        ``evidence_dir`` is asserted equal to the record's — that equality is what
        stops an approval from being written against the wrong run. The follow-up
        ``run_tape`` passing proves the approved baseline is the one the next run
        compares against.
        """
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
        """A FAIL is recorded as ``assertion`` (not ``harness_error``) with a named first failure.

        ``_aggregate`` maps the status to the kind, and it deliberately reports
        ASSERTION for FAIL even though a transport error can also end in FAIL — this
        test pins the mapping on the *record*, and ``first_failure`` naming the
        missing literal is what an operator sees in the TUI.
        """
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
    """The operator surface: exit codes, ``--json`` body plus the TB-TAPE trailer.

    Every test runs a real ``python -m hermes_testbench`` subprocess, because the
    contract includes process-level facts a direct call cannot show: the marker is
    the *last* stdout line, a policy refusal exits non-zero with the message on
    stderr, and approval without a TTY fails before writing. ``stdin`` defaults to
    ``DEVNULL`` for exactly that last reason.
    """

    def _env(self) -> dict:
        """Child environment: importable runner, scratch inside the test root, no sandbox marker.

        ``TMPDIR`` points at the test root so a nested CLI's own scratch and sandbox
        directories stay inside the temporary tree; the marker is popped so the
        approval test exercises the TTY gate rather than an inherited sandbox gate.
        """
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["TMPDIR"] = str(self.root)
        env.pop("TESTBENCH_SANDBOX", None)
        return env

    def _cli(self, *args: str, **kwargs) -> subprocess.CompletedProcess:
        """Invoke the CLI with ``--home`` pointed at the test store.

        ``--home`` is a global option, so it precedes the subcommand; capturing
        output as text lets a failure be asserted on stderr. The default timeout
        (180 s) sits above the tape's own 120 s budget so a wedged PTY is reported
        as a test failure instead of hanging the suite.
        """
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
        """Exit 0, JSON body first, marker last, and the artifacts exist on disk.

        The marker is extracted from the final line rather than searched for, which
        proves the trailer cannot be interleaved with ``--json`` output; the body is
        then everything up to ``rindex(marker)`` and must still parse. Exit 0 is
        derived from the run status, so a pass that reported non-zero here would
        break every CI caller.
        """
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
        """A failed tape exits 1 and the marker carries the failure classification.

        Two separate channels are checked on purpose: the exit code is what a shell
        caller sees, the marker is what the adapter parses. A mismatch between them
        would make the same run look failed to CI and clean to the engine.
        """
        tape = self.write_tape(TAPE_BODY + 'ExpectScreen contains "nope"\n', name="clifail")
        proc = self._cli("tape", "run", str(tape), "--project-root", str(self.project), "--json")
        self.assertEqual(proc.returncode, 1)
        marker = json.loads(proc.stdout.strip().splitlines()[-1][len("TB-TAPE ") :])
        self.assertEqual(marker["status"], RunStatus.FAIL)
        self.assertEqual(marker["failure_kind"], FailureKind.ASSERTION)

    def test_cli_refuses_vhs_without_spawning_anything(self) -> None:
        """The refusal precedes the spawn; the canary file is the negative control.

        The tape's ``Command`` would create ``SPAWNED`` if anything executed it. Exit
        1 with the deferral message on stderr (``TapePolicyError`` is a
        ``ValueError``, which the CLI's top-level handler prints instead of a
        traceback) and no canary together prove the policy check happens in the
        parser, not in the runner.
        """
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
        """``--yes`` through the CLI still cannot approve without a terminal.

        The run happens first because approval needs evidence; stdin is
        ``DEVNULL``, so the guard's ``isatty()`` is false in the child and the
        error lands on stderr with no golden written. This is the CLI-level
        counterpart to ``test_refused_without_a_tty``.
        """
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
        """A bare ``tape`` is a usage error (2), distinct from a runtime failure (1)."""
        proc = self._cli("tape")
        self.assertEqual(proc.returncode, 2)


# ---------------------------------------------------------------- adapter wiring


class TerminalTapeAdapterTests(TapeTestBase):
    """Adapter registration, marker parsing, discovery, and the network gate.

    The adapter is the only consumer of ``TB-TAPE``; if registration or the payload
    shape drifts, a run reports zero cases instead of failing loudly. The network
    test lives here rather than with the isolation suite because it is a property
    of *this* adapter: a tape drives a local PTY, so network access is a policy
    violation rather than a capability the runner could grant.
    """

    def test_adapter_is_registered(self) -> None:
        """Discovery name, parser mapping, and function identity all line up.

        The ``is`` check (not equality) is deliberate: a wrapping parser would keep
        the registry green while changing what the marker means.
        """
        self.assertIn(TAPE_ADAPTER, KNOWN_ADAPTERS)
        self.assertIn(TAPE_ADAPTER, ADAPTER_PARSERS)
        self.assertIs(ADAPTER_PARSERS[TAPE_ADAPTER], parse_terminal_tape)

    def test_parser_reads_marker_payload(self) -> None:
        """Counts come from the check list, and the human line is the name plus status.

        The marker also declares ``total``/``passed``/``failed``; the parser derives
        them from ``checks[].ok`` instead, so a marker whose counters disagree with
        its own check list cannot inflate a result. ``structured`` being true is
        what stops the generic summary heuristics from also firing.
        """
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
        """Status outranks per-check text: BLOCKED is an error, not a wall of failures.

        The blocked marker carries a check whose detail says "golden missing", and
        the expected classification is still BLOCKED_POLICY with ``errors == 1`` and
        ``failed == 0`` — that is the difference between "the gate refused" and
        "an assertion was wrong". Unmarked text leaves ``structured`` false so the
        caller must fall back to the exit status rather than invent counts.
        """
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
        """One ``.tape`` file becomes one terminal suite whose argv the adapter can parse.

        The suite name is ``tape:<stem>``, the profile is TERMINAL, and the argv ends
        in ``--json`` because that is what emits the marker this adapter reads.
        ``allow_network`` false is the discovery layer declining to grant what the
        adapter would refuse.
        """
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
        """A hand-built tape suite asking for the network is blocked before it runs.

        The command would print if it executed; ``usage`` being ``None`` and the
        evidence directory holding no ``*.log`` both show the gate returns before
        spawn, so there is nothing to parse and nothing to leak. A local PTY has no
        legitimate network need, which is why this is a hard block rather than a
        warning.
        """
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
