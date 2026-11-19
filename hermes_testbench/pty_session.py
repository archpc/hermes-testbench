"""Real-PTY sessions and a deterministic VT screen interpreter.

Two objects, one job each:

``VTScreen``
    Turns the child's decoded byte stream into the text grid a terminal would
    show. Pure and synchronous: ``feed()`` mutates a cell grid plus cursor,
    ``to_text()`` renders it. It covers cursor motion, erase, alt-screen
    save/restore, cursor visibility, wide cells, and consumption of SGR/OSC/DCS
    so no control byte can land in a cell. It models a *screen*, not a
    terminal: no scrollback, no reflow, no sixel/kitty graphics, no hyperlink
    targets, and no SGR attributes at all — colour and bold are dropped on
    purpose, because assertions and goldens are about text.

``PtySession``
    Owns exactly one ``pty.openpty()`` master/slave pair, one child in its own
    session (``pgid == pid``), an isolated environment
    (:func:`isolation.build_isolated_env`), and optionally a Seatbelt wrapper.
    Single-threaded by design: nothing reads the master except an explicit
    ``pump()`` call, so every read is bracketed by the caller's timeout
    arithmetic. No reader thread can interleave with an assertion, which is
    what makes replay reproducible enough to diff.

Data ownership (all RAM; this module never writes an artifact):

* ``raw_bytes`` — the last ``max_raw_bytes`` of the byte stream (``ExpectRaw``).
* ``text_parts`` / ``text_len`` — the decoded stream, same window measured in
  *characters* (see ``_absorb``), used by ``wait_for``.
* ``screen`` — the :class:`VTScreen`; never truncated, so the visible frame
  survives after the stream window rolls over.
* ``sandbox_dir`` — disposable ``0700`` scratch, removed by ``close()`` unless
  a survivor leaked out of it.

This is a *second* execution path, parallel to
``isolation.run_isolated_command``: it does not pipe output through
``StreamRedactor`` (``tape.replay_tape`` sanitizes when it writes artifacts), it
has no run-level timeout kill (``replay_tape`` checks a budget between steps),
and it performs its own teardown instead of the fingerprinted SIGTERM/SIGKILL
reaper. Those differences are deliberate — a PTY cannot be a pipe — but they
mean the operating guide's isolation section describes the *other* path.

Geometry is declared, not observed: ``TIOCSWINSZ`` tells the kernel a size,
``resize()`` records what was requested, and nothing verifies that the child
read it. A program that ignores ``SIGWINCH`` still passes, and its requested
size still lands in the run record and the golden manifest.

Pure stdlib. No GUI, no browser, no ambient tmux.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import termios
import time
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from .isolation import (
    _safe_rmtree_no_symlinks,
    _sweep_sandbox_survivors,
    build_isolated_env,
    build_seatbelt_profile,
    can_apply_seatbelt,
    default_scratch_root,
    terminate_process_group,
)

MAX_RAW_BYTES = 4 * 1024 * 1024
_READ_CHUNK = 65536
_WRITE_DEADLINE_S = 10.0

# Named keys understood by ``Key <NAME>`` in tape scripts.
KEY_SEQUENCES: Dict[str, str] = {
    "enter": "\r",
    "return": "\r",
    "tab": "\t",
    "esc": "\x1b",
    "escape": "\x1b",
    "space": " ",
    "backspace": "\x7f",
    "delete": "\x1b[3~",
    "insert": "\x1b[2~",
    "up": "\x1b[A",
    "down": "\x1b[B",
    "right": "\x1b[C",
    "left": "\x1b[D",
    "home": "\x1b[H",
    "end": "\x1b[F",
    "pgup": "\x1b[5~",
    "pgdn": "\x1b[6~",
    "shifttab": "\x1b[Z",
    "backtab": "\x1b[Z",
    "ctrl-c": "\x03",
    "ctrlc": "\x03",
    "ctrl-d": "\x04",
    "ctrld": "\x04",
    "ctrl-l": "\x0c",
    "ctrl-u": "\x15",
}


class PtyError(RuntimeError):
    """Base error for PTY transport failures."""


class PtyTimeout(PtyError):
    """Raised when a wait exceeds its deadline."""


def key_sequence(name: str) -> str:
    """Resolve a tape ``Key`` token to the bytes a terminal would send."""
    if name is None:
        raise ValueError("empty key name")
    token = str(name)
    lowered = token.lower()
    if lowered in KEY_SEQUENCES:
        return KEY_SEQUENCES[lowered]
    if len(token) == 1:
        return token
    raise ValueError(f"unknown key name: {token!r}")


def char_cells(ch: str) -> int:
    """Display cells one character occupies (0 for zero-width/combining).

    Widths are East-Asian-width based, not grapheme-cluster based: a ZWJ emoji
    sequence counts 2 cells per pictographic codepoint plus 0 for the joiner,
    where a modern terminal draws 2 for the whole cluster. That over-counts
    such text and shifts the cursor right; it is stable and deterministic, which
    is what goldens need, and no assertion in-tree depends on emoji layout.

    Controls (C0, DEL, C1) are 0 here, but ``VTScreen._put_char`` handles CR,
    LF, BS, TAB, BEL, and NUL before asking, so only stray controls that are
    neither printable nor placeable reach this branch.
    """
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


_WIDE_CONT = "\x00"  # placeholder occupying the second cell of a wide glyph
_DEFAULT_TAB = 8

_ESC_DESIGNATORS = frozenset("()*+-./#%")


@dataclass
class CursorState:
    row: int = 0
    col: int = 0
    visible: bool = True


class VTScreen:
    """Deterministic text-grid interpreter for a subset of ANSI/ECMA-48.

    Invariants:

    * Every live grid row is exactly ``cols`` cells long. ``resize()`` maintains
      that; the alt-screen snapshot does *not* (see ``_mode``).
    * A cell is either one visible character, a character with combining marks
      appended, ``" "``, or ``_WIDE_CONT`` (the placeholder for the second cell
      of a wide glyph). ``to_lines()`` drops the placeholders.
    * ``_pending`` holds only a prefix that could still become a valid escape
      sequence; incomplete tails are re-examined on the next ``feed()``. It has
      no length cap — see ``_escape_end``.

    Unsupported by design, and therefore invisible rather than approximated:
    scroll regions (DECSTBM), insert/delete line/character (IL/DL/ICH/DCH), and
    the ``ESC 7``/``ESC 8`` cursor save/restore pair. A program that scrolls
    with DECSTBM will be rendered as if it did not.
    """

    def __init__(self, cols: int = 80, rows: int = 24) -> None:
        if cols <= 0 or rows <= 0:
            raise ValueError(f"invalid screen geometry: {cols}x{rows}")
        self.cols = int(cols)
        self.rows = int(rows)
        self.grid: List[List[str]] = [self._blank_row() for _ in range(self.rows)]
        self.cursor = CursorState()
        self._saved: Optional[Tuple[List[List[str]], int, int, bool]] = None
        self._pending = ""

    # -- construction helpers -------------------------------------------------
    def _blank_row(self) -> List[str]:
        return [" "] * self.cols

    def _blank_grid(self) -> List[List[str]]:
        return [self._blank_row() for _ in range(self.rows)]

    # -- public API ----------------------------------------------------------
    def resize(self, cols: int, rows: int) -> None:
        """Resize the grid, preserving the top-left content and cursor position.

        Truncation, not reflow: cells past the new width and rows past the new
        height are discarded, new cells are blank, and the cursor is clamped.
        A real terminal may re-wrap its content (iTerm2 does); emulating that is
        out of scope because the child redraws from its own model anyway.

        The alt-screen snapshot in ``_saved`` is deliberately left at its old
        dimensions: it belongs to the other buffer. That is what makes a resize
        while the alternate screen is active restore a grid narrower than
        ``self.cols`` (documented in ``_mode``; reported as a defect, not fixed
        here).
        """
        if cols <= 0 or rows <= 0:
            raise ValueError(f"invalid screen geometry: {cols}x{rows}")
        old_cols = self.cols
        for row in self.grid:
            if cols < len(row):
                del row[cols:]
            else:
                row.extend([" "] * (cols - len(row)))
        while rows < len(self.grid):
            self.grid.pop()
        self.cols = int(cols)
        self.rows = int(rows)
        while rows > len(self.grid):
            self.grid.append(self._blank_row())
        if old_cols != self.cols:
            self.cursor.col = min(self.cursor.col, self.cols - 1)
        self.cursor.row = min(self.cursor.row, self.rows - 1)
        self.cursor.col = max(0, min(self.cursor.col, self.cols - 1))

    def feed(self, text: str) -> None:
        """Consume a chunk of terminal output; incomplete escapes are buffered.

        Chunk boundaries are invisible to the result: a CSI split across two
        ``feed()`` calls yields the same grid as one call. The trailing
        ``_pending`` slice is re-parsed next time, so a caller may write one
        byte at a time (the chunked-feed test does exactly that).
        """
        if not text:
            return
        self._pending += text
        pos = 0
        total = len(self._pending)
        while pos < total:
            ch = self._pending[pos]
            if ch == "\x1b":
                end = self._escape_end(self._pending, pos)
                if end is None:
                    break
                self._dispatch_escape(self._pending[pos:end])
                pos = end
                continue
            pos += 1
            self._put_char(ch)
        self._pending = self._pending[pos:]

    def to_lines(self) -> List[str]:
        """Screen rows with zero-width placeholders dropped and right padding removed."""
        out: List[str] = []
        for row in self.grid:
            out.append("".join(cell for cell in row if cell != _WIDE_CONT).rstrip())
        return out

    def to_text(self) -> str:
        """Normalized screen text: stripped lines, trailing blank rows removed.

        This is the string assertions and goldens see, so its normalization is
        part of the contract: per-line right padding is gone, trailing empty
        rows are gone, and there is no trailing newline (``tape.normalized_screen``
        adds exactly one). Consequently a literal ending in spaces, or one that
        spans a wrapped row boundary, can never match an ``ExpectScreen``.
        """
        lines = self.to_lines()
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def text(self) -> str:
        return self.to_text()

    def non_blank_lines(self) -> List[str]:
        return [line for line in self.to_lines() if line.strip()]

    # -- character placement -------------------------------------------------
    def _put_char(self, ch: str) -> None:
        """Place one printable character and advance the cursor.

        Auto-wrap happens *before* placement, so a glyph is never split across
        the right edge: a wide char that only half-fits moves to the next row
        first. After a glyph lands in the last cell the cursor sits at ``cols``
        (one past the edge, the "pending wrap" state); the next character
        notices that and wraps. A cursor at ``cols`` therefore behaves like a
        real terminal for writing, but differs for ``\\x1b[K`` and ``\\x1b[G``,
        which clamp instead of wrapping.

        Combining marks are appended to the previous cell rather than placed,
        which keeps ``to_lines()`` width-consistent with ``char_cells`` for
        NFD text. A mark with no preceding cell is dropped.
        """
        if ch == "\r":
            self.cursor.col = 0
            return
        if ch == "\n":
            self._linefeed()
            return
        if ch == "\b":
            self.cursor.col = max(0, self.cursor.col - 1)
            return
        if ch == "\t":
            nxt = ((self.cursor.col // _DEFAULT_TAB) + 1) * _DEFAULT_TAB
            self.cursor.col = min(self.cols - 1, nxt)
            return
        if ch == "\x07" or ch == "\x00":
            return
        width = char_cells(ch)
        if width == 0:
            # Combining mark: attach to the previous cell when one exists.
            if self.cursor.col > 0 and ord(ch) > 0x7F:
                row = self.grid[self.cursor.row]
                idx = self.cursor.col - 1
                if row[idx] == _WIDE_CONT and idx > 0:
                    idx -= 1
                prev = row[idx]
                row[idx] = prev + ch
            return
        if self.cursor.col >= self.cols or self.cursor.col + width > self.cols:
            self.cursor.col = 0
            self._linefeed()
        row = self.grid[self.cursor.row]
        row[self.cursor.col] = ch
        if width == 2 and self.cursor.col + 1 < self.cols:
            row[self.cursor.col + 1] = _WIDE_CONT
        # A full last cell leaves the cursor one past the edge (pending wrap).
        self.cursor.col = self.cursor.col + width

    def _linefeed(self) -> None:
        """Advance one row, scrolling the grid by discarding row 0.

        Scrollback does not exist: content that scrolls off the top is gone, so
        an assertion can only see the last ``rows`` lines. That is why tapes are
        written against the *current* frame (``WaitFor`` + ``ExpectScreen``)
        rather than against the whole session transcript.
        """
        if self.cursor.row + 1 >= self.rows:
            self.grid.pop(0)
            self.grid.append(self._blank_row())
        else:
            self.cursor.row += 1

    # -- escape handling -----------------------------------------------------
    @staticmethod
    def _escape_end(text: str, start: int) -> Optional[int]:
        """Index just past the escape sequence at ``start``; None when incomplete.

        Only enough grammar is recognised to find the end of a sequence: CSI
        parameter/intermediate bytes then one final byte; OSC/DCS/PM/APC/SOS up
        to BEL (OSC only) or ST ``ESC \\\\``; the two-byte ``ESC <designator>``
        forms and the single-character remainder.

        Returning ``None`` is the whole chunk-boundary story: the tail stays in
        ``_pending`` and is rescanned from its start next call. There is no cap
        on how long that tail may grow, and the rescan is O(n) per chunk, so a
        child that emits a never-terminated OSC (or a long CSI parameter run)
        accumulates memory and quadratic work outside ``MAX_RAW_BYTES``. Test
        corpora include exactly such payloads.
        """
        n = len(text)
        if start + 1 >= n:
            return None
        nxt = text[start + 1]
        if nxt == "[":
            j = start + 2
            while j < n and ("\x30" <= text[j] <= "\x3f" or "\x20" <= text[j] <= "\x2f"):
                j += 1
            if j >= n:
                return None
            return j + 1
        if nxt in ("]", "P", "X", "^", "_"):
            # OSC/DCS/PM/APC/SOS: consume to BEL (OSC only) or ST (ESC \\).
            j = start + 2
            while j < n:
                if text[j] == "\x07" and nxt == "]":
                    return j + 1
                if text[j] == "\x1b" and j + 1 < n and text[j + 1] == "\\":
                    return j + 2
                j += 1
            return None
        if nxt in _ESC_DESIGNATORS:
            if start + 2 >= n:
                return None
            return start + 3
        return start + 2

    def _dispatch_escape(self, seq: str) -> None:
        """Apply one complete escape sequence; anything unrecognised is ignored.

        Non-CSI escapes (``ESC 7``, ``ESC M``, charset designators) have already
        been consumed by ``_escape_end`` and do nothing here. That is the
        deliberate failure mode: an unsupported sequence never leaves bytes in
        a cell, it just has no effect.
        """
        if not seq or seq[0] != "\x1b" or len(seq) < 2:
            return
        if seq[1] != "[":
            return
        final = seq[-1]
        body = seq[2:-1]
        private = body.startswith("?")
        params_raw = body[1:] if private else body
        params = self._params(params_raw)
        if private:
            self._mode(final, params)
            return
        if final in ("H", "f"):
            row = (params[0] if params and params[0] else 1) - 1
            col = (params[1] if len(params) > 1 and params[1] else 1) - 1
            self.cursor.row = max(0, min(self.rows - 1, row))
            self.cursor.col = max(0, min(self.cols - 1, col))
        elif final == "A":
            self.cursor.row = max(0, self.cursor.row - (params[0] or 1))
        elif final == "B":
            self.cursor.row = min(self.rows - 1, self.cursor.row + (params[0] or 1))
        elif final == "C":
            self.cursor.col = min(self.cols - 1, self.cursor.col + (params[0] or 1))
        elif final == "D":
            self.cursor.col = max(0, self.cursor.col - (params[0] or 1))
        elif final == "E":
            self.cursor.row = min(self.rows - 1, self.cursor.row + (params[0] or 1))
            self.cursor.col = 0
        elif final == "F":
            self.cursor.row = max(0, self.cursor.row - (params[0] or 1))
            self.cursor.col = 0
        elif final == "G":
            self.cursor.col = max(0, min(self.cols - 1, (params[0] or 1) - 1))
        elif final == "d":
            self.cursor.row = max(0, min(self.rows - 1, (params[0] or 1) - 1))
        elif final == "J":
            self._erase_display(params[0] if params else 0)
        elif final == "K":
            self._erase_line(params[0] if params else 0)
        elif final == "m":
            return  # SGR never affects the text grid
        elif final in ("h", "l"):
            self._mode(final, params)

    @staticmethod
    def _params(raw: str) -> List[int]:
        if not raw:
            return []
        out: List[int] = []
        for piece in raw.split(";"):
            piece = piece.strip()
            if not piece:
                out.append(0)
                continue
            try:
                out.append(int(piece))
            except ValueError:
                out.append(0)
        return out

    def _mode(self, final: str, params: Sequence[int]) -> None:
        """Handle DEC private modes: alternate screen (1049/1047) and cursor (25).

        ``1049`` saves the main grid and blanks the alternate one; ``1047`` saves
        and blanks but leaves the cursor where it was. Both restore the saved
        grid wholesale, including its *width*.

        Hazard, not a design: ``resize()`` does not resize ``_saved``, so
        ``alt-screen → resize wider → leave alt-screen`` restores rows that are
        narrower than ``self.cols``. The next character (or ``EL``/``ED`` past
        the old width) indexes a short row and raises ``IndexError``, which
        escapes ``replay_tape`` because it is neither a ``TapeError`` nor an
        ``OSError``. Reported with a minimal repro; left unchanged here.

        Every other private mode (bracketed paste, mouse, focus events, 2026
        sync output) is ignored: those change how a terminal *talks to* the
        program, not which cells exist.
        """
        on = final == "h"
        for mode in params:
            if mode == 1049:
                if on:
                    if self._saved is None:
                        self._saved = ([row[:] for row in self.grid], self.cursor.row, self.cursor.col, self.cursor.visible)
                    self.grid = self._blank_grid()
                    self.cursor.row = 0
                    self.cursor.col = 0
                elif self._saved is not None:
                    grid, row, col, visible = self._saved
                    self.grid = grid
                    self.cursor.row = min(row, self.rows - 1)
                    self.cursor.col = min(col, self.cols - 1)
                    self.cursor.visible = visible
                    self._saved = None
            elif mode == 1047:
                if on and self._saved is None:
                    self._saved = ([r[:] for r in self.grid], self.cursor.row, self.cursor.col, self.cursor.visible)
                    self.grid = self._blank_grid()
                elif not on and self._saved is not None:
                    grid, row, col, visible = self._saved
                    self.grid = grid
                    self.cursor.row = min(row, self.rows - 1)
                    self.cursor.col = min(col, self.cols - 1)
                    self.cursor.visible = visible
                    self._saved = None
            elif mode == 25:
                self.cursor.visible = on

    def _erase_line(self, mode: int) -> None:
        """EL: blank from the cursor (0), to the cursor (1), or the row (2).

        Assumes the row is ``cols`` long, which ``resize()`` guarantees for the
        live grid and the alt-screen restore violates (see ``_mode``).
        """
        row = self.grid[self.cursor.row]
        if mode == 0:
            for idx in range(self.cursor.col, self.cols):
                row[idx] = " "
        elif mode == 1:
            for idx in range(0, min(self.cursor.col + 1, self.cols)):
                row[idx] = " "
        elif mode == 2:
            for idx in range(self.cols):
                row[idx] = " "

    def _erase_display(self, mode: int) -> None:
        if mode == 0:
            self._erase_line(0)
            for r in range(self.cursor.row + 1, self.rows):
                self.grid[r] = self._blank_row()
        elif mode == 1:
            self._erase_line(1)
            for r in range(0, self.cursor.row):
                self.grid[r] = self._blank_row()
        elif mode in (2, 3):
            self.grid = self._blank_grid()


class PtySession:
    """One isolated child process attached to a real PTY.

    Lifecycle: construct (sandbox, env, optional Seatbelt, ``openpty``, spawn) →
    ``pump``/``wait_*``/``write``/``resize`` (any number, any order) → ``close``
    exactly once. Nothing runs in the background and nothing is scheduled: a
    session with no ``pump`` calls produces no screen updates, which is why the
    wait helpers are the only way to make progress.

    Invariants established by ``__init__``: ``pgid == proc.pid`` (the child is a
    session leader), the master fd is non-blocking, the slave fd is closed in
    the parent, ``sandbox_dir`` is ``0700`` and owned by this object, and the
    initial window size was set *before* ``exec`` so the child's first
    ``TIOCGWINSZ`` already sees it.

    ``close()`` is the teardown contract: SIGTERM the group, bounded wait,
    SIGKILL if needed, drain remaining bytes (so a program's exit sequence still
    reaches the screen), close the master, sweep survivors when the child failed,
    then delete the sandbox if nothing leaked. It is not re-entrant on the error
    path — see the comment on the early return.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        cols: int = 80,
        rows: int = 24,
        run_id: str = "",
        nonce: str = "",
        env_overrides: Optional[Dict[str, str]] = None,
        base_env: Optional[Dict[str, str]] = None,
        use_seatbelt: bool = True,
        scratch_root: Optional[Path] = None,
        sandbox_dir: Optional[Path] = None,
        max_raw_bytes: int = MAX_RAW_BYTES,
    ) -> None:
        if not argv or any(not isinstance(a, str) or "\x00" in a for a in argv):
            raise ValueError(f"invalid argv: {argv!r}")
        self.argv = list(argv)
        self.cwd = Path(cwd).expanduser().resolve(strict=True)
        self.cols = int(cols)
        self.rows = int(rows)
        self.max_raw_bytes = int(max_raw_bytes)
        self.seatbelt_state = "disabled"
        self.truncated = False
        self.raw_bytes = bytearray()
        self.text_parts: List[str] = []
        self.text_len = 0
        self.screen = VTScreen(self.cols, self.rows)
        self._decoder = _IncrementalUtf8()
        self._eof = False
        self._closed = False
        self.survivors_reaped = 0
        self.survivors_leaked = 0
        nonce = nonce or os.urandom(6).hex()

        if sandbox_dir is None:
            root = (scratch_root.expanduser().resolve() if scratch_root else default_scratch_root())
            root.mkdir(parents=True, exist_ok=True)
            sandbox_dir = root / f"tbpty-{os.getpid()}-{nonce[:8]}"
            sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_dir = Path(sandbox_dir)
        os.chmod(str(self.sandbox_dir), 0o700)

        self.env = build_isolated_env(
            self.sandbox_dir,
            run_id=run_id or f"tape_{os.getpid()}",
            nonce=nonce,
            env_overrides=env_overrides,
            base_env=base_env,
        )

        effective_argv = list(self.argv)
        if use_seatbelt:
            import shutil

            sb_bin = shutil.which("sandbox-exec")
            if sb_bin is None:
                self.seatbelt_state = "unavailable"
            elif can_apply_seatbelt():
                profile = build_seatbelt_profile(sandbox_dir=self.sandbox_dir, cwd=self.cwd)
                effective_argv = [sb_bin, "-p", profile, *effective_argv]
                self.env["TESTBENCH_SEATBELT_ACTIVE"] = "1"
                self.seatbelt_state = "active"
            else:
                self.env["TESTBENCH_SEATBELT_ACTIVE"] = "1"
                self.seatbelt_state = "inherited_outer"

        self.master, slave = pty.openpty()
        try:
            # Window size goes on the slave before exec, so the child inherits a
            # terminal that is already the right shape; there is no race with a
            # first paint at 80x24. The master is non-blocking because every
            # read is guarded by select().
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", self.rows, self.cols, 0, 0))
            os.set_blocking(self.master, False)
            self.proc = subprocess.Popen(
                effective_argv,
                cwd=str(self.cwd),
                env=self.env,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                start_new_session=True,
                close_fds=True,
            )
        except Exception:
            try:
                os.close(slave)
            except OSError:
                pass
            try:
                os.close(self.master)
            except OSError:
                pass
            _safe_rmtree_no_symlinks(self.sandbox_dir)
            raise
        os.close(slave)
        # No termios setup here on purpose: echo, canonical mode, and the
        # window-change signal all belong to the program under test, exactly as
        # on a real terminal. A program that never calls tcsetattr/tty.setraw
        # will echo its own input back into the captured stream, and that echo
        # becomes part of the screen a tape asserts on.
        self.pgid = self.proc.pid
        self._last_output_at = time.monotonic()
        self._started_at = time.monotonic()

    # -- reads ---------------------------------------------------------------
    def _absorb(self, chunk: bytes) -> None:
        """Append raw bytes, decoded text, and screen cells; enforce the caps.

        Two independent windows, both front-truncated:

        * ``raw_bytes`` is bounded in *bytes* (``max_raw_bytes``), which is what
          ``ExpectRaw`` sees.
        * ``text_parts`` is bounded by the same number but counted in
          *characters* (``text_len``), so a UTF-8-heavy stream can hold up to
          four times the intended bytes. Harmless for substring waits; it is a
          cap on slices, not a memory guarantee.

        The screen is never truncated. A tape that waits on a needle, then
        asserts on the frame, must not lose the frame just because the program
        printed more than 4 MiB of scroll.
        """
        self.raw_bytes.extend(chunk)
        if len(self.raw_bytes) > self.max_raw_bytes:
            del self.raw_bytes[: len(self.raw_bytes) - self.max_raw_bytes]
            self.truncated = True
        decoded = self._decoder.decode(chunk)
        if not decoded:
            return
        self.text_parts.append(decoded)
        self.text_len += len(decoded)
        if self.text_len > self.max_raw_bytes:
            joined = "".join(self.text_parts)
            self.text_parts = [joined[-self.max_raw_bytes :]]
            self.text_len = len(self.text_parts[0])
            self.truncated = True
        self.screen.feed(decoded)

    def pump(self, timeout_s: float = 0.05) -> int:
        """Read whatever arrives within ``timeout_s`` without blocking the child.

        Returns the byte count absorbed; 0 means nothing arrived. EOF is sticky
        (``_eof``) but not terminal: later calls keep probing and returning 0,
        so a wait loop keeps honouring its own deadline instead of spinning.
        ``_last_output_at`` is refreshed only when bytes actually arrived, which
        is the clock ``wait_quiet`` reads.
        """
        if self._closed:
            return 0
        deadline = time.monotonic() + max(0.0, timeout_s)
        received = 0
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                ready, _, _ = select.select([self.master], [], [], remaining)
            except (OSError, ValueError):
                self._eof = True
                break
            if not ready:
                break
            try:
                chunk = os.read(self.master, _READ_CHUNK)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    continue
                self._eof = True
                break
            if not chunk:
                self._eof = True
                break
            received += len(chunk)
            self._absorb(chunk)
            if time.monotonic() >= deadline:
                break
        if received:
            self._last_output_at = time.monotonic()
        return received

    @property
    def raw_text(self) -> str:
        """The byte window decoded wholesale (may contain U+FFFD at the cut)."""
        return bytes(self.raw_bytes).decode("utf-8", errors="replace")

    @property
    def text(self) -> str:
        """The incrementally decoded stream window; chunk boundaries respected."""
        return "".join(self.text_parts)

    def screen_text(self) -> str:
        """Current VT screen: right-stripped rows, no trailing blank rows."""
        return self.screen.to_text()

    # -- writes --------------------------------------------------------------
    def write(self, data: str, *, deadline_s: float = _WRITE_DEADLINE_S) -> None:
        """Write text to the child, retrying while the PTY reports backpressure.

        The master is non-blocking, so ``EAGAIN`` means the kernel buffer is
        full (the child is not reading). Retry until ``deadline_s``, then raise
        ``PtyTimeout`` rather than blocking the whole run on a wedged program.
        Partial writes are normal: only the acknowledged prefix is retried.
        """
        if self._closed:
            raise PtyError("pty session is closed")
        payload = data.encode("utf-8")
        sent = 0
        deadline = time.monotonic() + deadline_s
        while sent < len(payload):
            try:
                written = os.write(self.master, payload[sent:])
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    raise PtyError(f"pty write failed: {exc}") from exc
                written = 0
            sent += written
            if written == 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PtyTimeout(f"pty write blocked for {deadline_s}s")
                try:
                    select.select([], [self.master], [], min(0.2, remaining))
                except (OSError, ValueError):
                    raise PtyError("pty master became unusable during write")

    def send_text(self, text: str) -> None:
        self.write(text)

    def send_key(self, name: str) -> None:
        self.write(key_sequence(name))

    # -- geometry ------------------------------------------------------------
    def resize(self, cols: int, rows: int) -> None:
        """Apply a new window size to the kernel, the child, and the interpreter.

        Order matters: record the requested size, then ``TIOCSWINSZ`` on the
        master (which updates the pty pair), then ``SIGWINCH`` to the process
        group, then resize the local grid. Bytes the child emits in response are
        read on the next ``pump`` and are interpreted at the *new* geometry,
        which is what the child painted them for.

        Linux already signals the foreground group from ``TIOCSWINSZ``; the
        explicit ``killpg`` is for macOS and is a harmless duplicate elsewhere.
        A dead child (``ESRCH``) is not an error here — teardown will report the
        exit status.

        Note the failure mode: ``self.cols/rows`` are updated before the ioctl,
        so an ``OSError`` from ``TIOCSWINSZ`` leaves the object advertising a
        size the kernel never accepted (and ``self.screen`` unresized). Only a
        closed master should do that in practice; it is reported, not fixed.
        """
        if cols <= 0 or rows <= 0:
            raise ValueError(f"invalid geometry: {cols}x{rows}")
        self.cols = int(cols)
        self.rows = int(rows)
        fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack("HHHH", self.rows, self.cols, 0, 0))
        try:
            os.killpg(self.pgid, signal.SIGWINCH)
        except OSError:
            pass
        self.screen.resize(self.cols, self.rows)

    # -- waits ---------------------------------------------------------------
    def _contains(self, needle: str) -> bool:
        return needle in self.text or needle in self.screen.to_text()

    def wait_for(self, needle: str, timeout_ms: int = 5000) -> str:
        """Block until ``needle`` appears on screen or in the decoded stream.

        Two domains on purpose. The screen matches what a user would read (an
        escape-split word still matches after interpretation), while the stream
        matches control sequences a tape may want to prove were emitted. A line
        that wrapped is only visible in the stream, not as one screen substring.

        The stream window is truncated, so a needle that scrolled past the last
        ``max_raw_bytes`` will never match even though it appeared earlier; the
        screen copy is unaffected. The timeout message carries the last 600
        characters of the stream — enough to see what the program was doing.
        """
        if not needle:
            raise ValueError("wait_for requires a non-empty needle")
        deadline = time.monotonic() + max(0.0, timeout_ms / 1000.0)
        while True:
            if self._contains(needle):
                return self.text
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                tail = self.text[-600:]
                raise PtyTimeout(
                    f"timed out after {timeout_ms}ms waiting for {needle!r}; tail={tail!r}"
                )
            self.pump(min(0.05, remaining))

    def wait_quiet(self, quiet_ms: int = 250, timeout_ms: int = 10000) -> float:
        """Block until the child has produced no output for ``quiet_ms``.

        Settling is inferred from two signals: the padded stream length stopped
        changing, and ``_last_output_at`` is older than the quiet window. The
        length check is only a fast path (the stream truncates, so the length
        can plateau while output continues); the timestamp is authoritative.

        Returns the observed idle time in seconds. It does **not** raise on
        timeout: if the deadline passes while the child is still printing, the
        current idle value is returned anyway. Callers therefore cannot
        distinguish "settled" from "gave up", which matters for a golden capture
        of an animating screen (a mid-frame baseline, reported as a defect).
        """
        deadline = time.monotonic() + max(0.0, timeout_ms / 1000.0)
        quiet_s = max(0.0, quiet_ms / 1000.0)
        last_len = -1
        while True:
            self.pump(min(0.05, max(0.001, deadline - time.monotonic())))
            current = len(self.raw_bytes) + self.text_len
            if current != last_len:
                last_len = current
                continue
            idle = time.monotonic() - self._last_output_at
            if idle >= quiet_s:
                return idle
            if time.monotonic() >= deadline:
                return idle

    def is_running(self) -> bool:
        return self.proc.poll() is None

    # -- teardown ------------------------------------------------------------
    def close(self, *, grace_s: float = 2.0) -> int:
        """SIGTERM then SIGKILL the process group, drain, sweep, and clean up.

        The drain loop after ``wait`` is load-bearing: a program that restores
        the main screen and shows the cursor on exit only does so after its
        leader exits, and ``tape.replay_tape`` reads the screen *after* this
        returns. It is bounded (50 x 10 ms) so a still-flooding writer cannot
        stall teardown.

        Survivors are swept only when the child exited non-zero. On a zero exit
        the sweep is skipped, ``survivors_leaked`` stays 0, and the sandbox
        directory is deleted unconditionally — a daemon left behind by a clean
        exit is not detected here.

        Early return: ``_closed`` is set before the kill/wait sequence, and
        ``exit_code`` is only assigned at the very end. If ``proc.wait`` raises
        (an unkillable process, twice), a second ``close()`` finds ``_closed``
        true and ``exit_code`` unset; the single-call
        contract is what callers actually use.
        """
        if self._closed:
            return int(self.exit_code or 0)
        self._closed = True
        self.pump(0.05)
        if self.proc.poll() is None:
            try:
                os.killpg(self.pgid, signal.SIGTERM)
            except OSError:
                pass
            deadline = time.monotonic() + max(0.05, grace_s)
            while time.monotonic() < deadline and self.proc.poll() is None:
                self.pump(0.05)
            if self.proc.poll() is None:
                try:
                    os.killpg(self.pgid, signal.SIGKILL)
                except OSError:
                    pass
        try:
            rc = self.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            terminate_process_group(self.pgid, grace_s=0.2)
            rc = self.proc.wait(timeout=2.0)
        # Drain whatever the child left in the kernel buffer (exit sequences).
        for _ in range(50):
            if self.pump(0.01) == 0:
                break
        try:
            os.close(self.master)
        except OSError:
            pass
        if rc != 0:
            reaped, leaked = _sweep_sandbox_survivors(self.sandbox_dir)
            self.survivors_reaped += reaped
            self.survivors_leaked += leaked
        if self.survivors_leaked == 0:
            _safe_rmtree_no_symlinks(self.sandbox_dir)
        self.exit_code = rc
        return rc

    def terminate(self) -> int:
        return self.close()

    def __enter__(self) -> "PtySession":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def isolation_meta(self) -> Dict[str, object]:
        return {
            "seatbelt": self.seatbelt_state,
            "pty": True,
            "cols": self.cols,
            "rows": self.rows,
            "truncated": self.truncated,
            "reaped_survivors": self.survivors_reaped,
            "leaked_children": self.survivors_leaked,
        }


class _IncrementalUtf8:
    """UTF-8 decoder that never splits a multi-byte codepoint across chunks.

    Needed because ``os.read`` splits at arbitrary byte offsets: a naive
    ``chunk.decode()`` would emit one U+FFFD per boundary and corrupt both the
    stream window and the screen. ``errors="replace"`` still applies to bytes
    that are genuinely invalid.

    ``final=True`` is never passed, so an incomplete codepoint at true EOF stays
    buffered and is dropped with the session. That is the right trade: an
    assertion should never see a half-character.
    """

    def __init__(self) -> None:
        import codecs

        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def decode(self, chunk: bytes) -> str:
        return self._decoder.decode(chunk, final=False)
