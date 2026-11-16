"""Reusable real-PTY sessions plus a deterministic VT screen interpreter.

``PtySession`` owns one ``pty.openpty()`` master/slave pair, a child process
group started from an isolated environment (:func:`isolation.build_isolated_env`,
optionally wrapped in Seatbelt), non-blocking reads/writes with backpressure
retry, a bounded raw capture buffer, ``TIOCSWINSZ`` + ``SIGWINCH`` resize, and
process-group teardown that reaps survivors.

``VTScreen`` consumes the child's byte stream and reproduces the text grid a
terminal would show: cursor motion, erase, alt-screen save/restore, cursor
visibility, wide-cell widths, and escape-sequence consumption so no control
bytes ever leak into screen cells.

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
    """Display cells one character occupies (0 for zero-width/combining)."""
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
    """Deterministic text-grid interpreter for a subset of ANSI/ECMA-48."""

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
        """Resize the grid, preserving the top-left content and cursor position."""
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
        """Consume a chunk of terminal output; incomplete escapes are buffered."""
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
        """Normalized screen text: stripped lines, trailing blank rows removed."""
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
        if self.cursor.row + 1 >= self.rows:
            self.grid.pop(0)
            self.grid.append(self._blank_row())
        else:
            self.cursor.row += 1

    # -- escape handling -----------------------------------------------------
    @staticmethod
    def _escape_end(text: str, start: int) -> Optional[int]:
        """Index just past the escape sequence at ``start``; None when incomplete."""
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
    """One isolated child process attached to a real PTY."""

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
        self.pgid = self.proc.pid
        self._last_output_at = time.monotonic()
        self._started_at = time.monotonic()

    # -- reads ---------------------------------------------------------------
    def _absorb(self, chunk: bytes) -> None:
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
        """Read whatever arrives within ``timeout_s`` without blocking the child."""
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
        return bytes(self.raw_bytes).decode("utf-8", errors="replace")

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    def screen_text(self) -> str:
        return self.screen.to_text()

    # -- writes --------------------------------------------------------------
    def write(self, data: str, *, deadline_s: float = _WRITE_DEADLINE_S) -> None:
        """Write text to the child, retrying while the PTY reports backpressure."""
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
        """Block until ``needle`` appears on screen or in the decoded stream."""
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
        """Block until the child has produced no output for ``quiet_ms``."""
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
        """SIGTERM then SIGKILL the process group, drain, sweep, and clean up."""
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
    """UTF-8 decoder that never splits a multi-byte codepoint across chunks."""

    def __init__(self) -> None:
        import codecs

        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def decode(self, chunk: bytes) -> str:
        return self._decoder.decode(chunk, final=False)
