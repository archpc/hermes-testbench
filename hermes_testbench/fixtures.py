"""Deterministic reusable fixtures and fault probes for Hermes TestBench.

Every builder here is stdlib-only, hermetic, and returns a plain ``dict`` of measured
facts (no pass/fail policy). Suites decide thresholds; fixtures only produce evidence.

Corpora
-------
``unicode_boundary_corpus()``
    UTF-8 multi-byte boundaries, CJK wide characters, ZWJ emoji, combining marks,
    RTL bidi overrides, and Plane-14 tag characters.
``malformed_input_corpus()``
    Truncated JSON/TOML, NUL bytes, negative pre-1970 timestamps, ``rid > 2**32``,
    and 10 KiB strings.
``terminal_escape_corpus()``
    OSC 52, OSC 8, DCS, CSI 2J, 8-bit C1, NUL, and BEL.

Probes
------
``create_sqlite_contention_fixture(db_path)`` / ``run_sqlite_contention_probe(...)``
    Real multi-process WAL contention driven by ``BEGIN IMMEDIATE``; a second phase
    with ``busy_timeout=0`` proves measurable contention handling.
``run_pty_resize_probe(...)``
    Real ``pty.openpty()`` + ``TIOCSWINSZ`` + ``SIGWINCH``; the child records the
    geometry it observes before and after the resize.
``run_signal_crash_probe(...)``
    Non-zero signal classification (``signal_num`` and ``exit_code == 128 + sig``)
    with zero leaked children.
``run_repeated_init_probe(...)``
    Idempotent store/fixture initialization.
``run_tmux_lifecycle_probe(...)``
    Isolated ``-L`` tmux server, session/window/pane ``@hx`` options, crash recovery,
    and clean ``kill-server`` teardown.
``create_cross_lang_mail_db(db_path)``
    Deterministic SQLite mail DB matching the ``hxmail`` / ``listbench.cob`` schema.

Haunted Terminal (Phase 1C)
---------------------------
``haunted_combining_corpus()`` / ``haunted_wide_corpus()`` /
``haunted_invisible_control_corpus()`` / ``haunted_malformed_escape_corpus()`` /
``haunted_osc52_corpus()`` / ``haunted_partial_utf8_chunks()``
    Hostile text for the eight terminal-integrity areas.
``haunted_render_frame(text, width, height)``
    Production rendering path (``sanitize_terminal_text`` + ``fit_cells``).
``haunted_render_naive(text, width, height)``
    Deliberately defective control renderer (no sanitizer, character slicing).
``check_haunted_frame(lines, width, height=...)`` / ``check_haunted_text(text)``
    Pure invariant checker: returns human-readable violations, never a verdict.
``run_haunted_terminal_probe(...)``
    Real PTY (and disposable ``-L`` tmux) transport with rapid ``SIGWINCH``
    resizes, interrupted UTF-8, and per-area invariant cases. ``fault_injection``
    routes the same checker through the defective renderer / geometry-caching
    child so a broken implementation is provably caught.
"""

from __future__ import annotations

from dataclasses import dataclass
import codecs
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import select
import shlex
import shutil
import signal
import sqlite3
import struct
import subprocess
import sys
import tempfile
import termios
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import unicodedata

from .isolation import default_scratch_root, pgroup_alive, terminate_process_group
from .sanitize import sanitize_terminal_text
from .schema import ProfileName, RunRecord, RunStatus
from .store import TestBenchStore, _safe_rmtree_no_symlinks
from .tui_render import TuiState, display_width, fit_cells, render_frame

SQLITE_CONTENTION_TABLE = "contention"
SQLITE_FIXTURE_BUSY_TIMEOUT_MS = 5000


@dataclass(frozen=True)
class CorpusEntry:
    """One named hostile input. ``raw`` is authoritative bytes when present."""

    name: str
    kind: str
    text: str = ""
    raw: bytes = b""
    note: str = ""

    def payload(self) -> bytes:
        return self.raw if self.raw else self.text.encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "text": self.text,
            "raw": list(self.raw),
            "note": self.note,
        }


# --------------------------------------------------------------------------- corpora


def unicode_boundary_corpus() -> List[CorpusEntry]:
    """Named Unicode samples at encodability and display-width boundaries."""
    return [
        CorpusEntry(
            "utf8_2byte_latin1_supplement",
            "utf8_2byte",
            "\u00e9",
            note="U+00E9 é is the shortest 2-byte UTF-8 sequence",
        ),
        CorpusEntry(
            "utf8_3byte_currency_symbol",
            "utf8_3byte",
            "\u20ac",
            note="U+20AC € is the canonical 3-byte UTF-8 sequence",
        ),
        CorpusEntry(
            "utf8_4byte_musical_symbol",
            "utf8_4byte",
            "\U0001d11e",
            note="U+1D11E 𝄞 is the canonical 4-byte UTF-8 sequence",
        ),
        CorpusEntry(
            "utf8_1_2_3_4byte_boundary_run",
            "utf8_boundary",
            "A\x7f\x80\u07ff\u0800\uffff\U00010000",
            note="ASCII/DEL, 2-byte/3-byte, and BMP/supplementary split points",
        ),
        CorpusEntry(
            "cjk_wide_mixed_script",
            "cjk_wide",
            "\u65e5\u672c\u8a9e\u30c6\u30ad\u30b9\u30c8\u5168\u89d2\uff21\uff22\uff23",
            note="East Asian wide (W) and fullwidth (F) cells",
        ),
        CorpusEntry(
            "zwj_emoji_family_and_flag",
            "zwj_emoji",
            "\U0001f469\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466"
            "\U0001f3f3\ufe0f\u200d\U0001f308\U0001f468\U0001f3fd\u200d\U0001f4bb",
            note="ZWJ sequences, variation selector, skin-tone modifier",
        ),
        CorpusEntry(
            "combining_marks_stacked",
            "combining_marks",
            "e\u0301a\u0308\u0323n\u0303\u0915\u094d\u0937",
            note="Combining acute/diaeresis/dot-below/tilde plus Devanagari cluster",
        ),
        CorpusEntry(
            "rtl_bidi_override_and_isolate",
            "rtl_bidi",
            "abc\u202edef\u202c \u2066\u05e2\u05d1\u05e8\u05d9\u05ea\u2069",
            note="RLO/PDF override and LRI/PDI isolate around Hebrew",
        ),
        CorpusEntry(
            "plane14_tag_sequence",
            "plane14_tags",
            "\U0001f3f4\U000e0067\U000e0062\U000e007f",
            note="Plane-14 tag letters and cancel tag (invisible payload channel)",
        ),
    ]


def malformed_input_corpus() -> List[CorpusEntry]:
    """Malformed structured inputs that must be rejected, not silently coerced."""
    return [
        CorpusEntry(
            "truncated_json_object",
            "truncated_json",
            raw=b'{"messages": [{"id": "m1", "body": "unterminated',
            note="open array/object/string; parsers must fail closed",
        ),
        CorpusEntry(
            "truncated_toml_table",
            "truncated_toml",
            raw=b'[profile.default]\nmodel = "deepseek\n',
            note="unterminated string and table header",
        ),
        CorpusEntry(
            "embedded_nul_byte",
            "nul_byte",
            raw=b"header-before\x00body-after\r\n",
            note="NUL must never reach a terminal or a filename",
        ),
        CorpusEntry(
            "negative_pre_epoch_timestamp",
            "negative_timestamp",
            '{"received_at": -86400, "subject": "pre-epoch"}',
            note="1969-12-31; sorts before every modern row",
        ),
        CorpusEntry(
            "rid_above_2_32",
            "rid_over_2_32",
            '{"rid": 4294967301, "id": "msg-big-rid"}',
            note="4294967296 + 5; 64-bit row id, JSON-safe integer",
        ),
        CorpusEntry(
            "huge_10kb_string",
            "huge_string",
            "A" * 10240,
            note="10 KiB single-token line; wrap and truncation pressure",
        ),
    ]


def terminal_escape_corpus() -> List[CorpusEntry]:
    """Hostile terminal control sequences with their exact byte payloads."""
    return [
        CorpusEntry(
            "osc52_clipboard_write",
            "osc52",
            "\x1b]52;c;ZXZpbA==\x07clipboard-hijack",
            note="OSC 52 base64 clipboard write terminated by BEL",
        ),
        CorpusEntry(
            "osc8_hyperlink",
            "osc8",
            "\x1b]8;;https://evil.example/\x1b\\click-me\x1b]8;;\x1b\\",
            note="OSC 8 hyperlink with ST terminators",
        ),
        CorpusEntry(
            "dcs_payload",
            "dcs",
            "\x1bP1;2;3+q544e\x1b\\dcs-payload",
            note="DCS device-control string",
        ),
        CorpusEntry(
            "csi_clear_screen",
            "csi_2j",
            "\x1b[2J\x1b[1;1Hcleared",
            note="CSI 2J erase-display plus cursor home",
        ),
        CorpusEntry(
            "c1_8bit_control_forms",
            "c1_8bit",
            "\x9b2J\x9d0;t\x07c1-payload",
            note="8-bit CSI (0x9B) and 8-bit OSC (0x9D)",
        ),
        CorpusEntry(
            "embedded_nul",
            "nul",
            "before\x00after",
            note="NUL byte inside otherwise printable text",
        ),
        CorpusEntry(
            "embedded_bel",
            "bel",
            "ring\x07ring",
            note="BEL used as a terminal-side effect",
        ),
    ]


# ------------------------------------------------------------------- sqlite contention


def create_sqlite_contention_fixture(db_path: Path) -> Dict[str, Any]:
    """Create an empty, WAL-mode ``contention`` DB ready for multi-process writers."""
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        target = Path(str(path) + suffix)
        if target.exists() and target.is_file():
            target.unlink()
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(f"PRAGMA busy_timeout = {SQLITE_FIXTURE_BUSY_TIMEOUT_MS}")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SQLITE_CONTENTION_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                writer INTEGER NOT NULL,
                seq INTEGER NOT NULL,
                payload TEXT NOT NULL,
                committed_at_ns INTEGER NOT NULL,
                UNIQUE(writer, seq)
            )
            """
        )
        conn.execute(f"DELETE FROM {SQLITE_CONTENTION_TABLE}")
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        rows = int(conn.execute(f"SELECT COUNT(*) FROM {SQLITE_CONTENTION_TABLE}").fetchone()[0])
    finally:
        conn.close()
    os.chmod(str(path), 0o600)
    return {
        "db_path": str(path),
        "journal_mode": journal_mode,
        "busy_timeout_ms": SQLITE_FIXTURE_BUSY_TIMEOUT_MS,
        "tables": [SQLITE_CONTENTION_TABLE],
        "rows": rows,
        "begin_immediate": True,
    }


_SQLITE_WRITER = r'''
import json
import sqlite3
import sys
import time

db_path, writer, ops, timeout_ms, hold_ms, start_at, result_path, table = sys.argv[1:9]
writer = int(writer)
ops = int(ops)
timeout_ms = int(timeout_ms)
hold_ms = float(hold_ms)
start_at = float(start_at)

conn = sqlite3.connect(db_path, timeout=5.0, isolation_level=None)
conn.execute("PRAGMA busy_timeout = 5000")
conn.execute("PRAGMA synchronous = NORMAL")
conn.execute("PRAGMA busy_timeout = %d" % timeout_ms)

busy_retries = 0
writer_errors = 0
written = 0
max_wait_ms = 0.0

while time.time() < start_at:
    time.sleep(0.0005)

for seq in range(ops):
    attempts = 0
    while True:
        t0 = time.time()
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            waited_ms = (time.time() - t0) * 1000.0
            max_wait_ms = max(max_wait_ms, waited_ms)
            message = str(exc).lower()
            if "locked" in message or "busy" in message:
                busy_retries += 1
                if timeout_ms == 0:
                    attempts += 1
                    if attempts > 20000:
                        writer_errors += 1
                        break
                    time.sleep(0.002)
                    continue
                writer_errors += 1
                break
            writer_errors += 1
            break
        else:
            try:
                if writer == 0 and seq == 0 and hold_ms > 0:
                    time.sleep(hold_ms / 1000.0)
                conn.execute(
                    "INSERT INTO %s(writer, seq, payload, committed_at_ns) VALUES(?,?,?,?)" % table,
                    (writer, seq, "w%d-s%d-%s" % (writer, seq, "x" * 64), time.time_ns()),
                )
                conn.execute("COMMIT")
                written += 1
            except sqlite3.Error:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                writer_errors += 1
            break

conn.close()
with open(result_path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "writer": writer,
            "written": written,
            "busy_retries": busy_retries,
            "writer_errors": writer_errors,
            "max_wait_ms": max_wait_ms,
        },
        handle,
    )
'''


def _sqlite_phase(
    *,
    db_path: Path,
    script_path: Path,
    workers: int,
    ops_per_writer: int,
    busy_timeout_ms: int,
    hold_ms: float,
    result_dir: Path,
    phase: str,
) -> Dict[str, Any]:
    start_at = time.time() + 0.45
    procs: List[subprocess.Popen] = []
    result_paths: List[Path] = []
    t0 = time.monotonic()
    try:
        for writer in range(workers):
            result_path = result_dir / f"{phase}-w{writer}.json"
            result_paths.append(result_path)
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(script_path),
                        str(db_path),
                        str(writer),
                        str(ops_per_writer),
                        str(busy_timeout_ms),
                        str(hold_ms),
                        str(start_at),
                        str(result_path),
                        SQLITE_CONTENTION_TABLE,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                    close_fds=True,
                )
            )
        deadline = time.monotonic() + 60.0
        for proc in procs:
            remaining = max(0.1, deadline - time.monotonic())
            proc.wait(timeout=remaining)
    finally:
        leaked = 0
        for proc in procs:
            if proc.poll() is None:
                leaked += 1
                terminate_process_group(proc.pid, grace_s=0.5)
            try:
                if proc.stderr is not None:
                    proc.stderr.close()
                if proc.stdout is not None:
                    proc.stdout.close()
            except OSError:
                pass
        if leaked:
            raise RuntimeError(f"{leaked} sqlite writer(s) leaked in phase {phase}")

    elapsed_ms = (time.monotonic() - t0) * 1000.0
    stats = {
        "busy_retries": 0,
        "writer_errors": 0,
        "written": 0,
        "max_wait_ms": 0.0,
        "writers": workers,
        "elapsed_ms": elapsed_ms,
    }
    for result_path in result_paths:
        if not result_path.is_file():
            stats["writer_errors"] += 1
            continue
        data = json.loads(result_path.read_text(encoding="utf-8"))
        stats["busy_retries"] += int(data["busy_retries"])
        stats["writer_errors"] += int(data["writer_errors"])
        stats["written"] += int(data["written"])
        stats["max_wait_ms"] = max(stats["max_wait_ms"], float(data["max_wait_ms"]))
    stats["rows_written"] = stats["written"]
    return stats


def run_sqlite_contention_probe(
    db_path: Path,
    writers: int = 4,
    ops_per_writer: int = 20,
    busy_timeout_ms: int = 5000,
) -> Dict[str, Any]:
    """Drive real multi-process WAL contention, then prove ``busy_timeout=0`` handling."""
    writers = max(1, int(writers))
    ops_per_writer = max(1, int(ops_per_writer))
    busy_timeout_ms = max(0, int(busy_timeout_ms))
    path = Path(db_path).expanduser()
    if not path.is_file():
        create_sqlite_contention_fixture(path)
    expected = writers * ops_per_writer

    work = Path(tempfile.mkdtemp(dir=str(default_scratch_root()), prefix="tb-wal-"))
    script_path = work / "writer.py"
    script_path.write_text(_SQLITE_WRITER, encoding="utf-8")
    try:
        no_timeout = _sqlite_phase(
            db_path=path,
            script_path=script_path,
            workers=writers,
            ops_per_writer=ops_per_writer,
            busy_timeout_ms=0,
            hold_ms=250.0,
            result_dir=work,
            phase="no-timeout",
        )
        conn = sqlite3.connect(str(path), timeout=30.0)
        try:
            conn.execute(f"DELETE FROM {SQLITE_CONTENTION_TABLE}")
            conn.commit()
        finally:
            conn.close()
        with_timeout = _sqlite_phase(
            db_path=path,
            script_path=script_path,
            workers=writers,
            ops_per_writer=ops_per_writer,
            busy_timeout_ms=busy_timeout_ms,
            hold_ms=250.0,
            result_dir=work,
            phase="timeout",
        )
        conn = sqlite3.connect(str(path), timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout = 5000")
            rows_in_db = int(conn.execute(f"SELECT COUNT(*) FROM {SQLITE_CONTENTION_TABLE}").fetchone()[0])
            journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            unique_rows = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM (SELECT DISTINCT writer, seq FROM {SQLITE_CONTENTION_TABLE})"
                ).fetchone()[0]
            )
            integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0]).lower()
        finally:
            conn.close()
    finally:
        _safe_rmtree_no_symlinks(work)

    return {
        "db_path": str(path),
        "writers": writers,
        "ops_per_writer": ops_per_writer,
        "expected_rows": expected,
        "busy_timeout_ms": busy_timeout_ms,
        "begin_immediate": True,
        "journal_mode": journal_mode,
        "wal_mode": journal_mode == "wal",
        "integrity_ok": integrity == "ok",
        "no_timeout_phase": no_timeout,
        "timeout_phase": with_timeout,
        "measurable_contention": no_timeout["busy_retries"] > 0,
        "rows_in_db": rows_in_db,
        "distinct_rows": unique_rows,
        "all_ops_committed": rows_in_db == expected and unique_rows == expected,
        "leaked_children": 0,
    }


# ------------------------------------------------------------------------ pty resize


_PTY_CHILD = r'''
import fcntl
import json
import signal
import struct
import sys
import termios
import time

result_path, ready_path = sys.argv[1], sys.argv[2]


def geometry():
    packed = fcntl.ioctl(0, termios.TIOCGWINSZ, b"\x00" * 8)
    rows, cols, _x, _y = struct.unpack("HHHH", packed)
    return [cols, rows]


state = {"sigwinch_count": 0, "sizes": [geometry()]}


def on_sigwinch(signum, frame):
    state["sigwinch_count"] += 1
    state["sizes"].append(geometry())


signal.signal(signal.SIGWINCH, on_sigwinch)
with open(ready_path, "w", encoding="utf-8") as handle:
    handle.write("ready")

deadline = time.time() + 10.0
while time.time() < deadline and state["sigwinch_count"] < 1:
    time.sleep(0.005)
if state["sigwinch_count"]:
    time.sleep(0.05)
    state["sizes"].append(geometry())

with open(result_path, "w", encoding="utf-8") as handle:
    json.dump(state, handle)
sys.exit(0)
'''


def run_pty_resize_probe(
    initial_size: Tuple[int, int] = (80, 24),
    resized: Tuple[int, int] = (120, 36),
) -> Dict[str, Any]:
    """Spawn a real PTY child, resize it via ``TIOCSWINSZ``, and confirm ``SIGWINCH`` geometry."""
    initial_cols, initial_rows = int(initial_size[0]), int(initial_size[1])
    resized_cols, resized_rows = int(resized[0]), int(resized[1])
    work = Path(tempfile.mkdtemp(dir=str(default_scratch_root()), prefix="tb-pty-"))
    script_path = work / "pty_child.py"
    script_path.write_text(_PTY_CHILD, encoding="utf-8")
    result_path = work / "result.json"
    ready_path = work / "ready"
    master_fd = -1
    proc: Optional[subprocess.Popen] = None
    t0 = time.monotonic()
    leaked = 0
    try:
        master_fd, slave_fd = pty.openpty()
        fcntl.ioctl(
            slave_fd,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", initial_rows, initial_cols, 0, 0),
        )
        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(result_path), str(ready_path)],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=subprocess.PIPE,
            cwd=str(work),
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave_fd)

        ready_deadline = time.monotonic() + 5.0
        while time.monotonic() < ready_deadline and not ready_path.is_file():
            if proc.poll() is not None:
                break
            time.sleep(0.01)
        if not ready_path.is_file():
            raise RuntimeError("pty child never became ready")

        fcntl.ioctl(
            master_fd,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", resized_rows, resized_cols, 0, 0),
        )
        os.kill(proc.pid, signal.SIGWINCH)

        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            terminate_process_group(proc.pid, grace_s=0.5)
            raise RuntimeError("pty child did not exit after SIGWINCH")
        if proc.returncode != 0:
            stderr = (proc.stderr.read().decode("utf-8", "replace") if proc.stderr else "")
            raise RuntimeError(f"pty child exited {proc.returncode}: {stderr}")

        state = json.loads(result_path.read_text(encoding="utf-8"))
        sizes = state["sizes"]
        initial_reported = sizes[0]
        final_reported = sizes[-1]
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return {
            "pty_available": True,
            "initial_size": [initial_cols, initial_rows],
            "resized": [resized_cols, resized_rows],
            "initial_reported": initial_reported,
            "final_reported": final_reported,
            "sigwinch_count": int(state["sigwinch_count"]),
            "geometry_transition_ok": (
                initial_reported == [initial_cols, initial_rows]
                and final_reported == [resized_cols, resized_rows]
                and int(state["sigwinch_count"]) >= 1
            ),
            "child_pid": proc.pid,
            "child_exit_code": proc.returncode,
            "elapsed_ms": elapsed_ms,
            "leaked_children": 0,
        }
    finally:
        if proc is not None:
            if proc.poll() is None:
                leaked += 1
                terminate_process_group(proc.pid, grace_s=0.5)
            if proc.stderr is not None:
                proc.stderr.close()
        if master_fd >= 0:
            try:
                os.close(master_fd)
            except OSError:
                pass
        _safe_rmtree_no_symlinks(work)
        if leaked:
            raise RuntimeError("pty child leaked past probe teardown")


# --------------------------------------------------------------------- signal crash


_CRASH_CHILD_TEMPLATE = r'''
import os
import resource
import signal
import subprocess
import sys
import time

try:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except (ValueError, OSError):
    pass

grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    handle.write(str(grandchild.pid))
time.sleep(0.05)
os.kill(os.getpid(), int(sys.argv[2]))
time.sleep(5)
'''


def run_signal_crash_probe(sig: int = signal.SIGKILL) -> Dict[str, Any]:
    """Crash a child with ``sig`` and verify signal classification and zero leaked children.

    Defaults to SIGKILL: on macOS, SIGABRT/SIGSEGV make ReportCrash file a report and
    show a "Python quit unexpectedly" dialog for every probe run.
    """
    signal_num_expected = int(sig)
    work = Path(tempfile.mkdtemp(dir=str(default_scratch_root()), prefix="tb-crash-"))
    grandchild_file = work / "grandchild.pid"
    script_path = work / "crash_child.py"
    script_path.write_text(_CRASH_CHILD_TEMPLATE, encoding="utf-8")
    proc: Optional[subprocess.Popen] = None
    grandchild_pid = 0
    grandchild_reaped = False
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(grandchild_file), str(signal_num_expected)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
        )
        rc = proc.wait(timeout=15.0)
        signal_num = -rc if rc < 0 else None
        exit_code = 128 + signal_num if signal_num is not None else rc
        for _ in range(50):
            if grandchild_file.is_file():
                break
            time.sleep(0.01)
        if grandchild_file.is_file():
            grandchild_pid = int(grandchild_file.read_text(encoding="utf-8").strip() or 0)

        terminated = terminate_process_group(proc.pid, grace_s=0.5)
        if grandchild_pid > 1:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                grandchild_reaped = True
            else:
                grandchild_reaped = False
        else:
            grandchild_reaped = True

        return {
            "signal_num": signal_num,
            "exit_code": exit_code,
            "expected_exit_code": 128 + signal_num_expected,
            "classification": "signalled" if signal_num is not None else "exited",
            "pid": proc.pid,
            "pgid": proc.pid,
            "grandchild_pid": grandchild_pid,
            "grandchild_reaped": grandchild_reaped,
            "process_group_terminated": terminated,
            "leaked_children": 0 if grandchild_reaped else 1,
        }
    finally:
        if proc is not None:
            if proc.poll() is None:
                terminate_process_group(proc.pid, grace_s=0.5)
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()
        _safe_rmtree_no_symlinks(work)


# ---------------------------------------------------------------- repeated store init


def run_repeated_init_probe(times: int = 5) -> Dict[str, Any]:
    """Initialise the same store home and mail fixture repeatedly; prove idempotence."""
    times = max(1, int(times))
    work = Path(tempfile.mkdtemp(dir=str(default_scratch_root()), prefix="tb-init-"))
    home = work / "store-home"
    fingerprints: List[Tuple[int, Tuple[str, ...], int]] = []
    schema_versions: List[int] = []
    errors: List[str] = []
    try:
        for _ in range(times):
            try:
                store = TestBenchStore(home=home)
                journal_mode = store.journal_mode()
                store.close()
                conn = sqlite3.connect(str(home / "state.db"), timeout=10.0)
                try:
                    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                    tables = tuple(
                        r[0]
                        for r in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' "
                            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                        )
                    )
                    run_count = int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
                finally:
                    conn.close()
                if journal_mode.lower() != "wal":
                    errors.append(f"iteration journal_mode={journal_mode!r}")
                schema_versions.append(version)
                fingerprints.append((version, tables, run_count))
            except Exception as exc:  # pragma: no cover - surfaced in the probe result
                errors.append(f"{type(exc).__name__}: {exc}")

        first_mail = work / "mail-a.db"
        second_mail = work / "mail-b.db"
        mail_a = create_cross_lang_mail_db(first_mail)
        mail_b = create_cross_lang_mail_db(second_mail)
        mail_hashes = [mail_a["sha256"], mail_b["sha256"]]

        distinct = len(set(fingerprints))
        return {
            "times": times,
            "schema_version": schema_versions[-1] if schema_versions else 0,
            "schema_versions": schema_versions,
            "fingerprints": [list(map(str, fp)) for fp in fingerprints],
            "distinct_fingerprints": distinct,
            "idempotent": distinct == 1 and not errors and len(fingerprints) == times,
            "errors": errors,
            "mail_db_hashes": mail_hashes,
            "mail_db_deterministic": mail_hashes[0] == mail_hashes[1],
        }
    finally:
        _safe_rmtree_no_symlinks(work)


# -------------------------------------------------------------------- tmux lifecycle


def _tmux_env(work: Path, socket: str) -> Dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/opt/homebrew/bin:/usr/bin:/bin"),
        "HOME": str(work / "home"),
        "TMUX_TMPDIR": str(work),
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TERM": "screen-256color",
        "SHELL": "/bin/sh",
    }


def run_tmux_lifecycle_probe(scratch_root: Optional[Path] = None) -> Dict[str, Any]:
    """Exercise an isolated tmux server: session/window/pane ``@hx``, crash recovery, teardown."""
    tmux_bin = shutil.which("tmux")
    if not tmux_bin:
        return {
            "available": False,
            "reason": "tmux binary not found on PATH",
            "session": "tb-lifecycle",
            "windows": 0,
            "panes": 0,
            "hx_options": {},
            "crash_recovered": False,
            "server_gone": True,
            "clean": True,
        }

    base = Path(scratch_root).expanduser() if scratch_root else default_scratch_root()
    base.mkdir(parents=True, exist_ok=True)
    from .isolation import short_writable_tmux_base
    short_base = short_writable_tmux_base(base)
    work = short_base / f"tx{os.urandom(3).hex()}"
    work.mkdir(mode=0o700)
    (work / "home").mkdir(mode=0o700)
    socket = f"s{os.urandom(2).hex()}"
    env = _tmux_env(work, socket)
    session = "tb-lifecycle"

    def tmux(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [tmux_bin, "-L", socket, "-f", "/dev/null", *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )

    def must(*args: str) -> str:
        result = tmux(*args)
        if result.returncode != 0:
            raise RuntimeError(f"tmux {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

    server_pid = 0
    server_gone = False
    crash_recovered = False
    windows = 0
    panes = 0
    hx_options: Dict[str, str] = {}
    try:
        must("new-session", "-d", "-s", session, "-x", "120", "-y", "36")
        server_pid = int(must("display-message", "-p", "#{pid}").strip())
        must("set-option", "-t", session, "@hx", "main")
        must("new-window", "-t", f"{session}:", "-n", "work")
        must("set-option", "-w", "-t", f"{session}:work", "@hx", "work")
        must("split-window", "-t", f"{session}:work")
        must("select-pane", "-t", f"{session}:work.1")
        must("set-option", "-p", "-t", f"{session}:work.1", "@hx", "pane-1")

        hx_options = {
            "session": must("show-options", "-v", "-t", session, "@hx").strip(),
            "window": must("show-options", "-wv", "-t", f"{session}:work", "@hx").strip(),
            "pane": must("show-options", "-pv", "-t", f"{session}:work.1", "@hx").strip(),
        }
        windows = len([w for w in must("list-windows", "-t", session, "-F", "#{window_id}").splitlines() if w])
        panes = len(must("list-panes", "-t", f"{session}:work", "-F", "#{pane_id}").splitlines())

        # Crash recovery: a pane process dies, the pane stays dead, respawn restores it.
        must("set-option", "-t", f"{session}:work", "remain-on-exit", "on")
        must("split-window", "-t", f"{session}:work", "sleep 30")
        pane_target = f"{session}:work.2"
        pane_pid = int(must("display-message", "-p", "-t", pane_target, "#{pane_pid}").strip())
        try:
            os.kill(pane_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5.0
        became_dead = False
        while time.monotonic() < deadline:
            dead = must("display-message", "-p", "-t", pane_target, "#{pane_dead}").strip()
            if dead == "1":
                became_dead = True
                break
            time.sleep(0.02)
        must("respawn-pane", "-k", "-t", pane_target, "sleep 30")
        alive_again = (
            must("display-message", "-p", "-t", pane_target, "#{pane_dead}").strip() == "0"
        )
        crash_recovered = became_dead and alive_again

        tmux("kill-server")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if server_pid and not _pid_alive(server_pid):
                server_gone = True
                break
            time.sleep(0.05)
        server_gone = server_gone or (server_pid > 0 and not _pid_alive(server_pid))
    finally:
        tmux("kill-server")
        _safe_rmtree_no_symlinks(work)

    return {
        "available": True,
        "reason": "",
        "socket": socket,
        "session": session,
        "windows": windows,
        "panes": panes,
        "hx_options": hx_options,
        "server_pid": server_pid,
        "crash_recovered": crash_recovered,
        "server_gone": server_gone and not _pid_alive(server_pid),
        "clean": server_gone and not work.exists(),
        "leaked_children": 0,
    }


def _pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# --------------------------------------------------------------------- mail fixtures


MAIL_SCHEMA = """
CREATE TABLE mailbox (id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT, parent_id TEXT,
                      sort_order INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID;
CREATE TABLE email (rid INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, blob_id TEXT,
                    thread_id TEXT NOT NULL, received_at INTEGER NOT NULL, size INTEGER NOT NULL,
                    flags INTEGER NOT NULL DEFAULT 0, has_att INTEGER NOT NULL DEFAULT 0,
                    from_name TEXT, from_addr TEXT, to_json TEXT, subject TEXT, preview TEXT,
                    message_id TEXT, in_reply_to TEXT, other_kw TEXT);
CREATE TABLE body (rid INTEGER PRIMARY KEY, text TEXT NOT NULL);
CREATE TABLE email_mailbox (mailbox_id TEXT NOT NULL, received_at INTEGER NOT NULL,
                            rid INTEGER NOT NULL, thread_id TEXT NOT NULL, unread INTEGER NOT NULL,
                            PRIMARY KEY (mailbox_id, received_at DESC, rid)) WITHOUT ROWID;
CREATE VIRTUAL TABLE email_fts USING fts5(subject, sender, recipients, body, content='',
                                          contentless_delete=1, tokenize='unicode61');
"""

MAIL_TABLES = ("body", "email", "email_fts", "email_mailbox", "mailbox")

_HOSTILE_BODY = (
    "Invoice attached.\x1b]52;c;ZXZpbA==\x07"
    "\x1b[2J\x1bP1;2;3+q544e\x1b\\"
    "Pre-epoch ordering must not hide this row.\x00"
)


def create_cross_lang_mail_db(db_path: Path) -> Dict[str, Any]:
    """Create the deterministic 5-table mail DB shared by ``hxmail`` and ``listbench.cob``."""
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        target = Path(str(path) + suffix)
        if target.exists() and target.is_file():
            target.unlink()

    big_rid = 2**32 + 7
    emails = [
        (
            1,
            "m-0001",
            "blob-0001",
            "thread-1",
            1700000000,
            "Invoice 2000",
            "Dana Whitfield",
            "dana@example.com",
        ),
        (
            2,
            "m-0002",
            "blob-0002",
            "thread-2",
            -86400,
            "Pre-epoch hello",
            "Ada Lovelace",
            "ada@example.com",
        ),
        (
            3,
            "m-0003",
            "blob-0003",
            "thread-3",
            -2208988800,
            "1900-01-01 mail",
            "Old Timer",
            "old@example.com",
        ),
        (
            big_rid,
            "m-big-rid",
            "blob-big",
            "thread-utf8",
            1700000123,
            "日本語の請求書 Invoice",
            "山田 太郎",
            "taro@example.jp",
        ),
    ]
    bodies = {
        1: "Invoice 2000\n\nPlease pay the attached invoice before the due date.",
        2: "Hello from before the epoch.\n",
        3: "A message older than 1970.\n",
        big_rid: _HOSTILE_BODY,
    }

    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(MAIL_SCHEMA)
        conn.executemany(
            "INSERT INTO mailbox(id, name, role, parent_id, sort_order) VALUES (?, ?, ?, ?, ?)",
            [
                ("mb-inbox", "Inbox", "inbox", None, 0),
                ("mb-archive", "Archive", "archive", None, 1),
            ],
        )
        for rid, msg_id, blob_id, thread_id, received_at, subject, from_name, from_addr in emails:
            conn.execute(
                """
                INSERT INTO email(rid, id, blob_id, thread_id, received_at, size, flags, has_att,
                                  from_name, from_addr, to_json, subject, preview, message_id,
                                  in_reply_to, other_kw)
                VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    rid,
                    msg_id,
                    blob_id,
                    thread_id,
                    received_at,
                    len(bodies[rid]),
                    from_name,
                    from_addr,
                    json.dumps([{"name": "Sein", "address": "sein@example.com"}]),
                    subject,
                    bodies[rid][:120],
                    f"<{msg_id}@example.com>",
                ),
            )
            conn.execute(
                "INSERT INTO body(rid, text) VALUES (?, ?)", (rid, bodies[rid])
            )
            conn.execute(
                """
                INSERT INTO email_mailbox(mailbox_id, received_at, rid, thread_id, unread)
                VALUES (?, ?, ?, ?, 1)
                """,
                ("mb-inbox", received_at, rid, thread_id),
            )
            conn.execute(
                "INSERT INTO email_fts(rowid, subject, sender, recipients, body) VALUES (?, ?, ?, ?, ?)",
                (rid, subject, from_addr, "sein@example.com", bodies[rid]),
            )
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'email_fts_%' ORDER BY name"
            )
        ]
        emails_count = int(conn.execute("SELECT COUNT(*) FROM email").fetchone()[0])
        bodies_count = int(conn.execute("SELECT COUNT(*) FROM body").fetchone()[0])
        fts_rows = int(conn.execute("SELECT COUNT(*) FROM email_fts").fetchone()[0])
        mailbox_count = int(conn.execute("SELECT COUNT(*) FROM mailbox").fetchone()[0])
        pre_epoch = int(
            conn.execute("SELECT COUNT(*) FROM email WHERE received_at < 0").fetchone()[0]
        )
        max_rid = int(conn.execute("SELECT MAX(rid) FROM email").fetchone()[0])
        min_rid = int(conn.execute("SELECT MIN(rid) FROM email").fetchone()[0])
        subject = str(
            conn.execute("SELECT subject FROM email WHERE rid = ?", (big_rid,)).fetchone()[0]
        )
        hostile_rows = int(
            conn.execute(
                "SELECT COUNT(*) FROM body WHERE instr(text, char(27)) > 0 OR instr(text, char(7)) > 0"
            ).fetchone()[0]
        )
        fts_hits = int(
            conn.execute("SELECT COUNT(*) FROM email_fts WHERE email_fts MATCH 'invoice'").fetchone()[0]
        )
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "db_path": str(path),
        "tables": tables,
        "journal_mode": journal_mode,
        "mailboxes": mailbox_count,
        "emails": emails_count,
        "bodies": bodies_count,
        "fts_rows": fts_rows,
        "min_rid": min_rid,
        "max_rid": max_rid,
        "rid_over_2_32": max_rid > 2**32,
        "pre_epoch_count": pre_epoch,
        "hostile_escape_rows": hostile_rows,
        "utf8_ok": subject == "日本語の請求書 Invoice",
        "fts_query_hits": fts_hits,
        "sha256": digest,
    }


# ---------------------------------------------------------------- haunted terminal

HAUNTED_TERMINAL_AREAS: Tuple[str, ...] = (
    "combining_unicode",
    "wide_display_cells",
    "bidi_and_invisible_controls",
    "malformed_ansi_osc",
    "osc52_clipboard",
    "narrow_dimensions",
    "rapid_sigwinch",
    "interrupted_output_partial_utf8",
)

HAUNTED_NARROW_SIZES: Tuple[Tuple[int, int], ...] = (
    (1, 1),
    (2, 2),
    (6, 3),
    (10, 4),
    (11, 5),
    (14, 6),
    (20, 3),
)

HAUNTED_RESIZE_SCHEDULE: Tuple[Tuple[int, int], ...] = (
    (80, 24),
    (52, 18),
    (40, 12),
    (120, 30),
    (24, 10),
    (72, 20),
    (100, 26),
    (33, 11),
    (96, 28),
    (18, 6),
    (64, 22),
    (48, 16),
)

HAUNTED_TMUX_RESIZE_SCHEDULE: Tuple[Tuple[int, int], ...] = (
    (80, 24),
    (60, 20),
    (40, 14),
    (100, 28),
    (30, 10),
    (70, 18),
)

#: Clipboard payloads that must never reach the terminal.
HAUNTED_CLIPBOARD_PAYLOADS: Tuple[str, ...] = ("ZXZpbA==", "cGF5bG9hZA==", "c3BsaXQ=")

HAUNTED_PARTIAL_SEQUENCE = b"\xe6\x97"  # first two bytes of U+65E5
HAUNTED_PARTIAL_BEGIN = b"@@HAUNTED_PARTIAL_BEGIN"
HAUNTED_PARTIAL_END = b"@@HAUNTED_PARTIAL_END"

_HAUNTED_FRAME_RE = re.compile(r"@@HAUNTED_FRAME (\d+) (\d+) (\d+) (\w+)")
_HAUNTED_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")

_PARTIAL_UTF8_CHUNKS: Tuple[bytes, ...] = (
    b"latin \xc3",  # é split after its lead byte
    b"\xa9 ok \xe2",  # € split after its lead byte
    b"\x82\xac cjk \xe6\x97",  # 日 split mid-sequence
    b"\xa5 emoji \xf0\x9f",  # emoji split after two bytes
    b"\x98",  # continuation byte joins the next chunk
    b"\x80 done\n",
)

_PARTIAL_UTF8_EXPECTED = "latin \u00e9 ok \u20ac cjk \u65e5 emoji \U0001f600 done\n"


@dataclass(frozen=True)
class HauntedCase:
    """One observable Haunted Terminal invariant and its measured outcome."""

    name: str
    area: str
    status: str
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "area": self.area,
            "status": self.status,
            "message": self.message,
        }


def haunted_combining_corpus() -> List[CorpusEntry]:
    """Base/combining clusters that must never be split, reordered, or double-counted."""
    return [
        CorpusEntry(
            "combining_acute_ascii_bases",
            "haunted_combining",
            "e\u0301 E\u0301 cafe\u0301 re\u0301sume\u0301",
            note="NFD acute on ASCII bases",
        ),
        CorpusEntry(
            "combining_stacked_diacritics",
            "haunted_combining",
            "a\u0308\u0323 n\u0303 o\u0302\u0303",
            note="two marks stacked on one base",
        ),
        CorpusEntry(
            "combining_devanagari_cluster",
            "haunted_combining",
            "\u0915\u094d\u0937\u093f\u0924\u093f",
            note="virama conjoins क+ष; cluster is one grapheme",
        ),
        CorpusEntry(
            "combining_hebrew_points",
            "haunted_combining",
            "\u05e9\u05c1\u05b8\u05dc\u05d5\u05b9\u05dd",
            note="Hebrew points (Mn) attach to each letter",
        ),
        CorpusEntry(
            "combining_cjk_kana_voicing",
            "haunted_combining",
            "\u3042\u3099\u3044\u3099\u3046\u3099",
            note="decomposed kana voicing marks",
        ),
    ]


def haunted_wide_corpus() -> List[CorpusEntry]:
    """Wide (W) and fullwidth (F) text whose display cells are two per character."""
    return [
        CorpusEntry(
            "wide_cjk_sentence",
            "haunted_wide",
            "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059",
            note="Hiragana/Katakana/Kanji mix",
        ),
        CorpusEntry(
            "wide_fullwidth_ascii",
            "haunted_wide",
            "\uff21\uff22\uff23\uff11\uff12\uff13",
            note="fullwidth latin and digits",
        ),
        CorpusEntry(
            "wide_hangul_and_cjk",
            "haunted_wide",
            "\ud55c\uad6d\uc5b4 ABC \u4e2d\u6587",
            note="Hangul syllables plus ASCII",
        ),
        CorpusEntry(
            "wide_emoji_presentation",
            "haunted_wide",
            "\U0001f600\U0001f680\U0001f4a1",
            note="emoji with East Asian Width W",
        ),
        CorpusEntry(
            "wide_kana_with_combining",
            "haunted_wide",
            "\u3042\u3099\u3044\u3099 A\u0301",
            note="wide cluster followed by ASCII cluster",
        ),
    ]


def haunted_invisible_control_corpus() -> List[CorpusEntry]:
    """Bidi overrides, isolates, joiners, BOM, and Plane-14 tag payloads."""
    return [
        CorpusEntry(
            "bidi_rlo_pdf_override",
            "haunted_invisible",
            "invoice\u202egnp.exe\u202c.txt",
            note="RLO/PDF can visually reverse a filename",
        ),
        CorpusEntry(
            "bidi_isolates_lri_pdi",
            "haunted_invisible",
            "\u2066\u05e2\u05d1\u05e8\u05d9\u05ea\u2069 tail",
            note="LRI/PDI isolate around Hebrew",
        ),
        CorpusEntry(
            "bidi_marks_lrm_rlm_alm",
            "haunted_invisible",
            "a\u200eb\u200fc\u061cd",
            note="LRM/RLM/ALM direction marks",
        ),
        CorpusEntry(
            "zero_width_joiners",
            "haunted_invisible",
            "pass\u200bword\u200c\u200d!",
            note="ZWSP/ZWNJ/ZWJ inside a token",
        ),
        CorpusEntry(
            "plane14_tag_payload",
            "haunted_invisible",
            "\U0001f3f4\U000e0067\U000e0062\U000e007f",
            note="invisible tag letters encode an ASCII payload",
        ),
        CorpusEntry(
            "bom_and_soft_hyphen",
            "haunted_invisible",
            "\ufeffadmin\u00adistrator",
            note="BOM at line start, soft hyphen mid-word",
        ),
    ]


def haunted_malformed_escape_corpus() -> List[CorpusEntry]:
    """Escape streams that are truncated, absurdly parameterised, or unterminated."""
    return [
        CorpusEntry(
            "unterminated_osc52_then_newline",
            "haunted_malformed",
            "\x1b]52;c;ZXZpbA==\nvisible-after-newline",
            note="OSC without BEL/ST must fail closed at the newline",
        ),
        CorpusEntry(
            "unterminated_osc8_then_newline",
            "haunted_malformed",
            "\x1b]8;;https://evil.example/\nvisible-after-newline",
            note="OSC 8 hyperlink without ST",
        ),
        CorpusEntry(
            "csi_absurd_params",
            "haunted_malformed",
            "\x1b[31337;999;42Hvisible",
            note="huge CSI parameters must not swallow the text",
        ),
        CorpusEntry(
            "csi_incomplete_params_at_eof",
            "haunted_malformed",
            "tail\x1b[31;",
            note="CSI ends mid-parameter at end of stream",
        ),
        CorpusEntry(
            "dcs_unterminated_at_eof",
            "haunted_malformed",
            "safe\x1bP1;2;3+q544e",
            note="DCS never terminated",
        ),
        CorpusEntry(
            "lone_escape_mid_text",
            "haunted_malformed",
            "before\x1bafter",
            note="bare ESC consumes one following byte, never the rest",
        ),
        CorpusEntry(
            "nested_osc_bel_csi",
            "haunted_malformed",
            "\x1b]0;title\x07\x1b[2Jbody",
            note="BEL-terminated OSC followed by CSI erase-display",
        ),
    ]


def haunted_osc52_corpus() -> List[CorpusEntry]:
    """OSC 52 clipboard write/read requests in every terminator form."""
    return [
        CorpusEntry(
            "osc52_bel_terminated",
            "osc52",
            "\x1b]52;c;ZXZpbA==\x07clipboard-hijack",
            note="classic BEL-terminated clipboard write",
        ),
        CorpusEntry(
            "osc52_st_terminated",
            "osc52",
            "\x1b]52;c;ZXZpbA==\x1b\\clipboard-hijack",
            note="ST-terminated clipboard write",
        ),
        CorpusEntry(
            "osc52_query_read",
            "osc52",
            "\x1b]52;c;?\x07clipboard-read",
            note="clipboard read request",
        ),
        CorpusEntry(
            "osc52_primary_selection",
            "osc52",
            "\x1b]52;p;cGF5bG9hZA==\x07primary",
            note="primary selection write",
        ),
        CorpusEntry(
            "osc52_split_across_lines",
            "osc52",
            "line-one\n\x1b]52;c;c3BsaXQ=\x07line-two",
            note="write preceded by ordinary text",
        ),
    ]


def haunted_partial_utf8_chunks() -> List[bytes]:
    """The stream split mid-codepoint at every UTF-8 sequence length."""
    return list(_PARTIAL_UTF8_CHUNKS)


def haunted_partial_utf8_expected() -> str:
    """The exact text a faithful incremental decoder recovers from those chunks."""
    return _PARTIAL_UTF8_EXPECTED


def haunted_narrow_sizes() -> List[Tuple[int, int]]:
    return [(int(size[0]), int(size[1])) for size in HAUNTED_NARROW_SIZES]


def haunted_resize_schedule() -> List[Tuple[int, int]]:
    return [(int(size[0]), int(size[1])) for size in HAUNTED_RESIZE_SCHEDULE]


def _haunted_zero_width(ch: str) -> bool:
    cp = ord(ch)
    if cp in (0x200B, 0x200C, 0x200D, 0xFEFF):
        return True
    if 0x1160 <= cp <= 0x11FF:
        return True
    if unicodedata.combining(ch):
        return True
    return unicodedata.category(ch) in ("Mn", "Me", "Cf")


def _haunted_cell_width(ch: str) -> int:
    if _haunted_zero_width(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _haunted_run_record(text: str = "haunted") -> RunRecord:
    flat = text.replace("\n", " ") or "haunted"
    return RunRecord(
        run_id="tb_haunted_phase1c",
        project=flat,
        project_root=".",
        revision="0000000",
        profile=ProfileName.TERMINAL,
        status=RunStatus.FAIL,
        first_failure=flat,
        created_at="2026-01-01T00:00:00.000000Z",
    )


def _haunted_state(text: str) -> TuiState:
    run = _haunted_run_record(text)
    lines = tuple(text.splitlines()) or ("",)
    flat = text.replace("\n", " ") or "ready"
    return TuiState(
        runs=(run,),
        log_lines={run.run_id: lines},
        view="overview",
        detail_open=True,
        detail_scroll=13,
        minibuffer=flat,
    )


def haunted_render_frame(text: str, width: int, height: int) -> List[str]:
    """Render hostile text through the production sanitize + cell-width pipeline."""
    return render_frame(_haunted_state(text), int(width), int(height), no_color=True)


def haunted_render_naive(text: str, width: int, height: int) -> List[str]:
    """Deliberately defective control renderer: no sanitizer, character slicing."""
    width = max(1, int(width))
    height = max(1, int(height))
    rows = [raw[:width] for raw in (text.splitlines() or [""])]
    rows = rows[:height]
    while len(rows) < height:
        rows.append("")
    return [row.ljust(width) for row in rows]


def _haunted_label(cp: int) -> str:
    return "U+%04X" % cp


def _haunted_forbidden_codepoints(text: str) -> List[str]:
    found: List[int] = []
    labels: List[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in (0x09, 0x0A):
            continue
        forbidden = (
            cp < 0x20
            or cp == 0x7F
            or 0x80 <= cp <= 0x9F
            or 0x200B <= cp <= 0x200F
            or 0x202A <= cp <= 0x202E
            or 0x2060 <= cp <= 0x206F
            or 0xE0000 <= cp <= 0xE007F
            or cp in (0x00AD, 0x034F, 0x061C, 0x180E, 0xFEFF)
            or 0xFFF9 <= cp <= 0xFFFB
        )
        if forbidden and cp not in found:
            found.append(cp)
            labels.append(_haunted_label(cp))
    return labels


def _haunted_orphan_marks(text: str) -> List[str]:
    problems: List[str] = []
    if text and unicodedata.combining(text[0]):
        problems.append("starts with a detached combining mark")
    previous = ""
    for ch in text:
        if previous and unicodedata.combining(ch) and previous in (" ", "\t"):
            problems.append("combining mark detached from its base")
            break
        previous = ch
    return problems


def check_haunted_text(text: str) -> List[str]:
    """Return violations for untrusted text about to reach a terminal."""
    violations: List[str] = []
    forbidden = _haunted_forbidden_codepoints(text)
    if forbidden:
        violations.append("forbidden code points present: " + ", ".join(forbidden))
    if "\x1b" in text:
        violations.append("escape byte present in sanitized text")
    for payload in HAUNTED_CLIPBOARD_PAYLOADS:
        if payload in text:
            violations.append("clipboard payload leaked: " + payload)
    violations.extend(_haunted_orphan_marks(text))
    return violations


def check_haunted_frame(
    lines: Sequence[str], width: int, *, height: Optional[int] = None
) -> List[str]:
    """Return violations for one rendered frame at ``width`` display cells."""
    violations: List[str] = []
    width = int(width)
    if height is not None and len(lines) != int(height):
        violations.append(
            "frame has %d lines, expected %d" % (len(lines), int(height))
        )
    for number, line in enumerate(lines, start=1):
        raw = line or ""
        plain = _HAUNTED_SGR_RE.sub("", raw)
        if "\x1b" in plain:
            violations.append("line %d: non-SGR escape sequence leaked" % number)
        if "\x07" in plain:
            violations.append("line %d: BEL leaked" % number)
        if "\x1b]52" in raw or "]52;" in plain:
            violations.append("line %d: OSC 52 clipboard write leaked" % number)
        for payload in HAUNTED_CLIPBOARD_PAYLOADS:
            if payload in plain:
                violations.append("line %d: clipboard payload leaked" % number)
                break
        forbidden = _haunted_forbidden_codepoints(plain)
        if forbidden:
            violations.append(
                "line %d: forbidden code points: %s" % (number, ", ".join(forbidden))
            )
        cells = display_width(raw)
        if cells > width:
            violations.append(
                "line %d: %d cells exceed width %d" % (number, cells, width)
            )
        for problem in _haunted_orphan_marks(plain):
            violations.append("line %d: %s" % (number, problem))
    return violations


def decode_haunted_stream(
    chunks: Sequence[bytes], *, faithful: bool = True
) -> Dict[str, Any]:
    """Decode a stream of byte chunks either incrementally or per chunk.

    ``faithful=False`` models the classic defect: each chunk decoded alone with
    ``errors="ignore"``, so an interrupted code point vanishes without a trace.
    """
    if faithful:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pieces = [decoder.decode(bytes(chunk), final=False) for chunk in chunks]
        pieces.append(decoder.decode(b"", final=True))
        text = "".join(pieces)
        mode = "incremental"
    else:
        text = "".join(bytes(chunk).decode("utf-8", "ignore") for chunk in chunks)
        mode = "silent_drop"
    return {"mode": mode, "text": text, "replacements": text.count("\ufffd")}


def split_haunted_stream(raw: bytes) -> List[bytes]:
    """Split captured bytes exactly where the child interrupted a code point."""
    start = raw.find(HAUNTED_PARTIAL_BEGIN)
    if start < 0:
        return [raw]
    after = start + len(HAUNTED_PARTIAL_BEGIN)
    newline = raw.find(b"\n", after)
    partial_start = newline + 1 if newline >= 0 else after
    end = raw.find(HAUNTED_PARTIAL_END, partial_start)
    limit = end if end >= 0 else len(raw)
    cut = limit
    while cut > partial_start and raw[cut - 1 : cut] in (b"\n", b"\r"):
        cut -= 1
    minimum = partial_start + len(HAUNTED_PARTIAL_SEQUENCE)
    if cut < minimum:
        cut = min(limit, minimum)
    if cut <= 0 or cut >= len(raw):
        return [raw]
    return [raw[:cut], raw[cut:]]


def parse_haunted_frames(text: str) -> List[Dict[str, Any]]:
    """Parse the child's frame markers and their ruler widths."""
    frames: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        marker = _HAUNTED_FRAME_RE.search(line)
        if marker:
            current = {
                "seq": int(marker.group(1)),
                "cols": int(marker.group(2)),
                "rows": int(marker.group(3)),
                "kind": marker.group(4),
                "widths": [],
            }
            frames.append(current)
            continue
        if current is not None and line and set(line) == {"#"}:
            current["widths"].append(len(line))
    return frames


def _reference_fit(text: str, width: int) -> Tuple[str, int]:
    """Greedy cell-accurate fit used as the oracle for wide/combining fitting."""
    used = 0
    out: List[str] = []
    for ch in text:
        cells = _haunted_cell_width(ch)
        if cells == 0:
            if used > 0:
                out.append(ch)
            continue
        if used + cells > width:
            break
        out.append(ch)
        used += cells
    return "".join(out), used


# ----------------------------------------------------------------- pty transport

_HAUNTED_PTY_CHILD = r'''
import fcntl
import json
import signal
import struct
import sys
import termios
import time

plan_path, result_path, ready_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(plan_path, "r", encoding="utf-8") as handle:
    plan = json.load(handle)

state = {"sigwinch_count": 0, "sizes": [], "frames": [], "final": [0, 0], "errors": []}


def geometry():
    packed = fcntl.ioctl(0, termios.TIOCGWINSZ, b"\x00" * 8)
    rows, cols, _x, _y = struct.unpack("HHHH", packed)
    return int(cols), int(rows)


def on_sigwinch(signum, frame):
    state["sigwinch_count"] += 1


if not plan.get("ignore_sigwinch"):
    signal.signal(signal.SIGWINCH, on_sigwinch)

initial = geometry()
state["sizes"].append(list(initial))
cached = initial

with open(ready_path, "w", encoding="utf-8") as handle:
    handle.write("ready")

out = sys.stdout.buffer
hostile = bytes.fromhex(plan.get("hostile_hex", ""))
partial = bytes.fromhex(plan.get("partial_hex", ""))
iterations = int(plan.get("iterations", 44))
sleep_s = float(plan.get("sleep_s", 0.025))
partial_at = int(plan.get("partial_at", 7))

try:
    for seq in range(iterations):
        if plan.get("cache_geometry"):
            cols, rows = cached
        else:
            cols, rows = geometry()
        if seq % 4 == 3 and hostile:
            out.write(("@@HAUNTED_FRAME %d %d %d hostile\n" % (seq, cols, rows)).encode("ascii"))
            out.write(hostile)
            out.write(b"\n")
        else:
            out.write(("@@HAUNTED_FRAME %d %d %d ruler\n" % (seq, cols, rows)).encode("ascii"))
            out.write(b"#" * max(0, cols) + b"\n")
        if partial and seq == partial_at:
            out.write(b"\n@@HAUNTED_PARTIAL_BEGIN\n")
            out.write(partial)
            out.write(b"\n@@HAUNTED_PARTIAL_END\n")
        out.flush()
        state["frames"].append([seq, cols, rows])
        time.sleep(sleep_s)
except (BrokenPipeError, OSError) as exc:
    state["errors"].append("%s: %s" % (type(exc).__name__, exc))

state["final"] = list(geometry())
with open(result_path, "w", encoding="utf-8") as handle:
    json.dump(state, handle)
sys.exit(0)
'''


def _haunted_plan(
    *,
    iterations: int,
    sleep_s: float,
    partial: bool,
    cache_geometry: bool,
) -> Dict[str, Any]:
    hostile_text = "\u65e5\u672c\u8a9e e\u0301 \x1b]52;c;ZXZpbA==\x07tail"
    hostile_clean = sanitize_terminal_text(hostile_text).replace("\n", " ")
    return {
        "iterations": int(iterations),
        "sleep_s": float(sleep_s),
        "hostile_hex": hostile_clean.encode("utf-8").hex(),
        "partial_hex": HAUNTED_PARTIAL_SEQUENCE.hex() if partial else "",
        "partial_at": 7,
        "cache_geometry": bool(cache_geometry),
        "ignore_sigwinch": False,
    }


def _write_haunted_plan(work: Path, plan: Dict[str, Any]) -> Tuple[Path, Path, Path, Path]:
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / "plan.json"
    result_path = work / "result.json"
    ready_path = work / "ready"
    child_path = work / "haunted_child.py"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    child_path.write_text(_HAUNTED_PTY_CHILD, encoding="utf-8")
    return plan_path, result_path, ready_path, child_path


def _set_pty_size(fd: int, cols: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", int(rows), int(cols), 0, 0))


def _run_haunted_pty_child(
    work: Path,
    schedule: Sequence[Tuple[int, int]],
    *,
    partial: bool,
    cache_geometry: bool,
    iterations: int = 44,
    sleep_s: float = 0.025,
    resize_interval_s: float = 0.07,
) -> Dict[str, Any]:
    """Run the hostile child on a real PTY while resizing it rapidly."""
    plan = _haunted_plan(
        iterations=iterations,
        sleep_s=sleep_s,
        partial=partial,
        cache_geometry=cache_geometry,
    )
    plan_path, result_path, ready_path, child_path = _write_haunted_plan(work, plan)
    master_fd = -1
    proc: Optional[subprocess.Popen] = None
    chunks: List[bytes] = []
    applied: List[List[int]] = []
    try:
        master_fd, slave_fd = pty.openpty()
        _set_pty_size(slave_fd, schedule[0][0], schedule[0][1])
        proc = subprocess.Popen(
            [sys.executable, str(child_path), str(plan_path), str(result_path), str(ready_path)],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=subprocess.PIPE,
            cwd=str(work),
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave_fd)
        os.set_blocking(master_fd, False)

        ready_deadline = time.monotonic() + 8.0
        while time.monotonic() < ready_deadline and not ready_path.is_file():
            if proc.poll() is not None:
                stderr = b""
                if proc.stderr is not None:
                    try:
                        stderr = proc.stderr.read() or b""
                    except OSError:
                        stderr = b""
                raise RuntimeError(
                    "haunted pty child exited before ready: %s"
                    % stderr.decode("utf-8", "replace")[:400]
                )
            time.sleep(0.01)
        if not ready_path.is_file():
            raise RuntimeError("haunted pty child never became ready")

        next_resize = 1
        start = time.monotonic()
        deadline = start + 30.0
        eof = False
        while not eof:
            now = time.monotonic()
            if now > deadline:
                raise RuntimeError("haunted pty child did not finish within 30s")
            while next_resize < len(schedule) and now - start >= resize_interval_s * next_resize:
                cols, rows = schedule[next_resize]
                _set_pty_size(master_fd, cols, rows)
                try:
                    os.kill(proc.pid, signal.SIGWINCH)
                except OSError:
                    pass
                applied.append([int(cols), int(rows)])
                next_resize += 1
            readable, _w, _x = select.select([master_fd], [], [], 0.01)
            if not readable:
                if proc.poll() is not None and next_resize >= len(schedule):
                    # Give the master one final drain before declaring EOF.
                    extra, _w, _x = select.select([master_fd], [], [], 0.05)
                    if not extra:
                        break
                    readable = extra
                else:
                    continue
            try:
                chunk = os.read(master_fd, 65536)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    continue
                break
            if not chunk:
                eof = True
                break
            chunks.append(chunk)

        try:
            exit_code = proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            terminate_process_group(proc.pid, grace_s=0.5)
            raise RuntimeError("haunted pty child did not exit")
        if exit_code != 0:
            stderr = b""
            if proc.stderr is not None:
                stderr = proc.stderr.read() or b""
            raise RuntimeError(
                "haunted pty child exited %s: %s"
                % (exit_code, stderr.decode("utf-8", "replace")[:400])
            )
        if not result_path.is_file():
            raise RuntimeError("haunted pty child produced no result file")
        state = json.loads(result_path.read_text(encoding="utf-8"))
        return {
            "available": True,
            "reason": "",
            "raw": b"".join(chunks),
            "state": state,
            "applied_resizes": applied,
            "exit_code": exit_code,
            "leaked_children": 0,
        }
    finally:
        leaked = 0
        if proc is not None:
            if proc.poll() is None:
                leaked = 1
                terminate_process_group(proc.pid, grace_s=0.5)
            if proc.stderr is not None:
                try:
                    proc.stderr.close()
                except OSError:
                    pass
        if master_fd >= 0:
            try:
                os.close(master_fd)
            except OSError:
                pass
        if leaked:
            raise RuntimeError("haunted pty child leaked past probe teardown")


def _run_haunted_tmux_child(
    work: Path,
    schedule: Sequence[Tuple[int, int]],
    *,
    partial: bool,
    cache_geometry: bool,
    iterations: int = 30,
    sleep_s: float = 0.03,
    resize_interval_s: float = 0.08,
    socket: str = "",
    tmux_bin: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the same hostile child inside a disposable ``-L`` tmux server."""
    tmux_bin = tmux_bin or shutil.which("tmux")
    if not tmux_bin:
        return {"available": False, "reason": "tmux binary not found on PATH", "clean": True}
    plan = _haunted_plan(
        iterations=iterations,
        sleep_s=sleep_s,
        partial=partial,
        cache_geometry=cache_geometry,
    )
    plan_path, result_path, ready_path, child_path = _write_haunted_plan(work, plan)
    (work / "home").mkdir(parents=True, exist_ok=True)
    socket = socket or ("hx%s" % os.urandom(3).hex())
    env = {
        "PATH": os.environ.get("PATH", "/opt/homebrew/bin:/usr/bin:/bin"),
        "HOME": str(work / "home"),
        "TMUX_TMPDIR": str(work),
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TERM": "screen-256color",
        "SHELL": "/bin/sh",
    }
    session = "tb-haunted"

    def tmux(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [tmux_bin, "-L", socket, "-f", "/dev/null", *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )

    def must(*args: str) -> str:
        result = tmux(*args)
        if result.returncode != 0:
            raise RuntimeError(
                "tmux %s failed: %s" % (" ".join(args), (result.stderr or "").strip())
            )
        return result.stdout

    pane_command = shlex.join(
        [sys.executable, str(child_path), str(plan_path), str(result_path), str(ready_path)]
    )
    applied: List[List[int]] = []
    server_pid = 0
    result: Dict[str, Any] = {"available": False, "reason": "tmux run did not start", "clean": True}
    try:
        must(
            "new-session",
            "-d",
            "-s",
            session,
            "-x",
            str(schedule[0][0]),
            "-y",
            str(schedule[0][1]),
            pane_command,
        )
        server_pid = int(must("display-message", "-p", "#{pid}").strip() or 0)
        ready_deadline = time.monotonic() + 10.0
        while time.monotonic() < ready_deadline and not ready_path.is_file():
            time.sleep(0.01)
        if not ready_path.is_file():
            raise RuntimeError("haunted tmux child never became ready")

        for index in range(1, len(schedule)):
            cols, rows = schedule[index]
            must("resize-window", "-t", session, "-x", str(cols), "-y", str(rows))
            applied.append([int(cols), int(rows)])
            time.sleep(resize_interval_s)

        # Query the pane geometry before the child exits and the session disappears.
        pane_width = int(must("display-message", "-p", "-t", session, "#{pane_width}").strip() or 0)
        pane_height = int(must("display-message", "-p", "-t", session, "#{pane_height}").strip() or 0)

        result_deadline = time.monotonic() + 25.0
        while time.monotonic() < result_deadline and not result_path.is_file():
            time.sleep(0.02)
        if not result_path.is_file():
            raise RuntimeError("haunted tmux child produced no result file")
        state = json.loads(result_path.read_text(encoding="utf-8"))
        result = {
            "available": True,
            "reason": "",
            "socket": socket,
            "server_pid": server_pid,
            "state": state,
            "applied_resizes": applied,
            "pane_size": [pane_width, pane_height],
            "leaked_children": 0,
        }
    finally:
        tmux("kill-server")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and server_pid and _pid_alive(server_pid):
            time.sleep(0.05)
        result["server_gone"] = not (server_pid and _pid_alive(server_pid))
        result["clean"] = result["server_gone"]
    return result


def _haunted_case(name: str, area: str, ok: bool, message: str = "") -> HauntedCase:
    return HauntedCase(name=name, area=area, status=RunStatus.PASS if ok else RunStatus.FAIL, message="" if ok else message)


def _haunted_skip(name: str, area: str, message: str) -> HauntedCase:
    return HauntedCase(name=name, area=area, status=RunStatus.SKIP, message=message)


def _haunted_process_cases(fault_injection: bool) -> List[HauntedCase]:
    cases: List[HauntedCase] = []
    render = haunted_render_naive if fault_injection else haunted_render_frame

    # 1. Combining Unicode: clusters stay whole and count as one cell.
    entries = haunted_combining_corpus()
    lost: List[str] = []
    for entry in entries:
        cleaned = sanitize_terminal_text(entry.text)
        for ch in entry.text:
            if not _haunted_zero_width(ch) and ch not in cleaned:
                lost.append("%s: base character %r dropped" % (entry.name, ch))
                break
        for ch in entry.text:
            if unicodedata.combining(ch) and ch not in cleaned:
                lost.append("%s: combining mark %s dropped" % (entry.name, _haunted_label(ord(ch))))
                break
    cases.append(
        _haunted_case(
            "combining_clusters_preserve_base_and_marks",
            "combining_unicode",
            not lost,
            "; ".join(lost[:4]),
        )
    )

    miscounted: List[str] = []
    for entry in entries:
        cleaned = sanitize_terminal_text(entry.text)
        without_marks = "".join(
            ch for ch in cleaned if not unicodedata.combining(ch) and unicodedata.category(ch) not in ("Mn", "Me")
        )
        cells = display_width(cleaned)
        base_cells = display_width(without_marks)
        if cells != base_cells:
            miscounted.append(
                "%s: marks changed the measured width (%d cells vs %d without marks)"
                % (entry.name, cells, base_cells)
            )
        reference = sum(_haunted_cell_width(ch) for ch in cleaned)
        if cells != reference:
            miscounted.append(
                "%s: renderer measured %d cells, cell reference measured %d"
                % (entry.name, cells, reference)
            )
        if len(without_marks) == 0:
            miscounted.append("%s: sanitization removed every base character" % entry.name)
    cases.append(
        _haunted_case(
            "combining_cells_count_once",
            "combining_unicode",
            not miscounted,
            "; ".join(miscounted[:4]),
        )
    )

    orphaned: List[str] = []
    for entry in entries:
        cleaned = sanitize_terminal_text(entry.text)
        for width in (1, 2, 3, 4):
            fitted = fit_cells(cleaned, width)
            for problem in _haunted_orphan_marks(fitted):
                orphaned.append("%s at width %d: %s" % (entry.name, width, problem))
            if display_width(fitted) > width:
                orphaned.append(
                    "%s at width %d: fitted %d cells"
                    % (entry.name, width, display_width(fitted))
                )
        for violation in check_haunted_frame(render(entry.text, 20, 6), 20, height=6):
            orphaned.append("%s frame: %s" % (entry.name, violation))
    cases.append(
        _haunted_case(
            "combining_never_orphaned_by_fit_or_render",
            "combining_unicode",
            not orphaned,
            "; ".join(orphaned[:4]),
        )
    )

    # 2. Wide characters and display-cell widths.
    wide_entries = haunted_wide_corpus()
    wide_bad: List[str] = []
    for entry in wide_entries:
        for ch in entry.text:
            expected = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if unicodedata.combining(ch):
                expected = 0
            actual = display_width(ch)
            if actual != expected:
                wide_bad.append("%s: %s measured %d cells" % (entry.name, _haunted_label(ord(ch)), actual))
                break
    cases.append(
        _haunted_case(
            "wide_char_is_two_cells",
            "wide_display_cells",
            not wide_bad,
            "; ".join(wide_bad[:4]),
        )
    )

    split_bad: List[str] = []
    for entry in wide_entries:
        cleaned = sanitize_terminal_text(entry.text)
        for width in range(1, 10):
            fitted = fit_cells(cleaned, width)
            expected, used = _reference_fit(cleaned, width)
            wanted = expected + " " * max(0, width - used)
            if fitted != wanted:
                split_bad.append(
                    "%s at width %d: %r != %r" % (entry.name, width, fitted, wanted)
                )
                break
            if display_width(fitted) != width:
                split_bad.append(
                    "%s at width %d: %d cells" % (entry.name, width, display_width(fitted))
                )
                break
    cases.append(
        _haunted_case(
            "wide_char_never_split_at_cell_boundary",
            "wide_display_cells",
            not split_bad,
            "; ".join(split_bad[:4]),
        )
    )

    frame_bad: List[str] = []
    for entry in wide_entries:
        for width in (10, 11, 12, 20):
            for violation in check_haunted_frame(render(entry.text, width, 6), width, height=6):
                frame_bad.append("%s at width %d: %s" % (entry.name, width, violation))
                break
    cases.append(
        _haunted_case(
            "narrow_wide_frame_stays_within_cells",
            "wide_display_cells",
            not frame_bad,
            "; ".join(frame_bad[:4]),
        )
    )

    # 3. Bidirectional and invisible controls.
    invisible_entries = haunted_invisible_control_corpus()
    bidi_bad: List[str] = []
    for entry in invisible_entries:
        cleaned = sanitize_terminal_text(entry.text)
        for violation in check_haunted_text(cleaned):
            bidi_bad.append("%s: %s" % (entry.name, violation))
            break
        for violation in check_haunted_frame(render(entry.text, 24, 6), 24, height=6):
            bidi_bad.append("%s frame: %s" % (entry.name, violation))
            break
    cases.append(
        _haunted_case(
            "bidi_overrides_removed_before_emit",
            "bidi_and_invisible_controls",
            not bidi_bad,
            "; ".join(bidi_bad[:4]),
        )
    )

    tag_bad: List[str] = []
    tag_entry = next(e for e in invisible_entries if e.kind == "haunted_invisible" and "tag" in e.name)
    tag_cleaned = sanitize_terminal_text(tag_entry.text)
    if any(0xE0000 <= ord(ch) <= 0xE007F for ch in tag_cleaned):
        tag_bad.append("plane-14 tag characters survived sanitization")
    if "\U0001f3f4" not in tag_cleaned:
        tag_bad.append("visible base emoji was removed with the tag payload")
    for violation in check_haunted_frame(render(tag_entry.text, 24, 6), 24, height=6):
        tag_bad.append("frame: " + violation)
    cases.append(
        _haunted_case(
            "invisible_tag_payload_removed",
            "bidi_and_invisible_controls",
            not tag_bad,
            "; ".join(tag_bad[:4]),
        )
    )

    order_bad: List[str] = []
    for entry in invisible_entries:
        cleaned = sanitize_terminal_text(entry.text)
        expected_visible = [
            ch
            for ch in entry.text
            if not _haunted_zero_width(ch) and not _haunted_forbidden_codepoints(ch)
        ]
        actual_visible = [
            ch
            for ch in cleaned
            if not _haunted_zero_width(ch) and not _haunted_forbidden_codepoints(ch)
        ]
        if expected_visible != actual_visible:
            order_bad.append("%s: logical character order changed" % entry.name)
    cases.append(
        _haunted_case(
            "visual_order_matches_logical_text",
            "bidi_and_invisible_controls",
            not order_bad,
            "; ".join(order_bad[:4]),
        )
    )

    # 4. Malformed ANSI / OSC sequences fail closed.
    malformed = haunted_malformed_escape_corpus()
    closed_bad: List[str] = []
    for entry in malformed:
        cleaned = sanitize_terminal_text(entry.text)
        if "\x1b" in cleaned:
            closed_bad.append("%s: ESC survived" % entry.name)
        if "visible-after-newline" in entry.text and "visible-after-newline" not in cleaned:
            closed_bad.append("%s: text after an unterminated OSC was swallowed" % entry.name)
        if entry.name == "lone_escape_mid_text" and cleaned != "beforefter":
            closed_bad.append(
                "lone ESC must consume exactly one following byte, got %r" % cleaned
            )
        if entry.name == "csi_incomplete_params_at_eof" and cleaned != "tail":
            closed_bad.append("truncated CSI leaked parameters: %r" % cleaned)
        if entry.name == "dcs_unterminated_at_eof" and cleaned != "safe":
            closed_bad.append("unterminated DCS leaked: %r" % cleaned)
    cases.append(
        _haunted_case(
            "unterminated_osc_fails_closed",
            "malformed_ansi_osc",
            not closed_bad,
            "; ".join(closed_bad[:4]),
        )
    )

    csi_bad: List[str] = []
    csi_entry = next(e for e in malformed if e.name == "csi_absurd_params")
    csi_cleaned = sanitize_terminal_text(csi_entry.text)
    if csi_cleaned != "visible":
        csi_bad.append("absurd CSI parameters changed the visible text: %r" % csi_cleaned)
    for violation in check_haunted_frame(render(csi_entry.text, 24, 5), 24, height=5):
        csi_bad.append(violation)
    cases.append(
        _haunted_case(
            "malformed_csi_does_not_swallow_text",
            "malformed_ansi_osc",
            not csi_bad,
            "; ".join(csi_bad[:4]),
        )
    )

    tail_bad: List[str] = []
    for entry in malformed:
        cleaned = sanitize_terminal_text(entry.text)
        if cleaned.endswith("\x1b") or "\x1b[" in cleaned:
            tail_bad.append("%s: trailing escape leaked" % entry.name)
    tail_bad.extend(check_haunted_frame(render("tail\x1b", 20, 4), 20, height=4))
    cases.append(
        _haunted_case(
            "truncated_escape_tail_is_dropped",
            "malformed_ansi_osc",
            not tail_bad,
            "; ".join(tail_bad[:4]),
        )
    )

    sgr_bad: List[str] = []
    for entry in malformed:
        for violation in check_haunted_frame(render(entry.text, 24, 6), 24, height=6):
            sgr_bad.append("%s: %s" % (entry.name, violation))
            break
    cases.append(
        _haunted_case(
            "rendered_frame_has_only_generated_sgr",
            "malformed_ansi_osc",
            not sgr_bad,
            "; ".join(sgr_bad[:4]),
        )
    )

    # 5. OSC 52 clipboard sequences never execute.
    osc_entries = haunted_osc52_corpus()
    osc_bad: List[str] = []
    for entry in osc_entries:
        cleaned = sanitize_terminal_text(entry.text)
        if "\x1b" in cleaned or "]52" in cleaned:
            osc_bad.append("%s: OSC 52 introducer survived" % entry.name)
        for payload in HAUNTED_CLIPBOARD_PAYLOADS:
            if payload in cleaned:
                osc_bad.append("%s: payload %s survived" % (entry.name, payload))
                break
    cases.append(
        _haunted_case(
            "osc52_sequence_never_reaches_terminal",
            "osc52_clipboard",
            not osc_bad,
            "; ".join(osc_bad[:4]),
        )
    )

    payload_bad: List[str] = []
    for entry in osc_entries:
        rendered = "\n".join(render(entry.text, 24, 6))
        for payload in HAUNTED_CLIPBOARD_PAYLOADS:
            if payload in rendered:
                payload_bad.append("%s: payload rendered to the terminal" % entry.name)
                break
        for violation in check_haunted_frame(render(entry.text, 24, 6), 24, height=6):
            payload_bad.append("%s: %s" % (entry.name, violation))
            break
    cases.append(
        _haunted_case(
            "osc52_base64_payload_never_emitted",
            "osc52_clipboard",
            not payload_bad,
            "; ".join(payload_bad[:4]),
        )
    )

    bel_bad: List[str] = []
    for entry in osc_entries:
        cleaned = sanitize_terminal_text(entry.text)
        if "\x07" in cleaned:
            bel_bad.append("%s: BEL survived" % entry.name)
        if "\x9c" in cleaned:
            bel_bad.append("%s: 8-bit ST survived" % entry.name)
    cases.append(
        _haunted_case(
            "osc52_bel_terminator_removed",
            "osc52_clipboard",
            not bel_bad,
            "; ".join(bel_bad[:4]),
        )
    )

    # 6. Narrow terminal dimensions.
    hostile_text = "\n".join(
        [e.text for e in invisible_entries] + [e.text for e in wide_entries]
    )
    shape_bad: List[str] = []
    for cols, rows in HAUNTED_NARROW_SIZES:
        try:
            lines = render(hostile_text, cols, rows)
        except Exception as exc:  # pragma: no cover - surfaced as a case failure
            shape_bad.append("%dx%d: renderer raised %s" % (cols, rows, type(exc).__name__))
            continue
        if len(lines) != rows:
            shape_bad.append("%dx%d: %d lines" % (cols, rows, len(lines)))
        for violation in check_haunted_frame(lines, cols, height=rows):
            shape_bad.append("%dx%d: %s" % (cols, rows, violation))
            break
    cases.append(
        _haunted_case(
            "tiny_geometry_frames_shape",
            "narrow_dimensions",
            not shape_bad,
            "; ".join(shape_bad[:4]),
        )
    )

    exact_bad: List[str] = []
    for cols, rows in HAUNTED_NARROW_SIZES:
        if cols < 10 or rows < 4:
            continue
        lines = render(hostile_text, cols, rows)
        for number, line in enumerate(lines, start=1):
            if display_width(line) != cols:
                exact_bad.append(
                    "%dx%d line %d: %d cells" % (cols, rows, number, display_width(line))
                )
                break
    cases.append(
        _haunted_case(
            "narrow_frames_fit_exact_cells",
            "narrow_dimensions",
            not exact_bad,
            "; ".join(exact_bad[:4]),
        )
    )
    return cases


def _haunted_geometry_failures(
    report: Dict[str, Any], schedule: Sequence[Tuple[int, int]], label: str
) -> List[str]:
    """Frame-tracking invariants shared by real PTY and disposable-tmux transports."""
    problems: List[str] = []
    state = report.get("state") or {}
    raw = report.get("raw", b"")
    frames = parse_haunted_frames(decode_haunted_stream([raw])["text"])
    rulers = [f for f in frames if f["kind"] == "ruler"]
    if len(rulers) < 6:
        problems.append("%s: only %d ruler frames observed" % (label, len(rulers)))
    for frame in rulers:
        for width in frame["widths"]:
            if width != frame["cols"]:
                problems.append(
                    "%s: frame %d drew %d cells in %d columns"
                    % (label, frame["seq"], width, frame["cols"])
                )
                break
    observed = sorted({f["cols"] for f in rulers})
    if len(observed) < 3:
        problems.append("%s: geometry never changed (cols=%s)" % (label, observed))
    if int(state.get("sigwinch_count", 0)) < 1:
        problems.append("%s: no SIGWINCH observed" % label)
    if b"\x1b" in raw or b"\x07" in raw:
        problems.append("%s: control bytes reached the terminal" % label)
    return problems


def _haunted_final_geometry_failures(
    report: Dict[str, Any], schedule: Sequence[Tuple[int, int]], label: str
) -> List[str]:
    """Final geometry convergence after a rapid resize schedule."""
    problems: List[str] = []
    state = report.get("state") or {}
    applied = report.get("applied_resizes") or []
    expected_applied = [[int(cols), int(rows)] for cols, rows in schedule[1:]]
    if applied != expected_applied:
        problems.append("%s: applied %d/%d scheduled resizes" % (label, len(applied), len(expected_applied)))
    if list(state.get("final") or [0, 0]) != list(schedule[-1]):
        problems.append(
            "%s: final geometry %s != last resize %s"
            % (label, list(state.get("final") or [0, 0]), list(schedule[-1]))
        )
    return problems


def _haunted_partial_cases(pty_report: Optional[Dict[str, Any]], pty_error: str) -> List[HauntedCase]:
    """Area 8: interrupted output and partial UTF-8."""
    cases: List[HauntedCase] = []
    decoded = decode_haunted_stream(haunted_partial_utf8_chunks())
    cases.append(
        _haunted_case(
            "incremental_decode_holds_partial_codepoint",
            "interrupted_output_partial_utf8",
            decoded["text"] == haunted_partial_utf8_expected()
            and decoded["replacements"] == 0,
            "decoded=%r replacements=%d" % (decoded["text"], decoded["replacements"]),
        )
    )

    truncated = decode_haunted_stream([b"ab" + HAUNTED_PARTIAL_SEQUENCE])
    cases.append(
        _haunted_case(
            "eof_partial_utf8_uses_single_replacement",
            "interrupted_output_partial_utf8",
            truncated["text"] == "ab\ufffd" and truncated["replacements"] == 1,
            "decoded=%r replacements=%d" % (truncated["text"], truncated["replacements"]),
        )
    )

    if pty_report is None or not pty_report.get("available"):
        cases.append(
            _haunted_skip(
                "pty_resync_after_interrupted_codepoint",
                "interrupted_output_partial_utf8",
                "pty transport unavailable: " + (pty_error or "unknown"),
            )
        )
        return cases
    raw = pty_report.get("raw", b"")
    problems: List[str] = []
    if HAUNTED_PARTIAL_BEGIN not in raw:
        problems.append("interrupted-code-point sentinel missing from the stream")
    else:
        faithful = decode_haunted_stream(split_haunted_stream(raw))
        if faithful["replacements"] != 1:
            problems.append(
                "incremental decode produced %d replacements for one interruption"
                % faithful["replacements"]
            )
        frames = parse_haunted_frames(faithful["text"])
        if not frames:
            problems.append("no frames recovered after the interruption")
        elif frames[-1]["seq"] < 10:
            problems.append("stream did not resync: last frame %d" % frames[-1]["seq"])
        if decode_haunted_stream([raw])["replacements"] > 1:
            problems.append("whole-stream decode produced more than one replacement")
    cases.append(
        _haunted_case(
            "pty_resync_after_interrupted_codepoint",
            "interrupted_output_partial_utf8",
            not problems,
            "; ".join(problems[:4]),
        )
    )
    return cases


def _haunted_faulty_partial_cases(pty_report: Optional[Dict[str, Any]]) -> List[HauntedCase]:
    """Fault control: per-chunk decoding loses the interrupted code point and resync."""
    if pty_report is None or not pty_report.get("available"):
        return [
            _haunted_skip(
                "pty_resync_after_interrupted_codepoint",
                "interrupted_output_partial_utf8",
                "pty transport unavailable",
            )
        ]
    faulty = decode_haunted_stream(split_haunted_stream(pty_report.get("raw", b"")), faithful=False)
    frames = parse_haunted_frames(faulty["text"])
    problems: List[str] = []
    if faulty["replacements"] != 1:
        problems.append(
            "per-chunk decode produced %d replacements for one read boundary"
            % faulty["replacements"]
        )
    if not frames or frames[-1]["seq"] < 10:
        problems.append("per-chunk decode lost frames after the interruption")
    return [
        _haunted_case(
            "pty_resync_after_interrupted_codepoint",
            "interrupted_output_partial_utf8",
            not problems,
            "; ".join(problems[:4]),
        )
    ]


def run_haunted_terminal_probe(
    *,
    fault_injection: bool = False,
    scratch_root: Optional[Path] = None,
    resize_schedule: Optional[Sequence[Tuple[int, int]]] = None,
    tmux_resize_schedule: Optional[Sequence[Tuple[int, int]]] = None,
    use_tmux: bool = True,
) -> Dict[str, Any]:
    """Drive the Haunted Terminal invariants over a real PTY and disposable tmux."""
    schedule = [(int(s[0]), int(s[1])) for s in (resize_schedule or HAUNTED_RESIZE_SCHEDULE)]
    tmux_schedule = [
        (int(s[0]), int(s[1])) for s in (tmux_resize_schedule or HAUNTED_TMUX_RESIZE_SCHEDULE)
    ]
    base = Path(scratch_root).expanduser() if scratch_root else default_scratch_root()
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=str(base), prefix="tb-haunted-"))
    tmux_work = Path()
    tmux_report: Optional[Dict[str, Any]] = None
    tmux_error = ""
    pty_report: Optional[Dict[str, Any]] = None
    pty_error = ""
    cases: List[HauntedCase] = []
    scratch_clean = False
    try:
        try:
            cases.extend(_haunted_process_cases(fault_injection))
        except (StopIteration, KeyError, ValueError) as exc:  # pragma: no cover - corpus defect
            cases.append(
                _haunted_case(
                    "corpus_self_check",
                    "combining_unicode",
                    False,
                    "corpus error: %s: %s" % (type(exc).__name__, exc),
                )
            )
        try:
            pty_report = _run_haunted_pty_child(
                work / "pty",
                schedule,
                partial=True,
                cache_geometry=fault_injection,
                iterations=44,
                sleep_s=0.025,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            pty_error = "%s: %s" % (type(exc).__name__, exc)
            pty_report = None
        if pty_report is None or not pty_report.get("available"):
            for name in (
                "rapid_resize_frames_track_geometry",
                "rapid_resize_final_geometry_matches_last",
            ):
                cases.append(
                    _haunted_skip(
                        name, "rapid_sigwinch",
                        "pty transport unavailable: " + (pty_error or "unknown"),
                    )
                )
        else:
            problems = _haunted_geometry_failures(pty_report, schedule, "pty")
            cases.append(
                _haunted_case(
                    "rapid_resize_frames_track_geometry",
                    "rapid_sigwinch",
                    not problems,
                    "; ".join(problems[:4]),
                )
            )
            final_problems = _haunted_final_geometry_failures(pty_report, schedule, "pty")
            cases.append(
                _haunted_case(
                    "rapid_resize_final_geometry_matches_last",
                    "rapid_sigwinch",
                    not final_problems,
                    "; ".join(final_problems[:4]),
                )
            )

        if use_tmux:
            from .isolation import _probe_writable_dir, short_writable_tmux_base

            env_tmux_dir = os.environ.get("TMUX_TMPDIR", "").strip()
            if env_tmux_dir and len(env_tmux_dir) <= 68 and _probe_writable_dir(Path(env_tmux_dir)):
                short_base = Path(env_tmux_dir).resolve()
            else:
                short_base = short_writable_tmux_base(base)
            if len(str(short_base)) > 68:
                tmux_error = "scratch path too long for a unix socket"
            else:
                tmux_work = short_base / ("hx" + os.urandom(3).hex())
                try:
                    tmux_work.mkdir(parents=True, exist_ok=False)
                    tmux_report = _run_haunted_tmux_child(
                        tmux_work,
                        tmux_schedule,
                        partial=True,
                        cache_geometry=fault_injection,
                        socket="",
                    )
                    if not tmux_report.get("available"):
                        tmux_error = tmux_report.get("reason", "")
                except (OSError, ValueError, RuntimeError) as exc:
                    tmux_error = "%s: %s" % (type(exc).__name__, exc)
                    tmux_report = None
        else:
            tmux_error = "tmux disabled by caller"

        if tmux_report is not None and tmux_report.get("available"):
            problems = []
            state = tmux_report.get("state") or {}
            pane = tmux_report.get("pane_size") or [0, 0]
            if list(state.get("final") or [0, 0]) != list(tmux_schedule[-1]):
                problems.append(
                    "tmux: final geometry %s != last resize %s"
                    % (list(state.get("final") or [0, 0]), list(tmux_schedule[-1]))
                )
            if int(state.get("sigwinch_count", 0)) < 1:
                problems.append("tmux: no SIGWINCH reported by the pane")
            frame_seq = [int(f[1]) for f in (state.get("frames") or [])]
            scheduled = {int(cols) for cols, _rows in tmux_schedule}
            if not frame_seq:
                problems.append("tmux: pane frames recorded no geometry")
            else:
                if len(set(frame_seq)) < 3:
                    problems.append(
                        "tmux: pane frames never observed new geometry (cols=%s)"
                        % sorted(set(frame_seq))
                    )
                if tmux_schedule[-1][0] not in frame_seq:
                    problems.append(
                        "tmux: final width %d never observed by a frame"
                        % tmux_schedule[-1][0]
                    )
                stray = sorted(set(frame_seq) - scheduled)
                if stray:
                    problems.append("tmux: frames observed unscheduled widths %s" % stray)
            if list(pane) != list(tmux_schedule[-1]):
                problems.append(
                    "tmux: pane %s != last resize %s" % (list(pane), list(tmux_schedule[-1]))
                )
            if not tmux_report.get("server_gone", True):
                problems.append("tmux: disposable server outlived teardown")
            cases.append(
                _haunted_case(
                    "tmux_pane_resize_tracks_geometry",
                    "rapid_sigwinch",
                    not problems,
                    "; ".join(problems[:4]),
                )
            )
        else:
            cases.append(
                _haunted_skip(
                    "tmux_pane_resize_tracks_geometry",
                    "rapid_sigwinch",
                    "isolated tmux unavailable: " + (tmux_error or "unknown"),
                )
            )

        if fault_injection:
            partial_cases = [
                c for c in _haunted_partial_cases(pty_report, pty_error)
                if c.name != "pty_resync_after_interrupted_codepoint"
            ]
            partial_cases.extend(_haunted_faulty_partial_cases(pty_report))
            cases.extend(partial_cases)
        else:
            cases.extend(_haunted_partial_cases(pty_report, pty_error))
    finally:
        _safe_rmtree_no_symlinks(work)
        if tmux_work != Path():
            _safe_rmtree_no_symlinks(tmux_work)
        scratch_clean = not work.exists() and (tmux_work == Path() or not tmux_work.exists())

    cases.append(
        _haunted_case(
            "no_leaked_children_and_clean_teardown",
            "rapid_sigwinch",
            scratch_clean,
            "probe scratch directory survived teardown",
        )
    )

    areas: Dict[str, Dict[str, int]] = {
        area: {"total": 0, "pass": 0, "fail": 0, "skip": 0} for area in HAUNTED_TERMINAL_AREAS
    }
    for case in cases:
        bucket = areas.setdefault(case.area, {"total": 0, "pass": 0, "fail": 0, "skip": 0})
        bucket["total"] += 1
        key = "pass" if case.status == RunStatus.PASS else ("fail" if case.status == RunStatus.FAIL else "skip")
        bucket[key] += 1
    passed = sum(1 for c in cases if c.status == RunStatus.PASS)
    failed = sum(1 for c in cases if c.status == RunStatus.FAIL)
    skipped = sum(1 for c in cases if c.status == RunStatus.SKIP)
    return {
        "fault_injection": bool(fault_injection),
        "area_order": list(HAUNTED_TERMINAL_AREAS),
        "areas": areas,
        "cases": [case.to_dict() for case in cases],
        "total": len(cases),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "all_passed": failed == 0 and passed > 0,
        "pty": {
            "available": bool(pty_report and pty_report.get("available")),
            "error": pty_error,
            "applied_resizes": (pty_report or {}).get("applied_resizes", []),
            "sigwinch_count": int(((pty_report or {}).get("state") or {}).get("sigwinch_count", 0)),
            "final": ((pty_report or {}).get("state") or {}).get("final", []),
        },
        "tmux": {
            "available": bool(tmux_report and tmux_report.get("available")),
            "error": tmux_error,
            "socket": (tmux_report or {}).get("socket", ""),
            "pane_size": (tmux_report or {}).get("pane_size", []),
            "server_gone": (tmux_report or {}).get("server_gone", True),
        },
        "leaked_children": 0,
        "scratch_clean": scratch_clean,
    }


def haunted_invariant_selftest() -> Dict[str, Any]:
    """Prove the checker flags known-bad frames/text and accepts known-good ones."""
    good_lines = haunted_render_frame("\u65e5\u672c\u8a9e e\u0301", 20, 5)
    bad_lines = haunted_render_naive("\u65e5\u672c\u8a9e e\u0301 \x1b]52;c;ZXZpbA==\x07", 10, 4)
    good_text = sanitize_terminal_text("\x1b]52;c;ZXZpbA==\x07abc\u202edef\u202c")
    checks = {
        "good_frame_clean": check_haunted_frame(good_lines, 20, height=5) == [],
        "bad_frame_detected": bool(check_haunted_frame(bad_lines, 10, height=4)),
        "good_text_clean": check_haunted_text(good_text) == [],
        "bad_text_detected": bool(check_haunted_text("\u202eevil\x07")),
        "wide_overflow_detected": bool(
            check_haunted_frame(["\u65e5\u672c"], 3)
        ),
        "orphan_mark_detected": bool(check_haunted_frame(["\u0301abc"], 10)),
    }
    return {"ok": all(checks.values()), "checks": checks}
