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
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import select
import shutil
import signal
import sqlite3
import struct
import subprocess
import sys
import tempfile
import termios
import time
from typing import Any, Dict, List, Optional, Tuple

from .isolation import default_scratch_root, pgroup_alive, terminate_process_group
from .store import TestBenchStore, _safe_rmtree_no_symlinks

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
