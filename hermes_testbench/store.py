"""Authoritative SQLite WAL state store, lock reconciler, trust registry, and retention.

This module owns everything that outlives a process. Three surfaces live here
and they answer different questions:

* **``state.db``** (``0700`` dir, ``0600`` file, WAL) — the record of what was
  run and how it ended. Not a cache: ``run.json``, ``junit.xml``, ``repro.sh``
  and every API/TUI view are derived from the row, never the other way round.
  Losing the artifacts costs evidence; losing the row costs the history.
* **``runs/<run_id>/``** — the evidence directory: the child's logs, the
  derived artifacts, and ``run.lock``. The lock file's *held-ness* is the only
  liveness signal the store has, which is what ``reconcile_orphans`` keys on.
* **``projects`` rows** — the trust registry: a project root plus the SHA-256
  of the ``.testbench.toml`` that was approved. A digest mismatch is
  "untrusted", and the gate in ``engine`` reads that via
  ``is_manifest_trusted``.

Concurrency model. One ``sqlite3.Connection`` per ``TestBenchStore`` instance
(``check_same_thread=False``, autocommit, ``isolation_level=None``); all writes
go through ``BEGIN IMMEDIATE`` so writers queue in the database rather than
interleaving read-modify-write, and ``busy_timeout=5000`` absorbs contention
from the CLI, a detached ``_worker``, the TUI and the MCP server sharing one
file. Readers aren't blocked by the writer under WAL; the TUI polls
``PRAGMA data_version`` to notice external commits. ``synchronous=NORMAL`` is
the WAL-appropriate durability point: a power loss can drop the tail of the
WAL, a process crash cannot corrupt it. The single connection is *not* a
thread-safe write path — two threads entering ``_write_tx`` on one instance
would nest ``BEGIN`` and raise; give each thread its own instance.

Status truthfulness. ``QUEUED``/``RUNNING`` are claims about a live process,
and the store cannot verify them directly. The contract is: a run that is
genuinely being executed holds ``runs/<id>/run.lock`` (``flock``), so an
ACTIVE row whose lock is unheld is stale and gets reconciled to ``ERROR`` /
``abandoned_worker``. That inference is why an interrupted run leaves
``ERROR`` behind as *legitimate history*: SIGINT kills the foreground process,
the kernel drops the lock, and the next engine construction records the truth.
Nothing deletes or relabels those rows — an ``ERROR`` from a Ctrl-C is the
correct answer, not a defect to be tidied away. The same rule means pruning
treats them as ordinary history and the TUI shows them as failures.

Derived-vs-authoritative. ``finalize_run`` writes the artifacts *before* the
row, so a crash between the two leaves artifacts with no row (visible as a
directory with no history entry) rather than a row pointing at nothing. That
ordering is deliberate: a row without artifacts would be a claim we cannot
back.

Redaction. Every string persisted here passes ``sanitize.redact_secrets`` and
``sanitize.sanitize_terminal_text`` (``_upsert_run_row``), mutating the caller's
``RunRecord`` in place so the in-memory view and the stored view agree. Raw
streams never reach this module: ``isolation`` redacts before a log byte is
written.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
from typing import Any, Dict, Iterator, List, Optional

from .reporting import format_repro_script, render_junit_xml, render_run_json
from .sanitize import redact_secrets, sanitize_terminal_text
from .schema import (
    FailureKind,
    RunRecord,
    RunStatus,
    SCHEMA_VERSION,
    run_from_dict,
    run_to_dict,
    utc_now_iso,
)


class SchemaMismatchError(RuntimeError):
    """Raised when state.db PRAGMA user_version disagrees with SCHEMA_VERSION.

    Deliberately fatal and unrecoverable in-process: there is no migration
    path, so the only correct responses are "run the release that wrote this
    store" or "start a new home". Downgrading to a best-effort open would let
    one release read another's payloads through ``run_from_dict``'s strict
    constructor and fail later, mid-run, with a far worse error.
    """


def default_testbench_home() -> Path:
    """``$TESTBENCH_HOME``, else ``$XDG_DATA_HOME/testbench``, else ``~/.local/share/testbench``.

    Explicit env wins because the isolation layer points children at a
    disposable ``TESTBENCH_HOME``; the XDG step is what makes the installed
    CLI and a dev checkout share one history on Linux.
    """
    env_home = os.environ.get("TESTBENCH_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    if xdg:
        return Path(xdg).expanduser() / "testbench"
    return Path.home() / ".local" / "share" / "testbench"


def _exclude_from_time_machine(path: Path) -> None:
    """Best-effort ``com.apple.backupd`` exclusion on a *newly created* home.

    Called only when the home directory did not exist before this store opened
    it, so a pre-existing home that lost the xattr (restore, migration, a
    ``cp -R``) stays included in backups and nothing re-applies it. Failures
    are swallowed on purpose: backup policy is not a reason to fail a test
    run, and ``xattr`` may be missing off macOS.
    """
    if shutil.which("xattr") is None:
        return
    try:
        subprocess.run(
            ["xattr", "-w", "com.apple.metadata:com_apple_backup_excludeItem", "com.apple.backupd", str(path)],
            capture_output=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def atomic_write_file(path: Path, text: str, mode: int = 0o600) -> None:
    """Write ``text`` to ``path`` atomically at ``mode``, reader-visible all-or-nothing.

    Temp file in the same directory (so ``os.replace`` stays a rename, not a
    copy), ``O_EXCL`` plus a pid/random suffix so two writers — the CLI and a
    detached worker both finalising, or the TUI and a worker — cannot share a
    temp file, ``fsync`` before rename so the content is durable, then
    ``os.replace``. The directory entry itself is not fsynced: after a power
    loss the file may be old, but it is never truncated or half-written, which
    is the property ``run.json`` and ``repro.sh`` actually need.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{os.urandom(4).hex()}"
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), mode)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(path))
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _safe_rmtree_no_symlinks(target: Path) -> None:
    """Remove a directory tree without ever traversing symlinks.

    A run directory is written by whatever the suite did, so retention must
    assume it contains links pointing at ``~/.ssh`` or the checkout. Entries
    that are symlinks (or not directories when followed=False) are unlinked,
    never descended into; only real directories recurse. The tradeoff is that
    a symlinked *parent* would still be followed by ``os.scandir``, which is
    why callers only ever pass ``runs_dir/<run_id>``.
    """
    if not target.exists() and not target.is_symlink():
        return
    if target.is_symlink() or target.is_file():
        target.unlink(missing_ok=True)
        return
    for entry in list(os.scandir(str(target))):
        p = Path(entry.path)
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            p.unlink(missing_ok=True)
        else:
            _safe_rmtree_no_symlinks(p)
    target.rmdir()


# Initial schema, executed statement-by-statement inside one transaction and
# only when the file is genuinely empty (user_version 0 *and* no ``runs``
# table). Every column here is duplicated inside ``payload_json`` on purpose:
# the columns exist for indexing, filtering, ordering and reconciliation, and
# the payload is what is read back. ``created_at`` is TEXT, not a timestamp,
# because ordering is lexicographic on the stored ISO form and the normalising
# CASE expressions below (which append ``.000000`` to whole-second stamps) are
# part of the contract — see ``list_runs``.
#
# ``benchmarks.run_id`` cascades on delete so pruning a run takes its
# measurements with it; ``foreign_keys = ON`` is set per connection for that
# to hold.
_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    project_root TEXT NOT NULL,
    revision TEXT NOT NULL,
    branch TEXT NOT NULL DEFAULT '',
    profile TEXT NOT NULL,
    status TEXT NOT NULL,
    failure_kind TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    duration_ms REAL,
    exit_code INTEGER,
    total_count INTEGER NOT NULL DEFAULT 0,
    pass_count INTEGER NOT NULL DEFAULT 0,
    fail_count INTEGER NOT NULL DEFAULT 0,
    skip_count INTEGER NOT NULL DEFAULT 0,
    blocked_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    first_failure TEXT NOT NULL DEFAULT '',
    repro_instructions TEXT NOT NULL DEFAULT '',
    evidence_dir TEXT NOT NULL DEFAULT '',
    pinned INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_project_created ON runs(project, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);

CREATE TABLE IF NOT EXISTS projects (
    project_root TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    manifest_sha256 TEXT,
    trusted INTEGER NOT NULL DEFAULT 0,
    registered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmarks (
    bench_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    project TEXT NOT NULL,
    workload TEXT NOT NULL,
    revision TEXT NOT NULL,
    created_at TEXT NOT NULL,
    median_ms REAL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_bench_proj_workload ON benchmarks(project, workload, created_at DESC);
"""


class TestBenchStore:
    """Process-local handle on the shared store; not thread-safe for writes.

    Construction is also a maintenance step: it creates/repairs the directory
    modes, opens the database (or refuses on a schema mismatch), and callers
    are expected to run ``reconcile_orphans`` before trusting ACTIVE rows —
    ``TestBenchEngine.__init__`` and every MCP dispatch do exactly that, which
    is why opening a store can change history without any run happening.

    Use as a context manager (or ``close()``); ``__del__`` is a backstop for
    the interpreter exiting, not a lifecycle. ``self._conn`` is one connection
    with autocommit semantics enabled; see the module docstring for why writes
    are serialised with ``BEGIN IMMEDIATE`` rather than by an in-process lock.
    """

    def __init__(self, home: Optional[Path] = None) -> None:
        self.home = (home or default_testbench_home()).expanduser().resolve()
        self.runs_dir = self.home / "runs"
        self.projects_dir = self.home / "projects"
        is_new_home = not self.home.exists()
        self.home.mkdir(parents=True, exist_ok=True)
        os.chmod(str(self.home), 0o700)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(self.runs_dir), 0o700)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(self.projects_dir), 0o700)
        if is_new_home:
            _exclude_from_time_machine(self.home)

        self.db_path = self.home / "state.db"
        self._closed = False
        self._conn: sqlite3.Connection = self._open_and_init_db()

    def _open_and_init_db(self) -> sqlite3.Connection:
        """Open (creating if needed), apply pragmas, and gate on schema version.

        Order matters. The file is ``chmod 0600`` through a raw fd *before*
        sqlite3 ever opens it, so there is no window where a world-readable
        ``state.db`` exists. WAL is requested per connection (idempotent; the
        mode is a property of the file) and takes a brief exclusive lock, which
        is one more reason ``busy_timeout`` is set before it.

        The three-way branch is the whole migration policy: an empty file gets
        the DDL and the version stamp in one transaction; anything else must
        already carry exactly ``SCHEMA_VERSION``. A file with tables but
        ``user_version = 0`` (hand-made, or a truncated first init) is refused
        rather than "repaired", because guessing whether it is empty would
        decide whether history is about to be overwritten.
        """
        fd = os.open(
            str(self.db_path),
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        os.fchmod(fd, 0o600)
        os.close(fd)
        conn = sqlite3.connect(str(self.db_path), timeout=5.0, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA foreign_keys = ON")

            ver = int(conn.execute("PRAGMA user_version").fetchone()[0])
            has_runs = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'"
            ).fetchone() is not None

            if ver == 0 and not has_runs:
                with self._write_tx_conn(conn):
                    for stmt in [s.strip() for s in _DDL.split(";") if s.strip()]:
                        conn.execute(stmt)
                    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            elif ver != SCHEMA_VERSION:
                raise SchemaMismatchError(
                    f"state.db schema user_version={ver} != expected {SCHEMA_VERSION} ({self.db_path})"
                )
            return conn
        except Exception:
            conn.close()
            raise

    def close(self) -> None:
        """Close the connection, idempotently; an open transaction is rolled back.

        Safe to call twice, safe after ``__del__``. Closing is how an
        unterminated transaction (an exception that escaped ``_write_tx``
        without a ROLLBACK) gets discarded, so callers that catch interrupts
        must still close the store or the next write on that handle fails.
        """
        if getattr(self, "_closed", True):
            return
        self._closed = True
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def __enter__(self) -> "TestBenchStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @staticmethod
    @contextmanager
    def _write_tx_conn(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` / COMMIT, ROLLBACK on ``Exception``.

        IMMEDIATE (not DEFERRED) reserves the write lock up front, so a
        read-then-write inside the block cannot lose to a concurrent writer and
        cannot fail with SQLITE_BUSY at COMMIT; contention shows up as a wait
        on ``BEGIN`` instead, bounded by ``busy_timeout``.

        The ``except Exception`` is deliberately not ``BaseException``:
        an interrupt escaping here leaves the transaction open, and the next
        write on that connection raises "cannot start a transaction within a
        transaction" — survivable only because the CLI tears its store down.
        A long-lived in-process holder (TUI, MCP) that swallows KeyboardInterrupt
        would wedge. Narrow window, but it is a real one.
        """
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        else:
            conn.execute("COMMIT")

    @contextmanager
    def _write_tx(self) -> Iterator[sqlite3.Connection]:
        """Write transaction on the instance connection; see ``_write_tx_conn``."""
        with self._write_tx_conn(self._conn) as conn:
            yield conn

    def journal_mode(self) -> str:
        """Actual ``PRAGMA journal_mode`` ("wal" in normal operation).

        Read from SQLite rather than assumed, because the mode lives in the
        file: a store directory restored from a filesystem that was captured
        mid-checkpoint can come back in ``delete`` mode, and this is the
        cheapest way for a diagnosis to notice.
        """
        row = self._conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]) if row else ""

    def data_version(self) -> int:
        """SQLite's external-change counter for this connection.

        Advances when *another* connection commits, not when this one does —
        which is precisely the signal the TUI needs to redraw on a worker's or
        a CLI run's write, and why its own writes don't cause refresh loops.
        """
        row = self._conn.execute("PRAGMA data_version").fetchone()
        return int(row[0]) if row else 0

    def run_dir(self, run_id: str) -> Path:
        """Evidence directory for ``run_id``, created ``0700`` if absent.

        The traversal guard is not cosmetic: ``run_id`` reaches here from CLI
        arguments and MCP payloads, and ``runs/../../..`` would otherwise put
        evidence (and pruning, which rmtree's this path) outside the store. The
        check rejects separators and ``..`` while allowing the normal
        ``tb_<stamp>_<hex>`` shape; callers must not assume it resolves
        anything — it is a string guard only.
        """
        if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
            raise ValueError(f"Invalid run_id: {run_id!r}")
        d = self.runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(str(d), 0o700)
        return d

    def run_lock_path(self, run_id: str) -> Path:
        """Path of the liveness lock inside the run's evidence directory."""
        return self.run_dir(run_id) / "run.lock"

    @contextmanager
    def hold_run_lock(self, run_id: str) -> Iterator[Path]:
        """Hold ``flock(LOCK_EX)`` on the run's lock for the block's duration.

        This is the writer half of the liveness protocol: while the block runs,
        ``reconcile_orphans`` and ``prune_runs`` will treat the run as alive and
        leave it alone. The lock is advisory and tied to the open file
        description, so the kernel drops it when the process dies — which is
        exactly why an unheld lock is trustworthy evidence of death, and why
        this must never be replaced by a pid file or a timestamp heuristic.

        Non-blocking: contention raises ``OSError`` (``BlockingIOError``)
        immediately rather than queueing. A caller that loses the race must
        decide what to do — the engine lets it propagate, because "someone else
        already owns this run" is not a state to paper over.
        """
        lpath = self.run_lock_path(run_id)
        fd = os.open(
            str(lpath),
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        os.fchmod(fd, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise
        try:
            yield lpath
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def is_run_locked(self, run_id: str) -> bool:
        """Is some process holding this run's lock right now?

        "Unknown" is folded into "not held" on purpose, and that asymmetry is
        the whole reconciliation policy: a missing lock file, or one we cannot
        even open, means we have no evidence of a live owner, so the run is
        declared dead and the row is corrected. ``EACCES``/``EAGAIN`` from
        ``flock`` is the only positive signal (someone else holds it); any
        other error is treated as "no owner" rather than as an error to raise,
        because a reconciler that cannot run is a store that shows phantom
        RUNNING rows forever.

        Probing costs an open + flock + unlock; callers reconcile on every
        engine construction and TUI refresh, so this is deliberately cheap
        rather than cached — a cached answer would be the stale one.

        The cost of collapsing unknown into gone: a run that has not taken the
        lock *yet* is indistinguishable from one whose process died. See
        ``reconcile_orphans`` for the one window where that matters.
        """
        lpath = self.runs_dir / run_id / "run.lock"
        if not lpath.is_file():
            return False
        fd = -1
        try:
            fd = os.open(str(lpath), os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            return exc.errno in (errno.EACCES, errno.EAGAIN)
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            return False
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _prepare_run_paths(self, run: RunRecord) -> None:
        """Fill ``evidence_dir``/``run.json``/``junit.xml``/``repro.sh`` on the record.

        Called by every write path, so a record is always self-describing about
        where its evidence lives; ``created_at`` is only defaulted, never
        overwritten, because it is the immutable ordering key. The directory is
        created here as a side effect — writing a record always materialises a
        ``0700`` run directory, even for a run that never executes.
        """
        rdir = self.run_dir(run.run_id)
        run.evidence_dir = str(rdir)
        run.run_json_path = str(rdir / "run.json")
        run.junit_xml_path = str(rdir / "junit.xml")
        run.repro_path = str(rdir / "repro.sh")
        if not run.created_at:
            run.created_at = utc_now_iso()

    def _export_terminal_artifacts(self, run: RunRecord) -> None:
        """Write ``run.json`` / ``junit.xml`` / ``repro.sh`` at rest as derived files.

        Derived, not independent: all three are rendered from the same
        ``RunRecord`` the row will carry, so they can never disagree with the
        database except by a crash between the write and the commit (in which
        case the artifacts are the orphans). ``repro.sh`` gets ``0700`` because
        it is executable; the two data files are ``0600``.

        This runs *before* ``_upsert_run_row`` in ``finalize_run`` and in the
        reconciler, so a failure here leaves the row untouched rather than
        recording a terminal state whose evidence is missing.
        """
        self._prepare_run_paths(run)
        assert run.run_json_path and run.junit_xml_path and run.repro_path
        atomic_write_file(Path(run.run_json_path), render_run_json(run), mode=0o600)
        atomic_write_file(Path(run.junit_xml_path), render_junit_xml(run), mode=0o600)
        atomic_write_file(Path(run.repro_path), format_repro_script(run), mode=0o700)

    def _upsert_run_row(self, conn: sqlite3.Connection, run: RunRecord) -> None:
        """Insert-or-replace the row and its payload in the caller's transaction.

        Three things happen here that are easy to miss from the call sites:

        1. **Redaction, in place.** ``summary``, ``first_failure`` and
           ``repro_instructions`` are sanitized and written back onto the passed
           ``RunRecord``. The mutation is intentional — the caller's object and
           the row must agree, and ``payload`` is rendered after the rewrite —
           but it means a holder of that record sees redacted text after a
           write, not the raw text it constructed.
        2. **The payload is the record.** ``render_run_json`` applies the same
           sanitizers recursively, so commands, case names and environment all
           get the treatment, and ``get_run`` can return exactly what was
           stored.
        3. **Six columns are insert-only.** ``project``, ``project_root``,
           ``revision``, ``branch``, ``profile`` and ``created_at`` are absent
           from the DO UPDATE clause: identity and ordering are frozen at
           insert. A caller that moves a run between projects by editing those
           fields gets a silent no-op on the columns while ``payload_json``
           changes — the row then disagrees with its own payload.

        The upsert is what makes ``create_run`` idempotent (and what makes an
        accidental run_id collision overwrite rather than fail).
        """
        clean_summary = sanitize_terminal_text(redact_secrets(run.summary or ""))
        clean_fail = sanitize_terminal_text(redact_secrets(run.first_failure or ""))
        clean_repro = sanitize_terminal_text(redact_secrets(run.repro_instructions or ""))
        run.summary = clean_summary
        run.first_failure = clean_fail
        run.repro_instructions = clean_repro
        payload = render_run_json(run)
        conn.execute(
            """
            INSERT INTO runs (
                run_id, project, project_root, revision, branch, profile, status, failure_kind,
                created_at, started_at, finished_at, duration_ms, exit_code,
                total_count, pass_count, fail_count, skip_count, blocked_count, error_count,
                summary, first_failure, repro_instructions, evidence_dir, pinned, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status=excluded.status,
                failure_kind=excluded.failure_kind,
                started_at=excluded.started_at,
                finished_at=excluded.finished_at,
                duration_ms=excluded.duration_ms,
                exit_code=excluded.exit_code,
                total_count=excluded.total_count,
                pass_count=excluded.pass_count,
                fail_count=excluded.fail_count,
                skip_count=excluded.skip_count,
                blocked_count=excluded.blocked_count,
                error_count=excluded.error_count,
                summary=excluded.summary,
                first_failure=excluded.first_failure,
                repro_instructions=excluded.repro_instructions,
                evidence_dir=excluded.evidence_dir,
                pinned=excluded.pinned,
                payload_json=excluded.payload_json
            """,
            (
                run.run_id,
                run.project,
                run.project_root,
                run.revision,
                run.branch,
                run.profile,
                run.status,
                run.failure_kind,
                run.created_at,
                run.started_at,
                run.finished_at,
                run.duration_ms,
                run.exit_code,
                run.total_count,
                run.pass_count,
                run.fail_count,
                run.skip_count,
                run.blocked_count,
                run.error_count,
                clean_summary,
                clean_fail,
                clean_repro,
                run.evidence_dir,
                1 if run.pinned else 0,
                payload,
            ),
        )

    def create_run(self, run: RunRecord) -> RunRecord:
        """Persist a new run (QUEUED or RUNNING) and return it, paths filled.

        Despite the name this is an upsert: passing an existing ``run_id``
        replaces that row and its ``payload_json`` without complaint. Callers
        rely on that nowhere, but nothing prevents it either — see
        ``_upsert_run_row`` for which fields survive a re-create.
        """
        self._prepare_run_paths(run)
        with self._write_tx() as conn:
            self._upsert_run_row(conn, run)
        return run

    def update_run(self, run: RunRecord) -> RunRecord:
        """Persist mid-run progress (worker pid/pgid/fingerprint, session start).

        Same upsert as ``create_run``; the distinction is intent, not
        mechanics. Artifacts are *not* written here — a run in progress has no
        terminal report, and writing one would make a half-done run look
        finished to anything that reads the directory instead of the database.
        """
        self._prepare_run_paths(run)
        with self._write_tx() as conn:
            self._upsert_run_row(conn, run)
        return run

    def finalize_run(self, run: RunRecord) -> RunRecord:
        """Stamp ``finished_at``, write terminal artifacts, then commit the row.

        The only settlement path for a run. It does not verify that
        ``run.status`` is terminal and does not consult the row it is
        replacing, so it is last-writer-wins by construction — which is why
        ``engine._populate_and_finalize`` re-reads the row first and refuses to
        downgrade a ``CANCELLED`` it finds there. Any other pre-existing status
        (including an ``ERROR`` written by ``reconcile_orphans``) is simply
        overwritten by whoever calls this last.
        """
        if not run.finished_at:
            run.finished_at = utc_now_iso()
        self._export_terminal_artifacts(run)
        with self._write_tx() as conn:
            self._upsert_run_row(conn, run)
        return run

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Read one run out of ``payload_json``; ``None`` when the row is absent.

        The columns are never consulted on read — they exist for SQL. That
        keeps one code path for the shape of a record (``run_from_dict``) and
        means a payload/column disagreement can only be seen by SQL, never
        papered over on the way in.
        """
        row = self._conn.execute(
            "SELECT payload_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not row:
            return None
        return run_from_dict(json.loads(row["payload_json"]))

    def list_runs(
        self,
        *,
        project: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[RunRecord]:
        """Newest-first history, filtered on columns, ordered by a normalised timestamp.

        The ``CASE`` expression is the reason same-second runs still sort: a
        whole-second stamp (``...T22:00:00Z``) would compare *shorter* and thus
        older than a fractional one (``...T22:00:00.123456Z``), so it is padded
        before the string comparison. SQLite has no timestamp type here on
        purpose — the ISO form is the sort key, and it is stable as long as
        every writer goes through ``utc_now_iso``. A hand-written stamp with no
        ``Z`` and no fraction is neither padded nor rejected; it just sorts as
        if it were older than its own second.

        ``rowid DESC`` then ``run_id DESC`` break exact ties so paging and
        "most recent run" cannot flap between two rows with equal stamps
        (``tests/test_evidence_quality_fixes`` pins that).
        """
        clauses: List[str] = []
        params: List[Any] = []
        if project:
            clauses.append("project = ?")
            params.append(project)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        order_ts = "CASE WHEN instr(created_at, '.') > 0 THEN created_at ELSE replace(created_at, 'Z', '.000000Z') END"
        rows = self._conn.execute(
            f"SELECT payload_json FROM runs {where} ORDER BY {order_ts} DESC, rowid DESC, run_id DESC LIMIT ?",
            tuple(params),
        ).fetchall()
        return [run_from_dict(json.loads(r["payload_json"])) for r in rows]

    def last_verified_per_project(self) -> Dict[str, RunRecord]:
        """Most recent settled run per project — settled meaning *terminal*, not *good*.

        "Verified" is a slight misnomer that the TUI/MCP callers have baked in:
        the query accepts every terminal status, so a project whose last run
        was ``ERROR``, ``BLOCKED`` or ``CANCELLED`` reports that run here.
        Returning only PASS/FAIL would turn "this project's latest run died"
        into "no data", which is a worse lie. ACTIVE rows are excluded so a
        freshly queued run does not erase the last known result.

        Ordering repeats ``list_runs``' normalisation and tiebreakers; keeping
        the two in step matters because the overview and the history list are
        compared side by side.
        """
        key_expr = (
            "printf('%s|%020d|%s', "
            "CASE WHEN instr(created_at, '.') > 0 THEN created_at ELSE replace(created_at, 'Z', '.000000Z') END, "
            "rowid, run_id)"
        )
        key_expr_r = (
            "printf('%s|%020d|%s', "
            "CASE WHEN instr(r.created_at, '.') > 0 THEN r.created_at ELSE replace(r.created_at, 'Z', '.000000Z') END, "
            "r.rowid, r.run_id)"
        )
        rows = self._conn.execute(
            f"""
            SELECT r.payload_json
            FROM runs r
            INNER JOIN (
                SELECT project, MAX({key_expr}) AS max_key
                FROM runs
                WHERE status IN ('PASS', 'FAIL', 'SKIP', 'BLOCKED', 'TIMEOUT', 'CANCELLED', 'ERROR')
                GROUP BY project
            ) latest
            ON r.project = latest.project AND {key_expr_r} = latest.max_key
            """
        ).fetchall()
        out: Dict[str, RunRecord] = {}
        for r in rows:
            rec = run_from_dict(json.loads(r["payload_json"]))
            out[rec.project] = rec
        return out

    def reconcile_orphans(self, *, stale_queued_after_s: float = 10.0) -> List[str]:
        """Convert ACTIVE rows with no live owner into ``ERROR``/``abandoned_worker``.

        This is the store's only mechanism for making status honest retroactively,
        and it is deliberately one-way: it never resurrects a row, never deletes
        one, and never relabels a terminal row. An orphan is legitimate history —
        a Ctrl-C on a foreground run, a killed ``_worker``, a crash inside the
        engine — and ``ERROR`` with ``failure_kind=abandoned_worker`` is the
        truthful description of "this process stopped without settling the run".
        Do not "clean up" those rows: they are the record of what happened.

        Two grace rules, because "no lock right now" is not the same as "dead":

        * ``RUNNING`` — reconciled immediately when the lock is unheld. Live
          executors hold the lock for the whole run (``run_sync`` wraps the
          suite loop, ``run_worker`` the whole body), so absence is death.
          The one gap is that ``run_sync`` commits ``RUNNING`` *before* taking
          the lock; a reconciler in that few-microsecond window will call a
          live run dead. The eventual ``finalize_run`` overwrites it, so the
          damage is a transient mislabel plus a rewritten ``run.json``, not a
          lost result.
        * ``QUEUED`` — waits ``stale_queued_after_s`` from ``created_at`` first.
          A detached worker needs time to boot Python and import the package;
          without the grace, every slow spawn would be declared abandoned. Ten
          seconds covers interpreter start; a worker that takes longer is
          mislabelled until it takes the lock.

        Writes are per-row (export, then a write transaction each) rather than
        one big transaction: a long reconciler holding SQLite's single writer
        slot would block the very workers it is racing.

        Returns the ids it touched, for logging/tests.
        """
        rows = self._conn.execute(
            "SELECT run_id, status, created_at, payload_json FROM runs WHERE status IN ('QUEUED', 'RUNNING')"
        ).fetchall()
        reconciled: List[str] = []
        now = datetime.now(timezone.utc)
        for row in rows:
            rid = str(row["run_id"])
            if self.is_run_locked(rid):
                continue
            if row["status"] == RunStatus.QUEUED:
                try:
                    created = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
                    if (now - created).total_seconds() < stale_queued_after_s:
                        continue
                except ValueError:
                    pass
            rec = run_from_dict(json.loads(row["payload_json"]))
            rec.status = RunStatus.ERROR
            rec.failure_kind = FailureKind.ABANDONED_WORKER
            # Exactly one error is recorded, and no counts are invented: the
            # suite tallies of a dead run are unknown, and guessing them would
            # turn "we lost the process" into "the tests failed". The summary is
            # left as-is too, so an interrupted run shows a status and a
            # first_failure, not a fabricated pass/fail sentence.
            rec.error_count = max(1, rec.error_count)
            rec.finished_at = utc_now_iso()
            if not rec.first_failure:
                rec.first_failure = "Worker exited or crashed without releasing terminal run state (run.lock unheld)"
            self._export_terminal_artifacts(rec)
            with self._write_tx() as conn:
                self._upsert_run_row(conn, rec)
            reconciled.append(rid)
        return reconciled

    @staticmethod
    def _manifest_sha256(project_root: Path) -> Optional[str]:
        """Digest of the project's ``.testbench.toml``, or ``None`` if absent/unusable.

        Symlinks are refused (``None``, not "followed"): a manifest that is a
        link can be re-pointed after approval without changing the digest of
        anything the operator reviewed. ``None`` then flows into the trust
        comparison, where it means "we have no fingerprint" — see
        ``is_manifest_trusted`` for the consequence.
        """
        m = project_root / ".testbench.toml"
        if not m.is_file() or m.is_symlink():
            return None
        return hashlib.sha256(m.read_bytes()).hexdigest()

    def register_project(self, project_root: Path, name: Optional[str] = None) -> Dict[str, Any]:
        """Record a project root in the registry so scoped access will allow it.

        Registration is *not* trust: a registered project may be run, but its
        manifest still cannot request network or repo writes. Re-registering
        only refreshes ``name``/``updated_at`` — ``trusted`` and
        ``manifest_sha256`` are preserved, so a rename cannot silently clear an
        existing approval (or grant one).
        """
        resolved = project_root.expanduser().resolve(strict=True)
        proj_name = name or resolved.name
        now = utc_now_iso()
        with self._write_tx() as conn:
            conn.execute(
                """
                INSERT INTO projects (project_root, name, manifest_sha256, trusted, registered_at, updated_at)
                VALUES (?, ?, NULL, 0, ?, ?)
                ON CONFLICT(project_root) DO UPDATE SET
                    name=excluded.name,
                    updated_at=excluded.updated_at
                """,
                (str(resolved), proj_name, now, now),
            )
        row = self._conn.execute(
            "SELECT * FROM projects WHERE project_root = ?", (str(resolved),)
        ).fetchone()
        return dict(row) if row else {}

    def trust_project(self, project_root: Path, name: Optional[str] = None) -> Dict[str, Any]:
        """Approve the manifest *as it exists now* by pinning its SHA-256.

        Trust is content-addressed, not path-addressed: editing a single byte
        of ``.testbench.toml`` afterwards silently drops the project back to
        untrusted and the engine returns ``BLOCKED``/``untrusted_manifest``
        for suites requesting network or repo writes. That is the whole point —
        approval is of a document, not of a directory.

        Two consequences worth knowing. ``register_project`` is implied
        (upsert), so trusting also makes the path eligible for scoped access.
        And trusting a directory with **no** manifest stores
        ``manifest_sha256 = NULL`` with ``trusted = 1``; ``is_manifest_trusted``
        then compares ``None == None`` and answers True. Harmless in practice
        (nothing to gate, and a manifest appearing later flips it to
        untrusted), but it means "trusted" can be true in a state where no
        manifest exists.
        """
        resolved = project_root.expanduser().resolve(strict=True)
        proj_name = name or resolved.name
        digest = self._manifest_sha256(resolved)
        now = utc_now_iso()
        with self._write_tx() as conn:
            conn.execute(
                """
                INSERT INTO projects (project_root, name, manifest_sha256, trusted, registered_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(project_root) DO UPDATE SET
                    name=excluded.name,
                    manifest_sha256=excluded.manifest_sha256,
                    trusted=1,
                    updated_at=excluded.updated_at
                """,
                (str(resolved), proj_name, digest, now, now),
            )
        return {
            "project_root": str(resolved),
            "name": proj_name,
            "manifest_sha256": digest,
            "trusted": True,
        }

    def is_project_registered(self, project_root: Path) -> bool:
        """Is this path in the registry? A missing path is simply "not registered".

        ``resolve(strict=True)`` means a nonexistent target can never be
        registered, and a symlink is judged by its target — the same path the
        registry would have stored.
        """
        try:
            resolved = project_root.expanduser().resolve(strict=True)
        except OSError:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM projects WHERE project_root = ?", (str(resolved),)
        ).fetchone()
        return row is not None

    def is_manifest_trusted(self, project_root: Path) -> bool:
        """Is the project approved *and* is its manifest byte-identical to the approval?

        Both halves must hold: an untrusted row, or a digest that no longer
        matches, returns False. This is the function the security gate hangs
        on (``engine._fallback_execute_suite`` consults ``spec.trusted``, which
        ``discovery`` fills from here), so its failure mode must always be
        "untrusted": an unreadable path, a missing project, a missing manifest
        and a digest read failure all degrade to False, never to an exception
        the caller might swallow into "proceed".

        It re-reads and re-hashes the file on every call — no caching — because
        a cached digest would defeat the point of pinning. Note the small
        TOCTOU: ``discovery`` reads the manifest again to parse it, so a writer
        that swaps content between this hash and that read presents trusted
        suites from untrusted bytes. Closing it would require hashing the bytes
        actually parsed (single read, digest + parse the same buffer).
        """
        try:
            resolved = project_root.expanduser().resolve(strict=True)
        except OSError:
            return False
        row = self._conn.execute(
            "SELECT manifest_sha256, trusted FROM projects WHERE project_root = ?",
            (str(resolved),),
        ).fetchone()
        if not row or not int(row["trusted"]):
            return False
        current_digest = self._manifest_sha256(resolved)
        return row["manifest_sha256"] == current_digest

    def list_projects(self) -> List[Dict[str, Any]]:
        """Registry contents as plain dicts, name-ordered for display.

        Columns only, no payload: a project row is small and its meaning is
        entirely in the columns. ``manifest_sha256`` may be NULL (never
        trusted, or trusted with no manifest) and ``trusted`` is an int, so
        consumers must not treat it as a bool from JSON.
        """
        rows = self._conn.execute("SELECT * FROM projects ORDER BY name ASC").fetchall()
        return [dict(r) for r in rows]

    def record_benchmark(
        self,
        *,
        bench_id: str,
        run_id: str,
        project: str,
        workload: str,
        revision: str,
        median_ms: Optional[float],
        payload: Dict[str, Any],
    ) -> None:
        """Persist one benchmark measurement, keyed by ``bench_id``.

        ``INSERT OR REPLACE``, so a re-recorded id overwrites rather than
        accumulating: ``bench_id`` is the caller's identity for a measurement,
        not a surrogate. The FK to ``runs`` means a benchmark whose run was
        pruned disappears with it — measurements are evidence *of* a run, not
        an independent series. ``payload`` is stored ``sort_keys`` so byte-level
        diffs of two runs' rows are meaningful.
        """
        now = utc_now_iso()
        with self._write_tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmarks
                (bench_id, run_id, project, workload, revision, created_at, median_ms, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bench_id,
                    run_id,
                    project,
                    workload,
                    revision,
                    now,
                    median_ms,
                    json.dumps(payload, sort_keys=True),
                ),
            )

    def list_benchmarks(
        self, *, project: Optional[str] = None, workload: Optional[str] = None, limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Recent benchmarks, newest first, ``payload_json`` parsed back into ``payload``.

        Same timestamp normalisation and rowid tiebreak as ``list_runs``: two
        benchmarks recorded in the same second must still come back in the
        order they were written, or a comparison would silently compare a
        baseline against itself.
        """
        clauses: List[str] = []
        params: List[Any] = []
        if project:
            clauses.append("project = ?")
            params.append(project)
        if workload:
            clauses.append("workload = ?")
            params.append(workload)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        order_ts = "CASE WHEN instr(created_at, '.') > 0 THEN created_at ELSE replace(created_at, 'Z', '.000000Z') END"
        rows = self._conn.execute(
            f"SELECT * FROM benchmarks {where} ORDER BY {order_ts} DESC, rowid DESC LIMIT ?",
            tuple(params),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["payload"] = json.loads(item.pop("payload_json"))
            out.append(item)
        return out

    def prune_runs(
        self,
        *,
        keep_last: int = 50,
        older_than_days: Optional[int] = None,
        dry_run: bool = False,
    ) -> List[str]:
        """Delete unpinned, settled runs beyond the retention policy; return their ids.

        Selection is newest-first over the same normalised order as history.
        Three exclusion classes, each for a different reason:

        * ``pinned`` — an explicit "keep this". Today nothing in the shipped
          CLI sets it (``pinned`` is only ever passed by tests), so in practice
          the keep-last count is the only protection.
        * ACTIVE status — a queued/running run is not history yet.
        * held ``run.lock`` — belt and braces over the status check: a run
          whose row still says QUEUED but whose worker has started must not
          have its evidence removed under it.

        Nothing else is protected. ``ERROR``/``abandoned_worker`` rows are
        prunable history like any other: a crash is a fact, not a pin.

        ``keep_last < 0`` disables the count rule (age-only retention);
        ``older_than_days`` is optional and OR'd with it, so the union of the
        two policies is removed. Only *unpinned, unlocked, settled* runs count
        toward ``keep_last``, which means "keep the last N prunable runs
        around", not "keep N rows total".

        Ordering of side effects: rows first, then ``rmtree``. A failure to
        delete a directory therefore leaves an orphaned artifact tree that no
        row points at — recoverable by hand, and preferable to a row deleted
        after its evidence is already gone. ``dry_run`` returns the same list
        without touching anything, which is the only safe way to discover what
        a policy would take.
        """
        order_ts = "CASE WHEN instr(created_at, '.') > 0 THEN created_at ELSE replace(created_at, 'Z', '.000000Z') END"
        rows = self._conn.execute(
            f"""
            SELECT run_id, created_at, pinned, status
            FROM runs
            ORDER BY {order_ts} DESC, rowid DESC, run_id DESC
            """
        ).fetchall()
        cutoff: Optional[datetime] = None
        if older_than_days is not None and older_than_days >= 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

        unpinned_seen = 0
        to_remove: List[str] = []
        for r in rows:
            rid = str(r["run_id"])
            if int(r["pinned"]) or r["status"] in RunStatus.ACTIVE or self.is_run_locked(rid):
                continue
            unpinned_seen += 1
            over_count = keep_last >= 0 and unpinned_seen > keep_last
            over_age = False
            if cutoff is not None:
                try:
                    dt = datetime.fromisoformat(str(r["created_at"]).replace("Z", "+00:00"))
                    over_age = dt < cutoff
                except ValueError:
                    pass
            if over_count or over_age:
                to_remove.append(rid)

        if dry_run or not to_remove:
            return to_remove

        with self._write_tx() as conn:
            conn.executemany("DELETE FROM runs WHERE run_id = ?", [(rid,) for rid in to_remove])
        for rid in to_remove:
            _safe_rmtree_no_symlinks(self.runs_dir / rid)
        return to_remove
