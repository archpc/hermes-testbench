"""Authoritative SQLite WAL state store, lock reconciler, trust registry, and retention."""

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
    """Raised when state.db PRAGMA user_version disagrees with SCHEMA_VERSION."""


def default_testbench_home() -> Path:
    env_home = os.environ.get("TESTBENCH_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    if xdg:
        return Path(xdg).expanduser() / "testbench"
    return Path.home() / ".local" / "share" / "testbench"


def _exclude_from_time_machine(path: Path) -> None:
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
    """Remove a directory tree without ever traversing symlinks."""
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
        with self._write_tx_conn(self._conn) as conn:
            yield conn

    def journal_mode(self) -> str:
        row = self._conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]) if row else ""

    def data_version(self) -> int:
        row = self._conn.execute("PRAGMA data_version").fetchone()
        return int(row[0]) if row else 0

    def run_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
            raise ValueError(f"Invalid run_id: {run_id!r}")
        d = self.runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(str(d), 0o700)
        return d

    def run_lock_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.lock"

    @contextmanager
    def hold_run_lock(self, run_id: str) -> Iterator[Path]:
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
        rdir = self.run_dir(run.run_id)
        run.evidence_dir = str(rdir)
        run.run_json_path = str(rdir / "run.json")
        run.junit_xml_path = str(rdir / "junit.xml")
        run.repro_path = str(rdir / "repro.sh")
        if not run.created_at:
            run.created_at = utc_now_iso()

    def _export_terminal_artifacts(self, run: RunRecord) -> None:
        self._prepare_run_paths(run)
        assert run.run_json_path and run.junit_xml_path and run.repro_path
        atomic_write_file(Path(run.run_json_path), render_run_json(run), mode=0o600)
        atomic_write_file(Path(run.junit_xml_path), render_junit_xml(run), mode=0o600)
        atomic_write_file(Path(run.repro_path), format_repro_script(run), mode=0o700)

    def _upsert_run_row(self, conn: sqlite3.Connection, run: RunRecord) -> None:
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
        self._prepare_run_paths(run)
        with self._write_tx() as conn:
            self._upsert_run_row(conn, run)
        return run

    def update_run(self, run: RunRecord) -> RunRecord:
        self._prepare_run_paths(run)
        with self._write_tx() as conn:
            self._upsert_run_row(conn, run)
        return run

    def finalize_run(self, run: RunRecord) -> RunRecord:
        if not run.finished_at:
            run.finished_at = utc_now_iso()
        self._export_terminal_artifacts(run)
        with self._write_tx() as conn:
            self._upsert_run_row(conn, run)
        return run

    def get_run(self, run_id: str) -> Optional[RunRecord]:
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
        rows = self._conn.execute(
            f"SELECT payload_json FROM runs {where} ORDER BY created_at DESC, run_id DESC LIMIT ?",
            tuple(params),
        ).fetchall()
        return [run_from_dict(json.loads(r["payload_json"])) for r in rows]

    def last_verified_per_project(self) -> Dict[str, RunRecord]:
        rows = self._conn.execute(
            """
            SELECT r.payload_json
            FROM runs r
            INNER JOIN (
                SELECT project, MAX(created_at || '|' || run_id) AS max_key
                FROM runs
                WHERE status IN ('PASS', 'FAIL', 'SKIP', 'BLOCKED', 'TIMEOUT', 'CANCELLED', 'ERROR')
                GROUP BY project
            ) latest
            ON r.project = latest.project AND (r.created_at || '|' || r.run_id) = latest.max_key
            """
        ).fetchall()
        out: Dict[str, RunRecord] = {}
        for r in rows:
            rec = run_from_dict(json.loads(r["payload_json"]))
            out[rec.project] = rec
        return out

    def reconcile_orphans(self, *, stale_queued_after_s: float = 10.0) -> List[str]:
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
        m = project_root / ".testbench.toml"
        if not m.is_file() or m.is_symlink():
            return None
        return hashlib.sha256(m.read_bytes()).hexdigest()

    def register_project(self, project_root: Path, name: Optional[str] = None) -> Dict[str, Any]:
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
        try:
            resolved = project_root.expanduser().resolve(strict=True)
        except OSError:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM projects WHERE project_root = ?", (str(resolved),)
        ).fetchone()
        return row is not None

    def is_manifest_trusted(self, project_root: Path) -> bool:
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
        rows = self._conn.execute(
            f"SELECT * FROM benchmarks {where} ORDER BY created_at DESC LIMIT ?",
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
        rows = self._conn.execute(
            """
            SELECT run_id, created_at, pinned, status
            FROM runs
            ORDER BY created_at DESC, run_id DESC
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
