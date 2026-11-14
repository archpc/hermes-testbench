"""TDD tests for authoritative SQLite WAL store, locks, trust pins, and retention."""

from __future__ import annotations

import json
import multiprocessing
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path

from hermes_testbench.schema import (
    CaseResult,
    FailureRecord,
    RunRecord,
    RunStatus,
    SuiteResult,
    new_run_id,
)
from hermes_testbench.store import SchemaMismatchError, TestBenchStore


def _concurrent_writer(store_dir: str, worker_idx: int, count: int) -> int:
    with TestBenchStore(Path(store_dir)) as store:
        for i in range(count):
            rid = f"tb_20261114_170000_{worker_idx:02d}{i:06d}"
            rec = RunRecord(
                run_id=rid,
                project=f"proj-{worker_idx % 2}",
                project_root=f"/tmp/proj-{worker_idx % 2}",
                revision="abc1234",
                profile="unit",
                status=RunStatus.RUNNING,
                created_at="2026-11-14T22:00:00Z",
                started_at="2026-11-14T22:00:00Z",
            )
            store.create_run(rec)
            rec.status = RunStatus.PASS
            rec.finished_at = "2026-11-14T22:00:01Z"
            rec.duration_ms = 12.5
            rec.total_count = 3
            rec.pass_count = 3
            store.finalize_run(rec)
    return count


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.home() / ".hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-store-")
        self.home = Path(self.tmp.name) / "tb_home"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_permissions_wal_mode_crud_and_never_fabricates_tokens(self) -> None:
        store = TestBenchStore(self.home)
        self.assertEqual(stat.S_IMODE(self.home.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(store.db_path.stat().st_mode), 0o600)
        self.assertEqual(store.journal_mode().lower(), "wal")

        rid = new_run_id()
        suite = SuiteResult(
            name="unit-all",
            adapter="cargo",
            profile="unit",
            command=["cargo", "test", "--offline"],
            cwd="/tmp/hxmail",
            status=RunStatus.FAIL,
            exit_code=101,
            duration_ms=210.5,
            total_count=10,
            pass_count=9,
            fail_count=1,
            repro_command="cargo test --offline -- parse_bad_header",
            failures=[
                FailureRecord(
                    suite="unit-all",
                    case_name="parse_bad_header",
                    kind="assertion",
                    message="panicked at assertion failed",
                    excerpt="left: 1, right: 2",
                )
            ],
            cases=[
                CaseResult(name="parse_bad_header", suite="unit-all", status=RunStatus.FAIL, duration_ms=5.2, message="1 != 2")
            ],
        )
        run = RunRecord(
            run_id=rid,
            project="hxmail",
            project_root="/tmp/hxmail",
            revision="8234029",
            branch="main",
            profile="unit",
            status=RunStatus.QUEUED,
            commands=[["cargo", "test", "--offline"]],
            created_at="2026-11-14T22:01:00Z",
        )
        store.create_run(run)
        dv_before = store.data_version()
        run.status = RunStatus.FAIL
        run.failure_kind = "assertion"
        run.started_at = "2026-11-14T22:01:01Z"
        run.finished_at = "2026-11-14T22:01:02Z"
        run.duration_ms = 210.5
        run.exit_code = 101
        run.total_count = 10
        run.pass_count = 9
        run.fail_count = 1
        run.first_failure = "parse_bad_header: panicked at assertion failed"
        run.repro_instructions = "cargo test --offline -- parse_bad_header"
        run.suites = [suite]
        # Commit from a second store instance to verify PRAGMA data_version advances on external writes
        TestBenchStore(self.home).finalize_run(run)
        self.assertGreater(store.data_version(), dv_before)

        loaded = store.get_run(rid)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.status, RunStatus.FAIL)
        self.assertIsNone(loaded.model_usage, "model_usage must stay None when not measured")
        self.assertEqual(loaded.fail_count, 1)
        self.assertEqual(len(loaded.suites), 1)
        self.assertEqual(loaded.suites[0].failures[0].case_name, "parse_bad_header")
        self.assertTrue(Path(loaded.run_json_path or "").is_file())
        self.assertEqual(stat.S_IMODE(Path(loaded.run_json_path or "").stat().st_mode), 0o600)
        self.assertTrue(Path(loaded.junit_xml_path or "").is_file())
        self.assertTrue(Path(loaded.repro_path or "").is_file())

        exported = json.loads(Path(loaded.run_json_path or "").read_text(encoding="utf-8"))
        self.assertEqual(exported["run_id"], rid)
        self.assertIsNone(exported["model_usage"])

        last = store.last_verified_per_project()
        self.assertEqual(last["hxmail"].run_id, rid)

    def test_orphan_reconciler_marks_unlocked_running_run_as_error_while_preserving_locked_run(self) -> None:
        store = TestBenchStore(self.home)
        live_run = RunRecord(
            run_id="tb_20261114_170000_live0001",
            project="demo",
            project_root="/tmp/demo",
            revision="1111111",
            profile="unit",
            status=RunStatus.RUNNING,
            created_at="2026-11-14T22:00:00Z",
            started_at="2026-11-14T22:00:00Z",
        )
        dead_run = RunRecord(
            run_id="tb_20261114_170000_dead0001",
            project="demo",
            project_root="/tmp/demo",
            revision="1111111",
            profile="unit",
            status=RunStatus.RUNNING,
            created_at="2026-11-14T22:00:00Z",
            started_at="2026-11-14T22:00:00Z",
        )
        store.create_run(live_run)
        store.create_run(dead_run)

        with store.hold_run_lock(live_run.run_id):
            reconciled = store.reconcile_orphans()
            self.assertIn(dead_run.run_id, reconciled)
            self.assertNotIn(live_run.run_id, reconciled)
            self.assertEqual(store.get_run(live_run.run_id).status, RunStatus.RUNNING)
            dead_after = store.get_run(dead_run.run_id)
            self.assertEqual(dead_after.status, RunStatus.ERROR)
            self.assertEqual(dead_after.failure_kind, "abandoned_worker")
            self.assertTrue(Path(dead_after.run_json_path or "").is_file())

    def test_user_version_mismatch_refuses_connection(self) -> None:
        from contextlib import closing
        with TestBenchStore(self.home) as store:
            db_path = store.db_path
        with closing(sqlite3.connect(str(db_path))) as conn:
            conn.execute("PRAGMA user_version = 999")
            conn.commit()
        with self.assertRaises(SchemaMismatchError):
            TestBenchStore(self.home)

    def test_project_trust_registry_pins_manifest_sha256(self) -> None:
        store = TestBenchStore(self.home)
        proj = self.home / "sample_repo"
        proj.mkdir(parents=True)
        manifest = proj / ".testbench.toml"
        manifest.write_text('[project]\nname = "sample"\n', encoding="utf-8")

        self.assertFalse(store.is_manifest_trusted(proj))
        rec = store.trust_project(proj)
        self.assertTrue(store.is_manifest_trusted(proj))
        self.assertEqual(rec["project_root"], str(proj.resolve()))

        manifest.write_text('[project]\nname = "tampered"\n', encoding="utf-8")
        self.assertFalse(store.is_manifest_trusted(proj))

    def test_bounded_retention_prunes_oldest_unpinned_runs_and_artifacts(self) -> None:
        store = TestBenchStore(self.home)
        ids = []
        for idx in range(5):
            rid = f"tb_20261114_17000{idx}_0000000{idx}"
            ids.append(rid)
            r = RunRecord(
                run_id=rid,
                project="demo",
                project_root="/tmp/demo",
                revision="111",
                profile="unit",
                status=RunStatus.PASS,
                created_at=f"2026-11-14T22:00:0{idx}Z",
                finished_at=f"2026-11-14T22:00:0{idx}Z",
                pinned=(idx == 0),
            )
            store.create_run(r)
            store.finalize_run(r)

        removed = store.prune_runs(keep_last=2)
        self.assertEqual(sorted(removed), [ids[1], ids[2]])
        self.assertIsNotNone(store.get_run(ids[0]))
        self.assertIsNone(store.get_run(ids[1]))
        self.assertFalse((store.runs_dir / ids[1]).exists())
        self.assertIsNotNone(store.get_run(ids[4]))

    def test_multiprocess_concurrent_begin_immediate_writes_zero_sqlite_busy(self) -> None:
        TestBenchStore(self.home)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=4) as pool:
            results = pool.starmap(_concurrent_writer, [(str(self.home), w, 12) for w in range(4)])
        self.assertEqual(sum(results), 48)
        store = TestBenchStore(self.home)
        runs = store.list_runs(limit=100)
        self.assertEqual(len(runs), 48)
        self.assertTrue(all(r.status == RunStatus.PASS for r in runs))


if __name__ == "__main__":
    unittest.main()
