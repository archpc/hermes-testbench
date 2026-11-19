"""Contract tests for the authoritative store: modes, WAL, locks, trust pins, retention.

What is under test here is the set of guarantees that outlive a process and therefore
cannot be observed from one call site:

* creation side effects — a ``0700`` home, a ``0600`` ``state.db``, WAL journalling.
  Modes are asserted on the *file*, not on intent: ``state.db`` carries run history and
  evidence paths, so a world-readable store leaks both.
* the database is authoritative and the artifacts are derived. ``finalize_run`` writes
  ``run.json``/``junit.xml``/``repro.sh`` and then the row, so all three must exist with
  the expected modes whenever a terminal row does.
* truthfulness of ``ACTIVE`` status, which only the ``run.lock`` flock can establish.
  ``reconcile_orphans`` is the single path from "the process that owned this run is
  gone" to ``ERROR``, and an interrupted run left as ``ERROR`` is correct history.
* trust as a digest: editing the manifest invalidates the approval.
* retention ordering and its exclusion classes (pinned, ACTIVE, held lock).
* multi-process write contention, exercised with a spawn Pool so the children have their
  own connections and their own file descriptions rather than threads sharing one.

Fixtures: one ``TemporaryDirectory`` inside ``~/.hermes/cache/scratch`` per test, so each
store home is genuinely new — the Time Machine exclusion is applied only when the home does
not already exist, so a reused home would silently skip that branch — and nothing touches the
real default home. Nothing here is stubbed: every test runs against a real SQLite file and
real ``flock``, so a failure is behavioural rather than a mock-shape mismatch.
"""

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
    """Child-process body for the contention test: create then finalize ``count`` runs.

    Runs in a *spawned* interpreter, which is the point — it opens its own
    ``TestBenchStore``, so its connection and file descriptions are independent of the
    parent's and of its siblings', exactly as the CLI, a detached ``_worker`` and the TUI
    are. Each run gets two transactions (create, finalize) to keep the ``BEGIN IMMEDIATE``
    windows short and to make a lost update observable as a wrong status rather than as a
    retried statement.

    Run ids are hand-built (``tb_<stamp>_<worker><seq>``) rather than ``new_run_id()``
    because that helper's cross-process uniqueness is probabilistic — a 16-bit seq
    bucket plus two random bytes — and a collision would upsert two writers into one
    row, hiding the lost write this test is meant to detect.
    """
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
    """End-to-end tests against a real SQLite file, real ``flock``, and real child processes.

    Every test gets a scrubbed home inside ``self.tmp``, so the store's "is this a new
    home?" branch (mode repair plus the backup exclusion) runs each time, and
    ``tearDown`` removes the database, its WAL sidecars and every evidence directory
    together.
    """

    def setUp(self) -> None:
        scratch = Path.home() / ".hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-store-")
        self.home = Path(self.tmp.name) / "tb_home"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_permissions_wal_mode_crud_and_never_fabricates_tokens(self) -> None:
        """Create → external finalize → re-read, plus the modes and the WAL invariant.

        What each assertion discriminates:

        * ``0700``/``0600``/``wal`` — a dropped ``chmod``, or a store directory restored from a
          filesystem captured mid-checkpoint (which comes back in ``delete`` mode), fails here
          instead of silently exposing run history. ``journal_mode()`` is read from SQLite
          rather than assumed, so the assertion is about the file.
        * ``data_version()`` must advance because a *second* store instance committed the
          finalize. Reading the counter on the original connection is what makes this a test of
          external-commit visibility — the signal the TUI refresh polls — and not of the write
          itself.
        * ``model_usage is None`` in the record *and* in the exported ``run.json``: the store
          must never invent usage. A regression defaulting it to zeros would pass every count
          assertion above it.
        * the three artifact paths exist and ``run.json`` is ``0600``. ``finalize_run`` writes
          artifacts before the row, so their absence would mean a terminal row is a claim with
          no evidence behind it.

        ``last_verified_per_project`` then confirms the settled row is the project's latest,
        which is the lookup the overview pane and MCP status rely on.
        """
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
        """A held lock is the only proof of life; an unheld one must become ``ERROR``.

        The negative control is the pair of identical ``RUNNING`` rows: status, timestamps and
        project all match, and the only difference is ``hold_run_lock`` around the reconcile
        call, so nothing but liveness can explain the divergent outcome. The control is valid
        in-process because ``flock`` is per open file description: ``is_run_locked`` opens a
        second description and still conflicts with the holder in this same process — no helper
        process is needed to simulate a live owner.

        ``failure_kind == abandoned_worker`` pins the vocabulary the TUI, ``testbench status``
        and JUnit derivation consume. The ``run.json`` assertion separates "row relabelled" from
        "history rewritten with evidence": the reconciler exports artifacts before the row, so a
        reconciled run is a complete record, not a bare status.

        Both rows are ``RUNNING``, so the ``stale_queued_after_s`` grace that protects booting
        ``QUEUED`` workers is deliberately not exercised here.
        """
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
        """A foreign ``PRAGMA user_version`` refuses the connection instead of "repairing" it.

        Nothing is stubbed: a raw ``sqlite3`` handle stamps ``user_version = 999`` on the real
        file and the next ``TestBenchStore`` must raise :class:`SchemaMismatchError`. The failure
        is the feature — there is no migration code, so a best-effort open would feed another
        release's payloads to ``run_from_dict``'s strict constructor and fail later, mid-run,
        with a much worse error. A regression that reset the version would make this test pass
        on a store it has just corrupted.
        """
        from contextlib import closing
        with TestBenchStore(self.home) as store:
            db_path = store.db_path
        with closing(sqlite3.connect(str(db_path))) as conn:
            conn.execute("PRAGMA user_version = 999")
            conn.commit()
        with self.assertRaises(SchemaMismatchError):
            TestBenchStore(self.home)

    def test_project_trust_registry_pins_manifest_sha256(self) -> None:
        """Trust is a digest of the document as approved, not a boolean on a path.

        The three-step sequence is the whole contract: untrusted before approval, trusted after
        ``trust_project``, untrusted again once a byte of the manifest changes. The last step is
        the one that matters — a path-keyed flag would stay true across the edit, which is
        exactly the substitution a pinned SHA-256 exists to catch, and the engine's
        ``allow_network`` veto reads this function.

        ``rec["project_root"] == str(proj.resolve())`` pins that the approval is stored under the
        resolved path, so a symlinked root cannot approve one tree and then run another. The test
        asserts the ``is_manifest_trusted`` view rather than the row, because that is the surface
        the gate actually calls.
        """
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
        """``keep_last`` counts *prunable* runs; a pinned run neither consumes nor loses quota.

        Five runs with the oldest pinned, ``keep_last=2``: selection is newest-first, the two
        newest are kept, the next two exceed the count and are removed, and the pinned row at the
        end is skipped entirely — so it survives without occupying one of the two slots. The
        expected ``removed`` list is therefore the second- and third-oldest, asserted sorted
        because selection order (newest-first) and the assertion (a sorted list compared
        positionally) are different things.

        The final assertions cover both halves of the deletion: the row is gone *and* its evidence
        directory is gone. ``prune_runs`` deletes rows and then rmtree's each directory; a
        regression that dropped the second step would leave artifact trees no row points at,
        which is why both are checked, plus ``ids[4]`` (the newest) still being present.
        """
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
        """Four spawned writers, 48 runs, no ``SQLITE_BUSY`` escaping to the caller.

        The "zero busy errors" half is structural rather than explicitly asserted: an
        ``OperationalError`` in any child propagates through ``starmap`` and fails the test, so
        reaching the count already proves every ``BEGIN IMMEDIATE`` waited inside
        ``busy_timeout`` instead of raising. Nothing is retried anywhere in the store, so a
        missing wait cannot be masked.

        The read-back then checks for lost writes: 48 distinct rows (each hand-built id landed)
        and every one settled ``PASS``. Since the upsert is last-writer-wins, a create that
        arrived after its own finalize would leave ``RUNNING``/``QUEUED`` behind, which is the
        one corruption this test can catch that a count alone would miss.

        Timing: this is the only test using ``spawn`` (fresh interpreters, no ``fork``
        inheritance), so it is the slow, load-sensitive one; the assertions themselves contain no
        wall-clock dependency.
        """
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
