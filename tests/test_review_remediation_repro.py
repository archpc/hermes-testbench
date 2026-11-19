"""Regression tests for messaging-reproduction findings A1, A2, A3, A6.

A1: reproduce() of a ``repro:verification`` run must re-enter the messaging
    verification path (not generic suite replay) and must never stack
    ``repro:repro:...`` profiles.
A2: an exception inside messaging verification must finalize the run as
    ERROR/harness_error instead of leaving it RUNNING, from both the engine
    reproduce path and the ``messaging-verify`` CLI command.
A3: a repro must use the same artifact_root semantics as the original run
    (None when the original had no artifacts directory).
A6: the agent-messaging artifact / hermes-agent paths come from one shared
    helper derived from the real user home, not hard-coded /Users/sein paths.

How the tests isolate themselves: ``setUp`` patches ``engine.messaging_paths``
to two directories under the scratch root, so neither the CLI nor
``reproduce`` can reach the operator's real ``~/.hermes``. The patch is a
module attribute, and both call sites look it up at call time (the CLI imports
it inside the command handler, ``reproduce`` reads the module global), which is
why patching ``engine_mod`` is sufficient. ``run_agent_messaging_verification``
is imported *inside* ``reproduce``, so the tests patch its home module
(``hermes_testbench.messaging_suite``), not a re-export.

The seeded run's suite command is ``/usr/bin/false`` on purpose: a plain argv
replay of that record would FAIL, so a PASS can only come from re-entering the
messaging path. That is what makes A1's assertion discriminate instead of
merely observing "a run happened".
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hermes_testbench import engine as engine_mod
from hermes_testbench.cli import main as cli_main
from hermes_testbench.engine import TestBenchEngine
from hermes_testbench.isolation import default_scratch_root
from hermes_testbench.schema import (
    FailureKind,
    RunRecord,
    RunStatus,
    SuiteResult,
    SuiteSpec,
    new_run_id,
    utc_now_iso,
)
from hermes_testbench.store import TestBenchStore

# Captured before any test patches engine.messaging_paths for isolation.
_REAL_MESSAGING_PATHS = engine_mod.messaging_paths


def _pass_suite(name: str = "agent-messaging:contract") -> SuiteResult:
    # Minimal passing suite: enough for run-level counts, no failures to report.
    return SuiteResult(
        name=name,
        adapter="contract",
        profile="integration",
        command=["testbench", "messaging-verify"],
        cwd=".",
        status=RunStatus.PASS,
        total_count=1,
        pass_count=1,
    )


class MessagingReproRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Scratch root is already resolved, so ``art_root == art_root.resolve()``
        # holds literally; the artifact-identity checks below rely on that.
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-repro-")
        self.root = Path(self.tmp.name)
        self.tb_home = self.root / "tb_home"
        # Isolated stand-ins for ~/.hermes/artifacts/agent-messaging and ~/.hermes/hermes-agent
        self.art_root = self.root / "artifacts" / "agent-messaging"
        self.art_root.mkdir(parents=True)
        self.ha_root = self.root / "hermes-agent"
        self.ha_root.mkdir()
        self._paths_patch = mock.patch.object(
            engine_mod, "messaging_paths", lambda home=None: (self.art_root, self.ha_root)
        )
        self._paths_patch.start()
        self.addCleanup(self._paths_patch.stop)
        self.addCleanup(self.tmp.cleanup)

    # ----- helpers -------------------------------------------------------
    def _seed_run(
        self,
        store: TestBenchStore,
        *,
        profile: str,
        project_root: Path,
        artifact_env: object = "__unset__",
    ) -> RunRecord:
        """Record one settled ``agent-messaging`` run shaped like a real one.

        ``artifact_env`` distinguishes the three historical shapes:

        * ``"__unset__"`` — pre-marker run; only ``project_root`` exists to infer
          from (the legacy fallback path);
        * ``""`` — marker present and empty: the original used no artifact root;
        * any path — the artifact root the original actually used.

        The record is PASS with a failing command so replays must not use argv.
        """
        env = {"os": "test"}
        if artifact_env != "__unset__":
            env["messaging_artifact_root"] = artifact_env  # type: ignore[assignment]
        rec = RunRecord(
            run_id=new_run_id(),
            project="agent-messaging",
            project_root=str(project_root),
            revision="abc1234",
            profile=profile,
            status=RunStatus.PASS,
            commands=[["testbench", "messaging-verify"]],
            created_at=utc_now_iso(),
            started_at=utc_now_iso(),
            finished_at=utc_now_iso(),
            environment=env,
            suites=[
                SuiteResult(
                    name="agent-messaging:contract",
                    adapter="contract",
                    profile="integration",
                    # A raw replay of this argv would FAIL; only the messaging
                    # path can return PASS.
                    command=["/usr/bin/false"],
                    cwd=str(project_root),
                    status=RunStatus.PASS,
                    total_count=1,
                    pass_count=1,
                )
            ],
        )
        store.create_run(rec)
        return rec

    def _fake_messaging(self, *, result=None, exc=None):
        """Stand-in verifier that records every call's effective arguments.

        The signature mirrors the real one, so a caller passing the wrong keyword
        (or omitting ``artifact_root``) fails here rather than silently working.
        ``calls`` is the observable: how many entries says whether the path was
        entered once, and what each entry holds says with which semantics.
        """
        calls = []

        def _fn(evidence_dir, *, artifact_root=None, hermes_agent_root=None, run_hermes_pytest=False):
            calls.append(
                {
                    "evidence_dir": Path(evidence_dir),
                    "artifact_root": Path(artifact_root) if artifact_root is not None else None,
                    "hermes_agent_root": Path(hermes_agent_root) if hermes_agent_root is not None else None,
                    "run_hermes_pytest": run_hermes_pytest,
                }
            )
            if exc is not None:
                raise exc
            return result if result is not None else [_pass_suite()]

        return _fn, calls

    # ----- A1 ------------------------------------------------------------
    def test_a1_reproduce_of_repro_verification_reenters_messaging_path(self) -> None:
        # Original profile is already prefixed. base_profile() strips *all*
        # leading "repro:" segments, so the child profile is exactly
        # "repro:verification" — the pre-fix bug produced "repro:repro:verification".
        with TestBenchStore(self.tb_home) as store:
            orig = self._seed_run(
                store,
                profile="repro:verification",
                project_root=self.art_root,
                artifact_env=str(self.art_root),
            )
            fn, calls = self._fake_messaging()
            with mock.patch("hermes_testbench.messaging_suite.run_agent_messaging_verification", fn):
                out = TestBenchEngine(store).reproduce(orig.run_id)

            self.assertEqual(len(calls), 1, "repro of a repro must call the messaging verifier once")
            self.assertEqual(out["status"], RunStatus.PASS)
            repro = store.get_run(out["repro_run_id"])
            assert repro is not None
            self.assertEqual(repro.profile, "repro:verification", "profile must not stack repro: prefixes")
            self.assertEqual(repro.project, "agent-messaging")
            self.assertEqual(repro.pass_count, 1)

    def test_a1_original_verification_run_still_uses_messaging_path(self) -> None:
        # The unprefixed case: the guard matches on base_profile(profile), so the
        # original "verification" run takes the same branch. Asserting the child
        # profile separately proves the prefix is added exactly once.
        with TestBenchStore(self.tb_home) as store:
            orig = self._seed_run(
                store,
                profile="verification",
                project_root=self.art_root,
                artifact_env=str(self.art_root),
            )
            fn, calls = self._fake_messaging()
            with mock.patch("hermes_testbench.messaging_suite.run_agent_messaging_verification", fn):
                out = TestBenchEngine(store).reproduce(orig.run_id)
            self.assertEqual(len(calls), 1)
            self.assertEqual(out["status"], RunStatus.PASS)
            repro = store.get_run(out["repro_run_id"])
            assert repro is not None
            self.assertEqual(repro.profile, "repro:verification")

    # ----- A3 ------------------------------------------------------------
    def test_a3_repro_uses_recorded_artifact_root_none_when_original_had_none(self) -> None:
        # Marker present but empty is the *recorded* fact "no artifact root".
        # The repro must not substitute the default root just because it exists.
        with TestBenchStore(self.tb_home) as store:
            # Original ran without an artifacts directory: it recorded '' and used ha_root as root.
            orig = self._seed_run(
                store,
                profile="verification",
                project_root=self.ha_root,
                artifact_env="",
            )
            fn, calls = self._fake_messaging()
            with mock.patch("hermes_testbench.messaging_suite.run_agent_messaging_verification", fn):
                TestBenchEngine(store).reproduce(orig.run_id)
            self.assertEqual(len(calls), 1)
            self.assertIsNone(calls[0]["artifact_root"], "artifacts absent originally => artifact_root must be None")

    def test_a3_repro_uses_recorded_artifact_root_when_present(self) -> None:
        # Marker present and non-empty: the recorded path wins over the default,
        # even though the patched default happens to equal it here (the value is
        # asserted, not the code path taken).
        with TestBenchStore(self.tb_home) as store:
            orig = self._seed_run(
                store,
                profile="verification",
                project_root=self.art_root,
                artifact_env=str(self.art_root),
            )
            fn, calls = self._fake_messaging()
            with mock.patch("hermes_testbench.messaging_suite.run_agent_messaging_verification", fn):
                TestBenchEngine(store).reproduce(orig.run_id)
            self.assertEqual(calls[0]["artifact_root"], self.art_root.resolve())

    def test_a3_legacy_run_without_marker_falls_back_to_resolved_project_root(self) -> None:
        # No marker: the only evidence is project_root. It is accepted precisely
        # because it *is* the default artifact root (compared resolved), which is
        # what a pre-marker verification run recorded.
        with TestBenchStore(self.tb_home) as store:
            # Older run: no environment marker. Its project_root is the artifacts dir.
            orig = self._seed_run(
                store, profile="verification", project_root=self.art_root, artifact_env="__unset__"
            )
            fn, calls = self._fake_messaging()
            with mock.patch("hermes_testbench.messaging_suite.run_agent_messaging_verification", fn):
                TestBenchEngine(store).reproduce(orig.run_id)
            self.assertEqual(calls[0]["artifact_root"], self.art_root.resolve())

    def test_a3_legacy_run_without_marker_ignores_non_artifact_project_root(self) -> None:
        # The other half of the fallback: hermes-agent checkout as project_root
        # is not an artifact root, so the repro keeps artifact_root None rather
        # than inventing one from an unrelated directory.
        with TestBenchStore(self.tb_home) as store:
            # Older run without marker whose project_root is the hermes-agent checkout:
            # it did not use an artifact root, so the repro must not invent one.
            orig = self._seed_run(
                store, profile="verification", project_root=self.ha_root, artifact_env="__unset__"
            )
            fn, calls = self._fake_messaging()
            with mock.patch("hermes_testbench.messaging_suite.run_agent_messaging_verification", fn):
                TestBenchEngine(store).reproduce(orig.run_id)
            self.assertIsNone(calls[0]["artifact_root"])

    def test_a3_repro_chain_keeps_artifact_root_semantics(self) -> None:
        # A repro is itself a source for the next repro. Two calls (not one) and
        # the same artifact root on the second proves the marker round-trips
        # through _populate_and_finalize into the child record.
        with TestBenchStore(self.tb_home) as store:
            orig = self._seed_run(
                store,
                profile="verification",
                project_root=self.art_root,
                artifact_env=str(self.art_root),
            )
            fn, calls = self._fake_messaging()
            with mock.patch("hermes_testbench.messaging_suite.run_agent_messaging_verification", fn):
                first = TestBenchEngine(store).reproduce(orig.run_id)
                second = TestBenchEngine(store).reproduce(first["repro_run_id"])
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1]["artifact_root"], self.art_root.resolve())
            repro2 = store.get_run(second["repro_run_id"])
            assert repro2 is not None
            self.assertEqual(repro2.profile, "repro:verification")

    def test_a3_cli_messaging_verify_records_artifact_root_marker(self) -> None:
        # Two different strings on purpose: the *marker* is str(art_root) as the
        # CLI computed it, while the verifier receives art_root.resolve(). Storing
        # the marker is what makes a later repro reproduce these semantics.
        fn, calls = self._fake_messaging()
        buf = io.StringIO()
        with mock.patch("hermes_testbench.messaging_suite.run_agent_messaging_verification", fn):
            with mock.patch("sys.stdout", buf):
                rc = cli_main(["--home", str(self.tb_home), "messaging-verify", "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["environment"].get("messaging_artifact_root"), str(self.art_root))
        self.assertEqual(calls[0]["artifact_root"], self.art_root.resolve())

    # ----- A2 ------------------------------------------------------------
    def test_a2_engine_reproduce_exception_finalizes_error_not_running(self) -> None:
        # The failure must be *settled*: ERROR + harness_error + error_count 1,
        # a finished_at stamp, and the exception text in first_failure. A row
        # left RUNNING would later be reconciled as an orphan and lose the cause.
        with TestBenchStore(self.tb_home) as store:
            orig = self._seed_run(
                store,
                profile="repro:verification",
                project_root=self.art_root,
                artifact_env=str(self.art_root),
            )
            fn, _calls = self._fake_messaging(exc=RuntimeError("messaging exploded"))
            with mock.patch("hermes_testbench.messaging_suite.run_agent_messaging_verification", fn):
                out = TestBenchEngine(store).reproduce(orig.run_id)

            self.assertEqual(out["status"], RunStatus.ERROR)
            stored = store.get_run(out["repro_run_id"])
            assert stored is not None
            self.assertEqual(stored.status, RunStatus.ERROR)
            self.assertEqual(stored.failure_kind, FailureKind.HARNESS_ERROR)
            self.assertEqual(stored.error_count, 1)
            self.assertIn("messaging exploded", stored.first_failure)
            self.assertIsNotNone(stored.finished_at)
            # Lock is released after failure: a re-take must succeed immediately.
            # hold_run_lock is non-blocking, so this raises OSError if leaked.
            with store.hold_run_lock(stored.run_id):
                pass

    def test_a2_cli_messaging_verify_exception_finalizes_error_and_returns_nonzero(self) -> None:
        # Same settlement contract through the CLI, plus its own exit code:
        # ERROR is not a success, so rc must be nonzero even though the JSON
        # payload still prints (status "ERROR") for scripting.
        fn, _calls = self._fake_messaging(exc=RuntimeError("cli messaging exploded"))
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with mock.patch("hermes_testbench.messaging_suite.run_agent_messaging_verification", fn):
            with mock.patch("sys.stdout", out_buf), mock.patch("sys.stderr", err_buf):
                rc = cli_main(["--home", str(self.tb_home), "messaging-verify", "--json"])
        self.assertNotEqual(rc, 0)
        with TestBenchStore(self.tb_home) as store:
            store_runs = store.list_runs(limit=5)
        # Exactly one row: the failed attempt is recorded once, not duplicated by
        # a second finalize (or a retry) — the count is part of the assertion.
        self.assertEqual(len(store_runs), 1)
        rec = store_runs[0]
        self.assertEqual(rec.status, RunStatus.ERROR)
        self.assertEqual(rec.failure_kind, FailureKind.HARNESS_ERROR)
        self.assertEqual(rec.error_count, 1)
        self.assertIn("cli messaging exploded", rec.first_failure)
        self.assertIsNotNone(rec.finished_at)
        payload = json.loads(out_buf.getvalue())
        self.assertEqual(payload["status"], "ERROR")

    # ----- A6 ------------------------------------------------------------
    def test_a6_messaging_paths_derive_from_real_user_home(self) -> None:
        # Uses the *unpatched* helper (captured at import) so this test observes
        # the real implementation. Patching engine_mod.real_user_home works
        # because messaging_paths resolves it at call time, not at import.
        from hermes_testbench.isolation import real_user_home

        with mock.patch.object(engine_mod, "real_user_home", lambda: Path("/tmp/fakehome")):
            art, ha = _REAL_MESSAGING_PATHS()
        self.assertEqual(art, Path("/tmp/fakehome/.hermes/artifacts/agent-messaging"))
        self.assertEqual(ha, Path("/tmp/fakehome/.hermes/hermes-agent"))
        # Explicit home argument bypasses real_user_home entirely.
        art_arg, ha_arg = _REAL_MESSAGING_PATHS(Path("/tmp/otherhome"))
        self.assertEqual(art_arg, Path("/tmp/otherhome/.hermes/artifacts/agent-messaging"))
        self.assertEqual(ha_arg, Path("/tmp/otherhome/.hermes/hermes-agent"))
        # And it must agree with the real home on this host. This is the
        # regression itself: the helper may not hard-code a host path.
        art_real, ha_real = _REAL_MESSAGING_PATHS()
        self.assertEqual(art_real, real_user_home() / ".hermes" / "artifacts" / "agent-messaging")
        self.assertEqual(ha_real, real_user_home() / ".hermes" / "hermes-agent")


if __name__ == "__main__":
    unittest.main()
