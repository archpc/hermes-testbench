"""Regression suite for the REQUIRED Adversarial Review findings (R1-R6) and the Opus critique.

Each test is a red test written before its fix, so it encodes the *defect*, not the
implementation: it asserts an externally observable property that was false once and
must stay true. Keeping them in one module is deliberate — the findings share a theme
(a hostile or unlucky environment breaks an assumption the harness made), and a
regression here usually means an execution-boundary invariant was weakened.

Mapping and the invariant each test pins:

* R1 — ``cancel()`` must never signal the caller's process group. The fix narrowed the
  signal to a group that is (a) recorded, (b) fingerprint-verified and (c) provably not
  ``os.getpgrp()``. The test spawns a bystander *inside the test runner's own group*
  because that was the blast radius of the bug: a group signal is not selective, so
  anyone sharing the caller's group was a casualty.
* R2 — a double-fork + ``setsid()`` descendant escapes the owned process group by
  construction; only the sandbox-survivor sweep can find it. The test therefore checks
  the escape *and* the sweep, since killing the group alone would pass a weaker test.
* R3 — the Seatbelt profile must deny by default: writes outside the sandbox, reads of
  Hermes state, ambient tmux sockets, DNS and CDP.
* R4 — a manifest error (``cwd`` pointing at a file) must finalize ``ERROR`` with
  evidence, never leave a row stuck in ``RUNNING``.
* R5 — JUnit attributes must agree with the elements actually emitted, including the
  synthetic back-fill elements; this is the contract CI consumers count on.
* R6 — the stream redactor must stay bounded inside an unterminated PEM block, and must
  not enter PEM state on a mid-line mention of a BEGIN marker.

The Opus test covers the TUI projection rules: the overview is one row per project with
benchmark runs excluded, and the header counters must be computed over that same
deduped set or the header and the list disagree on screen.

Tests here are mostly process-level and time-sensitive: they poll the store for a run
row, wait for threads to join, and rely on a 15 s sleeping suite outliving a 5 s
discovery window. They create all state under ``default_scratch_root()`` so nothing
touches the real TestBench home, and they clean up with ``TemporaryDirectory`` plus an
explicit bystander kill in ``finally`` (a failed assertion must not leak the child).
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import xml.etree.ElementTree as ET

from hermes_testbench.engine import TestBenchEngine
from hermes_testbench.isolation import build_seatbelt_profile, default_scratch_root, real_user_home, run_isolated_command
from hermes_testbench.reporting import render_junit_xml
from hermes_testbench.sanitize import StreamRedactor
from hermes_testbench.schema import FailureKind, FailureRecord, ProfileName, RunRecord, RunStatus, SuiteResult
from hermes_testbench.store import TestBenchStore
from hermes_testbench.tui_render import TuiState, render_frame

PY = sys.executable


class AdversarialAndOpusRegressionTests(unittest.TestCase):
    """One test per finding; each is independent and safe to run alone.

    ``setUp`` puts the run store and every project fixture inside a fresh scratch
    directory. ``tb_home`` is a *TestBench home*, not ``HOME``: the engine writes
    ``state.db`` and ``runs/<id>/`` beneath it, which is what lets a test read a run
    row back without touching the developer's real store. Because the tests spawn
    real processes, the per-test cleanup is not enough on its own — each test that
    plants a child also terminates it explicitly.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-adv-")
        self.root = Path(self.tmp.name)
        self.tb_home = self.root / "tb_home"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_r1_cancel_never_signals_caller_process_group_or_bystander_processes(self) -> None:
        """R1: cancelling a run_sync job must never SIGTERM our own pgroup or bystanders.

        The suite is a 15 s sleep, run on a background thread so this thread can
        cancel it while it executes. ``cancel()`` is cooperative first: it writes the
        flag, the in-suite read loop notices within ~50 ms and tears its own group
        down, and the run finalizes ``CANCELLED``. If the run is still locked after a
        1.5 s wait, ``cancel()`` escalates to ``terminate_process_group`` on a
        recorded worker pgid whose start time is fingerprint-verified — never on
        ``os.getpgrp()`` and never on an unverified pid. This test exercises the
        cooperative path; the invariant under test is that no unverified group,
        especially the caller's, is ever signalled. The two negative controls are
        the point of the test:

        * a SIGTERM handler installed in *this* process, so a stray group signal is
          recorded instead of silently killing the test runner;
        * an unaffiliated ``sleep 30`` child deliberately left in this process's
          group — the exact victim class R1 was about.

        The run id is discovered by polling the store rather than by intercepting the
        engine, because ``run_sync`` commits the ``RUNNING`` row before it takes the
        lock; that ordering is what makes an external cancel possible at all.

        Known diagnostic weak point: the ``assertTrue(rid, ...)`` message says the run
        "did not enter RUNNING state", but the loop can also expire because the run
        already reached a terminal state (or because ``list_runs`` returned a row for a
        different run) within the 5 s budget. If this assert fires, read the store —
        the message names one cause among several. The 5 s poll is also the original
        value and leaves little headroom on a loaded machine; the suite itself sleeps
        15 s, a second 5 s is spent joining the thread, and ``cancel`` may spend 1.5 s
        waiting for cooperative teardown.
        """
        proj = self.root / "proj_r1"
        (proj / "tests").mkdir(parents=True)
        (proj / "tests" / "test_slow.py").write_text(
            "import time, unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_wait(self):\n"
            "        time.sleep(15)\n",
            encoding="utf-8",
        )
        # Spawn an unaffiliated bystander sharing THIS process's process group!
        bystander = subprocess.Popen([PY, "-c", "import time; time.sleep(30)"])
        got_sigterm = False
        old_handler = signal.getsignal(signal.SIGTERM)

        def _catch_term(_s: int, _f: object) -> None:
            nonlocal got_sigterm
            got_sigterm = True

        signal.signal(signal.SIGTERM, _catch_term)
        try:
            done_rec: list[RunRecord] = []

            def _runner() -> None:
                with TestBenchStore(self.tb_home) as st:
                    eng = TestBenchEngine(st)
                    done_rec.append(eng.run_sync(proj, profile="unit"))

            t = threading.Thread(target=_runner)
            t.start()

            rid = ""
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not rid:
                if self.tb_home.exists():
                    with TestBenchStore(self.tb_home) as st2:
                        runs = st2.list_runs(limit=5)
                        if runs and runs[0].status == RunStatus.RUNNING:
                            rid = runs[0].run_id
                            break
                time.sleep(0.05)
            self.assertTrue(rid, "run_sync did not enter RUNNING state")

            with TestBenchStore(self.tb_home) as st_cancel:
                c_rec = TestBenchEngine(st_cancel).cancel(rid)
                self.assertEqual(c_rec.status, RunStatus.CANCELLED)

            t.join(timeout=5.0)
            self.assertFalse(t.is_alive(), "run_sync thread did not stop after cancel")
            self.assertFalse(got_sigterm, "cancel() sent SIGTERM to caller's own process group!")
            self.assertIsNone(bystander.poll(), "cancel() killed unaffiliated bystander in caller pgroup!")
            self.assertEqual(done_rec[0].status, RunStatus.CANCELLED)
        finally:
            signal.signal(signal.SIGTERM, old_handler)
            bystander.terminate()
            bystander.wait(timeout=2.0)

    def test_r2_setsid_daemonized_descendant_is_reaped_on_timeout(self) -> None:
        """R2: a double-fork + ``setsid()`` descendant holding sandbox cwd/env is reaped.

        The child forks, calls ``setsid()`` in the first child (a new session *and*
        group), forks again, and the grandchild ``chdir``s into the sandbox and writes
        its pid before sleeping. That is a deliberate escape from the owned group, so
        ``terminate_process_group`` can never reach it; it is found only because the
        post-timeout survivor sweep matches processes that hold or name the per-run
        sandbox directory.

        Discrimination: the test asserts ``timed_out`` first, because the sweep only
        runs for timeout/cancel/non-zero exits. The survivor sweep finds the grandchild
        through an ``lsof +D`` holder scan (which includes a cwd inside the tree) and a
        ``ps`` command-line scan of the per-run sandbox path. The pid file is the proof
        it started; ``os.kill(pid, 0)`` polling with a 1.5 s budget allows for
        SIGTERM-then-SIGKILL and reaping latency.

        Timing sensitivity: the 0.6 s timeout must exceed interpreter startup plus two
        forks plus the ``chdir``/write, or the pid file will be missing and the test
        fails as "did not start before timeout" rather than as a reap failure. A pid
        recycled into an unrelated process would also keep ``kill(pid, 0)`` alive;
        the sweep runs immediately after the timeout, so that window is small but not
        zero. ``use_seatbelt=False`` keeps the test about reaping, not confinement.
        """
        pid_file = self.root / "daemon_pid.txt"
        script = (
            "import os, pathlib, sys, time\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    os.setsid()\n"
            "    pid2 = os.fork()\n"
            "    if pid2 == 0:\n"
            "        os.chdir(os.environ['TMPDIR'])\n"
            f"        pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
            "        time.sleep(30)\n"
            "    os._exit(0)\n"
            "time.sleep(30)\n"
        )
        res = run_isolated_command(
            [PY, "-c", script],
            cwd=self.root,
            evidence_dir=self.root / "ev_r2",
            step_name="setsid_escape",
            run_id="tb_r2",
            timeout_s=0.6,
            use_seatbelt=False,
            scratch_root=self.root,
        )
        self.assertTrue(res.timed_out)
        self.assertTrue(pid_file.is_file(), "daemon grandchild did not start before timeout")
        daemon_pid = int(pid_file.read_text(encoding="utf-8").strip())
        deadline = time.monotonic() + 1.5
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(daemon_pid, 0)
                time.sleep(0.05)
            except ProcessLookupError:
                alive = False
                break
        self.assertFalse(alive, f"setsid escaped grandchild {daemon_pid} survived timeout!")

    @unittest.skipUnless(shutil.which("sandbox-exec"), "macOS sandbox-exec unavailable")
    def test_r3_and_opus_sbpl_deny_by_default_writes_scoped_unix_sockets_and_hermes_read_denies(self) -> None:
        """R3 + Opus: the SBPL profile denies by default, not just at the obvious paths.

        One probe process runs under a freshly built profile and reports a dict of
        booleans; the parent asserts every value is True. Covered denials: reads of
        Hermes config/state/sessions/profiles, writes to ``~/Library/LaunchAgents``,
        ``~/.zshrc`` and ``~/.local/bin``, connect() to the ambient tmux socket, DNS
        resolution, and localhost:9222 (CDP).

        Two honest limits are encoded in that probe rather than hidden:

        * A path that does not exist is reported as ``*_read_denied=True``. That is an
          *inferred* deny (nothing was there to read), not an observed EPERM, so the
          read half can pass on a machine without a Hermes home. The write half is a
          real attempt, so its True is an observed failure rather than an inference —
          but any ``OSError`` counts, not only a Seatbelt denial.
        * The tmux check targets the hard-coded ``/private/tmp/tmux-501/default``. When
          no server is listening there — or the uid is not 501 — connect() fails with
          ENOENT and the assertion passes for the wrong reason. It proves the probe
          cannot reach *that* socket, not that all sockets are unreachable.

        Skips (conditional tests): absent ``sandbox-exec`` and a failing
        ``can_apply_seatbelt()`` probe. The second case is read as "already inside an
        outer Seatbelt", because a nested ``sandbox-exec`` cannot apply a second
        profile; that is an inference, and the skip message says so.
        """
        from hermes_testbench.isolation import can_apply_seatbelt
        if not can_apply_seatbelt():
            self.skipTest("already inside outer Seatbelt; cannot apply fresh profile")

        uh = real_user_home()
        probe = (
            "import json, pathlib, socket\n"
            "res = {}\n"
            f"for k, p in [('cfg', {str(uh / '.hermes/config.yaml')!r}), ('db', {str(uh / '.hermes/state.db')!r}), ('sess', {str(uh / '.hermes/sessions')!r}), ('prof', {str(uh / '.hermes/profiles')!r})]:\n"
            "    pth = pathlib.Path(p)\n"
            "    if not pth.exists():\n"
            "        res[k + '_read_denied'] = True\n"
            "    else:\n"
            "        try:\n"
            "            (list(pth.iterdir()) if pth.is_dir() else pth.read_bytes()[:16])\n"
            "            res[k + '_read_denied'] = False\n"
            "        except OSError:\n"
            "            res[k + '_read_denied'] = True\n"
            f"for k, p in [('launchagents', {str(uh / 'Library/LaunchAgents/_tb_probe.plist')!r}), ('zshrc', {str(uh / '.zshrc._tb_probe')!r}), ('tb_bin', {str(uh / '.local/bin/_tb_probe')!r})]:\n"
            "    try:\n"
            "        pathlib.Path(p).write_text('x')\n"
            "        res[k + '_write_denied'] = False\n"
            "    except OSError:\n"
            "        res[k + '_write_denied'] = True\n"
            "s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            "try:\n"
            "    s.connect('/private/tmp/tmux-501/default')\n"
            "    res['live_tmux_denied'] = False\n"
            "except OSError:\n"
            "    res['live_tmux_denied'] = True\n"
            "finally:\n"
            "    s.close()\n"
            "try:\n"
            "    socket.getaddrinfo('example.com', 80)\n"
            "    res['dns_denied'] = False\n"
            "except OSError:\n"
            "    res['dns_denied'] = True\n"
            "try:\n"
            "    socket.create_connection(('127.0.0.1', 9222), 0.25)\n"
            "    res['cdp_9222_denied'] = False\n"
            "except OSError:\n"
            "    res['cdp_9222_denied'] = True\n"
            "print('PERMS=' + json.dumps(res, sort_keys=True))\n"
        )
        r = run_isolated_command(
            [PY, "-c", probe],
            cwd=self.root,
            evidence_dir=self.root / "ev_r3",
            step_name="r3_opus_sbpl",
            run_id="tb_r3",
            allow_network=False,
            allow_repo_write=False,
            use_seatbelt=True,
            scratch_root=self.root,
        )
        self.assertEqual(r.exit_code, 0, r.stderr_text)
        line = [ln for ln in r.stdout_text.splitlines() if ln.startswith("PERMS=")][0]
        import json
        perms = json.loads(line.split("=", 1)[1])
        for k, val in perms.items():
            self.assertTrue(val, f"Expected {k}=True under Seatbelt, got {perms}")

    def test_r4_invalid_manifest_cwd_finalizes_run_as_error_not_orphaned_running(self) -> None:
        """R4: a suite whose ``cwd`` is a file finalizes the run ``ERROR``, with evidence.

        ``run_isolated_command`` resolves ``cwd`` strictly and raises ``ValueError``
        before spawning anything; the adapter maps that to a suite-level ``ERROR``
        with a ``spawn_error`` failure, and ``run_sync``'s exception path plus
        ``_populate_and_finalize`` turn it into a persisted ``ERROR`` run. The failure
        mode being protected against is a row left ``RUNNING`` (which the TUI shows as
        an active run and the reconciler must later repair) or a missing evidence dir.

        The test asserts three separate things because they fail independently: the
        returned record, the *persisted* record (the store is the source of truth for
        other processes), and the presence of ``run.json`` (evidence). A fix that only
        updated the in-memory record would pass the first assert and fail the second.
        """
        proj = self.root / "proj_r4"
        proj.mkdir()
        (proj / "afile.txt").write_text("not a dir\n", encoding="utf-8")
        (proj / ".testbench.toml").write_text(
            '[project]\nname = "r4"\n\n[[suite]]\nname = "bad"\nadapter = "shell"\nprofile = "unit"\ncommand = ["/usr/bin/true"]\ncwd = "afile.txt"\n',
            encoding="utf-8",
        )
        with TestBenchStore(self.tb_home) as store:
            eng = TestBenchEngine(store)
            rec = eng.run_sync(proj, profile="unit")
            self.assertEqual(rec.status, RunStatus.ERROR)
            persisted = store.get_run(rec.run_id)
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertEqual(persisted.status, RunStatus.ERROR)
            self.assertTrue((Path(rec.evidence_dir) / "run.json").is_file())

    def test_r5_junit_xml_attributes_match_emitted_elements_for_timeout_and_blocked(self) -> None:
        """R5: JUnit ``tests``/``failures``/``errors``/``skipped`` match emitted elements.

        The record is hand-built rather than produced by a run, which isolates the
        renderer: a TIMEOUT suite with one failure record and a BLOCKED suite with no
        cases at all. ``render_junit_xml`` back-fills synthetic testcases for counters
        that the adapter never reported, so the invariant is not "counts equal input"
        but "attributes equal the elements in the document". The test recomputes each
        count from the parsed XML rather than trusting the renderer's own arithmetic,
        which is why it can catch a mismatch between the two.

        The document-level assert (``testsuites@tests`` equals the sum over suites)
        catches the case where per-suite counts are internally consistent but the
        roll-up is stale. Note that the TIMEOUT failure must become ``<error>`` and the
        BLOCKED one ``<skipped>``; the negative-control value here is that a renderer
        classifying both as ``<failure>`` would still pass the sum check but fail the
        per-element counts.
        """
        rec = RunRecord(
            run_id="tb_r5_junit",
            project="demo",
            project_root="/tmp/demo",
            revision="abc",
            profile="unit",
            status=RunStatus.FAIL,
            total_count=2,
            fail_count=1,
            blocked_count=1,
            suites=[
                SuiteResult(
                    name="s_timeout",
                    adapter="shell",
                    profile="unit",
                    command=["/bin/sleep", "10"],
                    cwd=".",
                    status=RunStatus.TIMEOUT,
                    total_count=1,
                    fail_count=1,
                    failures=[FailureRecord(suite="s_timeout", case_name="hang", kind=FailureKind.TIMEOUT, message="timed out")],
                ),
                SuiteResult(
                    name="s_blocked",
                    adapter="agent_eval",
                    profile="agent-eval",
                    command=["python3", "eval.py"],
                    cwd=".",
                    status=RunStatus.BLOCKED,
                    total_count=1,
                    blocked_count=1,
                    summary_excerpt="billable blocked",
                ),
            ],
        )
        xml_str = render_junit_xml(rec)
        root = ET.fromstring(xml_str)
        for ts in root.findall("testsuite"):
            cases = ts.findall("testcase")
            n_fail = sum(1 for c in cases if c.find("failure") is not None)
            n_err = sum(1 for c in cases if c.find("error") is not None)
            n_skip = sum(1 for c in cases if c.find("skipped") is not None)
            self.assertEqual(int(ts.attrib["tests"]), len(cases), xml_str)
            self.assertEqual(int(ts.attrib["failures"]), n_fail, xml_str)
            self.assertEqual(int(ts.attrib["errors"]), n_err, xml_str)
            self.assertEqual(int(ts.attrib["skipped"]), n_skip, xml_str)
        self.assertEqual(
            int(root.attrib["tests"]),
            sum(int(ts.attrib["tests"]) for ts in root.findall("testsuite")),
        )

    def test_r6_stream_redactor_caps_memory_inside_unclosed_pem_block(self) -> None:
        """R6: 1.28 MB inside an unterminated PEM block stays bounded and redacted.

        On the first ``feed`` the line-anchored BEGIN marker puts the redactor into PEM
        state; the following 20 chunks contain no newline, so no line is ever
        complete. The buffer therefore has to be trimmed by the cap path: it keeps
        only ``OVERLAP_TAIL`` (512 chars) and emits exactly one ``[REDACTED]``.
        ``MAX_BUFFER_CHARS * 2`` is a loose upper bound. In this scenario every feed
        pushes the buffer past the cap while in PEM state, so it ends at
        ``<= OVERLAP_TAIL``; a sub-cap final feed would leave a larger buffer, so that
        tighter bound holds only here. The loose bound is chosen so the assertion does
        not pin an internal constant to the exact byte.

        The second half is the negative control for the *other* direction: prose that
        mentions the BEGIN marker mid-line must not be treated as a PEM header, because
        otherwise every subsequent line of a test log would be swallowed. Both halves
        matter: a cap that redacts everything unconditionally would pass the first and
        fail the second.
        """
        red = StreamRedactor()
        out = red.feed("-----BEGIN RSA PRIVATE KEY-----\n")
        for _ in range(20):
            out += red.feed("MIIEpAIBAAKCAQEA" * 4096)  # 64 KB * 20 = 1.28 MB without END
        self.assertLessEqual(len(red._buf), StreamRedactor.MAX_BUFFER_CHARS * 2)
        out += red.finish()
        self.assertNotIn("MIIEpAIBAAKCAQEA", out)
        self.assertIn("REDACTED", out)

        # Prose mentioning -----BEGIN PRIVATE KEY----- mid-line must NOT swallow subsequent lines!
        red2 = StreamRedactor()
        prose_out = red2.feed("Docstring mentioning -----BEGIN PRIVATE KEY----- mid-sentence ... ok\nRan 188 tests in 1.0s\n\nOK\n") + red2.finish()
        self.assertIn("Ran 188 tests in 1.0s", prose_out)

    def test_opus_tui_overview_excludes_benchmarks_aligns_header_counts_and_surfaces_failures(self) -> None:
        """TUI: the overview is one row per project, and the header counts that set.

        Four synthetic runs set up three independent traps:

        * a *newer* benchmark run for the same project — must not shadow the
          regression PASS, because benchmarks live in their own view;
        * an older UNIT failure superseded by the newer PASS — must not inflate the
          fail count, because the overview keeps only the latest terminal run per
          project;
        * a failing run in a different project — must be visible and selected, so the
          inspector shows its repro command and its ``[FAIL]`` log line.

        ``1 pass`` / ``1 fail`` in the header is therefore computed over the deduped
        overview set, not over ``state.runs``. ``assertNotIn("5/5", ...)`` is a coarse
        guard against the benchmark row leaking into the body, and could in principle
        collide with unrelated text; it is kept because the positive assertions
        (``180/180``) already pin the intended row.

        The narrow 80x24 check pins the hint band's fallback: the full hint line does
        not fit, and the compact variant must still offer ``? help`` on the
        second-to-last line (the minibuffer is the last line).
        """
        r_pass = RunRecord(
            run_id="tb_20261114_100000_pass0001",
            project="hermes-testbench",
            project_root="/Users/sein/Code/hermes-testbench",
            revision="1111aaa",
            profile=ProfileName.REGRESSION,
            status=RunStatus.PASS,
            created_at="2026-11-14T10:00:00Z",
            total_count=180,
            pass_count=180,
            duration_ms=14500.0,
        )
        # Later benchmark run for same project must NOT shadow regression PASS in overview!
        r_bench = RunRecord(
            run_id="tb_20261114_100500_bnch0001",
            project="hermes-testbench",
            project_root="/Users/sein/Code/hermes-testbench",
            revision="1111aaa",
            profile=ProfileName.BENCHMARK,
            status=RunStatus.PASS,
            created_at="2026-11-14T10:05:00Z",
            total_count=5,
            pass_count=5,
            duration_ms=45.0,
        )
        # Historical old failure superseded by r_pass must NOT inflate overview fail count!
        r_old_fail = RunRecord(
            run_id="tb_20261114_090000_fail0000",
            project="hermes-testbench",
            project_root="/Users/sein/Code/hermes-testbench",
            revision="0000zzz",
            profile=ProfileName.UNIT,
            status=RunStatus.FAIL,
            created_at="2026-11-14T09:00:00Z",
            total_count=180,
            pass_count=179,
            fail_count=1,
        )
        r_cur_fail = RunRecord(
            run_id="tb_20261114_100600_apfw0001",
            project="apple-power-firmware",
            project_root="/Users/sein/.hermes/artifacts/apple-power-firmware",
            revision="4963730",
            profile=ProfileName.REGRESSION,
            status=RunStatus.FAIL,
            created_at="2026-11-14T10:06:00Z",
            total_count=13,
            pass_count=12,
            fail_count=1,
            first_failure="verify-docs: 1 failed",
            repro_instructions="python3 scripts/verify_docs.py .",
            evidence_dir="/Users/sein/.local/share/testbench/runs/tb_20261114_100600_apfw0001",
        )
        state = TuiState(
            runs=(r_cur_fail, r_bench, r_pass, r_old_fail),
            log_lines={
                r_cur_fail.run_id: (
                    "[PASS] check 1",
                    "[PASS] check 2",
                    "[PASS] check 3",
                    "[PASS] check 4",
                    "[FAIL] dispatcher inventory: 3.2.0 mismatch",
                )
            },
        )
        wide = "\n".join(render_frame(state, 136, 26, no_color=True))
        # Header counts match displayed overview (1 pass, 1 fail — NOT 2 fail!)
        self.assertIn("1 pass", wide.splitlines()[0])
        self.assertIn("1 fail", wide.splitlines()[0])
        # Overview shows 180/180 regression, NOT 5/5 benchmark!
        self.assertIn("180/180", wide)
        self.assertNotIn("5/5", wide)
        # Inspector shows testbench reproduce <run_id> AND surfaces [FAIL] line!
        self.assertIn("testbench reproduce tb_20261114_100600_apfw0001", wide)
        self.assertIn("[FAIL] dispatcher inventory", wide)

        # Narrow 80x24 key hints must NOT truncate "? help"
        narrow_lines = render_frame(state, 80, 24, no_color=True)
        self.assertIn("? help", narrow_lines[-2])


if __name__ == "__main__":
    unittest.main()
