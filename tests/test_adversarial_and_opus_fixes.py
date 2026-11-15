"""TDD regression tests for all REQUIRED findings from Adversarial Review (R1-R6) and Opus 5.5 Critique."""

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
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-adv-")
        self.root = Path(self.tmp.name)
        self.tb_home = self.root / "tb_home"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_r1_cancel_never_signals_caller_process_group_or_bystander_processes(self) -> None:
        """Probe A regression: cancelling a run_sync job must NEVER SIGTERM our own pgroup or bystanders."""
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
        """Probe B regression: double-fork + os.setsid() child holding sandbox cwd/env is reaped on timeout."""
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
        """Probe C + Opus SBPL verification: LaunchAgents write, live tmux socket, DNS, 9222, config.yaml/state.db reads all denied."""
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
        """Probe E regression: suite with invalid cwd (file instead of dir) finalizes ERROR cleanly."""
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
        """R5 regression: JUnit XML tests/failures/errors/skipped attributes match child elements."""
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
        """R6 regression: unclosed -----BEGIN PRIVATE KEY----- followed by 1 MB single line stays bounded."""
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
        """Opus TUI critique: overview excludes benchmark runs, header counts match overview, FAIL logs lead."""
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
