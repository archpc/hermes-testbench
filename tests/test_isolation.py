"""Isolation contract tests: sandbox env, disk redaction, reaping, teardown safety.

Each test drives ``run_isolated_command`` (the production path) and asserts on
observable effects — files on disk, process liveness, what the child printed.
Nothing here trusts ``isolation_meta`` counters (``reaped_survivors``,
``leaked_children``): those describe what a run had to clean up, not whether the
suite is well-behaved, so the tests observe the processes and files directly.

Sandbox assumptions, because three of these tests are conditional and one
depends on where the suite itself runs:

* scratch space is created under ``~/.hermes/cache/scratch`` directly, not via
  ``default_scratch_root()``, so ``TESTBENCH_SCRATCH_DIR`` does not redirect it;
  that directory must be writable or every test in the class errors in ``setUp``.
* the Seatbelt tests skip when ``sandbox-exec`` is absent. When the suite itself
  runs inside an outer Seatbelt, ``can_apply_seatbelt()``'s probe fails and
  ``run_isolated_command`` records ``inherited_outer``; the inner denials then
  come from the outer profile, and the test cannot tell which layer denied.
* the tmux tests skip when ``tmux`` is absent. Their assertions are about a
  private server on a private socket, never the ambient one.

Liveness is checked with ``os.kill(pid, 0)``: ``ProcessLookupError`` proves no
process currently holds that pid. It does not prove the *same* process existed
and died, but pid reuse inside these sub-second windows is implausible; a test
that passed because the pid was recycled would be a flake, not a false green
worth modelling.
"""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from hermes_testbench.isolation import (
    build_isolated_env,
    run_isolated_command,
    terminate_process_group,
)

PY = sys.executable


class IsolationTests(unittest.TestCase):
    """One behaviour per test: env, redaction, streaming, reaping, teardown safety."""

    def setUp(self) -> None:
        scratch = Path.home() / ".hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-iso-")
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir(mode=0o700)
        self.evidence = self.root / "evidence"
        self.evidence.mkdir(mode=0o700)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_env_scrubbing_removes_credentials_and_live_tmux_vars_and_isolates_home(self) -> None:
        """Scrubbing is name-based; redirection moves HOME/TMPDIR; the tree is 0700.

        ``SAFE_FLAG`` is the negative control: it proves the builder filters by
        variable *name* rather than whitelisting a known environment — which is
        also why an unrecognised secret name would survive. ``base_env`` is a
        synthetic dict, so the ambient environment cannot mask a bug; nothing
        outside that literal is copied in. ``HXMAIL_DB`` is asserted only as
        "not the Application Support path": the test pins redirection off the
        real mailbox, not the replacement location.
        """
        fake_base = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "OPENAI_API_KEY": "sk-live-secret-1234567890",
            "FASTMAIL_TOKEN": "fmu1-secret-token",
            "CUSTOM_DB_PASSWORD": "super-secret-password",
            "TMUX": "/private/tmp/tmux-501/default,1234,0",
            "TMUX_PANE": "%1",
            "HERMES_SPAWN": "v1:abc:chat:1:1.0",
            "HXMAIL_DB": "/Users/sein/Library/Application Support/hxmail/mail.db",
            "SAFE_FLAG": "keep_me",
        }
        sb = self.root / "sb"
        env = build_isolated_env(sb, run_id="tb_test_1", nonce="n123", base_env=fake_base)
        for forbidden in (
            "OPENAI_API_KEY",
            "FASTMAIL_TOKEN",
            "CUSTOM_DB_PASSWORD",
            "TMUX",
            "TMUX_PANE",
            "HERMES_SPAWN",
        ):
            self.assertNotIn(forbidden, env)
        self.assertEqual(env["SAFE_FLAG"], "keep_me")
        self.assertEqual(env["HOME"], str(sb / "home"))
        self.assertEqual(env["TMPDIR"], str(sb / "tmp"))
        self.assertNotIn("Application Support", env["HXMAIL_DB"])
        self.assertEqual(stat.S_IMODE(sb.stat().st_mode), 0o700)

    def test_in_stream_secret_redaction_prevents_secrets_from_ever_reaching_disk_logs(self) -> None:
        """A token split across two writes is redacted in both files and the capture.

        The child writes the key prefix, flushes, then writes the remainder plus
        a newline: two ``read`` chunks that split the token. ``stdout.log`` is
        not a raw transcript — ``_emit_out`` writes the ``StreamRedactor``
        output — so the assertion covers ``stdout_path``, ``stderr_path``,
        ``sanitized_path``, and the in-memory texts, i.e. every place the bytes
        could land. ``[REDACTED]`` in both disk logs is the positive control
        against a writer that dropped the line entirely; the OSC 52 absence in
        ``sanitized.log`` shows the terminal-strip pass ran after redaction.
        Mode ``0600`` is asserted on the raw log only, as a spot check of
        ``_open_0600_write``. Seatbelt is left on here: the child writes only to
        stdout, which the profile permits.
        """
        res = run_isolated_command(
            [
                PY,
                "-c",
                (
                    "import sys; "
                    "sys.stdout.write('token=sk-proj-1234567890abcdef'); sys.stdout.flush(); "
                    "sys.stdout.write('ghijkl\\nok\\n'); sys.stdout.flush(); "
                    "sys.stderr.write('err \x1b]52;c;ZXZpbA==\x07 Bearer secretjwtvalue999\\n')"
                ),
            ],
            cwd=self.repo,
            evidence_dir=self.evidence,
            step_name="secret_probe",
            run_id="tb_iso_sec",
            scratch_root=self.root,
        )
        self.assertEqual(res.exit_code, 0)
        stdout_disk = Path(res.stdout_path).read_text(encoding="utf-8")
        stderr_disk = Path(res.stderr_path).read_text(encoding="utf-8")
        sanitized_disk = Path(res.sanitized_path).read_text(encoding="utf-8")

        for blob in (stdout_disk, stderr_disk, sanitized_disk, res.stdout_text, res.stderr_text):
            self.assertNotIn("1234567890abcdefghijkl", blob)
            self.assertNotIn("secretjwtvalue999", blob)
        self.assertIn("[REDACTED]", stdout_disk)
        self.assertIn("[REDACTED]", stderr_disk)
        # sanitized.log also strips OSC 52 escape sequence
        self.assertNotIn("\x1b", sanitized_disk)
        self.assertNotIn("52;c;", sanitized_disk)
        self.assertEqual(stat.S_IMODE(Path(res.stdout_path).stat().st_mode), 0o600)

    def test_sanitized_log_is_readable_while_the_command_still_runs(self) -> None:
        """The disk log streams: the marker is visible before the child exits.

        This is the property the TUI depends on (``_refresh_active_logs`` reads
        ``*.sanitized.log`` for an ACTIVE run). The child blocks on a release
        flag for up to 20s, and the test polls the file for 15s while the worker
        thread is still alive; ``worker.is_alive()`` in the poll condition is
        what makes the observation "mid-run" rather than "after the fact". A
        buffered writer would make the file either absent or empty until close.
        ``use_seatbelt=False`` keeps the timing signal about the writer rather
        than about profile setup; the child touches no file of its own.
        """
        release = self.evidence / "release.flag"
        script = (
            "import os, sys, time\n"
            "sys.stdout.write('STREAM-MARKER-ONE\\n'); sys.stdout.flush()\n"
            "deadline = time.monotonic() + 20\n"
            f"while time.monotonic() < deadline and not os.path.exists({str(release)!r}):\n"
            "    time.sleep(0.02)\n"
        )
        holder = {}

        def run_step() -> None:
            holder["res"] = run_isolated_command(
                [PY, "-c", script],
                cwd=self.repo,
                evidence_dir=self.evidence,
                step_name="stream_probe",
                run_id="tb_iso_stream",
                allow_repo_write=True,
                use_seatbelt=False,
                scratch_root=self.root,
            )

        worker = threading.Thread(target=run_step, daemon=True)
        worker.start()
        sanitized = self.evidence / "stream_probe.sanitized.log"
        seen_mid_run = False
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if worker.is_alive() and sanitized.is_file():
                try:
                    if "STREAM-MARKER-ONE" in sanitized.read_text(encoding="utf-8"):
                        seen_mid_run = True
                        break
                except OSError:
                    pass
            time.sleep(0.05)
        self.assertTrue(seen_mid_run, "sanitized.log only appeared after the command exited")
        release.write_text("go", encoding="utf-8")
        worker.join(timeout=20.0)
        self.assertFalse(worker.is_alive(), "stream probe never exited")
        res = holder["res"]
        self.assertEqual(res.exit_code, 0)
        self.assertIn("STREAM-MARKER-ONE", Path(res.sanitized_path).read_text(encoding="utf-8"))
        self.assertEqual(stat.S_IMODE(Path(res.sanitized_path).stat().st_mode), 0o600)

    def test_timeout_reaps_entire_process_group_including_spawned_grandchild(self) -> None:
        """A timed-out suite takes its whole process group with it, not just the leader.

        The child spawns a grandchild and records its pid, so the assertion can
        name a process the parent never waited on. ``os.kill(gc_pid, 0)``
        raising ``ProcessLookupError`` proves the grandchild is gone; killing
        only the leader would leave it running and fail here. ``elapsed < 5.0``
        is the timing half: a reaper that waited out the 60s sleep, or a grace
        period that ignored ``timeout_s=0.6``, would exceed it. The pid file is
        asserted to exist first — otherwise a child that never got that far
        would make the liveness check vacuous.
        """

        pid_file = self.evidence / "grandchild.pid"
        script = (
            "import os, subprocess, sys, time\n"
            f"gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"open({str(pid_file)!r}, 'w').write(str(gc.pid))\n"
            "time.sleep(60)\n"
        )
        t0 = time.monotonic()
        res = run_isolated_command(
            [PY, "-c", script],
            cwd=self.repo,
            evidence_dir=self.evidence,
            step_name="timeout_pgid",
            run_id="tb_iso_to",
            timeout_s=0.6,
            allow_repo_write=True,
            use_seatbelt=False,
            scratch_root=self.root,
        )
        elapsed = time.monotonic() - t0
        self.assertTrue(res.timed_out)
        self.assertLess(elapsed, 5.0)
        self.assertTrue(pid_file.is_file())
        gc_pid = int(pid_file.read_text().strip())
        # Verify grandchild was reaped with its process group
        with self.assertRaises(ProcessLookupError):
            os.kill(gc_pid, 0)

    def test_cancel_flag_stops_running_command_and_cleans_up_children(self) -> None:
        """A flag file observed mid-run cancels the command and reaps the child.

        Cancellation is cooperative at the loop boundary: the flag is polled
        between reads, so the test signals only once the child has written its
        pid file — that ordering keeps the assertion about a *running* child
        rather than a race with spawn. ``res.cancelled`` distinguishes this from
        a timeout, which is the other path that terminates the same group; the
        two flags are mutually exclusive by construction, so asserting
        ``cancelled`` also rules out "it only looked cancelled because it timed
        out". ``timeout_s=30`` is deliberately much larger than the 3s trigger
        deadline, so a slow start cannot be misread as cancellation.
        """

        cancel_flag = self.evidence / "cancel.flag"
        pid_file = self.evidence / "child.pid"

        def trigger_cancel() -> None:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not pid_file.is_file():
                time.sleep(0.03)
            cancel_flag.write_text("cancel\n", encoding="utf-8")

        t = threading.Thread(target=trigger_cancel, daemon=True)
        t.start()
        res = run_isolated_command(
            [
                PY,
                "-c",
                f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)",
            ],
            cwd=self.repo,
            evidence_dir=self.evidence,
            step_name="cancel_probe",
            run_id="tb_iso_cancel",
            timeout_s=30.0,
            use_seatbelt=False,
            scratch_root=self.root,
            cancel_flag_path=cancel_flag,
        )
        t.join(timeout=2.0)
        self.assertTrue(res.cancelled)
        child_pid = int(pid_file.read_text().strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)

    def test_symlink_inside_sandbox_cleanup_never_deletes_external_target(self) -> None:
        """Teardown unlinks a child-planted symlink instead of walking it.

        The child plants a link named ``escape_link`` inside its own ``HOME``
        pointing at a directory outside the sandbox. The external file surviving
        is the whole assertion: a recursive delete that followed links would
        remove ``must survive``. The second assertion — no ``tbw-*`` directories
        left under the scratch root — is the complementary half, proving cleanup
        still happened rather than "nothing was deleted anywhere", which would
        also keep the external file safe.
        """

        precious_dir = self.root / "precious"
        precious_dir.mkdir()
        precious_file = precious_dir / "keep.txt"
        precious_file.write_text("must survive", encoding="utf-8")

        res = run_isolated_command(
            [
                PY,
                "-c",
                f"import os, pathlib; os.symlink({str(precious_dir)!r}, os.path.join(os.environ['HOME'], 'escape_link'))",
            ],
            cwd=self.repo,
            evidence_dir=self.evidence,
            step_name="symlink_escape",
            run_id="tb_iso_sym",
            use_seatbelt=False,
            scratch_root=self.root,
        )
        self.assertEqual(res.exit_code, 0)
        self.assertTrue(precious_file.is_file(), "Symlink cleanup deleted external target!")
        self.assertEqual(precious_file.read_text(encoding="utf-8"), "must survive")
        # Sandbox directory itself was cleaned up
        leftover_sandboxes = list(self.root.glob("tbw-*"))
        self.assertEqual(leftover_sandboxes, [])

    @unittest.skipUnless(shutil.which("sandbox-exec"), "macOS sandbox-exec unavailable")
    def test_seatbelt_blocks_network_and_hermes_home_writes_while_allowing_scratch(self) -> None:
        """One profile enforces three different decisions at once.

        Negative control and positive control in the same child: outbound DNS
        and a write under the real ``~/.hermes`` must raise ``OSError``, while a
        write to ``$TMPDIR`` (the sandbox ``tmp``) must succeed and read back.
        Asserting the combined string pins all three outcomes in one run, so a
        profile that denied everything — including scratch — fails instead of
        looking maximally secure. The probe path under the real home is asserted
        absent afterwards: the exception is the in-sandbox view, and a file on
        disk would mean confinement was not applied at all.
        """

        from hermes_testbench.isolation import real_user_home
        hermes_probe = real_user_home() / ".hermes" / "_tb_forbidden_write_probe"
        script = (
            "import os, pathlib, socket\n"
            "net_blocked = False\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), 0.3)\n"
            "except OSError:\n"
            "    net_blocked = True\n"
            "hermes_blocked = False\n"
            "try:\n"
            f"    pathlib.Path({str(hermes_probe)!r}).write_text('bad')\n"
            "except OSError:\n"
            "    hermes_blocked = True\n"
            "scratch_ok = pathlib.Path(os.environ['TMPDIR']) / 'ok.txt'\n"
            "scratch_ok.write_text('good')\n"
            "print(f'NET={net_blocked} HERMES={hermes_blocked} SCRATCH={scratch_ok.read_text()}')\n"
        )
        res = run_isolated_command(
            [PY, "-c", script],
            cwd=self.repo,
            evidence_dir=self.evidence,
            step_name="seatbelt_probe",
            run_id="tb_iso_sb",
            allow_network=False,
            use_seatbelt=True,
            scratch_root=self.root,
        )
        self.assertEqual(res.exit_code, 0, res.stderr_text)
        self.assertIn("NET=True HERMES=True SCRATCH=good", res.stdout_text)
        self.assertFalse(hermes_probe.exists())

    @unittest.skipUnless(shutil.which("tmux"), "tmux unavailable")
    def test_disposable_tmux_server_is_isolated_and_killed_on_teardown(self) -> None:
        """A private tmux server is reachable from the suite and gone after teardown.

        ``needs_tmux=True`` gives the child a ``TMUX_TMPDIR`` and socket name of
        its own; the child starts a server there and prints its pid. The poll
        loop after teardown exists because the kill is asynchronous relative to
        the ``run_isolated_command`` return: it waits up to 2s for
        ``ProcessLookupError`` instead of sampling once. A socket-path too long
        for ``sun_path`` would fail inside the child as ``bind: File name too
        long`` and show up as ``exit_code != 0`` with stderr, not as a leaked
        server, which is why the exit status is asserted first.
        """

        script = (
            "import os, subprocess\n"
            "sock = os.environ['TESTBENCH_TMUX_SOCKET']\n"
            "subprocess.run(['tmux', '-L', sock, '-f', '/dev/null', 'new-session', '-d', '-s', 't1'], check=True)\n"
            "out = subprocess.check_output(['tmux', '-L', sock, 'display-message', '-p', '#{pid}'], text=True).strip()\n"
            "print('TMUX_PID=' + out)\n"
        )
        res = run_isolated_command(
            [PY, "-c", script],
            cwd=self.repo,
            evidence_dir=self.evidence,
            step_name="tmux_iso",
            run_id="tb_iso_tmux",
            needs_tmux=True,
            use_seatbelt=True,
            scratch_root=self.root,
        )
        self.assertEqual(res.exit_code, 0, res.stderr_text)
        self.assertIn("TMUX_PID=", res.stdout_text)
        tmux_pid = int(res.stdout_text.strip().split("TMUX_PID=")[1].splitlines()[0])
        deadline = time.monotonic() + 2.0
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(tmux_pid, 0)
                time.sleep(0.05)
            except ProcessLookupError:
                alive = False
                break
        self.assertFalse(alive, f"Disposable tmux server pid {tmux_pid} leaked after teardown!")

    @unittest.skipUnless(shutil.which("tmux"), "tmux unavailable")
    def test_teardown_refuses_child_planted_tmux_tmpdir_symlink(self) -> None:
        """A suite that swaps its TMUX_TMPDIR for a symlink must not make teardown follow it.

        Adversarial case: the sandboxed child deletes ``$TMUX_TMPDIR`` and links it at a decoy
        base hosting an unrelated tmux server. Teardown must refuse to walk the link, so the
        decoy server stays alive and the decoy directory is left untouched.

        Three assertions, each a different failure: the decoy pid still answers
        ``kill(pid, 0)`` (no server was signalled), the decoy socket is still a
        socket and its directory still exists (no directory sweep crossed the
        link), and the planted link itself is gone (teardown unlinked rather than
        ignored it, so the next run does not inherit the escape). ``realpath`` on
        both sides is required because macOS reports ``$TMPDIR`` as ``/var/...``
        while ``realpath`` yields ``/private/var/...``; comparing raw strings
        would fail for the right behaviour.
        """
        # Short TMPDIR-based base: a socket under $TMPDIR/tmux-<uid>/decoy stays far below the
        # macOS 104-byte sun_path limit even inside the nested (already sandboxed) test run.
        decoy_base = Path(tempfile.mkdtemp(prefix="tb-decoy-", dir=os.environ.get("TMPDIR") or None))
        decoy_base.chmod(0o700)
        # No TMUX/TMUX_PANE in decoy_env: the decoy is a private server, never the ambient one.
        decoy_env = dict(os.environ)
        for key in ("TMUX", "TMUX_PANE"):
            decoy_env.pop(key, None)
        decoy_env["TMUX_TMPDIR"] = str(decoy_base)

        decoy_pids: list[int] = []

        def _kill_decoy() -> None:
            for pid in decoy_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            subprocess.run(
                ["tmux", "-L", "decoy", "kill-server"],
                env=decoy_env,
                capture_output=True,
                timeout=5.0,
                check=False,
            )
            shutil.rmtree(decoy_base, ignore_errors=True)

        self.addCleanup(_kill_decoy)
        started = subprocess.run(
            ["tmux", "-L", "decoy", "-f", "/dev/null", "new-session", "-d", "-s", "decoy"],
            env=decoy_env,
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        self.assertEqual(started.returncode, 0, f"decoy tmux failed to start: {started.stderr}")
        decoy_pid = int(
            subprocess.check_output(
                ["tmux", "-L", "decoy", "display-message", "-p", "#{pid}"],
                env=decoy_env,
                text=True,
                timeout=10.0,
            ).strip()
        )
        decoy_socket = decoy_base / f"tmux-{os.getuid()}" / "decoy"
        self.assertTrue(stat.S_ISSOCK(os.lstat(decoy_socket).st_mode))
        decoy_pids.append(decoy_pid)

        script = (
            "import os, shutil, sys\n"
            "t = os.environ['TMUX_TMPDIR']\n"
            "shutil.rmtree(t)\n"
            "os.symlink(sys.argv[1], t)\n"
            "print('TMPDIR=' + t)\n"
            "print('TARGET=' + os.readlink(t))\n"
            "print('SYMLINKED=' + str(os.path.islink(t)))\n"
        )
        res = run_isolated_command(
            [PY, "-c", script, str(decoy_base)],
            cwd=self.repo,
            evidence_dir=self.evidence,
            step_name="tmux_symlink_escape",
            run_id="tb_iso_tmux_symlink",
            needs_tmux=True,
            use_seatbelt=True,
            scratch_root=self.root,
        )
        self.assertEqual(res.exit_code, 0, res.stderr_text)
        self.assertIn("SYMLINKED=True", res.stdout_text)
        planted = Path(res.stdout_text.split("TMPDIR=")[1].splitlines()[0].strip())
        target = res.stdout_text.split("TARGET=")[1].splitlines()[0].strip()
        # realpath both sides: macOS reports $TMPDIR as /var/... while realpath gives /private/var/...
        self.assertEqual(Path(os.path.realpath(target)), Path(os.path.realpath(decoy_base)))

        # The decoy server must still be alive: teardown refused to follow the planted link.
        os.kill(decoy_pid, 0)
        self.assertTrue(stat.S_ISSOCK(os.lstat(decoy_socket).st_mode))
        self.assertTrue(decoy_base.is_dir())

        # ...and the planted symlink itself was unlinked instead of traversed.
        self.assertFalse(os.path.lexists(planted))

    @unittest.skipUnless(shutil.which("sandbox-exec"), "macOS sandbox-exec unavailable")
    def test_nested_process_inherits_effective_seatbelt_restrictions_and_detects_outer_sandbox(self) -> None:
        """A nested run cannot widen the outer profile, and reports it as inherited.

        The outer run launches an inner runner that launches a leaf; the leaf
        probes the same four denials plus a scratch write. The inner run asks for
        its own Seatbelt, but ``can_apply_seatbelt()`` fails inside an outer
        sandbox, so it prints ``inherited_outer`` — that reported state, not a
        stacked profile, is what the assertions pin. The branch selecting
        ``protected_repo`` follows the same distinction: when Seatbelt is
        applicable the temp repo is the target expected to be read-only, and when
        the suite already runs inside an outer profile the source checkout is
        used instead. Only one branch is reachable on a given host; both assert
        that a write fails, so the test does not claim to prove *which* layer
        denied it.
        """

        from hermes_testbench.isolation import can_apply_seatbelt, real_user_home
        real_hermes_probe = real_user_home() / ".hermes" / "_tb_nested_effective_probe"
        real_ssh_dir = real_user_home() / ".ssh"
        repo_root_dir = Path(__file__).resolve().parent.parent
        # When already inside an outer TestBench Seatbelt, repo_root_dir is the protected read-only repo
        protected_repo = self.repo if can_apply_seatbelt() else repo_root_dir
        repo_ro_probe = protected_repo / "_tb_should_not_be_written.txt"

        leaf_file = self.root / "leaf_probe.py"
        leaf_file.write_text(
            "import json, os, pathlib, socket\n"
            "out = {}\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), 0.3)\n"
            "    out['net_denied'] = False\n"
            "except OSError:\n"
            "    out['net_denied'] = True\n"
            "try:\n"
            f"    pathlib.Path({str(real_hermes_probe)!r}).write_text('x')\n"
            "    out['hermes_write_denied'] = False\n"
            "except OSError:\n"
            "    out['hermes_write_denied'] = True\n"
            "try:\n"
            f"    pathlib.Path({str(repo_ro_probe)!r}).write_text('x')\n"
            "    out['repo_write_denied'] = False\n"
            "except OSError:\n"
            "    out['repo_write_denied'] = True\n"
            f"ssh_p = pathlib.Path({str(real_ssh_dir)!r})\n"
            "if ssh_p.exists():\n"
            "    try:\n"
            "        list(ssh_p.iterdir())\n"
            "        out['ssh_read_denied'] = False\n"
            "    except OSError:\n"
            "        out['ssh_read_denied'] = True\n"
            "else:\n"
            "    out['ssh_read_denied'] = True\n"
            "p = pathlib.Path(os.environ['TMPDIR']) / 'leaf_ok.txt'\n"
            "p.write_text('ok')\n"
            "out['leaf_tmp_ok'] = (p.read_text() == 'ok')\n"
            "print('LEAF_PERMS=' + json.dumps(out))\n",
            encoding="utf-8",
        )

        inner_file = self.root / "inner_runner.py"
        inner_file.write_text(
            "import json, os, pathlib, sys\n"
            f"sys.path.insert(0, {str(repo_root_dir)!r})\n"
            "from hermes_testbench.isolation import run_isolated_command\n"
            "ev = pathlib.Path(os.environ['TMPDIR']) / 'inner_ev'\n"
            "ev.mkdir(parents=True, exist_ok=True)\n"
            "inner_res = run_isolated_command(\n"
            f"    [sys.executable, {str(leaf_file)!r}],\n"
            f"    cwd=pathlib.Path({str(self.repo)!r}),\n"
            "    evidence_dir=ev,\n"
            "    step_name='leaf',\n"
            "    run_id='tb_inner',\n"
            "    allow_network=False,\n"
            "    allow_repo_write=False,\n"
            "    use_seatbelt=True,\n"
            ")\n"
            "print('INNER_META=' + json.dumps(inner_res.isolation_meta))\n"
            "print(inner_res.stdout_text)\n"
            "sys.exit(inner_res.exit_code or 0)\n",
            encoding="utf-8",
        )

        outer_res = run_isolated_command(
            [PY, str(inner_file)],
            cwd=self.repo,
            evidence_dir=self.evidence,
            step_name="outer_seatbelt",
            run_id="tb_outer",
            allow_network=False,
            allow_repo_write=False,
            use_seatbelt=True,
            scratch_root=self.root,
        )
        self.assertEqual(outer_res.exit_code, 0, outer_res.stderr_text + "\n" + outer_res.stdout_text)
        self.assertIn(outer_res.isolation_meta.get("seatbelt"), ("active", "inherited_outer"))
        self.assertIn('"seatbelt": "inherited_outer"', outer_res.stdout_text)
        self.assertIn('"net_denied": true', outer_res.stdout_text)
        self.assertIn('"hermes_write_denied": true', outer_res.stdout_text)
        self.assertIn('"repo_write_denied": true', outer_res.stdout_text)
        self.assertIn('"ssh_read_denied": true', outer_res.stdout_text)
        self.assertIn('"leaf_tmp_ok": true', outer_res.stdout_text)
        self.assertFalse(real_hermes_probe.exists())
        self.assertFalse(repo_ro_probe.exists())


if __name__ == "__main__":
    unittest.main()
