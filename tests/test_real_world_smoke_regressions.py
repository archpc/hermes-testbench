"""Regression tests for real-repository integration edge cases discovered during smoke testing.

Each test pins one defect that only appeared against a real repository or a real host,
so the fixtures are shaped like the failing environment (a path long enough to matter, a
polyglot tree, a sandboxed ``ps``) rather than like the smallest legal input.

Host-conditional by nature, and that is the honest part of this file:

* the tmux case skips without a ``tmux`` binary;
* the Seatbelt case asserts full network confinement only when ``sandbox-exec`` exists —
  on Linux it asserts loopback only, because network isolation is not implemented there
  (OPERATING-GUIDE 1.1);
* the messaging case skips unless the host has both a ``hermes-agent`` checkout and an
  ``agent-messaging`` artifact directory.

A green run on a bare host therefore proves less than the test count suggests; the skips
are the reason, not flakiness. Several tests also *execute* suites for real (``cargo``
in discovery only, ``clang`` for the C sanitizer, the repo's own verifier scripts), so a
missing toolchain shows up as a failing suite, not as a skip.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
import unittest

from hermes_testbench.adapters import execute_suite
from hermes_testbench.cli import main as cli_main
from hermes_testbench.discovery import discover_project
from hermes_testbench.fixtures import run_tmux_lifecycle_probe
from hermes_testbench.isolation import (
    build_isolated_env,
    build_seatbelt_profile,
    default_scratch_root,
    real_user_home,
    run_isolated_command,
)
from hermes_testbench.messaging_suite import run_agent_messaging_verification
from hermes_testbench.schema import RunRecord, RunStatus
from hermes_testbench.store import TestBenchStore


def _deepcheck_pids() -> set[int]:
    """PIDs whose argv is a ``tmux ... new-session -d -s deepcheck`` server, read from ``ps``.

    Read from a snapshot rather than from tmux itself: the whole point is to catch a
    server that tmux no longer acknowledges (an orphaned default server in a
    ``TMUX_TMPDIR`` the killer never learned about), so asking tmux would be circular.

    The filter is a substring match on ``new-session`` and ``-s deepcheck``. Two guards
    matter: the ``ps`` line is skipped (its own argv contains ``ps -axo``, and a naive
    match could catch the probe process), and non-numeric first fields are ignored.
    The match is still coarse — any unrelated command carrying both tokens is counted —
    which is why callers subtract a baseline instead of asserting an empty set.
    """
    out = subprocess.run(
        ["/bin/ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=5.0, check=False
    ).stdout
    pids: set[int] = set()
    for line in out.splitlines():
        pid_txt, _, argv = line.strip().partition(" ")
        if not pid_txt.isdigit() or "ps -axo" in argv:
            continue
        if "new-session" in argv and "-s deepcheck" in argv:
            pids.add(int(pid_txt))
    return pids


class RealWorldSmokeRegressionTests(unittest.TestCase):
    """One class: every test needs a scratch tree under the real scratch root."""

    def setUp(self) -> None:
        scratch = default_scratch_root()
        self.tmp = tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-reg-")
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux unavailable")
    def test_deeply_nested_scratch_root_keeps_tmux_unix_socket_under_macos_sun_path_limit(self) -> None:
        """A long scratch path must not leak a default tmux server, and must still work.

        ``sun_path`` is 104 bytes and tmux appends ``/tmux-<uid>/<name>``, so a scratch
        root this deep (asserted > 115 characters) cannot be used directly as the socket
        base. ``run_tmux_lifecycle_probe`` proves the product found a short base;
        ``available``/``crash_recovered``/``server_gone`` are its three verdicts and all
        three matter — a probe that failed to crash-test would report ``clean`` for the
        wrong reason.

        The leak check is the regression proper. A bare ``tmux new-session`` ignores the
        disposable ``-L`` socket and starts the *default* server under
        ``$TMUX_TMPDIR/tmux-<uid>/default``, which ``-L <socket> kill-server`` cannot
        reach; before the socket sweep this left one orphan per invocation (83
        reproduced). Subtracting ``baseline_deepcheck`` means a server that was already
        running before the test cannot be blamed on it, so the assertion fails only on a
        new survivor.

        The second isolated command is the positive control: a suite that *does* honour
        ``TESTBENCH_TMUX_SOCKET`` must still succeed in the same deep root, so the sweep
        cannot have been "fixed" by breaking the supported path.
        """
        deep = self.root / ("nested_level_" * 6) / "sub"
        deep.mkdir(parents=True, exist_ok=True)
        self.assertGreater(len(str(deep)), 115)

        probe = run_tmux_lifecycle_probe(deep)
        self.assertTrue(probe["available"])
        self.assertTrue(probe["crash_recovered"])
        self.assertTrue(probe["server_gone"])

        baseline_deepcheck = _deepcheck_pids()
        res = run_isolated_command(
            [
                shutil.which("tmux") or "tmux",
                "-f",
                "/dev/null",
                "new-session",
                "-d",
                "-s",
                "deepcheck",
            ],
            cwd=self.root,
            evidence_dir=self.root / "ev",
            step_name="deep_tmux",
            run_id="tb_deep_1",
            needs_tmux=True,
            scratch_root=deep,
        )
        # A bare `tmux new-session` lands on the default server inside TMUX_TMPDIR
        # ($TMUX_TMPDIR/tmux-<uid>/default), which `-L <socket> kill-server` never reaches.
        # Before the socket sweep in _kill_disposable_tmux this leaked one orphan per run
        # (83 reproduced when the finding was filed — provenance, not a count this tree
        # can reproduce); it must now be gone the moment run_isolated_command returns.
        self.assertEqual(
            _deepcheck_pids() - baseline_deepcheck,
            set(),
            "a plain `tmux new-session -d -s deepcheck` server outlived run_isolated_command",
        )
        sock_script = (
            "import os, subprocess; s=os.environ['TESTBENCH_TMUX_SOCKET']; "
            "subprocess.run(['tmux','-L',s,'-f','/dev/null','new-session','-d','-s','ok'], check=True)"
        )
        res2 = run_isolated_command(
            [sys.executable, "-c", sock_script],
            cwd=self.root,
            evidence_dir=self.root / "ev2",
            step_name="deep_tmux_sock",
            run_id="tb_deep_2",
            needs_tmux=True,
            scratch_root=deep,
        )
        self.assertEqual(res2.exit_code, 0, res2.stderr_text)

    def test_isolated_env_provides_readonly_cargo_and_uv_caches_without_credentials(self) -> None:
        """Suite environments get usable build caches that carry no credentials.

        ``CARGO_HOME`` is a fresh directory holding symlinks to ``registry``/``git``/
        ``bin`` only. The two ``credentials`` checks are the reason that layout matters:
        cargo reads credentials from the ``CARGO_HOME`` *root*, so sharing the caches by
        symlink while leaving the root fresh makes the cache safe to share. Both
        spellings are checked (``credentials.toml`` and the older ``credentials``)
        because a toolchain that still reads the legacy name would otherwise see a file
        this test declared absent. The ``uv`` cache links are part of the same isolation
        contract but are *not* asserted here — the suite name is broader than its
        assertions.

        The Seatbelt profile must deny ``.cargo/credentials`` by name and the Hermes
        dashboard port ``9119`` — a deny that is not in the profile is not a deny, and
        the port rule is what keeps a sandboxed suite from talking to the live Hermes
        UI.

        The network probe asserts three different things from one run: an ephemeral
        loopback bind+connect works (suites host mock servers), ``127.0.0.1:9119`` is
        blocked, and ``1.1.1.1:53`` is blocked. ``LOOP=ok`` is required unconditionally;
        the two ``*_BLOCKED`` flags are required only with ``sandbox-exec``, because
        Linux has no network confinement and asserting it there would fail for a
        documented reason.

        The final ``/bin/ps`` run checks the ``(with no-sandbox)`` exec rule: a
        sandboxed ``ps`` cannot read other processes, and tmux plus the hermes-tmux job
        checks shell out to it. Comparing stdout to ``os.getpid()`` proves the child saw
        the *real* process table rather than an empty sandboxed view.
        """
        sb_dir = self.root / "sb"
        env = build_isolated_env(sb_dir, run_id="r1", nonce="n1")
        self.assertIn("CARGO_HOME", env)
        self.assertIn("CARGO_TARGET_DIR", env)
        ch = Path(env["CARGO_HOME"])
        self.assertTrue(ch.is_dir())
        self.assertFalse((ch / "credentials.toml").exists())
        self.assertFalse((ch / "credentials").exists())

        prof = build_seatbelt_profile(sandbox_dir=sb_dir, cwd=self.root)
        self.assertIn(".cargo/credentials", prof)
        self.assertIn(".cargo", prof)
        self.assertIn('localhost:9119', prof)

        # Verify loopback ephemeral bind works while external network + 9119 dashboard stay blocked!
        probe_py = (
            "import socket\n"
            "s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.listen(1)\n"
            "c = socket.create_connection(('127.0.0.1', port), 0.5); c.close(); s.close()\n"
            "dash_blocked = False\n"
            "try:\n"
            "    socket.create_connection(('127.0.0.1', 9119), 0.3)\n"
            "except OSError:\n"
            "    dash_blocked = True\n"
            "ext_blocked = False\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), 0.3)\n"
            "except OSError:\n"
            "    ext_blocked = True\n"
            "print(f'LOOP=ok DASH_BLOCKED={dash_blocked} EXT_BLOCKED={ext_blocked}')\n"
        )
        r_net = run_isolated_command(
            [sys.executable, "-c", probe_py],
            cwd=self.root,
            evidence_dir=self.root / "ev_net",
            step_name="net_loopback",
            run_id="tb_net_loop",
            allow_network=False,
            scratch_root=self.root,
        )
        self.assertEqual(r_net.exit_code, 0, r_net.stderr_text)
        if shutil.which("sandbox-exec"):
            self.assertIn("LOOP=ok DASH_BLOCKED=True EXT_BLOCKED=True", r_net.stdout_text)
        else:
            # No Seatbelt (Linux): network is not confined (OPERATING-GUIDE 1.1); loopback must still work.
            self.assertIn("LOOP=ok", r_net.stdout_text)

        # Verify macOS setuid /bin/ps works inside Seatbelt (needed by hermes-tmux job/lock checks)
        r_ps = run_isolated_command(
            ["/bin/ps", "-o", "pid=", "-p", str(os.getpid())],
            cwd=self.root,
            evidence_dir=self.root / "ev_ps",
            step_name="ps_seatbelt",
            run_id="tb_ps_sb",
            scratch_root=self.root,
        )
        self.assertEqual(r_ps.exit_code, 0, r_ps.stderr_text)
        self.assertEqual(r_ps.stdout_text.strip(), str(os.getpid()))

    def test_discovery_handles_cargo_lock_sibling_c_sqlite_cobol_harness_and_verify_docs_arg(self) -> None:
        """A polyglot tree yields the right suite set, and two of those suites really run.

        Several behaviours, one fixture: ``Cargo.lock`` present ⇒ ``--locked`` is appended
        (a lockfile-less tree gets a different command, so its presence must be read, not
        assumed); the GnuCOBOL TUI harness exists ⇒ ``cobol-unit`` is *not* emitted,
        because sources are only scanned when there is no harness — emitting both would
        compile the same programs twice and count them as independent evidence;
        ``.c`` files are matched by ``*selftest*.c``, so ``term.c`` (a plain helper) is
        ignored while ``selftest.c`` becomes ``c-sanitizer``; and ``verify-docs`` gets
        ``"."`` appended because the script reads ``sys.argv[1]``, meaning discovery
        folded a source-level fact into the command.

        The last two thirds execute for real: ``c-sanitizer`` needs ``clang`` with
        ASan/UBSan, ``verify-docs`` needs the interpreter to run the repo's script. Their
        exit status is the assertion, with ``summary_excerpt`` attached so a failure
        carries the compiler or script output instead of just ``FAIL``. Discovery
        correctness is thus cross-checked against execution rather than against the
        spec alone.
        """
        proj = self.root / "poly"
        proj.mkdir()
        (proj / "Cargo.toml").write_text('[package]\nname = "p"\n', encoding="utf-8")
        (proj / "Cargo.lock").write_text("# lock\n", encoding="utf-8")
        (proj / "bench" / "tui" / "cobol").mkdir(parents=True)
        (proj / "bench" / "tui" / "cobol" / "test_cobol.py").write_text("import pyte\n", encoding="utf-8")
        (proj / "bench" / "tui" / "cobol" / "listbench.cob").write_text("IDENTIFICATION DIVISION.\n", encoding="utf-8")
        (proj / "bench" / "tui" / "cobol" / "term.c").write_text(
            "int helper_val(void) { return 42; }\n", encoding="utf-8"
        )
        (proj / "bench" / "tui" / "cobol" / "selftest.c").write_text(
            "#include <sqlite3.h>\nint helper_val(void);\nint main(void){ return helper_val() == 42 ? 0 : 1; }\n",
            encoding="utf-8",
        )
        (proj / "scripts").mkdir()
        (proj / "scripts" / "verify_docs.py").write_text(
            "import os, sys\nroot = os.path.abspath(sys.argv[1])\nprint('[PASS] root: ok')\n",
            encoding="utf-8",
        )

        spec = discover_project(proj)
        by_name = {s.name: s for s in spec.suites}
        self.assertEqual(by_name["cargo-unit"].command, ["cargo", "test", "--locked", "--offline", "--workspace"])
        self.assertIn("cobol-tui-harness", by_name)
        self.assertNotIn("cobol-unit", by_name)
        self.assertEqual(by_name["verify-docs"].command[-1], ".")

        # Execute c-sanitizer and verify-docs for real!
        c_res, _ = execute_suite(
            by_name["c-sanitizer"],
            project_root=proj,
            evidence_dir=self.root / "ev_c",
            run_id="tb_reg_c",
        )
        self.assertEqual(c_res.status, RunStatus.PASS, c_res.summary_excerpt)

        v_res, _ = execute_suite(
            by_name["verify-docs"],
            project_root=proj,
            evidence_dir=self.root / "ev_v",
            run_id="tb_reg_v",
        )
        self.assertEqual(v_res.status, RunStatus.PASS, v_res.summary_excerpt)

    def test_live_agent_messaging_verification_passes_all_28_non_skipped_cases(self) -> None:
        hermes_src = real_user_home() / ".hermes/hermes-agent"
        art = real_user_home() / ".hermes/artifacts/agent-messaging"
        if not (hermes_src.is_dir() and art.is_dir()):
            self.skipTest("host hermes-agent or agent-messaging artifact absent")
        suites = run_agent_messaging_verification(
            self.root / "ev_msg",
            artifact_root=art,
            hermes_agent_root=hermes_src,
        )
        for s in suites:
            failed = [(c.name, c.message) for c in s.cases if c.status not in (RunStatus.PASS, RunStatus.SKIP)]
            self.assertEqual(s.status, RunStatus.PASS, f"{s.name} failed: {failed}")

    def test_cli_results_supports_json_and_logs_flags(self) -> None:
        tb_home = self.root / "cli_home"
        with TestBenchStore(tb_home) as store:
            ev = store.run_dir("tb_20261114_231000_cli00001")
            ev.mkdir(parents=True, exist_ok=True)
            (ev / "unit.sanitized.log").write_text("line one\nline two\n", encoding="utf-8")
            rec = RunRecord(
                run_id="tb_20261114_231000_cli00001",
                project="demo",
                project_root=str(self.root),
                revision="1234abc",
                profile="unit",
                status=RunStatus.PASS,
                evidence_dir=str(ev),
            )
            store.create_run(rec)
            store.finalize_run(rec)

        buf = io.StringIO()
        from contextlib import redirect_stdout
        with redirect_stdout(buf):
            rc = cli_main(["--home", str(tb_home), "results", "tb_20261114_231000_cli00001", "--json"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(buf.getvalue())["run_id"], "tb_20261114_231000_cli00001")

        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            rc2 = cli_main(["--home", str(tb_home), "results", "tb_20261114_231000_cli00001", "--logs"])
        self.assertEqual(rc2, 0)
        self.assertIn("line two", buf2.getvalue())

    def test_cli_bench_run_parses_options_before_double_dash_and_compares_runs(self) -> None:
        tb_home = self.root / "bench_cli_home"
        from contextlib import redirect_stdout
        b1 = io.StringIO()
        with redirect_stdout(b1):
            rc1 = cli_main(
                [
                    "--home",
                    str(tb_home),
                    "bench",
                    "run",
                    str(Path(__file__).resolve().parent.parent),
                    "--name",
                    "noop-py",
                    "--warmup",
                    "0",
                    "--repeats",
                    "2",
                    "--",
                    sys.executable,
                    "-c",
                    "x = sum(range(1000))",
                ]
            )
        self.assertEqual(rc1, 0, b1.getvalue())
        self.assertIn("noop-py", b1.getvalue())
        rid1 = b1.getvalue().split()[1]

        b2 = io.StringIO()
        with redirect_stdout(b2):
            rc2 = cli_main(
                [
                    "--home",
                    str(tb_home),
                    "bench",
                    "run",
                    str(Path(__file__).resolve().parent.parent),
                    "--name",
                    "noop-py",
                    "--warmup",
                    "0",
                    "--repeats",
                    "2",
                    "--",
                    sys.executable,
                    "-c",
                    "x = sum(range(1000))",
                ]
            )
        self.assertEqual(rc2, 0)
        rid2 = b2.getvalue().split()[1]

        bc = io.StringIO()
        with redirect_stdout(bc):
            rc_cmp = cli_main(["--home", str(tb_home), "bench", "compare", rid1, rid2])
        self.assertEqual(rc_cmp, 0)
        self.assertIn("noop-py", bc.getvalue())
        self.assertIn("p50_ms", bc.getvalue())


if __name__ == "__main__":
    unittest.main()
