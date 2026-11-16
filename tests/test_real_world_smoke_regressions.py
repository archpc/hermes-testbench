"""Regression tests for real-repository integration edge cases discovered during smoke testing."""

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
    """PIDs whose argv is a ``tmux ... new-session -d -s deepcheck`` server, read from ``ps``."""
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
    def setUp(self) -> None:
        scratch = default_scratch_root()
        self.tmp = tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-reg-")
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @unittest.skipUnless(shutil.which("tmux"), "tmux unavailable")
    def test_deeply_nested_scratch_root_keeps_tmux_unix_socket_under_macos_sun_path_limit(self) -> None:
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
        # (83 reproduced); it must now be gone the moment run_isolated_command returns.
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
