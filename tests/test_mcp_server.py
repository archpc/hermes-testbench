"""TDD tests for TestBench local stdio MCP server: wire protocol, SDK compatibility, authz, async survival."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from hermes_testbench.isolation import real_user_home
from hermes_testbench.engine import TestBenchEngine
from hermes_testbench.mcp_server import serve_stdio
from hermes_testbench.schema import RunStatus
from hermes_testbench.store import TestBenchStore

PY = sys.executable
REPO_ROOT = Path(__file__).resolve().parent.parent
HERMES_VENV_PY = real_user_home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python"


class McpServerTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path.home() / ".hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-mcp-")
        self.root = Path(self.tmp.name)
        self.tb_home = self.root / "tb_home"
        self.code_root = self.root / "Code"
        self.code_root.mkdir(mode=0o700)
        self.proj = self.code_root / "mcp_demo"
        self.proj.mkdir(mode=0o700)
        (self.proj / ".testbench.toml").write_text(
            '[project]\nname = "mcp_demo"\n\n'
            '[[suites]]\nname = "unit_ok"\nadapter = "shell"\nprofile = "unit"\n'
            f'command = [{PY!r}, "-c", "print(\\"hello SECRET=topsecret123\\")"]\ntimeout_s = 10.0\n\n'
            '[[suites]]\nname = "slow_bg"\nadapter = "shell"\nprofile = "integration"\n'
            f'command = [{PY!r}, "-c", "import time; time.sleep(2.0); print(\\"settled_bg\\")"]\ntimeout_s = 15.0\n',
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _roundtrip(self, requests: list[dict]) -> list[dict]:
        raw_in = "".join(json.dumps(r) + "\n" for r in requests)
        instream = io.StringIO(raw_in)
        outstream = io.StringIO()
        with TestBenchStore(self.tb_home) as store:
            eng = TestBenchEngine(store, allowed_roots=[self.code_root])
            rc = serve_stdio(eng, instream=instream, outstream=outstream)
        self.assertEqual(rc, 0)
        lines = [ln for ln in outstream.getvalue().splitlines() if ln.strip()]
        return [json.loads(ln) for ln in lines]

    def test_handshake_ping_tools_list_annotations_and_malformed_args(self) -> None:
        replies = self._roundtrip(
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2.0-invalid-offer",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0.1"},
                    },
                },
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "ping"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {"name": "testbench_discover", "arguments": {"path": str(real_user_home() / ".hermes" / "skills")}},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {"name": "testbench_status", "arguments": "not-an-object"},
                },
            ]
        )
        # Notification produces no reply -> 5 responses for ids 1..5
        self.assertEqual([r["id"] for r in replies], [1, 2, 3, 4, 5])
        self.assertEqual(replies[0]["result"]["protocolVersion"], "2025-11-25")
        self.assertIn("tools", replies[0]["result"]["capabilities"])
        self.assertEqual(replies[1]["result"], {})

        tools = {t["name"]: t for t in replies[2]["result"]["tools"]}
        expected_names = {
            "testbench_discover",
            "testbench_run",
            "testbench_status",
            "testbench_results",
            "testbench_reproduce",
            "testbench_cancel",
            "testbench_benchmark",
        }
        self.assertEqual(set(tools.keys()), expected_names)
        self.assertTrue(tools["testbench_discover"]["annotations"]["readOnlyHint"])
        self.assertTrue(tools["testbench_status"]["annotations"]["readOnlyHint"])
        self.assertTrue(tools["testbench_results"]["annotations"]["readOnlyHint"])
        self.assertFalse(tools["testbench_run"]["annotations"]["readOnlyHint"])
        self.assertFalse(tools["testbench_cancel"]["annotations"]["readOnlyHint"])

        # Unauthorized path (~/.hermes/skills) returns structured tool error, never crashes
        self.assertTrue(replies[3]["result"]["isError"])
        # Malformed arguments return error cleanly
        self.assertTrue(replies[4]["result"]["isError"])

    def test_async_job_survives_mcp_subprocess_shutdown_and_settles_in_store(self) -> None:
        env = dict(os.environ)
        env["TESTBENCH_HOME"] = str(self.tb_home)
        env["TESTBENCH_ALLOWED_ROOTS"] = str(self.code_root)
        env["PYTHONPATH"] = str(REPO_ROOT)

        proc = subprocess.Popen(
            [PY, "-m", "hermes_testbench", "mcp"],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert proc.stdin and proc.stdout
        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "p", "version": "1"}},
                }
            )
            + "\n"
        )
        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "testbench_run",
                        "arguments": {"project": str(self.proj), "profile": "integration", "wait_up_to_s": 0.2},
                    },
                }
            )
            + "\n"
        )
        proc.stdin.flush()

        init_line = proc.stdout.readline()
        call_line = proc.stdout.readline()
        self.assertTrue(init_line and call_line)
        call_resp = json.loads(call_line)
        self.assertFalse(call_resp["result"].get("isError", False), call_line)
        payload = json.loads(call_resp["result"]["content"][0]["text"])
        rid = payload["run_id"]
        self.assertIn(payload["status"], ("QUEUED", "RUNNING"))

        # Immediately close stdin so MCP server exits WHILE background job is still running!
        proc.stdin.close()
        rc = proc.wait(timeout=3.0)
        proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        self.assertEqual(rc, 0)

        # Verify background job survived MCP server exit and settled to PASS in state.db!
        deadline = time.monotonic() + 8.0
        settled = None
        while time.monotonic() < deadline:
            with TestBenchStore(self.tb_home) as store:
                cur = store.get_run(rid)
                if cur and cur.status in RunStatus.TERMINAL:
                    settled = cur
                    break
            time.sleep(0.1)
        self.assertIsNotNone(settled, "Async MCP job did not settle after MCP server exited!")
        assert settled is not None
        self.assertEqual(settled.status, RunStatus.PASS, settled.first_failure)

    @unittest.skipUnless(HERMES_VENV_PY.is_file(), "Hermes venv python with mcp SDK unavailable")
    def test_real_hermes_mcp_2_sdk_client_handshake_discover_and_run(self) -> None:
        probe_code = (
            "import asyncio, json, os\n"
            "from mcp import ClientSession, StdioServerParameters\n"
            "from mcp.client.stdio import stdio_client\n"
            "async def main():\n"
            "    params = StdioServerParameters(\n"
            f"        command={PY!r},\n"
            "        args=['-m', 'hermes_testbench', 'mcp'],\n"
            "        env={\n"
            f"            'PATH': {os.environ.get('PATH', '/usr/bin:/bin')!r},\n"
            f"            'HOME': {str(Path.home())!r},\n"
            f"            'TMPDIR': {str(self.root)!r},\n"
            f"            'TESTBENCH_HOME': {str(self.tb_home)!r},\n"
            f"            'TESTBENCH_ALLOWED_ROOTS': {str(self.code_root)!r},\n"
            f"            'PYTHONPATH': {str(REPO_ROOT)!r},\n"
            "        },\n"
            "    )\n"
            "    async with stdio_client(params) as streams:\n"
            "        async with ClientSession(streams[0], streams[1]) as session:\n"
            "            init = await session.initialize()\n"
            "            tlist = await session.list_tools()\n"
            f"            disc = await session.call_tool('testbench_discover', {{'path': {str(self.proj)!r}}})\n"
            f"            run = await session.call_tool('testbench_run', {{'project': {str(self.proj)!r}, 'profile': 'unit', 'wait_up_to_s': 5.0}})\n"
            "            run_data = json.loads(run.content[0].text)\n"
            "            res = await session.call_tool('testbench_results', {'run_id': run_data['run_id']})\n"
            "            print('SDK_OK=' + json.dumps({\n"
            "                'proto': getattr(init, 'protocol_version', getattr(init, 'protocolVersion', '')),\n"
            "                'tool_count': len(tlist.tools),\n"
            "                'disc_ok': not getattr(disc, 'is_error', getattr(disc, 'isError', False)),\n"
            "                'run_status': run_data['status'],\n"
            "                'res_text': res.content[0].text,\n"
            "            }))\n"
            "asyncio.run(main())\n"
        )
        r = subprocess.run([str(HERMES_VENV_PY), "-c", probe_code], capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("SDK_OK=", r.stdout)
        sdk_out = json.loads(r.stdout.split("SDK_OK=", 1)[1].strip())
        self.assertEqual(sdk_out["tool_count"], 7)
        self.assertTrue(sdk_out["disc_ok"])
        self.assertEqual(sdk_out["run_status"], "PASS")
        self.assertNotIn("topsecret123", sdk_out["res_text"])


if __name__ == "__main__":
    unittest.main()
