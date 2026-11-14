"""TDD tests for the TestBench Hermes skill, .testbench.toml profiles, and bounded agent-eval probe."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from hermes_testbench.engine import TestBenchEngine
from hermes_testbench.isolation import default_scratch_root
from hermes_testbench.schema import RunStatus
from hermes_testbench.store import TestBenchStore

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


class SkillAndAgentEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-skev-")
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_skill_md_meets_hardline_hermes_authoring_standards(self) -> None:
        skill_path = REPO_ROOT / "skills" / "software-development" / "testbench" / "SKILL.md"
        self.assertTrue(skill_path.is_file(), f"Missing {skill_path}")
        content = skill_path.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("---\n"))
        m = re.search(r"\n---\s*\n", content[3:])
        self.assertIsNotNone(m)
        assert m is not None
        fm_text = content[4 : m.start() + 3]
        desc_match = re.search(r'^description:\s*"?([^"\n]+)"?\s*$', fm_text, re.M)
        self.assertIsNotNone(desc_match)
        assert desc_match is not None
        desc = desc_match.group(1)
        self.assertLessEqual(len(desc), 60, f"description too long ({len(desc)}): {desc!r}")
        self.assertTrue(desc.endswith("."))
        self.assertIn("author: Sein, Hermes Agent", fm_text)
        self.assertIn("platforms: [linux, macos]", fm_text)
        self.assertNotIn("/Users/", content)
        self.assertIn("terminal(", content)

    def test_repo_manifest_defines_composable_profiles_and_blocks_agent_eval_by_default(self) -> None:
        manifest = REPO_ROOT / ".testbench.toml"
        self.assertTrue(manifest.is_file(), "Missing .testbench.toml")
        with TestBenchStore(self.root / "tb_home") as store:
            eng = TestBenchEngine(store)
            spec = eng.discover(REPO_ROOT)
            profiles = {s.profile for s in spec.suites}
            for expected in ("unit", "integration", "terminal", "security", "regression", "agent-eval"):
                self.assertIn(expected, profiles)

            # Running agent-eval without opt-in must be BLOCKED with zero subprocess spawns
            rec = eng.run_sync(REPO_ROOT, profile="agent-eval")
            self.assertEqual(rec.status, RunStatus.BLOCKED)
            self.assertEqual(rec.blocked_count, 1)
            self.assertIsNone(rec.model_usage)

    def test_agent_eval_probe_reads_observable_usage_without_fabricating_or_claiming_private_cot(self) -> None:
        probe = REPO_ROOT / "scripts" / "agent_eval_probe.py"
        self.assertTrue(probe.is_file(), f"Missing {probe}")

        # Build a synthetic Hermes state.db + synthetic one-shot runner to verify observable accounting
        fake_hermes = self.root / "hermes_home"
        fake_hermes.mkdir()
        with closing(sqlite3.connect(str(fake_hermes / "state.db"))) as conn:
            conn.execute(
                "CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL, model TEXT)"
            )
            conn.execute(
                "CREATE TABLE session_model_usage ("
                "session_id TEXT, model TEXT, provider TEXT, input_tokens INTEGER, output_tokens INTEGER)"
            )
            conn.execute("INSERT INTO sessions VALUES ('sess_probe_1', 100.0, 'google/gemini-4-carbon')")
            conn.execute(
                "INSERT INTO session_model_usage VALUES ('sess_probe_1', 'google/gemini-4-carbon', 'nous', 1420, 310)"
            )
            conn.commit()

        proc = subprocess.run(
            [
                PY,
                str(probe),
                "--self-check-db",
                str(fake_hermes / "state.db"),
                "--session-id",
                "sess_probe_1",
            ],
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("TESTBENCH_MODEL_USAGE=", proc.stdout)
        usage_line = [ln for ln in proc.stdout.splitlines() if ln.startswith("TESTBENCH_MODEL_USAGE=")][0]
        usage = json.loads(usage_line.split("=", 1)[1])
        self.assertEqual(usage["input_tokens"], 1420)
        self.assertEqual(usage["output_tokens"], 310)
        self.assertNotIn("reasoning_trace", usage)


if __name__ == "__main__":
    unittest.main()
