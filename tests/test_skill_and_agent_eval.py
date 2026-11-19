"""TDD tests for the TestBench Hermes skill, .testbench.toml profiles, and bounded agent-eval probe.

Three contracts, each enforced structurally rather than by trusting the artifacts to
stay correct:

1. ``skills/software-development/testbench/SKILL.md`` meets the Hermes hardline
   authoring rules (YAML frontmatter, a ``<= 60`` character description ending in a
   period, declared author/platforms, no absolute home paths, and a body that actually
   shows how to invoke the tool).
2. The repository manifest declares the composable profile set, and ``agent-eval`` is
   *blocked* without opt-in — blocked, not skipped, and with no subprocess spawned.
3. ``scripts/agent_eval_probe.py`` reports only observable accounting from a Hermes
   ``state.db``: real token sums that exist as rows, and no fabricated reasoning trace.

Environment sensitivity: everything here reads the checkout at ``REPO_ROOT`` (derived
from this file, so it works from a worktree) and drives the *source* package, not an
installed one. The probe subprocess is run with ``PY`` (``sys.executable``) so it shares
the suite's interpreter; no Hermes install, network or model call is involved — the
``--self-check-db`` mode exists precisely to keep this test offline.
"""

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
    """Static checks on the shipped artifacts, plus one offline probe round-trip."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=str(default_scratch_root()), prefix="tb-skev-")
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_skill_md_meets_hardline_hermes_authoring_standards(self) -> None:
        """The skill file is machine-loadable, not merely readable prose.

        Each assertion maps to a loader constraint rather than a style preference: the
        file must *start* with ``---`` and close the frontmatter (otherwise Hermes
        treats the whole file as body), ``description`` must be a single unquoted-or-
        quoted line of at most 60 characters ending in a period (the index truncates
        longer ones, so a long description silently loses its trigger), and
        ``author``/``platforms`` must be present verbatim because the frontmatter is
        matched textually, not parsed as YAML.

        ``/Users/`` is refused because a skill is copied between machines and a home
        path in it is both wrong everywhere else and a leak. ``terminal(`` is the
        minimum evidence that the skill teaches the tool's real entry point instead of
        describing it abstractly — a weak check, deliberately: it cannot tell a working
        example from a wrong one.
        """
        skill_path = REPO_ROOT / "skills" / "software-development" / "testbench" / "SKILL.md"
        self.assertTrue(skill_path.is_file(), f"Missing {skill_path}")
        content = skill_path.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("---\n"))
        # Frontmatter spans content[4:m.start()+3]: skip "---\n", then stop at the
        # closing rule found by searching from content[3:] (whose match offsets are
        # relative to that slice, hence the +3).
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
        """The shipped manifest composes profiles, and ``agent-eval`` needs opt-in.

        The profile set is asserted by membership, not equality: extra profiles are fine,
        but a missing one means a documented invocation (``--profile security``) silently
        selects nothing.

        The second half is the security-relevant assertion. ``agent-eval`` invokes a real
        agent, so running it by accident must cost nothing: ``BLOCKED`` (not SKIP, not
        PASS), exactly one blocked suite, and ``model_usage is None``. The last one is
        what proves no model call happened — a run that blocked *after* spending tokens
        would still report BLOCKED, and only the absent usage record distinguishes the
        two. ``discover`` runs in a throwaway store, so no history is read or written.
        """
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
        """The probe reports the token sums that exist as rows, and nothing it cannot see.

        The synthetic ``state.db`` is minimal on purpose: only the two tables and the
        columns the probe's PRAGMA-driven query needs, with one session and one usage
        row. That makes the expected numbers exact (``1420``/``310``) — a probe that
        guessed, scaled or defaulted would produce different ones, and a probe that
        hardcoded them would fail the moment the fixture changed.

        ``--self-check-db`` is the offline contract: no Hermes install, no network, no
        model call. ``reasoning_trace`` must be *absent*, which is the assertion with
        teeth — a probe that inferred chain-of-thought from output length, or echoed a
        field it found in some other schema version, would fail here rather than ship
        an unverifiable claim. Absence is checked by key, so renaming the field does not
        count as compliance.

        The 15 s timeout bounds a probe that resolves the real ``state.db`` instead of
        the ``--self-check-db`` path; a hang would otherwise read as a slow test.
        """
        probe = REPO_ROOT / "scripts" / "agent_eval_probe.py"
        self.assertTrue(probe.is_file(), f"Missing {probe}")

        # Synthetic Hermes state.db only: a real chat run is not needed (and must not be
        # triggered) to verify observable accounting. Columns are exactly the ones the
        # probe's PRAGMA probes look for, so a schema expectation that drifts shows up
        # as a failing assertion rather than a silent None.
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
