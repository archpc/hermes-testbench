#!/usr/bin/env python3
"""Bounded opt-in agent-behavior verification probe for Hermes TestBench.

Modeled on the ``architecture-decisions`` fresh-session verification workflow:
evaluates ONLY observable outcomes (exit status, persisted TestBench run record,
wall-clock duration, observable tool-call metadata, and recorded
``session_model_usage`` rows from ``state.db``). Never fabricates token or cost
accounting and never claims access to private model reasoning.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import pwd
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional


def _real_hermes_db() -> Path:
    try:
        uh = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        uh = Path.home()
    return uh / ".hermes" / "state.db"


def read_observable_session_usage(db_path: Path, session_id: Optional[str] = None, since_ts: float = 0.0) -> Optional[Dict[str, Any]]:
    if not db_path.is_file():
        return None
    try:
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
            u_cols = {r[1] for r in conn.execute("PRAGMA table_info(session_model_usage)").fetchall()}
            s_cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
            if not u_cols:
                return None

            sid = session_id
            if not sid and "started_at" in s_cols:
                row = conn.execute(
                    "SELECT id FROM sessions WHERE started_at >= ? ORDER BY started_at DESC LIMIT 1",
                    (since_ts,),
                ).fetchone()
                if row:
                    sid = str(row[0])
            if not sid:
                return None

            prov_col = "billing_provider" if "billing_provider" in u_cols else ("provider" if "provider" in u_cols else "''")
            calls_col = "SUM(api_call_count)" if "api_call_count" in u_cols else "COUNT(*)"
            cost_col = "SUM(actual_cost_usd)" if "actual_cost_usd" in u_cols else "NULL"

            urow = conn.execute(
                f"SELECT model, {prov_col}, {calls_col}, SUM(input_tokens), SUM(output_tokens), {cost_col} "
                "FROM session_model_usage WHERE session_id = ? GROUP BY model",
                (sid,),
            ).fetchone()
            if not urow or (urow[3] is None and urow[4] is None):
                return None

            usage: Dict[str, Any] = {
                "session_id": sid,
                "model": str(urow[0] or ""),
                "provider": str(urow[1] or ""),
                "api_calls": int(urow[2] or 0),
                "input_tokens": int(urow[3] or 0),
                "output_tokens": int(urow[4] or 0),
            }
            if urow[5] is not None and float(urow[5]) > 0:
                usage["actual_cost_usd"] = float(urow[5])

            if "tool_call_count" in s_cols:
                srow = conn.execute(
                    "SELECT tool_call_count, tool_names FROM sessions WHERE id = ?",
                    (sid,),
                ).fetchone() if "tool_names" in s_cols else conn.execute(
                    "SELECT tool_call_count, '' FROM sessions WHERE id = ?",
                    (sid,),
                ).fetchone()
                if srow:
                    usage["observable_tool_calls"] = int(srow[0] or 0)
                    if srow[1]:
                        usage["observable_tool_names"] = str(srow[1])
            return usage
    except sqlite3.Error:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check-db", default="", help="Offline unit-test mode reading a synthetic state.db")
    ap.add_argument("--session-id", default="", help="Specific session_id to inspect")
    ap.add_argument("--max-tokens", type=int, default=16000)
    args = ap.parse_args()

    if args.self_check_db:
        usage = read_observable_session_usage(Path(args.self_check_db), session_id=args.session_id or None)
        if usage is not None:
            print("TESTBENCH_MODEL_USAGE=" + json.dumps(usage, sort_keys=True))
            print("PASS observable_usage_extracted")
            return 0
        print("FAIL no observable usage found")
        return 1

    hermes_bin = shutil.which("hermes")
    if not hermes_bin:
        print("SKIP: hermes CLI not on PATH")
        return 0

    scratch = Path(os.environ.get("TMPDIR") or "/tmp")
    with tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-aeval-") as td:
        tdp = Path(td)
        demo_repo = tdp / "unfamiliar_repo"
        (demo_repo / "tests").mkdir(parents=True)
        (demo_repo / "tests" / "test_math.py").write_text(
            "import unittest\nclass T(unittest.TestCase):\n    def test_add(self): self.assertEqual(2 + 2, 4)\n",
            encoding="utf-8",
        )
        tb_home = tdp / "tb_home"
        prompt = (
            f"Load skill 'testbench' if available, then use TestBench (`testbench --home {tb_home} ...`) "
            f"to discover and run the unit tests in `{demo_repo}` and report the run_id and pass count in one line."
        )
        t_start = time.time()
        proc = subprocess.run(
            [hermes_bin, "chat", "-q", prompt, "-Q"],
            capture_output=True,
            text=True,
            timeout=180.0,
            check=False,
        )
        elapsed_s = time.time() - t_start
        runs_dir = tb_home / "runs"
        persisted_runs = list(runs_dir.glob("tb_*/run.json")) if runs_dir.is_dir() else []

        usage = read_observable_session_usage(_real_hermes_db(), since_ts=t_start - 1.0)
        if usage is not None:
            print("TESTBENCH_MODEL_USAGE=" + json.dumps(usage, sort_keys=True))

        if proc.returncode == 0 and persisted_runs:
            rj = json.loads(persisted_runs[0].read_text(encoding="utf-8"))
            ok = rj.get("status") == "PASS" and int(rj.get("pass_count", 0)) >= 1
            if ok:
                print(
                    f"PASS fresh_session_skill_invocation run_id={rj['run_id']} "
                    f"duration={elapsed_s:.2f}s pass_count={rj['pass_count']}"
                )
                return 0
        print(f"FAIL fresh_session rc={proc.returncode} persisted={len(persisted_runs)} out={proc.stdout[-300:]!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
