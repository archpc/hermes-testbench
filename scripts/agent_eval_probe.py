#!/usr/bin/env python3
"""Bounded opt-in agent-behavior verification probe for Hermes TestBench.

Modeled on the ``architecture-decisions`` fresh-session verification workflow:
evaluates ONLY observable outcomes (exit status, persisted TestBench run record,
wall-clock duration, observable tool-call metadata, and recorded
``session_model_usage`` rows from ``state.db``). Never fabricates token or cost
accounting and never claims access to private model reasoning.

Two modes
---------
``--self-check-db PATH``
    Offline. Pure SQLite read of the given file; no subprocess, no network. Prints
    ``TESTBENCH_MODEL_USAGE=<json>`` and ``PASS observable_usage_extracted`` when a
    usage row was found, else ``FAIL no observable usage found``. This is the mode the
    unit tests use, because it needs no ``hermes`` binary and no model spend.
(default)
    Live. Spawns one ``hermes chat -q ... -Q`` session against a throwaway project with
    a single trivially passing unittest, and asks the agent to discover and run it
    through TestBench. Verification is on artifacts only: the agent's exit status, a
    persisted ``tb_*/run.json`` under the TestBench home passed in the prompt, and that
    record's ``status``/``pass_count``. The prompt text and the agent's prose are not
    parsed.

Cost of running the live mode is a real model session: it spends tokens and wall time
on the operator's own account, with a 180 s ceiling per call. Nothing outside a
``TemporaryDirectory`` is written; the usage query opens the state DB read-only.

Exit codes are 0 for pass *and* for the ``SKIP: hermes CLI not on PATH`` case, so a
caller must read the marker, not just the status, to tell "verified" from "not run".
Anything else -- missing run record, non-PASS record, zero passes, nonzero exit -- is 1.
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
    """Path to the active profile's ``state.db``, resolved from the passwd entry.

    ``pwd.getpwuid`` is preferred over ``Path.home()`` because it survives a HOME that
    an invoking harness has redefined; the ``Path.home()`` branch is only the fallback
    when the passwd lookup fails. Nothing here verifies the file exists -- a missing DB
    is a normal "no evidence" outcome for the callers below, not an error.
    """
    try:
        uh = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        uh = Path.home()
    return uh / ".hermes" / "state.db"


def read_observable_session_usage(db_path: Path, session_id: Optional[str] = None, since_ts: float = 0.0) -> Optional[Dict[str, Any]]:
    """Extract one session's recorded token/cost/tool-call rows, or ``None``.

    The DB is opened read-only through a ``file:...?mode=ro`` URI: this probe must never
    create, lock-for-write, or migrate the operator's live state file.

    Schema drift is absorbed by introspection rather than assumed: the column set is read
    from ``PRAGMA table_info`` and each optional column has an explicit fallback
    (``billing_provider`` -> ``provider`` -> empty, ``SUM(api_call_count)`` -> ``COUNT(*)``,
    ``actual_cost_usd`` absent -> SQL ``NULL``). A missing ``session_model_usage`` table,
    an unresolvable session, or a row whose token sums are both NULL returns ``None``;
    that is the "no observable evidence" answer, and it is deliberately not zeroes --
    reporting ``input_tokens: 0`` would be an assertion the DB never made.

    Session selection: without an explicit ``session_id`` the newest row with
    ``started_at >= since_ts`` is used. ``since_ts`` is therefore the only fence between
    "the session this probe just started" and a concurrent one, and it is a heuristic.

    ``actual_cost_usd`` is only included when the recorded value is positive, because the
    column is optional and a stored 0 usually means "not yet priced" rather than "free".

    Two limits a reader must not over-read:

    * the aggregate is ``GROUP BY model`` fetched with ``fetchone()``, so a session that
      used several models reports only the first group -- the totals are that model's,
      not the session's;
    * ``except sqlite3.Error`` returns ``None``, so a corrupt or locked DB is
      indistinguishable from "no rows" at this layer.
    """
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
    """CLI entry point; see the module docstring for the two modes and the exit contract.

    Emits, in order: an optional ``TESTBENCH_MODEL_USAGE=<json>`` line, then exactly one
    terminal line -- ``PASS ...``, ``FAIL ...``, or ``SKIP: ...``. The ``PASS`` line
    carries the run id, wall-clock duration, and pass count taken from the persisted
    record, never from the agent's own claim.

    Failure paths stay distinguishing: a spawn timeout raises
    ``subprocess.TimeoutExpired`` uncaught (traceback, non-zero from the interpreter), and
    a passing exit with no persisted record prints the exit code, the record count, and
    the last 300 characters of stdout so the two causes are separable from the message.
    """
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
        # Glob order is filesystem order, not recency: ``persisted_runs[0]`` is whichever
        # record sorts first, so a stale run left in the tree can be the one judged.
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
