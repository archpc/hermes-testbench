"""Local stdio MCP server for Hermes TestBench (ADR-0002).

Speaks newline-delimited JSON-RPC 2.0 over stdin/stdout with zero external dependencies.
Never writes anything except valid JSON-RPC frames to stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from .engine import AuthorizationError, TestBenchEngine
from .sanitize import sanitize_for_agent
from .schema import RunRecord

HANDSHAKE_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
DEFAULT_PROTOCOL_VERSION = "2025-11-25"

TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "name": "testbench_discover",
        "description": "Discover test/benchmark capabilities, suites, and profiles in an allowlisted repository.",
        "annotations": {"readOnlyHint": True, "destructiveHint": False},
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repository path under an allowlisted root."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "testbench_run",
        "description": "Execute a TestBench profile or named suite(s) in an isolated sandbox (asynchronous for long runs).",
        "annotations": {"readOnlyHint": False, "destructiveHint": False},
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Repository path to execute."},
                "profile": {"type": "string", "description": "Profile name (unit, integration, terminal, security, regression).", "default": "unit"},
                "suites": {"type": "array", "items": {"type": "string"}, "description": "Optional specific suite names."},
                "wait_up_to_s": {"type": "number", "description": "Seconds to wait before returning RUNNING handle (0..30, default 3).", "default": 3.0},
                "timeout_s": {"type": "number", "description": "Optional per-suite timeout override in seconds."},
            },
            "required": ["project"],
            "additionalProperties": False,
        },
    },
    {
        "name": "testbench_status",
        "description": "Inspect live/historical TestBench execution status by run_id or project without re-running tests.",
        "annotations": {"readOnlyHint": True, "destructiveHint": False},
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Optional specific run_id."},
                "project": {"type": "string", "description": "Optional project name filter."},
                "limit": {"type": "integer", "description": "Max recent runs to return (1..50).", "default": 10},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "testbench_results",
        "description": "Read persisted TestBench run results, counts, sanitized failure summary, and evidence file paths.",
        "annotations": {"readOnlyHint": True, "destructiveHint": False},
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Specific run_id (omit with project to read latest)."},
                "project": {"type": "string", "description": "Project slug when fetching latest run."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "testbench_reproduce",
        "description": "Return exact reproduction instructions or replay a persisted TestBench run in isolation.",
        "annotations": {"readOnlyHint": False, "destructiveHint": False},
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Target run_id to reproduce."},
                "print_only": {"type": "boolean", "description": "If true, return command without executing.", "default": False},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "testbench_cancel",
        "description": "Cancel an active TestBench-owned run by run_id (never affects arbitrary non-TestBench processes).",
        "annotations": {"readOnlyHint": False, "destructiveHint": False},
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Owned TestBench run_id to cancel."},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "testbench_benchmark",
        "description": "Execute a named benchmark suite or compare two recorded benchmark runs.",
        "annotations": {"readOnlyHint": False, "destructiveHint": False},
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["run", "compare", "list"], "default": "list"},
                "project": {"type": "string", "description": "Repository path (for run) or project name (for list)."},
                "workload": {"type": "string", "description": "Benchmark suite/workload name."},
                "baseline_run_id": {"type": "string", "description": "Baseline run_id for compare."},
                "candidate_run_id": {"type": "string", "description": "Candidate run_id for compare."},
                "wait_up_to_s": {"type": "number", "default": 5.0},
            },
            "additionalProperties": False,
        },
    },
]


def _compact_run_summary(run: RunRecord, *, include_suites: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "run_id": run.run_id,
        "project": run.project,
        "revision": run.revision,
        "profile": run.profile,
        "status": run.status,
        "failure_kind": run.failure_kind,
        "duration_ms": run.duration_ms,
        "counts": {
            "total": run.total_count,
            "pass": run.pass_count,
            "fail": run.fail_count,
            "skip": run.skip_count,
            "blocked": run.blocked_count,
            "error": run.error_count,
        },
        "first_failure": sanitize_for_agent(run.first_failure, max_chars=500) if run.first_failure else "",
        "repro_instructions": run.repro_instructions,
        "evidence_dir": run.evidence_dir,
        "run_json_path": run.run_json_path,
        "junit_xml_path": run.junit_xml_path,
        "model_usage": run.model_usage,
    }
    if include_suites:
        suites_summary = []
        for s in run.suites:
            suites_summary.append(
                {
                    "name": s.name,
                    "adapter": s.adapter,
                    "status": s.status,
                    "exit_code": s.exit_code,
                    "duration_ms": s.duration_ms,
                    "counts": {
                        "total": s.total_count,
                        "pass": s.pass_count,
                        "fail": s.fail_count,
                        "skip": s.skip_count,
                        "blocked": s.blocked_count,
                        "error": s.error_count,
                    },
                    "sanitized_log": s.sanitized_path,
                    "excerpt": sanitize_for_agent(s.summary_excerpt, max_chars=1200) if s.summary_excerpt else "",
                    "failures": [
                        {
                            "case": f.case_name,
                            "kind": f.kind,
                            "message": sanitize_for_agent(f.message or f.excerpt, max_chars=600),
                        }
                        for f in s.failures[:10]
                    ],
                }
            )
        out["suites"] = suites_summary
    return out


def _tool_ok(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2, sort_keys=True)}],
        "isError": False,
    }


def _tool_err(message: str, code: str = "error") -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps({"error": message, "code": code}, sort_keys=True)}],
        "isError": True,
    }


def _dispatch_tool(engine: TestBenchEngine, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    engine.store.reconcile_orphans()

    if name == "testbench_discover":
        raw_path = args.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return _tool_err("'path' must be a non-empty string", "invalid_arguments")
        spec = engine.discover(Path(raw_path), enforce_scope=True)
        profiles = sorted({s.profile for s in spec.suites})
        return _tool_ok(
            {
                "project": spec.name,
                "root": spec.root,
                "revision": spec.revision,
                "branch": spec.branch,
                "languages": spec.languages,
                "trusted": spec.trusted,
                "profiles": profiles,
                "suites": [
                    {
                        "name": s.name,
                        "adapter": s.adapter,
                        "profile": s.profile,
                        "billable": s.billable,
                        "needs_tmux": s.needs_tmux,
                        "needs_pty": s.needs_pty,
                        "timeout_s": s.timeout_s,
                    }
                    for s in spec.suites
                ],
            }
        )

    if name == "testbench_run":
        proj = args.get("project")
        if not isinstance(proj, str) or not proj.strip():
            return _tool_err("'project' must be a non-empty string", "invalid_arguments")
        profile = str(args.get("profile") or "unit")
        suites = args.get("suites")
        if suites is not None and (not isinstance(suites, list) or not all(isinstance(x, str) for x in suites)):
            return _tool_err("'suites' must be an array of strings", "invalid_arguments")
        wait_s = float(args.get("wait_up_to_s", 3.0))
        timeout_s = float(args["timeout_s"]) if args.get("timeout_s") is not None else None
        # Billable LLM-backed suites are never silently enabled through MCP
        rec = engine.spawn_async(
            Path(proj),
            profile=profile,
            suite_names=suites,
            allow_billable=False,
            max_tokens=0,
            timeout_s=timeout_s,
            wait_up_to_s=wait_s,
            enforce_scope=True,
        )
        return _tool_ok(_compact_run_summary(rec, include_suites=False))

    if name == "testbench_status":
        rid = args.get("run_id")
        if rid is not None:
            if not isinstance(rid, str) or not rid.strip():
                return _tool_err("'run_id' must be a non-empty string", "invalid_arguments")
            rec = engine.store.get_run(rid)
            if rec is None:
                return _tool_err(f"Unknown run_id: {rid}", "not_found")
            return _tool_ok(_compact_run_summary(rec, include_suites=False))
        proj = args.get("project")
        limit = min(max(1, int(args.get("limit", 10))), 50)
        runs = engine.store.list_runs(project=str(proj) if proj else None, limit=limit)
        last_map = engine.store.last_verified_per_project()
        return _tool_ok(
            {
                "runs": [_compact_run_summary(r, include_suites=False) for r in runs],
                "last_verified_per_project": {
                    k: {"run_id": v.run_id, "status": v.status, "revision": v.revision, "profile": v.profile}
                    for k, v in last_map.items()
                },
            }
        )

    if name == "testbench_results":
        rid = args.get("run_id")
        proj = args.get("project")
        rec: Optional[RunRecord] = None
        if isinstance(rid, str) and rid.strip():
            rec = engine.store.get_run(rid.strip())
        elif isinstance(proj, str) and proj.strip():
            recent = engine.store.list_runs(project=proj.strip(), limit=1)
            rec = recent[0] if recent else None
        else:
            recent = engine.store.list_runs(limit=1)
            rec = recent[0] if recent else None
        if rec is None:
            return _tool_err("No matching TestBench run found", "not_found")
        return _tool_ok(_compact_run_summary(rec, include_suites=True))

    if name == "testbench_reproduce":
        rid = args.get("run_id")
        if not isinstance(rid, str) or not rid.strip():
            return _tool_err("'run_id' must be a non-empty string", "invalid_arguments")
        print_only = bool(args.get("print_only", False))
        out = engine.reproduce(rid.strip(), print_only=print_only, enforce_scope=True)
        return _tool_ok(out)

    if name == "testbench_cancel":
        rid = args.get("run_id")
        if not isinstance(rid, str) or not rid.strip():
            return _tool_err("'run_id' must be a non-empty string", "invalid_arguments")
        cancelled = engine.cancel(rid.strip())
        return _tool_ok(_compact_run_summary(cancelled, include_suites=False))

    if name == "testbench_benchmark":
        action = str(args.get("action") or "list")
        if action == "list":
            proj = str(args["project"]) if args.get("project") else None
            wk = str(args["workload"]) if args.get("workload") else None
            rows = engine.store.list_benchmarks(project=proj, workload=wk, limit=20)
            return _tool_ok({"benchmarks": rows})
        if action == "compare":
            b_id = args.get("baseline_run_id")
            c_id = args.get("candidate_run_id")
            if not isinstance(b_id, str) or not isinstance(c_id, str):
                return _tool_err("'baseline_run_id' and 'candidate_run_id' required for compare", "invalid_arguments")
            rows = engine.store.list_benchmarks(limit=200)
            by_run = {r["run_id"]: r["payload"] for r in rows}
            if b_id not in by_run or c_id not in by_run:
                return _tool_err("Benchmark record not found for baseline or candidate run_id", "not_found")
            from .benchmark import compare_benchmarks  # type: ignore
            return _tool_ok(compare_benchmarks(by_run[b_id], by_run[c_id]))
        if action == "run":
            proj = args.get("project")
            if not isinstance(proj, str) or not proj.strip():
                return _tool_err("'project' path required for benchmark run", "invalid_arguments")
            wk = [str(args["workload"])] if args.get("workload") else None
            wait_s = float(args.get("wait_up_to_s", 5.0))
            rec = engine.spawn_async(
                Path(proj),
                profile="benchmark",
                suite_names=wk,
                wait_up_to_s=wait_s,
                enforce_scope=True,
            )
            return _tool_ok(_compact_run_summary(rec, include_suites=False))
        return _tool_err(f"Unsupported benchmark action: {action!r}", "invalid_arguments")

    return _tool_err(f"Unknown tool: {name!r}", "unknown_tool")


def _handle_rpc_message(engine: TestBenchEngine, msg: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(msg, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
    method = msg.get("method")
    has_id = "id" in msg
    req_id = msg.get("id")
    params = msg.get("params")

    # JSON-RPC notifications have no "id" and MUST NOT receive a response frame
    if not has_id:
        return None

    if not isinstance(method, str):
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32600, "message": "Missing method"}}

    if method == "initialize":
        offered = params.get("protocolVersion") if isinstance(params, dict) else None
        negotiated = offered if offered in HANDSHAKE_VERSIONS else DEFAULT_PROTOCOL_VERSION
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "testbench", "version": "0.1.0"},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS_SPEC}}

    if method == "tools/call":
        if not isinstance(params, dict):
            return {"jsonrpc": "2.0", "id": req_id, "result": _tool_err("params must be an object", "invalid_params")}
        tname = params.get("name")
        targs = params.get("arguments", {})
        if not isinstance(tname, str) or not isinstance(targs, dict):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": _tool_err("tool name must be string and arguments must be object", "invalid_arguments"),
            }
        try:
            res = _dispatch_tool(engine, tname, targs)
        except AuthorizationError as exc:
            res = _tool_err(str(exc), "unauthorized")
        except KeyError as exc:
            res = _tool_err(str(exc), "not_found")
        except ValueError as exc:
            res = _tool_err(str(exc), "invalid_arguments")
        except Exception as exc:
            sys.stderr.write(f"[testbench-mcp] internal error in {tname}: {exc}\n")
            sys.stderr.flush()
            res = _tool_err(f"Internal error: {exc}", "internal_error")
        return {"jsonrpc": "2.0", "id": req_id, "result": res}

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def serve_stdio(
    engine: Optional[TestBenchEngine] = None,
    *,
    instream: Optional[io.TextIOBase] = None,
    outstream: Optional[io.TextIOBase] = None,
) -> int:
    owns_engine = engine is None
    eng = engine if engine is not None else TestBenchEngine()
    inp = instream if instream is not None else sys.stdin
    out = outstream if outstream is not None else sys.stdout
    try:
        for raw_line in inp:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError as exc:
                err_frame = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                out.write(json.dumps(err_frame) + "\n")
                out.flush()
                continue
            reply = _handle_rpc_message(eng, msg)
            if reply is not None:
                out.write(json.dumps(reply) + "\n")
                out.flush()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if owns_engine:
            eng.close()
