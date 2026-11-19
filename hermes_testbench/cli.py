"""Standalone CLI for Hermes TestBench."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import List, Optional

from .engine import AuthorizationError, TestBenchEngine
from .flake import (
    FlakeHistory,
    FlakeReport,
    FlakeVerdict,
    TargetClassification,
    analyze_flake_history,
)
from .reporting import format_terminal_summary, render_junit_xml, render_run_json
from .schema import RunStatus
from .store import TestBenchStore


def _version_string() -> str:
    """Package version, deployed release SHA, store schema, interpreter, and sandbox backend."""
    import platform
    import shutil

    from . import __version__
    from .schema import SCHEMA_VERSION

    sandbox = "seatbelt" if shutil.which("sandbox-exec") else "none (env and process-group isolation only)"
    release = Path(__file__).resolve().parents[1] / "RELEASE"
    sha = f"release {release.read_text(encoding='utf-8').strip()}; " if release.is_file() else ""
    return (
        f"testbench {__version__} ({sha}schema {SCHEMA_VERSION}; python {platform.python_version()} "
        f"{sys.executable}; {platform.system()} {platform.machine()}; sandbox {sandbox})"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="testbench",
        description="Hermes TestBench: isolated testing, benchmarking, and verification CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--home", default="", help="Override TESTBENCH_HOME store directory")
    p.add_argument("--version", action="version", version=_version_string())
    sub = p.add_subparsers(dest="command")

    disc = sub.add_parser("discover", help="Discover capabilities, suites, and profiles in a repository")
    disc.add_argument("path", nargs="?", default=".")
    disc.add_argument("--json", action="store_true")

    runp = sub.add_parser("run", help="Execute a test profile or named suites in isolation")
    runp.add_argument("path", nargs="?", default=".")
    runp.add_argument("--profile", default="unit")
    runp.add_argument("--suite", action="append", default=[], help="Specific suite name (repeatable)")
    runp.add_argument("--async", dest="async_run", action="store_true", help="Detach background worker")
    runp.add_argument("--wait", type=float, default=0.5, help="Seconds to wait when --async is used")
    runp.add_argument("--timeout", type=float, default=None, help="Per-suite timeout override (s)")
    runp.add_argument("--allow-billable", action="store_true", help="Explicitly permit billable model-backed suites")
    runp.add_argument("--max-tokens", type=int, default=0, help="Token budget required with --allow-billable")
    runp.add_argument(
        "--mutation-gate",
        type=float,
        default=None,
        help="Fail the run when caught/(caught+survived) falls below this fraction (mutation profile)",
    )
    runp.add_argument("--json", action="store_true")
    runp.add_argument("--stream", action="store_true", help="Show sanitized output during a foreground run")

    statp = sub.add_parser("status", help="Show active and recent TestBench runs")
    statp.add_argument("run_id", nargs="?", default="")
    statp.add_argument("--project", default="")
    statp.add_argument("--limit", type=int, default=15)
    statp.add_argument("--json", action="store_true")

    resp = sub.add_parser("results", help="Inspect persisted run results without rerunning tests")
    resp.add_argument("run_id", nargs="?", default="")
    resp.add_argument("--last", action="store_true", help="Show most recent run")
    resp.add_argument("--project", default="")
    resp.add_argument("--format", choices=["text", "json", "junit"], default="text")
    resp.add_argument("--json", action="store_true", help="Alias for --format json")
    resp.add_argument("--logs", action="store_true", help="Include sanitized suite log tails")

    reprop = sub.add_parser("reproduce", help="Print or replay reproduction instructions for a run")
    reprop.add_argument("run_id")
    reprop.add_argument("--print-only", action="store_true")
    reprop.add_argument("--json", action="store_true")

    canp = sub.add_parser("cancel", help="Cancel a TestBench-owned active run")
    canp.add_argument("run_id")
    canp.add_argument("--json", action="store_true")

    bench = sub.add_parser("bench", help="Run or compare reproducible benchmarks")
    # Parent-level so `testbench bench` (no sub-action) lists without AttributeError.
    bench.add_argument("--project", default="")
    bench.add_argument("--json", action="store_true")
    bsub = bench.add_subparsers(dest="bench_action")
    brun = bsub.add_parser("run", help="Run a benchmark workload")
    brun.add_argument("path", nargs="?", default=".")
    brun.add_argument("--name", default="workload")
    brun.add_argument("--warmup", type=int, default=1)
    brun.add_argument("--repeats", type=int, default=5)
    brun.add_argument("--timeout", type=float, default=120.0)
    brun.add_argument("--json", action="store_true")
    bcomp = bsub.add_parser("compare", help="Compare two benchmark run IDs")
    bcomp.add_argument("baseline_run_id")
    bcomp.add_argument("candidate_run_id")
    bcomp.add_argument("--json", action="store_true")
    blist = bsub.add_parser("list", help="List recorded benchmarks")
    blist.add_argument("--project", default="")
    blist.add_argument("--json", action="store_true")

    projp = sub.add_parser("projects", help="Manage registered projects")
    psub = projp.add_subparsers(dest="proj_action")
    plist = psub.add_parser("list", help="List registered projects")
    plist.add_argument("--json", action="store_true")
    preg = psub.add_parser("register", help="Register a repository path for scoped access")
    preg.add_argument("path")
    preg.add_argument("--name", default="")
    preg.add_argument("--json", action="store_true")

    trustp = sub.add_parser("trust", help="Trust a repository's current .testbench.toml SHA-256")
    trustp.add_argument("path", nargs="?", default=".")
    trustp.add_argument("--json", action="store_true")

    prunep = sub.add_parser("prune", help="Prune old unpinned run artifacts")
    prunep.add_argument("--keep-last", type=int, default=50)
    prunep.add_argument("--older-than-days", type=int, default=None)
    prunep.add_argument("--dry-run", action="store_true")
    prunep.add_argument("--json", action="store_true")

    msgp = sub.add_parser("messaging-verify", help="Run offline agent-messaging verification suite")
    msgp.add_argument("--hermes-pytest", action="store_true", help="Also run Hermes offline bot_mode_dm pytest suite")
    msgp.add_argument("--json", action="store_true")

    tapep = sub.add_parser("tape", help="Replay declarative terminal tapes against approved goldens")
    tsub = tapep.add_subparsers(dest="tape_action")
    trun = tsub.add_parser("run", help="Replay a .tape script in an isolated PTY")
    trun.add_argument("tape_file")
    trun.add_argument("--project-root", default="", help="Program cwd (defaults to the tape's directory)")
    trun.add_argument("--golden-dir", default="", help="Golden directory (defaults to <tape dir>/golden)")
    trun.add_argument("--timeout", type=float, default=120.0)
    trun.add_argument("--json", action="store_true")
    tapp = tsub.add_parser("approve", help="Approve a captured screen as a golden (interactive TTY only)")
    tapp.add_argument("evidence_or_run_id")
    tapp.add_argument("--golden", default="", help="Golden name (defaults to the tape name)")
    tapp.add_argument("--golden-dir", required=True)
    tapp.add_argument("--yes", action="store_true", help="Skip the confirmation prompt (still requires a TTY)")
    tapp.add_argument("--json", action="store_true")

    tuip = sub.add_parser("tui", help="Launch interactive terminal UI (hx:testbench)")
    tuip.add_argument("--quit-on-q", action="store_true", help="Exit on lowercase q at top level")
    tuip.add_argument("--project", default="", help="Initial project filter")

    flakep = sub.add_parser("flake", help="Hunt flaky tests: repeat executions and classify variance")
    fsub = flakep.add_subparsers(dest="flake_action")
    frun = fsub.add_parser(
        "run",
        help="Repeat a profile/suite group and classify each target (exit 0 only when all always_pass)",
    )
    frun.add_argument("path", nargs="?", default=".")
    frun.add_argument("--repeats", type=int, default=None, help="Attempts (default: max SuiteSpec.repeat_runs)")
    frun.add_argument("--profile", default="unit")
    frun.add_argument("--suite", action="append", default=[], help="Specific suite name (repeatable)")
    frun.add_argument("--timeout", type=float, default=None, help="Per-suite timeout override (s)")
    frun.add_argument("--json", action="store_true")
    fana = fsub.add_parser("analyze", help="Classify flake groups already recorded in the store")
    fana.add_argument("--project", default="")
    fana.add_argument("--revision", default="")
    fana.add_argument("--group", default="", help="Restrict to one flake group_id")
    fana.add_argument("--limit", type=int, default=200, help="Max stored runs to scan")
    fana.add_argument("--json", action="store_true")

    sub.add_parser("mcp", help="Run local stdio MCP server")

    comp = sub.add_parser("completion", help="Print bash/zsh completion script (opt-in install)")
    comp.add_argument("shell", choices=["bash", "zsh"])

    workp = sub.add_parser("_worker", help=argparse.SUPPRESS)
    workp.add_argument("--run-id", required=True)
    workp.add_argument("--project-root", required=True)
    workp.add_argument("--profile", default="unit")
    workp.add_argument("--suite", action="append", default=[])
    workp.add_argument("--allow-billable", action="store_true")
    workp.add_argument("--max-tokens", type=int, default=0)
    workp.add_argument("--timeout", type=float, default=None)
    workp.add_argument("--mutation-gate", type=float, default=None)

    return p


def _format_flake_target(target: TargetClassification) -> List[str]:
    head = f"  [{target.level}] {target.target}: {target.verdict}  pass={target.pass_count} fail={target.fail_count}"
    if target.other_count:
        head += f" other={target.other_count}"
    head += f"  statuses={'/'.join(target.statuses)}"
    lines = [head]
    if target.passing_run_ids:
        lines.append(f"    pass_runs={','.join(target.passing_run_ids)}")
    if target.failing_run_ids:
        lines.append(f"    fail_runs={','.join(target.failing_run_ids)}")
        for rid, evidence in zip(target.failing_run_ids, target.failing_evidence_dirs):
            lines.append(f"      fail_evidence[{rid}]={evidence}")
    if target.other_run_ids:
        lines.append(f"    other_runs={','.join(target.other_run_ids)}")
    if target.missing_attempts:
        lines.append(f"    missing_attempts={','.join(str(a) for a in target.missing_attempts)}")
    return lines


def _format_flake_report(report: FlakeReport) -> str:
    lines = [
        f"FLAKE {report.verdict}  group={report.group_id}  project={report.project}  "
        f"revision={report.revision}  attempts={report.attempts}/{report.expected_repeats}"
    ]
    for run_id, status, evidence in zip(report.run_ids, report.statuses, report.evidence_dirs):
        lines.append(f"  run {run_id}: {status}  evidence={evidence}")
    for target in report.targets:
        lines.extend(_format_flake_target(target))
    if report.missing_attempts:
        lines.append(f"  missing_attempts={','.join(str(a) for a in report.missing_attempts)}")
    return "\n".join(lines)


def _format_flake_history(history: FlakeHistory) -> str:
    if not history.groups:
        return "no flake groups recorded"
    counts = history.verdict_counts()
    header = "flake groups: {total} ({detail})".format(
        total=len(history.groups),
        detail=", ".join(f"{name}={counts.get(name, 0)}" for name in FlakeVerdict.ALL),
    )
    blocks = [header]
    for group in history.groups:
        blocks.append("")
        blocks.append(_format_flake_report(group))
    return "\n".join(blocks)


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    extra_cmd: List[str] = []
    if "--" in raw_argv:
        sep = raw_argv.index("--")
        extra_cmd = raw_argv[sep + 1 :]
        raw_argv = raw_argv[:sep]
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if not args.command:
        parser.print_help()
        return 2

    if args.command == "completion":
        # Pure text generation: no store, no home, no side effects.
        from .shell_completion import render

        sys.stdout.write(render(args.shell))
        return 0

    home = Path(args.home).expanduser() if args.home else None

    gate = getattr(args, "mutation_gate", None)
    if gate is not None and (math.isnan(gate) or not 0.0 <= gate <= 1.0):
        sys.stderr.write(
            f"testbench: --mutation-gate must be a fraction between 0 and 1, got {gate:g}\n"
        )
        return 1

    if args.command == "mcp":
        from .mcp_server import serve_stdio
        with TestBenchStore(home) as store:
            eng = TestBenchEngine(store)
            return serve_stdio(eng)

    if args.command == "tui":
        from .tui import run_tui  # type: ignore
        return run_tui(home=home, permanent_tab=not args.quit_on_q, initial_project=args.project or None)

    try:
        with TestBenchStore(home) as store:
            eng = TestBenchEngine(store)

            if args.command == "_worker":
                return eng.run_worker(
                    run_id=args.run_id,
                    project_root=Path(args.project_root),
                    profile=args.profile,
                    suite_names=args.suite or None,
                    allow_billable=args.allow_billable,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout,
                    mutation_gate=args.mutation_gate,
                )

            if args.command == "flake":
                if args.flake_action == "run":
                    report = eng.flake_hunt(
                        Path(args.path),
                        repeats=args.repeats,
                        profile=args.profile,
                        suite_names=args.suite or None,
                        timeout_s=args.timeout,
                    )
                    if args.json:
                        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
                    else:
                        print(_format_flake_report(report))
                    # Only a clean sweep of always_pass may exit 0: flakes,
                    # deterministic failures, and inconclusive attempts never
                    # masquerade as success.
                    return 0 if report.verdict == FlakeVerdict.ALWAYS_PASS else 1
                if args.flake_action == "analyze":
                    runs = store.list_runs(project=args.project or None, limit=max(1, args.limit))
                    history = analyze_flake_history(
                        runs,
                        project=args.project or None,
                        revision=args.revision or None,
                        group_id=args.group or None,
                    )
                    if args.json:
                        print(json.dumps(history.to_dict(), indent=2, sort_keys=True))
                    else:
                        print(_format_flake_history(history))
                    if not history.groups:
                        sys.stderr.write("testbench: no flake groups recorded for the given filters\n")
                        return 1
                    return 0
                parser.print_help()
                return 2

            if args.command == "discover":
                spec = eng.discover(Path(args.path))
                if args.json:
                    print(json.dumps(asdict(spec), indent=2, sort_keys=True))
                else:
                    print(f"project:   {spec.name} ({spec.root})")
                    print(f"revision:  {spec.revision or '-'}  branch: {spec.branch or '-'}  trusted: {spec.trusted}")
                    print(f"languages: {', '.join(spec.languages) or 'auto'}")
                    for s in spec.suites:
                        flags = []
                        if s.needs_tmux:
                            flags.append("tmux")
                        if s.needs_pty:
                            flags.append("pty")
                        if s.billable:
                            flags.append("billable-opt-in")
                        fstr = f" ({','.join(flags)})" if flags else ""
                        print(f"  - [{s.profile}] {s.name} ({s.adapter}){fstr}: {' '.join(s.command)}")
                return 0

            if args.command == "run":
                if args.stream and (args.async_run or args.json):
                    parser.error("--stream requires a foreground text run (no --async or --json)")
                if args.async_run:
                    rec = eng.spawn_async(
                        Path(args.path),
                        profile=args.profile,
                        suite_names=args.suite or None,
                        allow_billable=args.allow_billable,
                        max_tokens=args.max_tokens,
                        timeout_s=args.timeout,
                        wait_up_to_s=args.wait,
                        mutation_gate=args.mutation_gate,
                    )
                else:
                    rec = eng.run_sync(
                        Path(args.path),
                        profile=args.profile,
                        suite_names=args.suite or None,
                        allow_billable=args.allow_billable,
                        max_tokens=args.max_tokens,
                        timeout_s=args.timeout,
                        mutation_gate=args.mutation_gate,
                        on_output=(lambda text: print(text, end="", flush=True)) if args.stream else None,
                    )
                if args.json:
                    sys.stdout.write(render_run_json(rec))
                else:
                    if args.stream:
                        print()
                    print(format_terminal_summary(rec))
                if args.async_run and rec.status in RunStatus.ACTIVE:
                    return 0
                if args.suite and rec.status == RunStatus.SKIP and not rec.suites:
                    print(f"testbench: no suites matched: {', '.join(args.suite)}", file=sys.stderr)
                    return 1
                return 0 if rec.status in (RunStatus.PASS, RunStatus.SKIP) else 1

            if args.command == "status":
                if args.run_id:
                    rec = store.get_run(args.run_id)
                    if rec is None:
                        sys.stderr.write(f"testbench: unknown run_id {args.run_id}\n")
                        return 1
                    if args.json:
                        sys.stdout.write(render_run_json(rec))
                    else:
                        print(format_terminal_summary(rec))
                    return 0
                runs = store.list_runs(project=args.project or None, limit=args.limit)
                if args.json:
                    print(
                        json.dumps(
                            [{"run_id": r.run_id, "project": r.project, "profile": r.profile, "status": r.status, "duration_ms": r.duration_ms} for r in runs],
                            indent=2,
                        )
                    )
                elif not runs:
                    print("no TestBench runs recorded")
                else:
                    for r in runs:
                        print(format_terminal_summary(r))
                return 0

            if args.command == "results":
                rec = None
                if args.run_id:
                    rec = store.get_run(args.run_id)
                else:
                    recent = store.list_runs(project=args.project or None, limit=1)
                    rec = recent[0] if recent else None
                if rec is None:
                    sys.stderr.write("testbench: no matching run found\n")
                    return 1
                fmt = "json" if args.json else args.format
                if fmt == "json":
                    sys.stdout.write(render_run_json(rec))
                elif fmt == "junit":
                    sys.stdout.write(render_junit_xml(rec))
                else:
                    print(format_terminal_summary(rec))
                    if args.logs:
                        ev = Path(rec.evidence_dir) if rec.evidence_dir else None
                        if ev and ev.is_dir():
                            for lf in sorted(ev.glob("*.sanitized.log")):
                                print(f"--- {lf.name} ---")
                                print(lf.read_text(encoding="utf-8", errors="replace")[-4000:])
                return 0

            if args.command == "reproduce":
                out = eng.reproduce(args.run_id, print_only=args.print_only)
                if args.json:
                    print(json.dumps(out, indent=2, sort_keys=True))
                elif args.print_only:
                    print(out["repro_command"])
                else:
                    print(f"{out['status']} {out['repro_run_id']} (from {out['source_run_id']})  evidence={out['evidence_dir']}")
                return 0 if args.print_only or out.get("status") in (RunStatus.PASS, RunStatus.SKIP) else 1

            if args.command == "cancel":
                rec = eng.cancel(args.run_id)
                if args.json:
                    sys.stdout.write(render_run_json(rec))
                else:
                    print(format_terminal_summary(rec))
                return 0

            if args.command == "tape":
                from .tape import approve_golden, run_tape, tape_marker_from_run  # type: ignore
                if args.tape_action == "run":
                    rec = run_tape(
                        store,
                        Path(args.tape_file),
                        project_root=Path(args.project_root) if args.project_root else None,
                        golden_dir=Path(args.golden_dir) if args.golden_dir else None,
                        timeout_s=args.timeout,
                    )
                    if args.json:
                        sys.stdout.write(render_run_json(rec))
                    else:
                        print(format_terminal_summary(rec))
                        for sres in rec.suites:
                            print(f"  golden: {(sres.isolation_meta or {}).get('golden') or '-'}")
                            print(f"  screen: {sres.sanitized_path or '-'}")
                    # Machine marker last: consumed by the terminal_tape adapter.
                    print(tape_marker_from_run(rec))
                    return 0 if rec.status in (RunStatus.PASS, RunStatus.SKIP) else 1
                if args.tape_action == "approve":
                    info = approve_golden(
                        args.evidence_or_run_id,
                        golden_name=args.golden,
                        golden_dir=Path(args.golden_dir),
                        store=store,
                        assume_yes=args.yes,
                    )
                    if args.json:
                        print(json.dumps(info, indent=2, sort_keys=True))
                    else:
                        print(
                            f"approved golden {info['name']} -> {info['screen_path']} "
                            f"(sha256 {info['screen_sha256'][:12]})"
                        )
                    return 0
                sys.stderr.write("testbench tape: expected 'run' or 'approve'\n")
                return 2

            if args.command == "trust":
                info = store.trust_project(Path(args.path))
                if args.json:
                    print(json.dumps(info, indent=2, sort_keys=True))
                else:
                    print(f"trusted {info['name']} ({info['project_root']}) sha256={info['manifest_sha256']}")
                return 0

            if args.command == "projects":
                if args.proj_action == "register":
                    info = store.register_project(Path(args.path), name=args.name or None)
                    if args.json:
                        print(json.dumps(info, indent=2, sort_keys=True))
                    else:
                        print(f"registered {info['name']} ({info['project_root']})")
                    return 0
                rows = store.list_projects()
                if args.json:
                    print(json.dumps(rows, indent=2, sort_keys=True))
                elif not rows:
                    print("no registered projects")
                else:
                    for r in rows:
                        t_str = "trusted" if r["trusted"] else "sandbox"
                        print(f"{r['name']:<26} {t_str:<8} {r['project_root']}")
                return 0

            if args.command == "prune":
                removed = store.prune_runs(
                    keep_last=args.keep_last,
                    older_than_days=args.older_than_days,
                    dry_run=args.dry_run,
                )
                if args.json:
                    print(json.dumps({"removed": removed, "dry_run": args.dry_run}, indent=2))
                else:
                    verb = "would prune" if args.dry_run else "pruned"
                    print(f"testbench: {verb} {len(removed)} run(s)")
                return 0

            if args.command == "bench":
                from .benchmark import compare_benchmarks, format_benchmark_comparison, run_workload  # type: ignore
                from .discovery import detect_git_metadata
                if args.bench_action == "run":
                    cmd_argv = list(extra_cmd)
                    target = Path(args.path).expanduser().resolve(strict=True)
                    if not cmd_argv:
                        rec = eng.run_sync(target, profile="benchmark")
                        if args.json:
                            sys.stdout.write(render_run_json(rec))
                        else:
                            print(format_terminal_summary(rec))
                        return 0 if rec.status in (RunStatus.PASS, RunStatus.SKIP) else 1
                    rev, _br = detect_git_metadata(target)
                    rec, payload = run_workload(
                        args.name,
                        cmd_argv,
                        cwd=target,
                        store=store,
                        project=target.name,
                        revision=rev,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        timeout_s=args.timeout,
                    )
                    if args.json:
                        print(json.dumps(payload, indent=2, sort_keys=True))
                    else:
                        st = payload.get("stats", {})
                        print(
                            f"BENCH {rec.run_id} {target.name}:{args.name}  "
                            f"p50={st.get('p50_ms', 0):.2f}ms p95={st.get('p95_ms', 0):.2f}ms "
                            f"iqr={st.get('iqr_ms', 0):.2f}ms (n={st.get('runs', 0)})"
                        )
                    return 0 if rec.status == RunStatus.PASS else 1
                if args.bench_action == "compare":
                    rows = store.list_benchmarks(limit=200)
                    by_run = {r["run_id"]: r["payload"] for r in rows}
                    if args.baseline_run_id not in by_run or args.candidate_run_id not in by_run:
                        sys.stderr.write("testbench bench compare: run_id not found in benchmarks table\n")
                        return 1
                    comp = compare_benchmarks(by_run[args.baseline_run_id], by_run[args.candidate_run_id])
                    if args.json:
                        print(json.dumps(comp, indent=2, sort_keys=True))
                    else:
                        print(format_benchmark_comparison(comp))
                    return 0
                rows = store.list_benchmarks(project=args.project or None, limit=30)
                if args.json:
                    print(json.dumps(rows, indent=2, sort_keys=True))
                elif not rows:
                    print("no benchmarks recorded")
                else:
                    for r in rows:
                        print(f"{r['run_id']}  {r['project']}:{r['workload']}  median={r['median_ms']}ms  rev={r['revision']}")
                return 0

            if args.command == "messaging-verify":
                from .discovery import detect_git_metadata
                from .engine import host_environment_snapshot, messaging_paths
                from .messaging_suite import run_agent_messaging_verification  # type: ignore
                from .schema import FailureKind, RunRecord, new_run_id, utc_now_iso
                import time as _time
                rid = new_run_id()
                now = utc_now_iso()
                art_root, ha_root = messaging_paths()
                has_art_root = art_root.is_dir()
                target_root = art_root if has_art_root else (ha_root if ha_root.is_dir() else Path.cwd())
                rev, branch = detect_git_metadata(target_root)
                ha_rev, _ha_br = detect_git_metadata(ha_root) if ha_root.is_dir() else ("unversioned", "")
                env_snap = host_environment_snapshot()
                env_snap["hermes_agent_revision"] = ha_rev
                # Recorded so `reproduce` replays with identical artifact_root semantics.
                env_snap["messaging_artifact_root"] = str(art_root) if has_art_root else ""
                repro_argv = ["testbench", "messaging-verify"] + (["--hermes-pytest"] if args.hermes_pytest else [])
                run = RunRecord(
                    run_id=rid,
                    project="agent-messaging",
                    project_root=str(target_root),
                    revision=rev,
                    branch=branch,
                    profile="verification",
                    status=RunStatus.RUNNING,
                    commands=[repro_argv],
                    created_at=now,
                    started_at=now,
                    environment=env_snap,
                )
                store.create_run(run)
                t0 = _time.monotonic()
                try:
                    with store.hold_run_lock(rid):
                        suites = run_agent_messaging_verification(
                            store.run_dir(rid),
                            artifact_root=art_root if has_art_root else None,
                            hermes_agent_root=ha_root,
                            run_hermes_pytest=args.hermes_pytest,
                        )
                        done = eng._populate_and_finalize(run, suites, [], t0)
                except Exception as exc:
                    run.status = RunStatus.ERROR
                    run.failure_kind = FailureKind.HARNESS_ERROR
                    run.error_count = 1
                    run.first_failure = f"Messaging verification error: {exc}"[:300]
                    done = store.finalize_run(run)
                if args.json:
                    sys.stdout.write(render_run_json(done))
                else:
                    print(format_terminal_summary(done))
                return 0 if done.status in (RunStatus.PASS, RunStatus.SKIP) else 1

    except (AuthorizationError, KeyError, ValueError) as exc:
        sys.stderr.write(f"testbench error: {exc}\n")
        return 1
    return 2
