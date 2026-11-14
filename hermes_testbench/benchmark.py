"""Reproducible benchmarking: environment provenance, statistics, workloads, comparison.

Benchmarks never carry pass/fail policy: ``run_workload`` records a ``RunRecord`` with
``profile='benchmark'`` whose status only reports whether the command itself ran, and
``compare_benchmarks`` emits deltas, never a threshold verdict.
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import sqlite3
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .isolation import run_isolated_command
from .schema import ProfileName, RunRecord, RunStatus, new_run_id, utc_now_iso
from .store import TestBenchStore

_VERSION_TOOLS = ("rustc", "cargo", "cobc", "clang")


def _scrub_identity(value: str) -> str:
    """Remove home-directory and host-name traces from a version string."""
    text = str(value)
    home = str(Path.home())
    if home:
        text = text.replace(home, "~")
    text = text.replace("Users/", "")
    try:
        nodename = os.uname().nodename
    except (AttributeError, OSError):  # pragma: no cover - non-POSIX fallback
        nodename = ""
    if nodename:
        text = text.replace(nodename, "<host>")
        short = nodename.split(".")[0]
        if short and short != nodename:
            text = text.replace(short, "<host>")
    return text


def _tool_version(tool: str) -> str:
    binary = shutil.which(tool)
    if binary is None:
        return ""
    try:
        proc = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = (proc.stdout or proc.stderr or "").strip().splitlines()
    return _scrub_identity(lines[0].strip()) if lines else ""


def _git(cwd: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def collect_environment_metadata(cwd: Path) -> Dict[str, Any]:
    """Capture provenance for a benchmark run without leaking local identity."""
    resolved = Path(cwd).expanduser()
    try:
        resolved = resolved.resolve(strict=True)
    except OSError:
        resolved = resolved.resolve()

    try:
        uname = os.uname()
        os_name = uname.sysname
        os_release = uname.release
        machine = uname.machine
    except (AttributeError, OSError):  # pragma: no cover - non-POSIX fallback
        os_name = platform.system()
        os_release = platform.release()
        machine = platform.machine()

    try:
        loadavg = [float(v) for v in os.getloadavg()]
    except (OSError, AttributeError):  # pragma: no cover - no loadavg support
        loadavg = [0.0, 0.0, 0.0]
    if len(loadavg) != 3:  # pragma: no cover - defensive
        loadavg = (loadavg + [0.0, 0.0, 0.0])[:3]

    revision = _git(resolved, "rev-parse", "HEAD")
    branch = _git(resolved, "rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        branch = ""

    env: Dict[str, Any] = {
        "os_name": _scrub_identity(os_name),
        "os_release": _scrub_identity(os_release),
        "arch": _scrub_identity(machine),
        "cpu_count": int(os.cpu_count() or 1),
        "loadavg": [round(v, 4) for v in loadavg],
        "python_version": platform.python_version(),
        "sqlite_version": sqlite3.sqlite_version,
        "git_revision": revision,
        "git_branch": _scrub_identity(branch),
        "python_implementation": platform.python_implementation(),
    }
    for tool in _VERSION_TOOLS:
        env[f"{tool}_version"] = _tool_version(tool)
    return env


def _quantile(sorted_values: List[float], q: float) -> float:
    """Linear-interpolation quantile (Type 7), matching common benchmark tooling."""
    if not sorted_values:
        raise ValueError("quantile of empty sample set")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def compute_stats(samples_ms: List[float]) -> Dict[str, Any]:
    """Describe a sample set. Dispersion fields are 0.0 for a single sample."""
    values = [float(v) for v in samples_ms]
    if not values:
        raise ValueError("compute_stats requires at least one sample")
    ordered = sorted(values)
    n = len(values)
    p50 = _quantile(ordered, 0.50)
    q25 = _quantile(ordered, 0.25)
    q75 = _quantile(ordered, 0.75)
    mad = statistics.median([abs(v - p50) for v in ordered])
    stdev = float(statistics.stdev(ordered)) if n >= 2 else 0.0
    return {
        "runs": n,
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
        "mean_ms": float(sum(ordered) / n),
        "p50_ms": float(p50),
        "p95_ms": float(_quantile(ordered, 0.95)),
        "iqr_ms": float(q75 - q25) if n >= 2 else 0.0,
        "mad_ms": float(mad) if n >= 2 else 0.0,
        "stdev_ms": stdev,
    }


def run_workload(
    name: str,
    command: List[str],
    *,
    cwd: Path,
    store: TestBenchStore,
    project: str,
    revision: str = "",
    warmup: int = 1,
    repeats: int = 5,
    timeout_s: float = 120.0,
    scratch_root: Optional[Path] = None,
) -> Tuple[RunRecord, Dict[str, Any]]:
    """Execute ``warmup`` + ``repeats`` iterations and persist timing evidence."""
    if not name or not isinstance(name, str):
        raise ValueError("workload name is required")
    if int(warmup) < 0:
        raise ValueError("warmup must be >= 0")
    if int(repeats) < 1:
        raise ValueError("repeats must be >= 1")

    run_id = new_run_id()
    work_dir = Path(cwd).expanduser().resolve(strict=True)
    evidence_dir = store.run_dir(run_id)

    run = RunRecord(
        run_id=run_id,
        project=project,
        project_root=str(work_dir),
        revision=revision,
        profile=ProfileName.BENCHMARK,
        status=RunStatus.RUNNING,
        branch=_git(work_dir, "rev-parse", "--abbrev-ref", "HEAD"),
        commands=[list(command)],
        started_at=utc_now_iso(),
    )
    store.create_run(run)

    warmup_ms: List[float] = []
    samples_ms: List[float] = []
    exit_codes: List[int] = []
    all_ok = True

    try:
        for index in range(int(warmup)):
            result = run_isolated_command(
                list(command),
                cwd=work_dir,
                evidence_dir=evidence_dir,
                step_name=f"{name}-warmup-{index + 1}",
                run_id=run_id,
                timeout_s=timeout_s,
                scratch_root=scratch_root,
            )
            warmup_ms.append(round(float(result.duration_ms), 3))
            if result.timed_out or result.exit_code != 0:
                all_ok = False

        for index in range(int(repeats)):
            result = run_isolated_command(
                list(command),
                cwd=work_dir,
                evidence_dir=evidence_dir,
                step_name=f"{name}-run-{index + 1}",
                run_id=run_id,
                timeout_s=timeout_s,
                scratch_root=scratch_root,
            )
            samples_ms.append(round(float(result.duration_ms), 3))
            exit_codes.append(int(result.exit_code if result.exit_code is not None else -1))
            if result.timed_out or result.exit_code != 0:
                all_ok = False
    except Exception:
        run.status = RunStatus.ERROR
        run.finished_at = utc_now_iso()
        run.failure_kind = "harness_error"
        run.duration_ms = 0.0
        store.finalize_run(run)
        raise

    stats = compute_stats(samples_ms)
    payload: Dict[str, Any] = {
        "workload": name,
        "project": project,
        "revision": revision,
        "warmup_ms": warmup_ms,
        "samples_ms": samples_ms,
        "exit_codes": exit_codes,
        "stats": stats,
        "environment": collect_environment_metadata(work_dir),
    }

    run.status = RunStatus.PASS if all_ok else RunStatus.FAIL
    run.finished_at = utc_now_iso()
    run.duration_ms = round(sum(warmup_ms) + sum(samples_ms), 3)
    run.exit_code = exit_codes[-1] if exit_codes else None
    run.environment = payload["environment"]
    run.suites = []
    run.total_count = len(samples_ms)
    run.pass_count = sum(1 for code in exit_codes if code == 0)
    run.fail_count = len(exit_codes) - run.pass_count
    run.summary = (
        f"benchmark {name}: {stats['runs']} samples, "
        f"p50={stats['p50_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms"
    )
    if not all_ok:
        run.first_failure = f"workload exit codes: {exit_codes}"
    run.repro_instructions = run.summary
    store.finalize_run(run)

    store.record_benchmark(
        bench_id=f"bench_{run_id}",
        run_id=run_id,
        project=project,
        workload=name,
        revision=revision,
        median_ms=stats["p50_ms"],
        payload=payload,
    )
    return run, payload


_COMPARED_STATS = ("p50_ms", "p95_ms", "iqr_ms")
_DRIFT_KEYS = ("revision", "cpu_count", "os_release", "arch", "python_version")


def _ratio(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 1.0 if candidate == 0 else float("inf")
    return candidate / baseline


def compare_benchmarks(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Pairwise deltas and environment drift notes. No verdicts, no policy."""
    base_stats = baseline.get("stats") or {}
    cand_stats = candidate.get("stats") or {}
    deltas: Dict[str, Dict[str, Any]] = {}
    for key in _COMPARED_STATS:
        base_val = float(base_stats.get(key, 0.0) or 0.0)
        cand_val = float(cand_stats.get(key, 0.0) or 0.0)
        delta = cand_val - base_val
        delta_pct = 0.0 if base_val == 0 else delta / base_val * 100.0
        deltas[key] = {
            "baseline": base_val,
            "candidate": cand_val,
            "delta_ms": delta,
            "delta_pct": delta_pct,
            "ratio": _ratio(cand_val, base_val),
        }

    base_env = baseline.get("environment") or {}
    cand_env = candidate.get("environment") or {}
    drift_notes: List[str] = []
    for key in _DRIFT_KEYS:
        if key == "revision":
            base_val = baseline.get("revision")
            cand_val = candidate.get("revision")
        else:
            base_val = base_env.get(key)
            cand_val = cand_env.get(key)
        if base_val != cand_val:
            drift_notes.append(f"{key} differs: baseline={base_val!r} candidate={cand_val!r}")

    return {
        "workload": candidate.get("workload") or baseline.get("workload") or "",
        "project": candidate.get("project") or baseline.get("project") or "",
        "baseline_revision": baseline.get("revision", ""),
        "candidate_revision": candidate.get("revision", ""),
        "deltas": deltas,
        "drift_notes": drift_notes,
        "comparable": len(drift_notes) == 0,
    }


def format_benchmark_comparison(comp: Dict[str, Any]) -> str:
    """Render a comparison as plain text: deltas and drift, never a verdict."""
    lines: List[str] = []
    lines.append(f"benchmark comparison: {comp.get('workload', '')}")
    lines.append(
        f"  revision  baseline={comp.get('baseline_revision', '')} "
        f"candidate={comp.get('candidate_revision', '')}"
    )
    lines.append("")
    lines.append(
        f"  {'metric':<10} {'baseline':>12} {'candidate':>12} "
        f"{'delta_ms':>12} {'delta_pct':>10} {'ratio':>8}"
    )
    for key in _COMPARED_STATS:
        entry = (comp.get("deltas") or {}).get(key)
        if not entry:
            continue
        lines.append(
            f"  {key:<10} {entry['baseline']:>12.3f} {entry['candidate']:>12.3f} "
            f"{entry['delta_ms']:>12.3f} {entry['delta_pct']:>9.2f}% {entry['ratio']:>8.3f}"
        )
    notes = comp.get("drift_notes") or []
    lines.append("")
    if notes:
        lines.append("environment drift (not comparable):")
        for note in notes:
            lines.append(f"  - {note}")
    else:
        lines.append("environment drift (not comparable): none")
    return "\n".join(lines)
