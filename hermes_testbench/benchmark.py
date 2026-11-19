"""Reproducible benchmarking: environment provenance, statistics, workloads, comparison.

Benchmarks carry no *performance* policy. ``run_workload`` records a ``RunRecord`` with
``profile='benchmark'`` whose status reports only whether every iteration exited 0 —
warmup included — never whether a number looked fast enough. ``compare_benchmarks``
emits deltas and environment drift, never a threshold verdict.

Why no percentage gate: a pass/fail cut on wall-clock noise converts machine state
into a verdict. A 5% ceiling fails on a laptop running a video call and passes on an
idle server for the same commit, so the signal says more about the host than the code.
The alternative kept here is to publish the distribution (p50/p95/IQR/MAD), publish the
provenance (CPU count, load average, compiler versions, git revision), and let the
reader — usually a human comparing two runs of the same workload — decide whether a
delta is interesting. ``comparable`` is the only gate, and it is about environment
drift, not speed.

Timing is measured by ``isolation.run_isolated_command`` around the whole child
process, so it includes fork/exec and interpreter startup and the number answers
"how long did this command take" rather than "how long did its kernel run". Subtract a
baseline workload if you need the harness overhead; do not subtract a constant here.
"""

from __future__ import annotations

import math
import os
import platform
import shlex
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
    """Remove home-directory and host-name traces from a version string.

    Applied to values that end up in the benchmark payload, which is persisted and
    shared. Ordering is home first, then the literal ``Users/`` prefix (so a path that
    lost its leading slash is still stripped), then the full nodename and its short
    form. The short-form replacement is a plain substring swap and can therefore rewrite
    unrelated text if the hostname is a common word — a cosmetic risk accepted over
    leaking a username, and the reason this helper is never used on opaque data.
    """
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
    # A git failure (not a repository, no HEAD, detached oddities) is "unknown
    # provenance", not an error: the benchmark still ran. Empty string is the
    # representation, so consumers must not read "" as "clean tree".
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
    """Capture provenance for a benchmark run without leaking local identity.

    One flat dict, all values JSON-safe, all strings scrubbed. It answers "was this the
    same machine and the same toolchain" — not "was this a fair comparison", which
    ``compare_benchmarks`` decides later by diffing this dict.

    Cost: up to six subprocesses (``git`` twice, one ``--version`` per tool), each
    capped at 10 s and failing to an empty string. Missing tools produce an empty
    version rather than an error, so the payload shape is identical everywhere; a
    reader must treat ``""`` as "unknown", not as "not installed at this version".

    ``loadavg`` is the 1/5/15-minute triplet at collection time — a coarse hint, kept
    because it is the cheapest signal that a sample set was taken under load. It is
    not compared as drift (it would always differ).
    """
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
    """Linear-interpolation quantile (Type 7), matching common benchmark tooling.

    Type 7 (NumPy/Excel ``PERCENTILE.INC`` default) interpolates between order
    statistics at ``q * (n - 1)``. With few samples the result sits between two real
    observations, so p95 of a five-sample run is an interpolation, not a measured
    point — quote ``runs`` alongside any percentile.

    Takes an already-sorted list and trusts the caller: sorting inside would hide the
    O(n log n) behind a helper that is called three times per sample set.
    """
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
    """Describe a sample set. Dispersion fields are 0.0 for a single sample.

    Fields: ``runs`` (n), min/max/mean, ``p50_ms`` (median), ``p95_ms``, ``iqr_ms``
    (q75-q25), ``mad_ms`` (median absolute deviation from p50) and ``stdev_ms``
    (sample standard deviation, n-1).

    Why both IQR and MAD: IQR describes the middle half, so one extreme sample moves
    it less than it moves a stdev; MAD describes typical distance from the median and
    survives a few. Neither is derived from the other, and reporting both makes a
    bimodal sample set (e.g. a GC pause in half the runs) visible in a way a single
    stdev is not. ``stdev_ms`` is the non-robust number kept for familiarity, not the
    one to trust on a noisy host.

    Raises on an empty set instead of returning zeros: zero dispersion for zero
    samples is a lie that a downstream threshold would happily consume.
    """
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
    """Execute ``warmup`` + ``repeats`` iterations and persist timing evidence.

    Semantics:

    * ``warmup`` iterations run first and are timed, but their durations go only into
      ``payload['warmup_ms']``. They exist to pay one-time costs (page cache, JIT,
      dynamic linking) so the sample set describes steady state.
    * ``repeats`` iterations produce ``samples_ms`` and ``exit_codes``; ``stats`` is
      computed over those samples only.
    * ``status`` is PASS only if *every* iteration — warmup included — exited 0 and
      none timed out. A non-zero exit means the workload itself is broken; that is a
      liveness verdict, not a performance one.
    * ``pass_count``/``fail_count`` are derived from the sample exit codes alone, so a
      warmup-only failure produces FAIL with ``pass_count == total_count``. Read
      ``status`` and ``first_failure``, not the counts, when a warmup can fail.
    * A timed-out sample is recorded with ``exit_code == -1`` and its (capped)
      duration, and still enters ``stats``. The payload therefore describes "what the
      command did within the timeout", including when what it did was time out.

    The ``RunRecord`` is created before the first iteration so an interrupted run is
    reconcilable from the store; on an exception mid-loop it is finalised as ERROR
    with ``failure_kind='harness_error'`` and the exception is re-raised. Durations are
    rounded to whole microseconds (3 decimal places of ms) to keep the payload stable
    across machines without pretending to more precision than the clock has.

    ``repro_instructions`` is an exact CLI line; ``--timeout`` is omitted when it
    equals the CLI default so the printed command matches what a user would type.
    """
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
    cmd_str = " ".join(shlex.quote(a) for a in command)
    timeout_flag = f" --timeout {float(timeout_s):g}" if float(timeout_s) != 120.0 else ""
    run.repro_instructions = (
        f"testbench bench run {shlex.quote(str(work_dir))} "
        f"--name {shlex.quote(name)} --warmup {int(warmup)} --repeats {int(repeats)}{timeout_flag} -- {cmd_str}"
    )
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


# Only three statistics are compared: p50 (typical), p95 (tail), IQR (predictability)
# — the three questions a reader usually has, answerable separately. min/max/mean/stdev
# stay in the payload for anyone who wants them; comparing all seven mostly produces
# deltas nobody acts on.
_COMPARED_STATS = ("p50_ms", "p95_ms", "iqr_ms")
# Keys that invalidate a comparison. ``revision`` changes what is being measured;
# the rest change how fast the machine measures it. Toolchain versions
# (rustc/cargo/cobc/clang) and sqlite are collected but not compared — a compiler
# upgrade can move a workload just as far as a CPU change, and today that drift is
# reported as comparable. Load average is excluded on purpose: it always differs.
_DRIFT_KEYS = ("revision", "cpu_count", "os_release", "arch", "python_version")


def _ratio(candidate: float, baseline: float) -> float:
    # 0/0 is "no change" (1.0) rather than NaN; x/0 is infinite rather than a crash,
    # because a zero baseline is a real payload state and the renderer formats inf.
    if baseline == 0:
        return 1.0 if candidate == 0 else float("inf")
    return candidate / baseline


def compare_benchmarks(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Pairwise deltas and environment drift notes. No verdicts, no policy.

    Output is per-stat ``{baseline, candidate, delta_ms, delta_pct, ratio}`` plus
    ``drift_notes`` and ``comparable`` (true only when no drift key differs). The
    caller decides what a delta means; nothing here knows a threshold.

    Missing input is coerced, not rejected: an absent stat becomes 0.0 and a zero
    baseline makes ``delta_pct`` 0.0 while ``ratio`` becomes ``inf``. Those two fields
    therefore disagree for the same comparison, and a payload without ``stats``
    produces a full table of zeros that looks like a real result. Compare stored
    payloads, not hand-built dicts.

    ``workload``/``project`` fall back to the baseline when the candidate omits them,
    so a partially populated row still labels itself.
    """
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
    """Render a comparison as plain text: deltas and drift, never a verdict.

    Fixed-width columns so two runs can be diffed in a terminal, and the drift section
    is printed even when empty (``none``) — an absent drift section would read as "no
    drift checked" rather than "no drift found". Ratio is rendered with three decimals
    so a 1.05 and a 1.5 cannot be confused at a glance.
    """
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
