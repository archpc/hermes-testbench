"""Mutation-testing adapters: disposable staging, external tools only, explicit gating.

Phase 3 contract
----------------
* The mutation tool is a third-party binary (``cargo-mutants`` or ``mutmut``); TestBench
  never generates mutants itself.
* The tool runs against a disposable copy of the project staged under
  ``scratch_root / f"tbmut-{nonce}"``. The real checkout is never mutated and is verified
  byte-for-byte untouched afterwards (``git status --porcelain`` unchanged, no mutant
  artefacts left behind).
* Seatbelt confinement is mandatory. The staged copy is handed to the sandbox as an
  internal extra writable directory (``run_isolated_command(extra_writable_dirs=...)``),
  which keeps ``allow_repo_write=False`` for the real repository.
* All mutant outcomes land in exactly five buckets exposed through
  ``SuiteResult.metrics["mutation"]``: ``caught``, ``survived``, ``invalid`` (unviable /
  compile failure), ``timeout`` and ``inconclusive`` (baseline failure, no tests, skipped,
  interrupted, not checked, harness error).
* Surviving mutants are findings, not failures: the suite passes unless an operator gate
  (``--mutation-gate``) or a *trusted* manifest ``mutation_score_gate`` says otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from ..isolation import (
    can_apply_seatbelt,
    default_scratch_root,
    extra_bin_dirs,
    run_isolated_command,
)
from ..sanitize import redact_secrets, sanitize_terminal_text
from ..schema import (
    CaseResult,
    FailureKind,
    FailureRecord,
    RunStatus,
    SuiteResult,
    SuiteSpec,
    utc_now_iso,
)
from ..store import _safe_rmtree_no_symlinks
from . import parsers

MUTATION_ADAPTERS = frozenset({"cargo_mutants", "mutmut"})
MUTATION_BUCKETS: Tuple[str, ...] = ("caught", "survived", "invalid", "timeout", "inconclusive")

EVIDENCE_MAX_BYTES = 8 * 1024 * 1024
_ARTIFACT_MAX_BYTES = 8 * 1024 * 1024
_EVIDENCE_CASE_LIMIT = 400

# Anything a mutation tool would drop inside the project tree. Their presence after a run
# proves the tool escaped its staging directory.
SENTINEL_NAMES: Tuple[str, ...] = ("mutants", "mutants.out", "mutants.out.old", ".mutmut-cache")

_TOOL_ENV_VAR = {"cargo_mutants": "TESTBENCH_CARGO_MUTANTS", "mutmut": "TESTBENCH_MUTMUT"}
_TOOL_BINARY = {"cargo_mutants": "cargo-mutants", "mutmut": "mutmut"}

# Directories never copied into the staging tree: they are large, regenerable, or carry
# state whose presence would make the copy differ from "the sources a tool should see".
_STAGE_SKIP_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mutmut-cache",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "mutants",
        "mutants.out",
        "mutants.out.old",
        "node_modules",
        "target",
        "venv",
    }
)

_BASELINE_MESSAGE = "baseline test suite failed; no mutant verdict is meaningful"


class MutationSpawnError(RuntimeError):
    """The mutation tool could not be spawned at all."""


# --------------------------------------------------------------------------- tool + staging


def resolve_tool(adapter: str) -> Optional[List[str]]:
    """Return the argv prefix for a mutation tool, or ``None`` when it is unavailable.

    Resolution order: ``$TESTBENCH_CARGO_MUTANTS`` / ``$TESTBENCH_MUTMUT`` (an explicit
    executable path), then ``PATH`` plus the standard toolchain bin dirs. ``cargo-mutants``
    is a cargo subcommand, so its argv always carries the ``mutants`` subcommand.
    """
    name = _TOOL_BINARY.get(adapter)
    if name is None:
        return None
    override = os.environ.get(_TOOL_ENV_VAR[adapter], "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return [str(candidate)] if adapter == "mutmut" else [str(candidate), "mutants"]
    search = os.pathsep.join([p for p in (os.environ.get("PATH", ""), *extra_bin_dirs()) if p])
    found = shutil.which(name, path=search)
    if not found:
        return None
    return [found] if adapter == "mutmut" else [found, "mutants"]


def _run_argv(adapter: str, base: List[str]) -> List[str]:
    """Full argv for a mutation run (``cargo-mutants`` already carries its subcommand)."""
    if adapter == "mutmut":
        return [*base, "run"]
    return [*base]


def _stage_ignore(_dir: str, names: List[str]) -> List[str]:
    return [name for name in names if name in _STAGE_SKIP_NAMES]


def stage_project_tree(project_root: Path, stage_dir: Path) -> Path:
    """Copy the project sources into a fresh disposable directory and return it."""
    source = Path(project_root).resolve(strict=True)
    target = Path(stage_dir)
    target.mkdir(parents=True, exist_ok=False)
    os.chmod(str(target), 0o700)
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        symlinks=True,
        ignore=_stage_ignore,
    )
    return target


def _git_status(project_root: Path) -> Optional[str]:
    """``git status --porcelain`` for the real checkout, or ``None`` when not a repo."""
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["LC_ALL"] = "C"
    try:
        proc = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15.0,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _untouched_violations(project_root: Path, before_status: Optional[str]) -> List[str]:
    """Reasons the real checkout is no longer pristine after a mutation run."""
    problems: List[str] = []
    for name in SENTINEL_NAMES:
        if (Path(project_root) / name).exists():
            problems.append(f"mutant artefact left in project root: {name}")
    after_status = _git_status(Path(project_root))
    if before_status is not None and after_status is not None and before_status != after_status:
        problems.append("git status --porcelain changed for the real checkout")
    return problems


def _read_bounded(path: Path, limit: int = _ARTIFACT_MAX_BYTES) -> str:
    try:
        with open(str(path), "rb") as handle:
            raw = handle.read(limit)
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- evidence


def _redacted(text: str, limit: int = 400) -> str:
    return sanitize_terminal_text(redact_secrets(text or ""), max_chars=limit)


def export_evidence(
    evidence_dir: Path,
    *,
    adapter: str,
    metrics: Dict[str, Any],
    cases: List[CaseResult],
    summary: str,
) -> Optional[str]:
    """Write a bounded, redacted mutation report under ``evidence_dir/mutation/``."""
    out_dir = Path(evidence_dir) / "mutation"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(out_dir), 0o700)
    except OSError:
        return None

    payload = {
        "adapter": adapter,
        "summary": _redacted(summary, 800),
        "metrics": metrics,
        "mutants": [
            {
                "name": _redacted(case.name, 200),
                "status": case.status,
                "detail": _redacted(case.message, 300),
            }
            for case in cases[:_EVIDENCE_CASE_LIMIT]
        ],
        "mutants_truncated": max(0, len(cases) - _EVIDENCE_CASE_LIMIT),
    }
    blob = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    encoded = blob.encode("utf-8")[:EVIDENCE_MAX_BYTES]
    dest = out_dir / f"{adapter}.json"
    try:
        fd = os.open(
            str(dest),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded.decode("utf-8", errors="ignore"))
    except OSError:
        return None
    return str(dest)


# --------------------------------------------------------------------------- result shaping


def _score(caught: int, survived: int) -> Optional[float]:
    tested = caught + survived
    if tested <= 0:
        return None
    return round(caught / tested, 6)


def _metrics(
    outcome: parsers.ParseOutcome,
    *,
    adapter: str,
    gate: Optional[float],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    caught = int(outcome.buckets.get("caught", 0))
    survived = int(outcome.buckets.get("survived", 0))
    payload: Dict[str, Any] = {
        "caught": caught,
        "survived": survived,
        "invalid": int(outcome.buckets.get("invalid", 0)),
        "timeout": int(outcome.buckets.get("timeout", 0)),
        "inconclusive": int(outcome.buckets.get("inconclusive", 0)),
        "tested": caught + survived,
        "score": _score(caught, survived),
        "gate": gate,
        "tool": _TOOL_BINARY.get(adapter, adapter),
        "baseline_failed": bool(outcome.baseline_failed),
    }
    if extra:
        payload.update(extra)
    return payload


def _cases(suite_name: str, outcome: parsers.ParseOutcome) -> List[CaseResult]:
    out: List[CaseResult] = []
    for item in outcome.cases:
        data = dict(item)
        data["suite"] = suite_name
        data.setdefault("status", RunStatus.SKIP)
        data.setdefault("message", "")
        out.append(CaseResult(**data))
    return out


def _bucket_counts(result: SuiteResult, metrics: Dict[str, Any]) -> None:
    result.total_count = int(metrics["tested"]) + int(metrics["invalid"]) + int(
        metrics["timeout"]
    ) + int(metrics["inconclusive"])
    result.pass_count = int(metrics["caught"])
    result.fail_count = int(metrics["survived"])
    result.skip_count = (
        int(metrics["invalid"]) + int(metrics["timeout"]) + int(metrics["inconclusive"])
    )


def _blocked(
    suite: SuiteSpec,
    *,
    cwd: Path,
    repro: str,
    started_at: str,
    kind: str,
    message: str,
) -> Tuple[SuiteResult, Optional[Dict[str, Any]]]:
    result = SuiteResult(
        name=suite.name,
        adapter=suite.adapter,
        profile=suite.profile,
        command=list(suite.command),
        cwd=str(cwd),
        status=RunStatus.BLOCKED,
        started_at=started_at,
        finished_at=utc_now_iso(),
        total_count=1,
        blocked_count=1,
        summary_excerpt=message,
        repro_command=repro,
        failures=[FailureRecord(suite=suite.name, case_name="<policy>", kind=kind, message=message)],
    )
    return result, None


def _skipped(
    suite: SuiteSpec, *, cwd: Path, repro: str, started_at: str, message: str
) -> Tuple[SuiteResult, Optional[Dict[str, Any]]]:
    result = SuiteResult(
        name=suite.name,
        adapter=suite.adapter,
        profile=suite.profile,
        command=list(suite.command),
        cwd=str(cwd),
        status=RunStatus.SKIP,
        started_at=started_at,
        finished_at=utc_now_iso(),
        total_count=1,
        skip_count=1,
        summary_excerpt=message,
        repro_command=repro,
    )
    return result, None


def _baseline_failure_record(suite_name: str, excerpt: str) -> FailureRecord:
    return FailureRecord(
        suite=suite_name,
        case_name="<baseline>",
        kind=FailureKind.ASSERTION,
        message=_BASELINE_MESSAGE,
        excerpt=_redacted(excerpt, 800),
    )


# --------------------------------------------------------------------------- execution


def run_mutation_suite(
    suite: SuiteSpec,
    *,
    project_root: Path,
    evidence_dir: Path,
    run_id: str,
    trusted_manifest: bool = False,
    cancel_flag_path: Optional[Path] = None,
    scratch_root: Optional[Path] = None,
    on_spawn: Optional[Any] = None,
) -> Tuple[SuiteResult, Optional[Dict[str, Any]]]:
    """Stage a disposable copy, run one mutation tool under Seatbelt, classify outcomes."""
    adapter = suite.adapter
    project_root = Path(project_root).resolve()
    evidence_dir = Path(evidence_dir)
    relative_cwd = Path(suite.cwd or ".")
    cwd = (project_root / relative_cwd).resolve()
    repro = (
        f"cd {cwd} && testbench run {project_root} --profile mutation "
        f"--suite {suite.name}"
    )
    started_at = utc_now_iso()
    gate = suite.mutation_score_gate

    # Gate A: a manifest-declared gate is only meaningful on a trusted manifest.
    if gate is not None and suite.custom_from_manifest and not suite.mutation_gate_from_cli:
        if not trusted_manifest:
            return _blocked(
                suite,
                cwd=cwd,
                repro=repro,
                started_at=started_at,
                kind=FailureKind.UNTRUSTED_MANIFEST,
                message=(
                    "blocked: mutation_score_gate from an untrusted manifest is ignored; "
                    "run 'testbench trust' or pass --mutation-gate explicitly"
                ),
            )

    # Gate B: never spawn a mutation tool without filesystem/network confinement.
    if not can_apply_seatbelt() and os.environ.get("TESTBENCH_SEATBELT_ACTIVE") != "1":
        return _blocked(
            suite,
            cwd=cwd,
            repro=repro,
            started_at=started_at,
            kind=FailureKind.SANDBOX_DENIED,
            message=(
                "mutation testing blocked: filesystem/network confinement unavailable on this "
                "platform"
            ),
        )

    # Gate C: a missing tool is a skip, never a pass.
    tool_argv = resolve_tool(adapter)
    binary = _TOOL_BINARY.get(adapter, adapter)
    if tool_argv is None:
        return _skipped(
            suite,
            cwd=cwd,
            repro=repro,
            started_at=started_at,
            message=f"skip: required binary not found on PATH: {binary}",
        )

    sroot = Path(scratch_root).expanduser().resolve() if scratch_root else default_scratch_root()
    sroot.mkdir(parents=True, exist_ok=True)
    stage_dir = sroot / f"tbmut-{os.urandom(6).hex()}"
    before_status = _git_status(project_root)

    exec_meta: Dict[str, Any] = {}
    outcome = parsers.ParseOutcome(buckets={name: 0 for name in MUTATION_BUCKETS})
    timed_out = False
    cancelled = False
    exit_code: Optional[int] = None
    duration_ms = 0.0
    stdout_path = stderr_path = sanitized_path = None
    run_text = ""
    results_text = ""
    try:
        try:
            stage_project_tree(project_root, stage_dir)
        except OSError as exc:
            raise RuntimeError(
                f"could not stage a disposable copy for mutation testing: {exc}"
            ) from exc

        workdir = stage_dir / relative_cwd
        if not workdir.is_dir():
            raise RuntimeError(f"staged suite cwd is missing: {relative_cwd}")

        try:
            exec_result = run_isolated_command(
                _run_argv(adapter, tool_argv),
                cwd=workdir,
                evidence_dir=evidence_dir,
                step_name=suite.name,
                run_id=run_id,
                timeout_s=suite.timeout_s,
                allow_network=suite.allow_network,
                allow_repo_write=False,  # the real checkout stays read-only
                extra_writable_dirs=(stage_dir,),  # only the disposable copy is writable
                env_overrides=suite.env_overrides,
                cancel_flag_path=cancel_flag_path,
                scratch_root=sroot,
                on_spawn=on_spawn,
            )
        except (OSError, ValueError) as exc:
            raise MutationSpawnError(str(exc)) from exc
        exit_code = exec_result.exit_code
        duration_ms = exec_result.duration_ms
        timed_out = exec_result.timed_out
        cancelled = exec_result.cancelled
        stdout_path = exec_result.stdout_path
        stderr_path = exec_result.stderr_path
        sanitized_path = exec_result.sanitized_path
        exec_meta = dict(exec_result.isolation_meta)
        run_text = (exec_result.stdout_text or "") + "\n" + (exec_result.stderr_text or "")

        if adapter == "mutmut" and not (timed_out or cancelled):
            # ``mutmut run`` only prints a tally; per-mutant verdicts come from its cache.
            results_result = run_isolated_command(
                [tool_argv[0], "results", "--all", "true"],
                cwd=workdir,
                evidence_dir=evidence_dir,
                step_name=f"{suite.name}-results",
                run_id=run_id,
                timeout_s=min(120.0, float(suite.timeout_s)),
                allow_network=suite.allow_network,
                allow_repo_write=False,
                extra_writable_dirs=(stage_dir,),
                env_overrides=suite.env_overrides,
                cancel_flag_path=cancel_flag_path,
                scratch_root=sroot,
            )
            results_text = (results_result.stdout_text or "") + "\n" + (
                results_result.stderr_text or ""
            )

        if adapter == "cargo_mutants":
            outcomes_json = _read_bounded(workdir / "mutants.out" / "outcomes.json")
            outcome = parsers.parse_cargo_mutants(run_text, outcomes_json=outcomes_json)
        else:
            outcome = parsers.parse_mutmut(run_text, results_text=results_text)
    except MutationSpawnError as exc:
        result = SuiteResult(
            name=suite.name,
            adapter=adapter,
            profile=suite.profile,
            command=list(suite.command),
            cwd=str(cwd),
            status=RunStatus.ERROR,
            started_at=started_at,
            finished_at=utc_now_iso(),
            total_count=1,
            error_count=1,
            summary_excerpt=f"spawn error: {exc}",
            repro_command=repro,
            failures=[
                FailureRecord(
                    suite=suite.name,
                    case_name="<spawn>",
                    kind=FailureKind.SPAWN_ERROR,
                    message=str(exc),
                )
            ],
        )
        return result, None
    except RuntimeError as exc:
        result = SuiteResult(
            name=suite.name,
            adapter=adapter,
            profile=suite.profile,
            command=list(suite.command),
            cwd=str(cwd),
            status=RunStatus.ERROR,
            started_at=started_at,
            finished_at=utc_now_iso(),
            total_count=1,
            error_count=1,
            summary_excerpt=str(exc),
            repro_command=repro,
            failures=[
                FailureRecord(
                    suite=suite.name,
                    case_name="<harness>",
                    kind=FailureKind.HARNESS_ERROR,
                    message=str(exc),
                )
            ],
        )
        return result, None
    finally:
        _safe_rmtree_no_symlinks(stage_dir)

    metrics = _metrics(outcome, adapter=adapter, gate=gate, extra={"isolation": exec_meta})
    violations = _untouched_violations(project_root, before_status)

    result = SuiteResult(
        name=suite.name,
        adapter=adapter,
        profile=suite.profile,
        command=list(suite.command),
        cwd=str(cwd),
        status=RunStatus.RUNNING,
        exit_code=exit_code,
        started_at=started_at,
        finished_at=utc_now_iso(),
        duration_ms=duration_ms,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        sanitized_path=sanitized_path,
        repro_command=repro,
        isolation_meta=exec_meta,
        metrics={"mutation": metrics},
    )
    result.cases = _cases(suite.name, outcome)
    _bucket_counts(result, metrics)
    result.summary_excerpt = outcome.summary_line or (
        f"mutation: {metrics['caught']} caught, {metrics['survived']} survived"
    )

    if violations:
        result.status = RunStatus.ERROR
        result.error_count = max(1, result.error_count)
        result.summary_excerpt = "mutation harness wrote inside the real checkout: " + "; ".join(
            violations
        )
        result.failures = [
            FailureRecord(
                suite=suite.name,
                case_name="<isolation>",
                kind=FailureKind.HARNESS_ERROR,
                message=result.summary_excerpt,
            )
        ]
        export_evidence(
            evidence_dir,
            adapter=adapter,
            metrics=metrics,
            cases=result.cases,
            summary=result.summary_excerpt,
        )
        return result, None

    if cancelled:
        result.status = RunStatus.CANCELLED
        result.summary_excerpt = "cancelled by operator"
        result.failures = [
            FailureRecord(
                suite=suite.name,
                case_name="<cancelled>",
                kind=FailureKind.CANCELLED,
                message=result.summary_excerpt,
            )
        ]
        return result, None

    if timed_out:
        result.status = RunStatus.TIMEOUT
        result.summary_excerpt = f"timed out after {suite.timeout_s}s"
        result.failures = [
            FailureRecord(
                suite=suite.name,
                case_name="<timeout>",
                kind=FailureKind.TIMEOUT,
                message=result.summary_excerpt,
            )
        ]
        export_evidence(
            evidence_dir,
            adapter=adapter,
            metrics=metrics,
            cases=result.cases,
            summary=result.summary_excerpt,
        )
        return result, None

    if outcome.baseline_failed:
        result.status = RunStatus.FAIL
        result.summary_excerpt = _BASELINE_MESSAGE
        result.failures = [_baseline_failure_record(suite.name, run_text[-800:])]
        export_evidence(
            evidence_dir,
            adapter=adapter,
            metrics=metrics,
            cases=result.cases,
            summary=result.summary_excerpt,
        )
        return result, None

    if not outcome.structured or (metrics["tested"] == 0 and metrics["inconclusive"] == 0):
        # The tool failed before producing any mutant outcome: that is a harness error,
        # never a silent pass.
        result.status = RunStatus.ERROR
        result.error_count = max(1, result.error_count)
        result.summary_excerpt = (
            f"mutation tool produced no mutant outcomes (exit {exit_code}): "
            + _redacted(run_text, 400)
        ).strip()
        result.failures = [
            FailureRecord(
                suite=suite.name,
                case_name="<harness>",
                kind=FailureKind.HARNESS_ERROR,
                message=result.summary_excerpt,
            )
        ]
        export_evidence(
            evidence_dir,
            adapter=adapter,
            metrics=metrics,
            cases=result.cases,
            summary=result.summary_excerpt,
        )
        return result, None

    if gate is not None:
        score = metrics["score"]
        if metrics["tested"] == 0 or score is None:
            result.status = RunStatus.FAIL
            result.summary_excerpt = (
                "mutation gate cannot be evaluated: no viable mutants were tested"
            )
            result.failures = [
                FailureRecord(
                    suite=suite.name,
                    case_name="<mutation-gate>",
                    kind=FailureKind.ASSERTION,
                    message=result.summary_excerpt,
                )
            ]
        elif float(score) < float(gate):
            result.status = RunStatus.FAIL
            result.summary_excerpt = (
                f"mutation score {float(score):.3f} < gate {float(gate):.3f} "
                f"({metrics['caught']} caught / {metrics['tested']} tested)"
            )
            result.failures = [
                FailureRecord(
                    suite=suite.name,
                    case_name="<mutation-gate>",
                    kind=FailureKind.ASSERTION,
                    message=result.summary_excerpt,
                    excerpt="\n".join(
                        case.name for case in result.cases if case.status == RunStatus.FAIL
                    )[:800],
                )
            ]
        else:
            result.status = RunStatus.PASS
    else:
        result.status = RunStatus.PASS

    export_evidence(
        evidence_dir,
        adapter=adapter,
        metrics=metrics,
        cases=result.cases,
        summary=result.summary_excerpt,
    )
    return result, None
