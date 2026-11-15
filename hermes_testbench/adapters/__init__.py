"""Native runner adapters: policy gates, isolated execution, and result accounting.

Execution policy is enforced before any process is spawned, the command runs
exactly once, and all counts come from the single captured stdout/stderr pair.
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
from typing import Any, Dict, List, Optional, Tuple

from ..isolation import extra_bin_dirs, run_isolated_command
from ..schema import (
    CaseResult,
    FailureKind,
    FailureRecord,
    RunStatus,
    SuiteResult,
    SuiteSpec,
    utc_now_iso,
)
from . import parsers
from .parsers import ADAPTER_PARSERS, ParseOutcome

__all__ = ["execute_suite", "ADAPTER_PARSERS", "ParseOutcome"]

_SUMMARY_TAIL_LINES = 12


def _resolve_cwd(project_root: Path, suite: SuiteSpec) -> Path:
    return (Path(project_root) / suite.cwd).expanduser().resolve()


def _repro_command(suite: SuiteSpec, cwd: Path) -> str:
    if not suite.command:
        return f"cd {shlex.quote(str(cwd))}"
    return f"cd {shlex.quote(str(cwd))} && {shlex.join(suite.command)}"


def _extra_bin_dirs() -> List[str]:
    return extra_bin_dirs()


def _search_path() -> str:
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    for extra in _extra_bin_dirs():
        if extra not in parts:
            parts.append(extra)
    return os.pathsep.join(parts)


def _missing_binary(binaries: List[str]) -> Optional[str]:
    path = _search_path()
    for binary in binaries:
        if not shutil.which(binary, path=path):
            return binary
    return None


def _base_result(
    suite: SuiteSpec,
    *,
    cwd: Path,
    status: str,
    repro: str,
    started_at: str,
    finished_at: Optional[str] = None,
) -> SuiteResult:
    return SuiteResult(
        name=suite.name,
        adapter=suite.adapter,
        profile=suite.profile,
        command=list(suite.command),
        cwd=str(cwd),
        status=status,
        started_at=started_at,
        finished_at=finished_at or utc_now_iso(),
        repro_command=repro,
    )


def _tail_excerpt(text: str, limit: int = 800) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    tail = "\n".join(lines[-_SUMMARY_TAIL_LINES:])
    return tail[-limit:]


def _to_failure_records(suite_name: str, raw: List[Dict[str, Any]]) -> List[FailureRecord]:
    records: List[FailureRecord] = []
    for item in raw:
        data = dict(item)
        data["suite"] = suite_name
        data.setdefault("kind", FailureKind.ASSERTION)
        data.setdefault("case_name", "<run>")
        data.setdefault("message", "")
        records.append(FailureRecord(**data))
    return records


def _to_case_results(suite_name: str, raw: List[Dict[str, Any]]) -> List[CaseResult]:
    cases: List[CaseResult] = []
    for item in raw:
        data = dict(item)
        data["suite"] = suite_name
        data.setdefault("status", RunStatus.PASS)
        cases.append(CaseResult(**data))
    return cases


def _finalize_counts(
    result: SuiteResult,
    outcome: ParseOutcome,
    exit_code: Optional[int],
    signal_num: Optional[int],
) -> None:
    result.total_count = outcome.total
    result.pass_count = outcome.passed
    result.fail_count = outcome.failed
    result.skip_count = outcome.skipped
    result.error_count = outcome.errors
    result.failures = _to_failure_records(result.name, outcome.failures)
    result.cases = _to_case_results(result.name, outcome.cases)

    exit_bad = exit_code not in (0, None)
    if outcome.structured:
        if (exit_bad or signal_num is not None) and outcome.failed == 0 and outcome.errors == 0:
            # The process died without a runner summary: honour the exit status.
            result.fail_count = max(1, result.fail_count)
            result.total_count = max(
                result.total_count, result.pass_count + result.fail_count + result.skip_count + result.error_count
            )
            reason = f"process exited with status {exit_code}" if exit_code is not None else f"killed by signal {signal_num}"
            result.failures.append(
                FailureRecord(suite=result.name, case_name="<exit>", kind=FailureKind.HARNESS_ERROR, message=reason)
            )
        failed_any = result.fail_count > 0 or result.error_count > 0
        result.status = RunStatus.FAIL if (failed_any or exit_bad or signal_num is not None) else RunStatus.PASS
        return

    # No machine-readable summary: fall back to the exit status, never invent counts.
    if exit_code == 0 and signal_num is None:
        result.total_count = 1
        result.pass_count = 1
        result.status = RunStatus.PASS
    else:
        result.total_count = 1
        result.fail_count = 1
        result.status = RunStatus.FAIL
        reason = (
            f"process exited with status {exit_code}"
            if exit_code is not None
            else f"killed by signal {signal_num}"
        )
        result.failures.append(
            FailureRecord(
                suite=result.name,
                case_name="<run>",
                kind=FailureKind.HARNESS_ERROR,
                message=reason,
                excerpt=result.summary_excerpt,
            )
        )


def execute_suite(
    suite: SuiteSpec,
    *,
    project_root: Path,
    evidence_dir: Path,
    run_id: str,
    trusted_manifest: bool = False,
    allow_billable: bool = False,
    max_tokens: int = 0,
    cancel_flag_path: Optional[Path] = None,
    scratch_root: Optional[Path] = None,
    on_spawn: Optional[Any] = None,
) -> Tuple[SuiteResult, Optional[Dict[str, Any]]]:
    """Run one suite exactly once under isolation and map its output to a result."""
    project_root = Path(project_root)
    evidence_dir = Path(evidence_dir)
    cwd = _resolve_cwd(project_root, suite)
    repro = _repro_command(suite, cwd)
    started_at = utc_now_iso()

    # Gate 1: billable suites need explicit approval *and* a positive token budget.
    if suite.billable and not (allow_billable and max_tokens > 0):
        result = _base_result(suite, cwd=cwd, status=RunStatus.BLOCKED, repro=repro, started_at=started_at)
        result.total_count = 1
        result.blocked_count = 1
        result.summary_excerpt = (
            "blocked: billable suite requires allow_billable=True and max_tokens > 0 "
            f"(allow_billable={allow_billable}, max_tokens={max_tokens})"
        )
        result.failures = [
            FailureRecord(
                suite=suite.name,
                case_name="<policy>",
                kind=FailureKind.BLOCKED_POLICY,
                message=result.summary_excerpt,
            )
        ]
        return result, None

    # Gate 2: untrusted manifest suites may not request network or repo writes.
    if suite.custom_from_manifest and not trusted_manifest and (
        suite.allow_network or suite.allow_repo_write
    ):
        caps = []
        if suite.allow_network:
            caps.append("allow_network")
        if suite.allow_repo_write:
            caps.append("allow_repo_write")
        result = _base_result(suite, cwd=cwd, status=RunStatus.BLOCKED, repro=repro, started_at=started_at)
        result.total_count = 1
        result.blocked_count = 1
        result.summary_excerpt = (
            "blocked: suite from an untrusted manifest requests "
            + ", ".join(caps)
            + "; trust the manifest or remove the capability"
        )
        result.failures = [
            FailureRecord(
                suite=suite.name,
                case_name="<policy>",
                kind=FailureKind.UNTRUSTED_MANIFEST,
                message=result.summary_excerpt,
            )
        ]
        return result, None

    # Gate 3: nothing to run, or prerequisites absent.
    skip_reason = ""
    if not suite.command:
        skip_reason = "skip: suite has no command"
    else:
        missing_bin = _missing_binary(suite.required_binaries)
        if missing_bin:
            skip_reason = f"skip: required binary not found on PATH: {missing_bin}"
        else:
            for rel in suite.required_paths:
                if not (project_root / rel).exists():
                    skip_reason = f"skip: required path missing under project root: {rel}"
                    break
    if skip_reason:
        result = _base_result(suite, cwd=cwd, status=RunStatus.SKIP, repro=repro, started_at=started_at)
        result.total_count = 1
        result.skip_count = 1
        result.summary_excerpt = skip_reason
        return result, None

    # Single execution: counts must come from this one captured run.
    try:
        exec_result = run_isolated_command(
            list(suite.command),
            cwd=cwd,
            evidence_dir=evidence_dir,
            step_name=suite.name,
            run_id=run_id,
            timeout_s=suite.timeout_s,
            needs_tmux=suite.needs_tmux,
            needs_pty=suite.needs_pty,
            allow_network=suite.allow_network,
            allow_repo_write=suite.allow_repo_write,
            env_overrides=suite.env_overrides,
            cancel_flag_path=cancel_flag_path,
            scratch_root=scratch_root,
            on_spawn=on_spawn,
        )
    except (OSError, ValueError) as exc:
        result = _base_result(suite, cwd=cwd, status=RunStatus.ERROR, repro=repro, started_at=started_at)
        result.total_count = 1
        result.error_count = 1
        result.summary_excerpt = f"spawn error: {exc}"
        result.failures = [
            FailureRecord(
                suite=suite.name,
                case_name="<spawn>",
                kind=FailureKind.SPAWN_ERROR,
                message=str(exc),
            )
        ]
        return result, None

    result = _base_result(suite, cwd=cwd, status=RunStatus.RUNNING, repro=repro, started_at=started_at)
    result.exit_code = exec_result.exit_code
    result.signal_num = exec_result.signal_num
    result.duration_ms = exec_result.duration_ms
    result.stdout_path = exec_result.stdout_path
    result.stderr_path = exec_result.stderr_path
    result.sanitized_path = exec_result.sanitized_path
    result.isolation_meta = dict(exec_result.isolation_meta)

    combined = (exec_result.stdout_text or "") + "\n" + (exec_result.stderr_text or "")

    if exec_result.timed_out:
        result.status = RunStatus.TIMEOUT
        result.total_count = 1
        result.fail_count = 1
        result.summary_excerpt = f"timed out after {suite.timeout_s}s"
        result.failures = [
            FailureRecord(
                suite=suite.name,
                case_name="<timeout>",
                kind=FailureKind.TIMEOUT,
                message=result.summary_excerpt,
                excerpt=_tail_excerpt(combined),
            )
        ]
        result.metrics = {"timeout_s": suite.timeout_s}
        return result, None

    if exec_result.cancelled:
        result.status = RunStatus.CANCELLED
        result.total_count = 1
        result.fail_count = 1
        result.summary_excerpt = "cancelled by operator"
        result.failures = [
            FailureRecord(
                suite=suite.name,
                case_name="<cancelled>",
                kind=FailureKind.CANCELLED,
                message=result.summary_excerpt,
                excerpt=_tail_excerpt(combined),
            )
        ]
        return result, None

    parser = ADAPTER_PARSERS.get(suite.adapter, parsers.parse_generic)
    outcome = parser(combined)
    result.summary_excerpt = outcome.summary_line or _tail_excerpt(combined)
    _finalize_counts(result, outcome, exec_result.exit_code, exec_result.signal_num)
    if result.isolation_meta.get("seatbelt") in ("active", "inherited_outer"):
        for frec in result.failures:
            blob = f"{frec.message}\n{frec.excerpt}"
            if frec.kind == FailureKind.HARNESS_ERROR and (
                "[Errno 1] Operation not permitted" in blob or "Operation not permitted" in frec.message
            ):
                frec.kind = FailureKind.SANDBOX_DENIED
    if not result.summary_excerpt:
        result.summary_excerpt = _tail_excerpt(combined)
    return result, outcome.usage
