"""Single orchestration engine shared by CLI, detached workers, MCP server, and TUI."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .isolation import real_user_home, run_isolated_command, terminate_process_group
from .sanitize import redact_secrets, sanitize_terminal_text
from .schema import (
    CaseResult,
    FailureKind,
    FailureRecord,
    ProjectSpec,
    RunRecord,
    RunStatus,
    SuiteResult,
    SuiteSpec,
    new_run_id,
    utc_now_iso,
)
from .store import TestBenchStore, atomic_write_file

def pinned_python() -> str:
    """The interpreter running TestBench: the one its installation environment provides.

    The detached worker must import the same package from the same environment as its
    parent, so it is never resolved from a host-specific candidate list.
    """
    return sys.executable or "python3"


def repo_package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def messaging_paths(home: Optional[Path] = None) -> Tuple[Path, Path]:
    """Resolve the ``agent-messaging`` artifact root and ``hermes-agent`` checkout.

    Derived from the real user home so neither the engine nor the CLI hard-codes a
    host-specific home path. Shared by ``messaging-verify`` and by
    ``reproduce`` so both surfaces look at exactly the same locations.
    """
    base = (Path(home).expanduser() if home is not None else real_user_home())
    hermes = base / ".hermes"
    return (hermes / "artifacts" / "agent-messaging"), (hermes / "hermes-agent")


def base_profile(profile: str) -> str:
    """Strip accumulated ``repro:`` prefixes so repro-of-repro never stacks them."""
    name = profile or ""
    while name.startswith("repro:"):
        name = name[len("repro:"):]
    return name


def _recorded_artifact_root(
    orig: RunRecord, resolved: Path, default_art_root: Path
) -> Optional[Path]:
    """Return the ``artifact_root`` the original messaging run actually used.

    Runs recorded since the messaging path exists store it explicitly in
    ``environment['messaging_artifact_root']`` (an empty string means "no
    artifact root was used"). Older runs lack the marker, so fall back to the
    recorded project_root only when it is in fact the artifact root.
    """
    env = orig.environment or {}
    if "messaging_artifact_root" in env:
        recorded = str(env.get("messaging_artifact_root") or "").strip()
        return Path(recorded).expanduser() if recorded else None
    try:
        if resolved == default_art_root.resolve():
            return resolved
    except OSError:
        return None
    return None


class AuthorizationError(PermissionError):
    """Raised when an interface attempts to access a path outside its scoped allowlist."""


def default_allowed_roots() -> List[Path]:
    raw = os.environ.get("TESTBENCH_ALLOWED_ROOTS", "").strip()
    if raw:
        out: List[Path] = []
        for part in raw.split(os.pathsep):
            p = part.strip()
            if p:
                out.append(Path(p).expanduser().resolve())
        if out:
            return out
    uhome = real_user_home()
    return [
        (uhome / "Code").resolve(),
        (uhome / ".hermes" / "artifacts").resolve(),
    ]


def check_scoped_access(
    target: Path,
    store: TestBenchStore,
    *,
    allowed_roots: Optional[Sequence[Path]] = None,
) -> Path:
    try:
        resolved = target.expanduser().resolve(strict=True)
    except OSError as exc:
        raise AuthorizationError(f"Target path does not exist or is inaccessible: {target}") from exc

    uhome = real_user_home()
    hermes_root = (uhome / ".hermes").resolve()
    forbidden_hermes_subpaths = (
        hermes_root / "skills",
        hermes_root / "plugins",
        hermes_root / "memories",
        hermes_root / "sessions",
        hermes_root / "profiles",
        hermes_root / "mcp-tokens",
        hermes_root / "logs",
        hermes_root / "config.yaml",
        hermes_root / ".env",
        hermes_root / "auth.json",
        hermes_root / "state.db",
    )
    if resolved == hermes_root or any(
        resolved == p.resolve() or resolved.is_relative_to(p.resolve())
        for p in forbidden_hermes_subpaths
        if p.exists() or not p.suffix
    ):
        raise AuthorizationError(
            f"Access denied to protected Hermes runtime path: {resolved}"
        )

    if store.is_project_registered(resolved):
        return resolved

    roots = [r.expanduser().resolve() for r in (allowed_roots if allowed_roots is not None else default_allowed_roots())]
    for root in roots:
        if resolved == root or resolved.is_relative_to(root):
            return resolved

    # Block any other path under ~/.hermes unless under ~/.hermes/artifacts or an explicit allowed_root above
    hermes_artifacts = (hermes_root / "artifacts").resolve()
    if resolved.is_relative_to(hermes_root) and not resolved.is_relative_to(hermes_artifacts):
        raise AuthorizationError(
            f"Access denied to protected Hermes runtime directory: {resolved}"
        )

    raise AuthorizationError(
        f"Path {resolved} is outside allowlisted roots ({[str(r) for r in roots]}) "
        "and is not registered in TestBench."
    )


def _fallback_git_metadata(root: Path) -> Tuple[str, str]:
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if sha.returncode != 0:
            return "unversioned", ""
        rev = (sha.stdout or "").strip()
        br = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        branch = (br.stdout or "").strip() if br.returncode == 0 else ""
        st = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "-uno"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if st.returncode == 0 and (st.stdout or "").strip():
            rev = f"{rev}+dirty"
        return rev or "unversioned", branch
    except (OSError, subprocess.SubprocessError):
        return "unversioned", ""


def _fallback_discover(root: Path, *, store: Optional[TestBenchStore] = None) -> ProjectSpec:
    try:
        from .discovery import discover_project  # type: ignore
        return discover_project(root, store=store)
    except ImportError:
        pass

    resolved = root.expanduser().resolve(strict=True)
    rev, branch = _fallback_git_metadata(resolved)
    manifest = resolved / ".testbench.toml"
    trusted = store.is_manifest_trusted(resolved) if store else False
    suites: List[SuiteSpec] = []
    name = resolved.name
    if manifest.is_file() and not manifest.is_symlink():
        raw_bytes = manifest.read_bytes()
        if len(raw_bytes) > 262144:
            raise ValueError(".testbench.toml exceeds 256 KiB limit")
        parsed = tomllib.loads(raw_bytes.decode("utf-8"))
        proj_tbl = parsed.get("project") or {}
        name = str(proj_tbl.get("name") or name)
        for item in parsed.get("suites") or []:
            suites.append(
                SuiteSpec(
                    name=str(item["name"]),
                    adapter=str(item.get("adapter", "shell")),
                    profile=str(item.get("profile", "unit")),
                    command=[str(x) for x in (item.get("command") or [])],
                    cwd=str(item.get("cwd", ".")),
                    timeout_s=float(item.get("timeout_s", 120.0)),
                    needs_tmux=bool(item.get("needs_tmux", False)),
                    needs_pty=bool(item.get("needs_pty", False)),
                    allow_network=bool(item.get("allow_network", False)),
                    allow_repo_write=bool(item.get("allow_repo_write", False)),
                    billable=bool(item.get("billable", False)),
                    custom_from_manifest=True,
                    env_overrides={str(k): str(v) for k, v in (item.get("env") or {}).items()},
                    required_paths=[str(p) for p in (item.get("required_paths") or [])],
                    required_binaries=[str(b) for b in (item.get("required_binaries") or [])],
                    description=str(item.get("description", "")),
                )
            )
    elif (resolved / "tests").is_dir():
        suites.append(
            SuiteSpec(
                name="python-unittest",
                adapter="python_unittest",
                profile="unit",
                command=[pinned_python(), "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                cwd=".",
                timeout_s=180.0,
            )
        )
    return ProjectSpec(
        name=name,
        root=str(resolved),
        languages=[],
        suites=suites,
        manifest_path=str(manifest) if manifest.is_file() else None,
        trusted=trusted,
        revision=rev,
        branch=branch,
    )


def _fallback_execute_suite(
    suite: SuiteSpec,
    *,
    project_root: Path,
    evidence_dir: Path,
    run_id: str,
    trusted_manifest: bool = False,
    allow_billable: bool = False,
    max_tokens: int = 0,
    cancel_flag_path: Optional[Path] = None,
    on_spawn: Optional[Any] = None,
) -> Tuple[SuiteResult, Optional[Dict[str, Any]]]:
    try:
        from .adapters import execute_suite  # type: ignore
        return execute_suite(
            suite,
            project_root=project_root,
            evidence_dir=evidence_dir,
            run_id=run_id,
            trusted_manifest=trusted_manifest,
            allow_billable=allow_billable,
            max_tokens=max_tokens,
            cancel_flag_path=cancel_flag_path,
            on_spawn=on_spawn,
        )
    except ImportError:
        pass

    started = utc_now_iso()
    cwd_path = (project_root / suite.cwd).resolve()
    repro = " ".join(shlex.quote(a) for a in suite.command)
    if suite.billable and not (allow_billable and max_tokens > 0):
        return (
            SuiteResult(
                name=suite.name,
                adapter=suite.adapter,
                profile=suite.profile,
                command=list(suite.command),
                cwd=str(cwd_path),
                status=RunStatus.BLOCKED,
                started_at=started,
                finished_at=utc_now_iso(),
                blocked_count=1,
                total_count=1,
                summary_excerpt="Billable suite blocked: pass --allow-billable and --max-tokens > 0",
                repro_command=repro,
            ),
            None,
        )
    if suite.custom_from_manifest and (suite.allow_network or suite.allow_repo_write) and not trusted_manifest:
        return (
            SuiteResult(
                name=suite.name,
                adapter=suite.adapter,
                profile=suite.profile,
                command=list(suite.command),
                cwd=str(cwd_path),
                status=RunStatus.BLOCKED,
                started_at=started,
                finished_at=utc_now_iso(),
                blocked_count=1,
                total_count=1,
                summary_excerpt="Untrusted .testbench.toml suite requesting network/repo-write requires 'testbench trust'",
                repro_command=repro,
            ),
            None,
        )

    res = run_isolated_command(
        suite.command,
        cwd=cwd_path,
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
    )
    finished = utc_now_iso()
    if res.cancelled:
        status = RunStatus.CANCELLED
        total, passed, failed = 1, 0, 0
    elif res.timed_out:
        status = RunStatus.TIMEOUT
        total, passed, failed = 1, 0, 0
    elif res.exit_code == 0:
        status = RunStatus.PASS
        total, passed, failed = 1, 1, 0
    else:
        status = RunStatus.FAIL
        total, passed, failed = 1, 0, 1

    combined = sanitize_terminal_text(redact_secrets((res.stdout_text + "\n" + res.stderr_text).strip()))
    failures: List[FailureRecord] = []
    if status == RunStatus.FAIL:
        first_line = next((ln.strip() for ln in combined.splitlines() if ln.strip()), f"exit {res.exit_code}")
        failures.append(FailureRecord(suite=suite.name, case_name=suite.name, message=first_line, excerpt=combined[:600]))

    return (
        SuiteResult(
            name=suite.name,
            adapter=suite.adapter,
            profile=suite.profile,
            command=list(suite.command),
            cwd=str(cwd_path),
            status=status,
            exit_code=res.exit_code,
            signal_num=res.signal_num,
            started_at=started,
            finished_at=finished,
            duration_ms=res.duration_ms,
            total_count=total,
            pass_count=passed,
            fail_count=failed,
            stdout_path=res.stdout_path,
            stderr_path=res.stderr_path,
            sanitized_path=res.sanitized_path,
            summary_excerpt=combined[:300],
            repro_command=repro,
            failures=failures,
            cases=[CaseResult(name=suite.name, suite=suite.name, status=status, duration_ms=res.duration_ms)],
            isolation_meta=res.isolation_meta,
        ),
        None,
    )


def host_environment_snapshot() -> Dict[str, Any]:
    try:
        l1, l5, l15 = os.getloadavg()
        load = [round(l1, 2), round(l5, 2), round(l15, 2)]
    except (AttributeError, OSError):
        load = []
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count() or 1,
        "loadavg": load,
    }


def _blocked_failure_kind(suites: List[SuiteResult]) -> str:
    for s in suites:
        if s.status != RunStatus.BLOCKED:
            continue
        for f in s.failures:
            if f.kind == FailureKind.UNTRUSTED_MANIFEST:
                return FailureKind.UNTRUSTED_MANIFEST
    return FailureKind.BLOCKED_POLICY


def _failed_suite_kind(suites: List[SuiteResult]) -> str:
    failing = [s for s in suites if s.status == RunStatus.FAIL]
    kinds = [f.kind for s in failing for f in s.failures if f.kind]
    if FailureKind.ASSERTION in kinds:
        return FailureKind.ASSERTION
    if FailureKind.SANDBOX_DENIED in kinds:
        return FailureKind.SANDBOX_DENIED
    if kinds:
        return kinds[0]
    if sum(s.error_count for s in failing) > 0 and sum(s.fail_count for s in failing) == 0:
        return FailureKind.HARNESS_ERROR
    return FailureKind.ASSERTION


def _apply_mutation_gate(suites: List[SuiteSpec], mutation_gate: Optional[float]) -> None:
    """Stamp an operator-supplied mutation gate onto the selected suites.

    A CLI gate is trusted by construction, unlike ``mutation_score_gate`` in a manifest,
    which only counts once the manifest itself is trusted.
    """
    if mutation_gate is None:
        return
    gate = float(mutation_gate)
    for suite in suites:
        suite.mutation_score_gate = gate
        suite.mutation_gate_from_cli = True


def _aggregate_run_status(suites: List[SuiteResult]) -> Tuple[str, Optional[str]]:
    if not suites:
        return RunStatus.SKIP, None
    statuses = [s.status for s in suites]
    if any(st == RunStatus.CANCELLED for st in statuses):
        return RunStatus.CANCELLED, FailureKind.CANCELLED
    if any(st == RunStatus.ERROR for st in statuses):
        return RunStatus.ERROR, FailureKind.HARNESS_ERROR
    if any(st == RunStatus.TIMEOUT for st in statuses):
        return RunStatus.TIMEOUT, FailureKind.TIMEOUT
    if any(st == RunStatus.FAIL for st in statuses):
        return RunStatus.FAIL, _failed_suite_kind(suites)
    if any(st == RunStatus.BLOCKED for st in statuses) and not any(st == RunStatus.PASS for st in statuses):
        return RunStatus.BLOCKED, _blocked_failure_kind(suites)
    if all(st == RunStatus.SKIP for st in statuses):
        return RunStatus.SKIP, None
    if any(st == RunStatus.BLOCKED for st in statuses):
        return RunStatus.BLOCKED, _blocked_failure_kind(suites)
    return RunStatus.PASS, None


class TestBenchEngine:
    def __init__(
        self,
        store: Optional[TestBenchStore] = None,
        *,
        allowed_roots: Optional[Sequence[Path]] = None,
        discover_fn: Optional[Callable[..., ProjectSpec]] = None,
        execute_suite_fn: Optional[Callable[..., Tuple[SuiteResult, Optional[Dict[str, Any]]]]] = None,
    ) -> None:
        self._owns_store = store is None
        self.store = store if store is not None else TestBenchStore()
        self.allowed_roots = list(allowed_roots) if allowed_roots is not None else None
        self._discover_fn = discover_fn or _fallback_discover
        self._execute_suite_fn = execute_suite_fn or _fallback_execute_suite
        # Reconcile any orphaned runs whose worker crashed without releasing run.lock
        self.store.reconcile_orphans()

    def close(self) -> None:
        if self._owns_store:
            self.store.close()

    def resolve_target(self, target: Path, *, enforce_scope: bool = False) -> Path:
        if enforce_scope:
            return check_scoped_access(target, self.store, allowed_roots=self.allowed_roots)
        return target.expanduser().resolve(strict=True)

    def discover(self, target: Path, *, enforce_scope: bool = False) -> ProjectSpec:
        resolved = self.resolve_target(target, enforce_scope=enforce_scope)
        return self._discover_fn(resolved, store=self.store)

    def _select_suites(
        self,
        spec: ProjectSpec,
        *,
        profile: Optional[str] = None,
        suite_names: Optional[Sequence[str]] = None,
    ) -> List[SuiteSpec]:
        wanted_names = set(suite_names or ())
        selected: List[SuiteSpec] = []
        for s in spec.suites:
            if wanted_names and s.name not in wanted_names:
                continue
            if not wanted_names and profile and profile != "all" and s.profile != profile:
                continue
            selected.append(s)
        return selected

    def _populate_and_finalize(
        self,
        run: RunRecord,
        suite_results: List[SuiteResult],
        usages: List[Dict[str, Any]],
        t0: float,
    ) -> RunRecord:
        run.suites = suite_results
        run.duration_ms = round((time.monotonic() - t0) * 1000.0, 2)
        run.total_count = sum(s.total_count for s in suite_results)
        run.pass_count = sum(s.pass_count for s in suite_results)
        run.fail_count = sum(s.fail_count for s in suite_results)
        run.skip_count = sum(s.skip_count for s in suite_results)
        run.blocked_count = sum(s.blocked_count for s in suite_results)
        run.error_count = sum(s.error_count for s in suite_results)
        status, fkind = _aggregate_run_status(suite_results)
        # Fencing against last-writer-wins: if cancel.flag was written or DB already records CANCELLED, keep CANCELLED
        ev_cancel = (Path(run.evidence_dir) / "cancel.flag") if run.evidence_dir else (self.store.run_dir(run.run_id) / "cancel.flag")
        existing = self.store.get_run(run.run_id)
        if ev_cancel.exists() or (existing is not None and existing.status == RunStatus.CANCELLED):
            status = RunStatus.CANCELLED
            fkind = FailureKind.CANCELLED
        run.status = status
        run.failure_kind = fkind

        non_zero = [s.exit_code for s in suite_results if s.exit_code not in (None, 0)]
        if status in (RunStatus.CANCELLED, RunStatus.TIMEOUT) and not non_zero:
            run.exit_code = None
        else:
            run.exit_code = non_zero[0] if non_zero else (0 if suite_results else None)

        first_fail = ""
        for s in suite_results:
            if s.failures:
                f = s.failures[0]
                first_fail = f"{f.case_name}: {f.message or f.excerpt}".strip()
                break
            if s.status in (RunStatus.BLOCKED, RunStatus.TIMEOUT, RunStatus.ERROR, RunStatus.FAIL) and s.summary_excerpt:
                first_fail = f"{s.name}: {s.summary_excerpt.splitlines()[0]}"
                break
        run.first_failure = first_fail[:300]

        repros = [s.repro_command for s in suite_results if s.repro_command and s.status != RunStatus.PASS]
        if not repros:
            repros = [s.repro_command for s in suite_results if s.repro_command]
        run.repro_instructions = " && ".join(dict.fromkeys(repros))
        run.summary = (
            f"{run.status}: {run.pass_count}/{run.total_count} passed, "
            f"{run.fail_count} failed, {run.error_count} errors, "
            f"{run.skip_count} skipped, {run.blocked_count} blocked"
        )

        # Never fabricate model token usage; record only when measured
        if usages:
            prompt_t = sum(int(u.get("prompt_tokens", 0) or 0) for u in usages)
            comp_t = sum(int(u.get("completion_tokens", 0) or 0) for u in usages)
            tot_t = sum(int(u.get("total_tokens", prompt_t + comp_t) or 0) for u in usages)
            costs = [float(u["estimated_cost_usd"]) for u in usages if u.get("estimated_cost_usd") is not None]
            run.model_usage = {
                "model": usages[0].get("model"),
                "provider": usages[0].get("provider"),
                "prompt_tokens": prompt_t,
                "completion_tokens": comp_t,
                "total_tokens": tot_t,
                "estimated_cost_usd": round(sum(costs), 6) if costs else None,
            }
        else:
            run.model_usage = None

        return self.store.finalize_run(run)

    def run_sync(
        self,
        target: Path,
        *,
        profile: str = "unit",
        suite_names: Optional[Sequence[str]] = None,
        allow_billable: bool = False,
        max_tokens: int = 0,
        timeout_s: Optional[float] = None,
        enforce_scope: bool = False,
        delegation_id: Optional[str] = None,
        mutation_gate: Optional[float] = None,
    ) -> RunRecord:
        resolved = self.resolve_target(target, enforce_scope=enforce_scope)
        spec = self._discover_fn(resolved, store=self.store)
        suites = self._select_suites(spec, profile=profile, suite_names=suite_names)
        if timeout_s is not None:
            for s in suites:
                s.timeout_s = float(timeout_s)
        _apply_mutation_gate(suites, mutation_gate)

        rid = new_run_id()
        now = utc_now_iso()
        run = RunRecord(
            run_id=rid,
            project=spec.name,
            project_root=str(resolved),
            revision=spec.revision or "unversioned",
            branch=spec.branch,
            profile=profile,
            status=RunStatus.RUNNING,
            commands=[list(s.command) for s in suites],
            created_at=now,
            started_at=now,
            environment=host_environment_snapshot(),
            delegation_id=delegation_id or os.environ.get("HERMES_DELEGATION_ID") or None,
            worker_pid=None,
            worker_pgid=None,
            worker_start_time=None,
        )
        self.store.create_run(run)
        ev_dir = self.store.run_dir(rid)
        cancel_flag = ev_dir / "cancel.flag"
        t0 = time.monotonic()

        def _record_child_spawn(cpid: int, cpgid: int, cfp: Optional[str]) -> None:
            run.worker_pid = cpid
            run.worker_pgid = cpgid
            run.worker_start_time = cfp
            self.store.update_run(run)

        try:
            with self.store.hold_run_lock(rid):
                if not suites:
                    run.status = RunStatus.SKIP
                    run.summary = f"No suites matched profile={profile!r}"
                    return self.store.finalize_run(run)

                results: List[SuiteResult] = []
                usages: List[Dict[str, Any]] = []
                for s in suites:
                    if cancel_flag.exists():
                        results.append(
                            SuiteResult(
                                name=s.name,
                                adapter=s.adapter,
                                profile=s.profile,
                                command=list(s.command),
                                cwd=str((resolved / s.cwd).resolve()),
                                status=RunStatus.CANCELLED,
                            )
                        )
                        break
                    try:
                        sres, usage = self._execute_suite_fn(
                            s,
                            project_root=resolved,
                            evidence_dir=ev_dir,
                            run_id=rid,
                            trusted_manifest=spec.trusted,
                            allow_billable=allow_billable,
                            max_tokens=max_tokens,
                            cancel_flag_path=cancel_flag,
                            on_spawn=_record_child_spawn,
                        )
                    except TypeError:
                        # Custom injected _execute_suite_fn in unit tests without on_spawn kwarg
                        sres, usage = self._execute_suite_fn(
                            s,
                            project_root=resolved,
                            evidence_dir=ev_dir,
                            run_id=rid,
                            trusted_manifest=spec.trusted,
                            allow_billable=allow_billable,
                            max_tokens=max_tokens,
                            cancel_flag_path=cancel_flag,
                        )
                    run.worker_pid = None
                    run.worker_pgid = None
                    run.worker_start_time = None
                    results.append(sres)
                    if usage:
                        usages.append(usage)
                return self._populate_and_finalize(run, results, usages, t0)
        except Exception as exc:
            run_err = self.store.get_run(rid) or run
            run_err.status = RunStatus.ERROR
            run_err.failure_kind = FailureKind.HARNESS_ERROR
            run_err.error_count = max(1, run_err.error_count)
            run_err.first_failure = f"Execution error: {exc}"[:300]
            return self.store.finalize_run(run_err)

    def spawn_async(
        self,
        target: Path,
        *,
        profile: str = "unit",
        suite_names: Optional[Sequence[str]] = None,
        allow_billable: bool = False,
        max_tokens: int = 0,
        timeout_s: Optional[float] = None,
        wait_up_to_s: float = 3.0,
        enforce_scope: bool = False,
        delegation_id: Optional[str] = None,
        mutation_gate: Optional[float] = None,
    ) -> RunRecord:
        resolved = self.resolve_target(target, enforce_scope=enforce_scope)
        spec = self._discover_fn(resolved, store=self.store)
        suites = self._select_suites(spec, profile=profile, suite_names=suite_names)

        rid = new_run_id()
        now = utc_now_iso()
        run = RunRecord(
            run_id=rid,
            project=spec.name,
            project_root=str(resolved),
            revision=spec.revision or "unversioned",
            branch=spec.branch,
            profile=profile,
            status=RunStatus.QUEUED,
            commands=[list(s.command) for s in suites],
            created_at=now,
            environment=host_environment_snapshot(),
            delegation_id=delegation_id or os.environ.get("HERMES_DELEGATION_ID") or None,
        )
        self.store.create_run(run)

        argv = [
            pinned_python(),
            "-m",
            "hermes_testbench",
            "--home",
            str(self.store.home),
            "_worker",
            "--run-id",
            rid,
            "--project-root",
            str(resolved),
            "--profile",
            profile,
        ]
        for sname in suite_names or ():
            argv.extend(["--suite", sname])
        if allow_billable:
            argv.append("--allow-billable")
        if max_tokens > 0:
            argv.extend(["--max-tokens", str(int(max_tokens))])
        if timeout_s is not None:
            argv.extend(["--timeout", str(float(timeout_s))])
        if mutation_gate is not None:
            argv.extend(["--mutation-gate", str(float(mutation_gate))])

        env = dict(os.environ)
        # The worker imports exactly the package this process imported. A source tree,
        # site-packages dir, or zipapp all work: PYTHONPATH accepts a zip file, but cwd
        # must be a directory.
        pkg_root = repo_package_root()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{pkg_root}{os.pathsep}{existing_pp}" if existing_pp else str(pkg_root)
        env["TESTBENCH_HOME"] = str(self.store.home)

        proc = subprocess.Popen(
            argv,
            cwd=str(pkg_root) if pkg_root.is_dir() else None,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        # Reap our direct Popen handle in a tiny daemon thread so exited detached workers never become zombies
        import threading
        threading.Thread(target=proc.wait, daemon=True).start()
        from .isolation import _process_start_fingerprint
        run.worker_pid = proc.pid
        run.worker_pgid = proc.pid
        run.worker_start_time = _process_start_fingerprint(proc.pid)
        self.store.update_run(run)

        bound = min(max(0.0, float(wait_up_to_s)), 30.0)
        deadline = time.monotonic() + bound
        while time.monotonic() < deadline:
            cur = self.store.get_run(rid)
            if cur is not None and cur.status in RunStatus.TERMINAL:
                return cur
            time.sleep(0.06)
        return self.store.get_run(rid) or run

    def run_worker(
        self,
        *,
        run_id: str,
        project_root: Path,
        profile: str = "unit",
        suite_names: Optional[Sequence[str]] = None,
        allow_billable: bool = False,
        max_tokens: int = 0,
        timeout_s: Optional[float] = None,
        mutation_gate: Optional[float] = None,
    ) -> int:
        rec = self.store.get_run(run_id)
        if rec is None:
            return 2
        ev_dir = self.store.run_dir(run_id)
        cancel_flag = ev_dir / "cancel.flag"

        def _on_term(_signum: int, _frame: Any) -> None:
            try:
                atomic_write_file(cancel_flag, "signal\n", mode=0o600)
            except OSError:
                pass

        old_term = signal.signal(signal.SIGTERM, _on_term)
        old_int = signal.signal(signal.SIGINT, _on_term)
        t0 = time.monotonic()
        try:
            with self.store.hold_run_lock(run_id):
                resolved = project_root.expanduser().resolve(strict=True)
                spec = self._discover_fn(resolved, store=self.store)
                suites = self._select_suites(spec, profile=profile, suite_names=suite_names)
                if timeout_s is not None:
                    for s in suites:
                        s.timeout_s = float(timeout_s)
                _apply_mutation_gate(suites, mutation_gate)

                rec.status = RunStatus.RUNNING
                rec.started_at = utc_now_iso()
                rec.worker_pid = os.getpid()
                rec.environment = {**(rec.environment or {}), "worker_python": sys.executable}
                rec.worker_pgid = os.getpgrp()
                rec.commands = [list(s.command) for s in suites]
                self.store.update_run(rec)

                if not suites:
                    rec.status = RunStatus.SKIP
                    rec.summary = f"No suites matched profile={profile!r}"
                    self.store.finalize_run(rec)
                    return 0

                results: List[SuiteResult] = []
                usages: List[Dict[str, Any]] = []
                for s in suites:
                    if cancel_flag.exists():
                        results.append(
                            SuiteResult(
                                name=s.name,
                                adapter=s.adapter,
                                profile=s.profile,
                                command=list(s.command),
                                cwd=str((resolved / s.cwd).resolve()),
                                status=RunStatus.CANCELLED,
                            )
                        )
                        break
                    sres, usage = self._execute_suite_fn(
                        s,
                        project_root=resolved,
                        evidence_dir=ev_dir,
                        run_id=run_id,
                        trusted_manifest=spec.trusted,
                        allow_billable=allow_billable,
                        max_tokens=max_tokens,
                        cancel_flag_path=cancel_flag,
                    )
                    results.append(sres)
                    if usage:
                        usages.append(usage)

                final = self._populate_and_finalize(rec, results, usages, t0)
                return 0 if final.status in (RunStatus.PASS, RunStatus.SKIP) else 1
        except Exception as exc:
            rec_err = self.store.get_run(run_id) or rec
            rec_err.status = RunStatus.ERROR
            rec_err.failure_kind = FailureKind.HARNESS_ERROR
            rec_err.error_count = max(1, rec_err.error_count)
            rec_err.first_failure = f"Worker error: {exc}"[:300]
            self.store.finalize_run(rec_err)
            return 1
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)

    def cancel(self, run_id: str) -> RunRecord:
        rec = self.store.get_run(run_id)
        if rec is None:
            raise KeyError(f"Unknown TestBench run_id: {run_id}")
        if rec.status in RunStatus.TERMINAL:
            return rec

        ev_dir = self.store.run_dir(run_id)
        cancel_flag = ev_dir / "cancel.flag"
        atomic_write_file(cancel_flag, f"cancelled at {utc_now_iso()}\n", mode=0o600)

        # Wait briefly for cooperative cancel_flag detection by active worker
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            cur = self.store.get_run(run_id)
            if cur is not None and cur.status in RunStatus.TERMINAL:
                return cur
            if not self.store.is_run_locked(run_id):
                break
            time.sleep(0.05)

        # Only signal a dedicated TestBench-owned session group with a verified start fingerprint;
        # NEVER signal our own caller process group or a group lacking worker_start_time!
        cur = self.store.get_run(run_id) or rec
        wpgid = int(cur.worker_pgid) if cur.worker_pgid else 0
        if (
            cur.status not in RunStatus.TERMINAL
            and self.store.is_run_locked(run_id)
            and wpgid > 1
            and wpgid != os.getpgrp()
            and bool(cur.worker_start_time)
        ):
            terminate_process_group(wpgid, expected_start_time=cur.worker_start_time, grace_s=0.5)
            time.sleep(0.1)

        cur = self.store.get_run(run_id) or rec
        if cur.status not in RunStatus.TERMINAL:
            cur.status = RunStatus.CANCELLED
            cur.failure_kind = FailureKind.CANCELLED
            cur.summary = "Run cancelled explicitly"
            self.store.finalize_run(cur)
        return cur

    def reproduce(
        self,
        run_id: str,
        *,
        print_only: bool = False,
        enforce_scope: bool = False,
    ) -> Dict[str, Any]:
        orig = self.store.get_run(run_id)
        if orig is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        cmd_str = orig.repro_instructions
        if not cmd_str and orig.commands:
            cmd_str = " && ".join(" ".join(shlex.quote(a) for a in c) for c in orig.commands if c)

        if print_only:
            return {
                "source_run_id": orig.run_id,
                "project": orig.project,
                "project_root": orig.project_root,
                "revision": orig.revision,
                "repro_command": cmd_str,
                "repro_script": orig.repro_path,
            }

        target = Path(orig.project_root)
        resolved = self.resolve_target(target, enforce_scope=enforce_scope)
        if orig.project == "agent-messaging" and base_profile(orig.profile) == "verification":
            from .messaging_suite import PYTEST_SUITE, run_agent_messaging_verification
            rid = new_run_id()
            now = utc_now_iso()
            use_pt = any(s.name == PYTEST_SUITE for s in orig.suites)
            default_art_root, ha_root = messaging_paths()
            art_root = _recorded_artifact_root(orig, resolved, default_art_root)
            env_snap = host_environment_snapshot()
            env_snap["messaging_artifact_root"] = str(art_root) if art_root is not None else ""
            replay_mv = RunRecord(
                run_id=rid,
                project=orig.project,
                project_root=str(resolved),
                revision=_fallback_git_metadata(resolved)[0],
                profile="repro:" + base_profile(orig.profile),
                status=RunStatus.RUNNING,
                commands=list(orig.commands),
                created_at=now,
                started_at=now,
                environment=env_snap,
            )
            self.store.create_run(replay_mv)
            ev_mv = self.store.run_dir(rid)
            t0_mv = time.monotonic()
            try:
                with self.store.hold_run_lock(rid):
                    mv_suites = run_agent_messaging_verification(
                        ev_mv,
                        artifact_root=art_root,
                        hermes_agent_root=ha_root,
                        run_hermes_pytest=use_pt,
                    )
                    done_mv = self._populate_and_finalize(replay_mv, mv_suites, [], t0_mv)
            except Exception as exc:
                replay_mv.status = RunStatus.ERROR
                replay_mv.failure_kind = FailureKind.HARNESS_ERROR
                replay_mv.error_count = 1
                replay_mv.first_failure = f"Reproduce error: {exc}"[:300]
                done_mv = self.store.finalize_run(replay_mv)
            return {
                "source_run_id": orig.run_id,
                "repro_run_id": done_mv.run_id,
                "status": done_mv.status,
                "pass_count": done_mv.pass_count,
                "fail_count": done_mv.fail_count,
                "first_failure": done_mv.first_failure,
                "evidence_dir": done_mv.evidence_dir,
            }
        # Replay recorded suites directly without rediscovery if suites were persisted
        if orig.suites:
            rid = new_run_id()
            now = utc_now_iso()
            replay = RunRecord(
                run_id=rid,
                project=orig.project,
                project_root=str(resolved),
                revision=_fallback_git_metadata(resolved)[0],
                profile=f"repro:{orig.profile}",
                status=RunStatus.RUNNING,
                commands=[list(s.command) for s in orig.suites],
                created_at=now,
                started_at=now,
                environment=host_environment_snapshot(),
            )
            self.store.create_run(replay)
            ev_dir = self.store.run_dir(rid)
            t0 = time.monotonic()
            trusted = self.store.is_manifest_trusted(resolved)
            try:
                with self.store.hold_run_lock(rid):
                    results: List[SuiteResult] = []
                    for prev_s in orig.suites:
                        spec_s = SuiteSpec(
                            name=prev_s.name,
                            adapter=prev_s.adapter,
                            profile=prev_s.profile,
                            command=list(prev_s.command),
                            cwd=prev_s.cwd,
                            needs_tmux=bool(prev_s.isolation_meta.get("needs_tmux", False)),
                            needs_pty=bool(prev_s.isolation_meta.get("needs_pty", False)),
                            allow_network=bool(prev_s.isolation_meta.get("allow_network", False)),
                            allow_repo_write=bool(prev_s.isolation_meta.get("allow_repo_write", False)),
                        )
                        sres, _ = self._execute_suite_fn(
                            spec_s,
                            project_root=resolved,
                            evidence_dir=ev_dir,
                            run_id=rid,
                            trusted_manifest=trusted,
                        )
                        results.append(sres)
                    done = self._populate_and_finalize(replay, results, [], t0)
            except Exception as exc:
                replay.status = RunStatus.ERROR
                replay.failure_kind = FailureKind.HARNESS_ERROR
                replay.error_count = 1
                replay.first_failure = f"Reproduce error: {exc}"[:300]
                done = self.store.finalize_run(replay)
            return {
                "source_run_id": orig.run_id,
                "repro_run_id": done.run_id,
                "status": done.status,
                "pass_count": done.pass_count,
                "fail_count": done.fail_count,
                "first_failure": done.first_failure,
                "evidence_dir": done.evidence_dir,
            }

        rerun = self.run_sync(resolved, profile=orig.profile, enforce_scope=enforce_scope)
        return {
            "source_run_id": orig.run_id,
            "repro_run_id": rerun.run_id,
            "status": rerun.status,
            "pass_count": rerun.pass_count,
            "fail_count": rerun.fail_count,
            "first_failure": rerun.first_failure,
            "evidence_dir": rerun.evidence_dir,
        }
