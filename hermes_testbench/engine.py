"""Single orchestration engine shared by CLI, detached workers, MCP server, and TUI.

The engine owns the *run lifecycle* and nothing else: ``discovery`` decides what to run,
``adapters`` decide how, ``isolation`` decides where, ``store`` decides what is true. Four
surfaces call these same methods, so everything documented here holds for ``testbench run``,
``testbench _worker``, the MCP server and the TUI alike.

Run status machine — one writer per run, enforced by the ``runs/<id>/run.lock`` flock::

    QUEUED --(worker takes lock)--> RUNNING --> PASS | FAIL | SKIP | BLOCKED
                                            \\-> TIMEOUT | CANCELLED | ERROR

* :meth:`run_sync` commits the row ``RUNNING`` before taking the lock; finalization may
  write without it (the store module states the one-writer rule precisely).
* :meth:`spawn_async` creates the row ``QUEUED`` and returns; a detached ``_worker`` takes
  the lock later, flips ``RUNNING``, executes, finalizes.
* A row left ``QUEUED``/``RUNNING`` with an *unheld* lock was abandoned (SIGKILL, ``^C`` on a
  foreground run, host power loss). ``TestBenchEngine.__init__`` reconciles it to
  ``ERROR``/``abandoned_worker`` and keeps it: that record is legitimate history. Nothing
  here deletes or hides an interrupted run — a half-finished run that disappears makes "no
  data" look like "no problem".
* Cancellation is cooperative first (``cancel.flag`` in the evidence dir, polled at most
  every 50ms by the isolation reader loop) and terminal second (fingerprinted
  SIGTERM→SIGKILL of the run's *own* process group). It never signals the caller's group.

Aggregation runs bottom-up: per-suite status → :func:`_aggregate_run_status` → run status.
Status writes are last-writer-wins in the store (``ON CONFLICT DO UPDATE``), and the one
fence that keeps a cancelled run from being rewritten as ``PASS`` is the ``cancel.flag``
re-check inside :meth:`TestBenchEngine._populate_and_finalize`.
"""

from __future__ import annotations

import os
import math
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
from .flake import FlakeReport, MIN_ATTEMPTS, build_report
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
    """Resolve ``target`` or raise :class:`AuthorizationError`.

    Order is the security property, so it is written top-down: (1) resolve strict and refuse
    anything that cannot be resolved, (2) refuse the enumerated Hermes runtime paths
    (``skills``, ``plugins``, ``memories``, ``sessions``, ``profiles``, ``mcp-tokens``,
    ``logs``, ``state.db``, credentials) *before* any allow rule can match, (3) honour an
    explicit registration, (4) honour ``allowed_roots``, (5) refuse every remaining path under
    ``~/.hermes`` except ``artifacts``. Registration therefore widens the allowlist for
    project trees but can never unlock the enumerated runtime paths.

    The ``p.exists() or not p.suffix`` filter only skips enumerated *files* that are absent;
    their parent directory deny and the step-5 blanket catch the rest, which is why a missing
    ``auth.json`` is not a hole. ``allowed_roots`` is passed by callers that construct an
    engine with a fixed scope (MCP) rather than reading the environment variable.
    """
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
    on_output: Optional[Callable[[str], None]] = None,
) -> Tuple[SuiteResult, Optional[Dict[str, Any]]]:
    """Degraded single-pass executor used only when ``hermes_testbench.adapters`` is absent.

    This path is the reason a stripped install still runs: it enforces the two pre-spawn
    gates itself (billable, untrusted-manifest network/repo-write), then runs the argv under
    isolation and reduces "exit 0" to one ``PASS``/``FAIL`` case with no structured parsing.

    The ``try`` spans both the import *and* the call: ``ImportError`` raised from inside
    ``execute_suite`` is indistinguishable here from "adapters not installed", so such a
    suite silently falls through to this bare executor and runs a second time. Narrow the
    ``try`` to the import if that is ever fixed; it is a defect, not a design choice.

    A blocked suite returns ``BLOCKED`` with ``failures=[]``; which gate fired is then only
    recoverable from ``summary_excerpt``, unlike the adapter path which records a
    ``FailureKind``.
    """
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
            on_output=on_output,
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
        on_output=on_output,
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


def _merge_environment(
    base: Dict[str, Any], extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Merge caller-supplied environment metadata over the host snapshot.

    Used by ``flake_hunt`` to tag each attempt with ``environment['flake']``
    while keeping every host fact. Nested ``flake`` dicts merge instead of
    replacing each other so future callers can add keys without clobbering the
    group/attempt/repeats contract.
    """
    env = dict(base)
    for key, value in (extra or {}).items():
        if key == "flake" and isinstance(value, dict):
            merged = dict(env.get("flake") or {})
            merged.update(value)
            env["flake"] = merged
        else:
            env[key] = value
    return env


def _blocked_failure_kind(suites: List[SuiteResult]) -> str:
    for s in suites:
        if s.status != RunStatus.BLOCKED:
            continue
        for f in s.failures:
            if f.kind == FailureKind.UNTRUSTED_MANIFEST:
                return FailureKind.UNTRUSTED_MANIFEST
    return FailureKind.BLOCKED_POLICY


def _failed_suite_kind(suites: List[SuiteResult]) -> str:
    """Pick the failure kind that best explains a mixed set of failing suites.

    ``assertion`` wins over ``sandbox_denied`` on purpose: a sandbox denial usually surfaces
    as an assertion or spawn error deeper in, and reporting the mechanism instead of the
    observed failure sends the reader to the wrong place. ``harness_error`` is only inferred
    when a failing suite counted errors and no failures — otherwise the absence of a kind
    defaults to ``assertion``, which is the honest "a suite failed, we do not know why".
    """
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
    """Stamp an operator-supplied mutation gate onto the selected suites (in place).

    A CLI gate is trusted by construction, unlike ``mutation_score_gate`` in a manifest,
    which only counts once the manifest itself is trusted. The value is validated as a
    finite fraction here — out of range would otherwise make every mutation suite pass or
    fail for the wrong reason, and the failure would look like a test result.
    """
    if mutation_gate is None:
        return
    gate = float(mutation_gate)
    if math.isnan(gate) or not 0.0 <= gate <= 1.0:
        # A gate is a fraction of caught mutants; anything else would silently make every
        # mutation suite pass or fail, so it is rejected before any suite is stamped.
        raise ValueError(
            f"mutation gate must be a fraction between 0 and 1, got {mutation_gate!r}"
        )
    for suite in suites:
        suite.mutation_score_gate = gate
        suite.mutation_gate_from_cli = True


def _aggregate_run_status(suites: List[SuiteResult]) -> Tuple[str, Optional[str]]:
    """Collapse per-suite statuses into one run status and failure kind.

    Strict precedence, worst-wins: ``CANCELLED`` > ``ERROR`` > ``TIMEOUT`` > ``FAIL`` >
    ``BLOCKED`` > ``SKIP``, with ``PASS`` last. Rationale for the top of that order: a
    cancelled run must never read as a pass (it fenced the whole attempt), and an
    ``ERROR`` — a harness fault — must not be reported as an ordinary assertion failure,
    because the suite's verdict is unknown. ``TIMEOUT`` outranks ``FAIL`` for the same
    reason: "did not finish" is not "failed".

    ``BLOCKED`` is checked twice on purpose. An all-blocked set is ``BLOCKED``; a mixed set
    with at least one ``PASS`` also ends ``BLOCKED`` (something was not evaluated — that is
    not a green run); a ``BLOCKED``+``SKIP`` set takes the ``not any(PASS)`` branch and lands
    in the same place. The ``SKIP``-only and empty cases are ``SKIP`` with no failure kind.
    """
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


MAX_FLAKE_REPEATS = 100


class TestBenchEngine:
    """Owns run lifecycle for every surface: CLI, ``_worker``, MCP, TUI.

    Collaboration: ``store`` is authoritative (rows, evidence dirs, run locks, trust registry);
    ``discover_fn`` produces the suite list; ``execute_suite_fn`` runs one suite and returns
    ``(SuiteResult, usage)``. Both are injectable for tests, which is why the shapes of those
    callables are effectively part of this module's contract.

    Constructing an engine calls ``store.reconcile_orphans()`` as a side effect: any run row
    left ``QUEUED``/``RUNNING`` whose ``run.lock`` is unheld is rewritten to ``ERROR`` /
    ``abandoned_worker``. Every surface constructs an engine, so a crashed worker is accounted
    for by whoever looks at the store next — including a TUI that only renders. That is why
    reconciliation happens here and not in ``run_sync``: readers must not be able to see an
    eternal ``RUNNING``, and the record must survive rather than be deleted.

    ``close()`` only closes a store this engine created; an injected store belongs to the
    caller (the CLI wraps one store around one engine; MCP and the TUI each own theirs).
    """

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
        """Filter suites by explicit names, else by profile.

        Naming suites overrides the profile filter entirely (including ``profile="all"``),
        so ``run X --suite a --suite b`` cannot be silently widened by a profile. An empty
        result is not an error at this layer: ``run_sync`` finalizes ``SKIP`` and the CLI
        turns "you named a suite that does not exist" into a non-zero exit.
        """
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
        """Sum the suite results, derive run-level fields, and finalize the row.

        Everything here is derivation from ``suite_results`` plus one fence:

        * counts are plain sums; ``duration_ms`` is wall-clock from ``t0`` (monotonic), which
          is why a run can take longer than the sum of its suites (discovery, teardown).
        * the **cancel fence** (``cancel.flag`` on disk, or ``CANCELLED`` already in the
          store) overrides the aggregate. Without it a worker that noticed the flag only
          after its last suite finished would rewrite a cancelled run as ``PASS``, because
          the store's upsert is last-writer-wins.
        * ``exit_code`` is the first non-zero suite exit, ``0`` when every suite exited 0, and
          ``None`` for a cancelled/timed-out run with no non-zero code — a cancelled run has
          no meaningful exit status and inventing one would invite scripting on it.
        * ``first_failure`` prefers a structured ``FailureRecord`` and falls back to the first
          line of a non-pass suite's summary, so a ``BLOCKED`` or ``TIMEOUT`` run still says
          something useful.
        * ``repro_instructions`` prefers the non-passing suites, then everything, deduped in
          order; ``&&``-joined because the field is read by humans and by ``repro.sh``.
        * ``model_usage`` is ``None`` unless a suite actually measured tokens; nothing here
          estimates usage.
        """
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
        env_meta: Optional[Dict[str, Any]] = None,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> RunRecord:
        """Run the selected suites now, in this process, and return the finalized record.

        Sequence, and why it is in this order:

        1. resolve + discover *outside* the lock. Discovery can be slow (git, manifest parse)
           and must not be able to hold a run's lock; a concurrently-granted cancel then
           applies to a run that has no suite rows yet, which the flag check below absorbs.
        2. ``store.create_run`` writes ``RUNNING`` and creates ``runs/<id>/`` (``0700``).
           From here on, a crash leaves a reconcilable row rather than nothing.
        3. everything else happens under ``hold_run_lock``. Only one process may execute a
           run id; a second holder gets ``EWOULDBLOCK`` and the exception path finalizes
           ``ERROR``/``harness_error`` instead of racing.
        4. per suite: re-check ``cancel.flag`` *before* spawning, then execute. A cancel that
           lands mid-suite is handled inside isolation (the reader loop polls the same flag),
           so the flag is read in two places on purpose — before, to avoid starting work, and
           during, to stop it.
        5. ``_populate_and_finalize`` aggregates and writes ``run.json``/``junit.xml``/
           ``repro.sh`` plus the DB row.

        Failure handling: any exception from discovery, execution or the store is caught and
        the run is finalized ``ERROR``/``harness_error`` with ``first_failure`` naming the
        exception. Swallowing it is deliberate — the alternative is an exception escaping into
        a TUI render loop or an MCP frame, and an unfinalized row that reads as ``RUNNING``
        forever. ``KeyboardInterrupt`` is *not* caught (it is a ``BaseException``): it unwinds,
        the lock is released, ``run_isolated_command``'s ``finally`` still kills the suite, and
        the row stays ``RUNNING`` until the next engine reconciles it to ``ERROR``. That
        interrupted record is correct history and is never rewritten to look deliberate.

        ``worker_pid``/``worker_pgid``/``worker_start_time`` are persisted at each spawn so a
        *different* process (``testbench cancel``) can signal the suite group it does not own.
        They are cleared in memory after each suite but not in the store: a stale pointer is
        harmless because the start-time fingerprint makes a recycled pid unmatchable, while a
        missing pointer would make cancellation impossible.
        """
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
            environment=_merge_environment(host_environment_snapshot(), env_meta),
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
                            **({"on_output": on_output} if on_output is not None else {}),
                        )
                    except TypeError:
                        # Retry once without on_output, for injected execute_suite callables
                        # that predate the streaming kwarg (test doubles, older adapters).
                        # Broad by necessity and therefore hazardous: a TypeError raised
                        # *inside* the call is indistinguishable from a signature mismatch
                        # here, and the suite is then executed a second time, bare, with no
                        # adapter parsing and no streaming.
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
                    # Cleared in memory only: the store keeps the last child's (pid, pgid,
                    # start_fp) until the next suite overwrites it. Deliberate — a stale
                    # pointer is unmatchable thanks to the fingerprint, whereas dropping it
                    # would leave an in-flight run uncancellable.
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

    @staticmethod
    def _resolve_flake_repeats(
        repeats: Optional[int], selected: Sequence[SuiteSpec]
    ) -> int:
        """Resolve attempt count: explicit overrides, else ``SuiteSpec.repeat_runs``."""
        if repeats is None:
            if not selected:
                return 1
            return max([MIN_ATTEMPTS] + [int(s.repeat_runs) for s in selected])
        try:
            count = int(repeats)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"flake repeats must be an integer, got {repeats!r}") from exc
        if count < 1 or count > MAX_FLAKE_REPEATS:
            raise ValueError(f"flake repeats must be between 1 and {MAX_FLAKE_REPEATS}, got {count}")
        return count

    def flake_hunt(
        self,
        target: Path,
        *,
        repeats: Optional[int] = None,
        profile: str = "unit",
        suite_names: Optional[Sequence[str]] = None,
        allow_billable: bool = False,
        max_tokens: int = 0,
        timeout_s: Optional[float] = None,
        enforce_scope: bool = False,
        group_id: Optional[str] = None,
    ) -> FlakeReport:
        """Execute the selected suites ``repeats`` times and classify flakiness.

        Every attempt is a complete :meth:`run_sync` execution: its own
        independent ``run_id``, its own ``runs/RUN_ID/`` evidence directory, and
        an ``environment['flake']`` tag of ``{group_id, attempt, repeats}``.
        A later passing retry never touches an earlier failed attempt; the
        failed row, its evidence, and its ``run.json`` stay in the store.
        """
        resolved = self.resolve_target(target, enforce_scope=enforce_scope)
        spec = self._discover_fn(resolved, store=self.store)
        selected = self._select_suites(spec, profile=profile, suite_names=suite_names)
        attempts = self._resolve_flake_repeats(repeats, selected)
        gid = (group_id or "").strip() or f"fg_{new_run_id()}"

        runs: List[RunRecord] = []
        for attempt in range(1, attempts + 1):
            runs.append(
                self.run_sync(
                    resolved,
                    profile=profile,
                    suite_names=suite_names,
                    allow_billable=allow_billable,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                    env_meta={
                        "flake": {
                            "group_id": gid,
                            "attempt": attempt,
                            "repeats": attempts,
                        }
                    },
                )
            )
        return build_report(
            runs,
            expected_repeats=attempts,
            suite_names=[s.name for s in selected] or None,
        )

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
        """Create a ``QUEUED`` run and hand it to a detached ``_worker`` process.

        The parent does only four things: allocate the run id, write the row (``QUEUED``, with
        the suite argv already recorded so the TUI can show what will run), spawn the worker,
        and persist the worker's ``(pid, pgid, start_time)``. Execution happens in the child,
        so the caller may exit immediately — that is the entire point of the split, and why
        this surface is also what MCP uses.

        Details that are easy to break:

        * ``start_new_session=True`` makes the worker's pgid equal to its pid, which is what
          makes the recorded ``worker_pgid`` a killable group rather than the caller's.
        * The worker must import *this* package, so ``PYTHONPATH`` is prefixed with
          ``repo_package_root()`` and ``cwd`` is that root. A source tree, a site-packages
          dir and a zipapp all work; a zipapp cannot be a ``cwd``, hence the ``is_dir`` test.
        * ``TESTBENCH_HOME`` is exported so the worker opens the same store the parent wrote
          the row into; without it the worker would use its own default home and never find
          the run.
        * ``proc.wait`` runs in a daemon thread purely to reap the direct child if it exits
          early; the run's own lifecycle is driven by the worker's DB rows, not by this handle.
        * The fingerprint is sampled *after* ``Popen``; a child that died during import
          fingerprints as ``None``, and ``cancel`` then refuses to signal it (no proof of
          ownership) and finalizes the row instead.

        ``wait_up_to_s`` bounds how long the caller blocks waiting for a terminal status
        (clamped to 0–30s). Returning a still-``QUEUED``/``RUNNING`` record is normal: MCP
        and the TUI poll the store, the CLI prints an id.
        """
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
        """Detached worker entry point (``testbench _worker``): execute a ``QUEUED`` run.

        Exit codes are a harness contract: ``0`` when the run finalized ``PASS`` or ``SKIP``,
        ``1`` for anything else (including a worker-side exception), ``2`` when the run id is
        unknown to the store — the last one means the worker was pointed at the wrong home.

        Two behaviors here that the foreground path does not have:

        * ``SIGTERM``/``SIGINT`` install a handler that writes ``cancel.flag`` and returns.
          This is what makes ``cancel()``'s fingerprinted SIGTERM a *cooperative* request
          rather than a kill: the worker keeps running long enough to let isolation stop the
          suite group and write the evidence. A second SIGKILL from the canceller, or any
          crash before the handler runs, is what ``reconcile_orphans`` exists for.
        * The worker records its *own* pid/pgid while running suites and never records the
          suite child's — there is no ``on_spawn`` wiring on this path. So a cancel of an
          async run signals the worker's group, and the suite group is only stopped if the
          worker survives long enough to notice the flag. (Documented, not fixed here.)

        ``RUNNING`` is written after the lock is acquired, so a row can never claim to be
        executing while no one holds its lock. It is written unconditionally, though: a
        ``CANCELLED`` row for a run cancelled before the worker started can be flipped back
        to ``RUNNING`` for as long as this worker lives. The finalize step re-reads the flag
        and lands on ``CANCELLED`` again, so the end state is right and the intermediate state
        is a known monotonicity break.
        """
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
        """Ask a run to stop, then make it stop. Idempotent for terminal runs.

        Three escalating steps, each narrower than the last:

        1. write ``cancel.flag`` (``0600``) into the run's evidence dir. Every execution path
           polls it: the engine between suites, ``run_isolated_command`` every ≤50ms inside a
           suite. Cooperative, so evidence and teardown still complete.
        2. wait up to 1.5s for the run to reach a terminal status while its lock is held,
           then **signal only a verified owned group**. All of these must hold:
           the run is still non-terminal, its ``run.lock`` is still held, ``worker_pgid > 1``,
           it is not *our* pgid, and ``worker_start_time`` is recorded and matches. That
           conjunction is ADR-0005's R1 fix — before it, a run whose ``worker_pgid`` had been
           recorded as ``os.getpgrp()`` could SIGTERM the caller's own shell job.
        3. finalize ``CANCELLED`` unconditionally if the row is still non-terminal.

        Step 3 does not require the lock, so this method can write ``CANCELLED`` while a
        worker still owns the run. That is intentional (a cancel must never be unable to
        complete because the worker is wedged) and it is why the worker re-checks the flag
        and why ``_populate_and_finalize`` fences on it. The cost is a visible
        ``CANCELLED → RUNNING → CANCELLED`` flap when a queued worker starts just after a
        cancel; the end state is correct.

        ``KeyError`` for an unknown id — the CLI turns that into a non-zero exit, and that
        check is also what keeps this from creating evidence dirs for arbitrary strings.
        """
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
        """Re-execute a historical run as a *new* run, or just print its command.

        Three replay paths, tried in order, because the recorded data differs by age:

        1. ``print_only`` — return the stored repro command (or rebuild it from
           ``orig.commands``) without touching the target. Always safe, always cheap.
        2. agent-messaging verification runs — they are not suite-argv runs at all, so they
           are replayed through ``run_agent_messaging_verification`` with the artifact root
           the *original* recorded (``environment['messaging_artifact_root']``; older runs
           fall back to comparing the recorded project root against the default).
        3. runs with persisted suites — replay the recorded argv verbatim, without
           rediscovery, so a changed ``.testbench.toml`` cannot silently alter the repro.
           Isolation flags (``needs_tmux``/``needs_pty``/``allow_*``) are recovered from each
           suite's ``isolation_meta``, which is why adapters must record them there; trust is
           re-read from the store (a repro of a now-untrusted manifest is blocked, correctly).

        A run with no persisted suites falls back to a plain ``run_sync`` by profile — the
        weakest path, and the only one where discovery can pick different suites.

        Invariants: the source record is never mutated, every path allocates a fresh
        ``new_run_id()`` and evidence dir, and the new profile is prefixed ``repro:``. Only the
        messaging path strips an existing prefix (``base_profile``), so a repro of a
        suite-replay record accumulates ``repro:repro:...`` — cosmetic, but do not "fix" one
        path without checking the other. A failure inside the replay is captured as
        ``ERROR``/``harness_error`` on the new record and returned. An unknown id raises
        ``KeyError``; ``resolve_target`` runs before the replay's ``try``, so a moved or
        deleted ``project_root`` can still raise ``OSError``.
        """
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
