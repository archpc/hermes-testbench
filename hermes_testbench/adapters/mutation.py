"""Mutation-testing adapters: disposable staging, external tools only, explicit gating.

Phase 3 contract
----------------
* The mutation tool is a third-party binary (``cargo-mutants`` or ``mutmut``); TestBench
  never generates mutants itself.
* The tool runs against a disposable copy of the project staged under
  ``scratch_root / f"tbmut-{nonce}"``. The real checkout is never handed to the tool;
  afterwards it is compared against a snapshot taken before staging (``git status
  --porcelain`` plus a sentinel scan) and any difference ends the suite ERROR /
  harness_error. See ``_untouched_violations`` for what that check can and cannot see.
* Seatbelt confinement is mandatory. The staged copy is handed to the sandbox as an
  internal extra writable directory (``run_isolated_command(extra_writable_dirs=...)``),
  which keeps ``allow_repo_write=False`` for the real repository.
* All mutant outcomes land in exactly five buckets exposed through
  ``SuiteResult.metrics["mutation"]``: ``caught``, ``survived``, ``invalid`` (unviable /
  compile failure), ``timeout`` and ``inconclusive`` (baseline failure, no tests, skipped,
  interrupted, not checked, harness error).
* Surviving mutants are findings, not failures: the suite passes unless an operator gate
  (``--mutation-gate``) or a *trusted* manifest ``mutation_score_gate`` says otherwise.

Gate order (``run_mutation_suite``) — each one is cheap and each one exists because the
alternative silently ran unconfined or silently passed:

A. a manifest-declared score gate from an *untrusted* manifest is refused (BLOCKED):
   the gate that can fail a build may not come from a file the agent under test writes.
   An operator ``--mutation-gate`` is stamped ``mutation_gate_from_cli`` and is trusted
   on its own.
B. ``can_apply_seatbelt()`` must be true (BLOCKED / ``sandbox_denied``). There is no
   environment-variable escape hatch — see the comment on the gate.
C. the tool must resolve (SKIP, never PASS): "could not measure" is not "clean".

Lifecycle of one run: snapshot ``git status`` → create a private stage dir → copy
sources → resolve the suite cwd *inside* the copy twice (lexical before staging, real
path after, because symlinks are copied verbatim) → one confined spawn (plus a second,
bounded one for ``mutmut results``) → parse → ``finally`` remove the stage with
``_safe_rmtree_no_symlinks`` → compare the real checkout → decide the verdict.
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
    """``copytree`` ignore hook: drop caches, VCS state, venvs and prior mutant output.

    Skipping ``.git`` is what makes the stage a source tree rather than a clone, and
    skipping prior artefacts (``mutants.out``, ``.mutmut-cache``) prevents an old run's
    verdicts from being read as this run's.
    """
    return [name for name in names if name in _STAGE_SKIP_NAMES]


def stage_project_tree(project_root: Path, stage_dir: Path) -> Path:
    """Copy the project sources into a fresh disposable directory and return it.

    ``exist_ok=False`` is load-bearing: staging into a directory that already exists
    would mix trees, and the nonce in ``run_mutation_suite`` exists to make that
    impossible. Mode 0700 keeps the copy private to this user.

    ``symlinks=True`` copies links *as links*, so a link inside the project can still
    point at the real checkout from inside the stage. That is not a bug in the copy —
    it is why ``run_mutation_suite`` re-checks the staged cwd with ``realpath`` after
    this returns, and why links are never followed here.
    """
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
    """Reasons the real checkout is no longer pristine after a mutation run.

    Two checks, both after the stage is deleted: a sentinel scan for artefacts a tool
    would drop in the project root, and a ``git status --porcelain`` diff against the
    pre-run snapshot.

    Limits, stated so nobody over-trusts a PASS: ``git status`` says nothing about
    ignored paths (``target/``, a ``.mutmut-cache`` in ``.gitignore``), nor about a
    file that was modified and restored, nor about edits that leave the index clean
    by definition. When the project is not a git repo both snapshots are ``None`` and
    the comparison silently degrades to the sentinel scan alone. The real guarantee is
    the sandbox: the tool is given write access to the stage only.
    """
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
    """Write a bounded, redacted mutation report under ``evidence_dir/mutation/``.

    Stored separately from the generic suite log so a mutant list survives stream
    truncation. Bounded twice (400 cases, 8 MiB) and redacted per field via
    ``redact_secrets``+``sanitize_terminal_text``: mutant names embed source text, which
    can embed a secret. Returns the path, or ``None`` when the directory is unwritable —
    a missing report is never allowed to fail the suite, because the verdict is already
    in the store.
    """
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
    """``caught / (caught + survived)``, or ``None`` when no viable mutant completed.

    ``None`` is the honest answer for "nothing was measured" and the gate turns it into
    a failure; returning 0.0 or 1.0 here would let an empty run look decisive.
    Timeouts and unviable mutants are excluded — ADR-0008 rejects counting a hang as a
    kill because it rewards a suite that never asserts.
    """
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
    """The ``metrics["mutation"]`` payload: five buckets, tested, score, gate, tool.

    ``gate`` is echoed so a stored run explains its own verdict without the manifest,
    and ``baseline_failed`` because a FAIL caused by a broken baseline looks identical
    to a FAIL caused by survivors in the counts alone.
    """
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
    """Project the five buckets onto the generic count fields.

    ``caught``→pass, ``survived``→fail, and invalid/timeout/inconclusive all collapse
    into ``skip`` (they are "not a verdict"), which is why ``fail_count == 0`` alone
    does not mean the suite proved anything — read ``metrics["mutation"]``.
    """
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


def _relativize_suite_cwd(suite_cwd: Optional[str], project_root: Path) -> Path:
    """Return the suite cwd as a path relative to ``project_root``.

    ``TestBenchEngine.reproduce`` replays recorded suites by feeding ``SuiteResult.cwd``
    straight back into a new ``SuiteSpec``, and that recorded value is absolute (it is
    ``str(project_root)`` for a mutation suite). Joining an absolute path onto
    ``stage_dir`` discards the stage entirely, so a recorded, trustworthy cwd inside the
    project must be rewritten as a relative path (``cwd == str(project_root)`` -> ``.``).

    Only a path that is lexically inside ``project_root``, is not itself a symlink, and
    whose resolved form stays inside the resolved project root is rewritten. Everything
    else is returned untouched so the containment check below rejects it and nothing is
    spawned.
    """
    rel = Path(suite_cwd or ".")
    if not rel.is_absolute():
        return rel
    try:
        lexical = rel.relative_to(project_root)
    except ValueError:
        return rel
    real_root = Path(os.path.realpath(project_root))
    if rel.is_symlink() or not Path(os.path.realpath(rel)).is_relative_to(real_root):
        return rel
    if not rel.resolve().is_relative_to(project_root.resolve()):
        return rel
    return lexical


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
    """Stage a disposable copy, run one mutation tool under Seatbelt, classify outcomes.

    Verdict precedence after the tool has run (first match wins, so a later check can
    never mask an earlier one):

    1. checkout violations → ERROR (harness wrote outside the stage);
    2. ``cancelled`` → CANCELLED;
    3. ``timed_out`` → TIMEOUT (a partial tally is *not* reported as a score);
    4. ``outcome.baseline_failed`` → FAIL with an assertion-kind baseline record: a
       broken baseline makes every mutant verdict meaningless;
    5. nothing structured / nothing tested and nothing inconclusive → ERROR
       (harness error), never a pass;
    6. score gate (operator or trusted manifest): ``None`` score fails as "cannot be
       evaluated", below-gate fails with the surviving mutants as the excerpt;
    7. otherwise PASS — surviving mutants are findings in ``metrics``/evidence.

    ``result.fail_count`` is the survivor count and ``result.pass_count`` the caught
    count (``_bucket_counts``): a mutation suite reads as "how many mutants did the
    suite notice", not "how many tests ran". ``metrics["mutation"]["tested"]`` is the
    denominator the gate uses, and it excludes invalid/timeout/inconclusive by design.

    Replay is deliberately weaker than the original run: ``engine.reproduce`` rebuilds
    a ``SuiteSpec`` without ``mutation_score_gate``/``mutation_gate_from_cli``, so a
    replayed gate failure reports PASS/FAIL purely on mutants. Reproducing a *gate*
    verdict is not supported; the recorded summary is the evidence for it.
    """
    adapter = suite.adapter
    project_root = Path(project_root).resolve()
    evidence_dir = Path(evidence_dir)
    relative_cwd = _relativize_suite_cwd(suite.cwd, project_root)
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
    #
    # There is deliberately *no* environment-variable escape hatch here. An inherited
    # ``TESTBENCH_SEATBELT_ACTIVE`` marker only proves that some outer sandbox exists; it
    # does not let ``run_isolated_command`` install its own per-stage Seatbelt profile
    # (cwd=stage_dir, allow_repo_write=False). Honouring it would run the mutation tool
    # unconfined on any host without ``sandbox-exec`` (i.e. every Linux box), so the
    # platform capability is the only accepted evidence.
    if not can_apply_seatbelt():
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
        # The declared suite cwd is only ever resolved *inside* the disposable copy. An
        # absolute path or a ``..`` traversal would otherwise pick an arbitrary working
        # directory for the mutation tool, so it is rejected before anything is spawned.
        stage_root = stage_dir.resolve()
        workdir = (stage_root / relative_cwd).resolve()
        if workdir != stage_root and not workdir.is_relative_to(stage_root):
            raise RuntimeError(
                f"suite cwd escapes the staged copy: {relative_cwd} -> {workdir}"
            )

        try:
            stage_project_tree(project_root, stage_dir)
        except OSError as exc:
            raise RuntimeError(
                f"could not stage a disposable copy for mutation testing: {exc}"
            ) from exc

        if not workdir.is_dir():
            raise RuntimeError(f"staged suite cwd is missing: {relative_cwd}")

        # Defense in depth: ``stage_project_tree`` copies symlinks verbatim, so a
        # symlinked suite cwd inside the staged copy can still point back at the real
        # checkout. The lexical containment check above runs *before* the copy and cannot
        # see links that the copy itself created, so re-verify the staged entry on disk
        # right before spawning.
        staged_workdir = stage_dir / relative_cwd
        staged_root_real = Path(os.path.realpath(stage_dir))
        if staged_workdir.is_symlink() or not Path(
            os.path.realpath(staged_workdir)
        ).is_relative_to(staged_root_real):
            raise RuntimeError(
                "staged suite cwd is not a real directory inside the staged copy "
                f"(symlink escape): {relative_cwd}"
            )

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
    # Runs after the ``finally`` above deleted the stage, so a sentinel here can only have
    # been written into the *real* checkout by something that escaped confinement.
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
