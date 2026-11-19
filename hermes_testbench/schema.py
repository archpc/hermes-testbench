"""Canonical pure data contracts for Hermes TestBench.

Every other module in the package sits downstream of this one (ADR-0004):
``schema`` -> ``{sanitize, reporting, store}`` -> ``isolation`` ->
``{adapters, engine}`` -> ``{cli, mcp_server, tui}``. Nothing here may import
``re``, ``xml``, ``sqlite3``, ``subprocess``, ``pty``, ``termios``,
``select`` or ``fcntl``; ``tests/test_schema.py`` asserts that set by walking
the AST, so adding one of those imports fails the suite rather than review.
The only ambient dependencies are ``threading`` (to make the clock and the ID
generator monotonic under concurrency) and ``os.urandom`` (ID entropy) — no
filesystem, process, or network calls.

``SCHEMA_VERSION`` is the on-disk contract, not a cosmetic number:
``TestBenchStore`` sets it as ``PRAGMA user_version`` when it creates
``state.db`` and refuses to open any database whose ``user_version`` differs
(``SchemaMismatchError``). There is no migration code. Bumping this constant
therefore makes every existing store unreadable to the new release, which is
why OPERATING-GUIDE ties schema bumps to an explicit migration step and
forbids switching a release across one.

Run status vocabulary (``RunStatus``) is partitioned: ``ACTIVE`` = {QUEUED,
RUNNING}, ``TERMINAL`` = everything else. Persistence and reconciliation both
lean on that partition, so a new intermediate state must be added to exactly
one of the two sets — a status in neither is invisible to pruning, to the TUI
refresh, and to ``reconcile_orphans``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import os
import threading
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


class RunStatus:
    """Lifecycle vocabulary; persisted verbatim into ``runs.status``.

    ``ACTIVE`` vs ``TERMINAL`` is load-bearing, not decorative:
    ``reconcile_orphans`` only inspects ACTIVE rows, ``prune_runs`` refuses to
    delete ACTIVE rows, the TUI tails logs only for ACTIVE rows, and
    ``spawn_async`` polls until the row leaves ACTIVE. The aggregate in
    ``engine._aggregate_run_status`` has no "unknown" state on purpose: every
    run ends in exactly one of these strings or stays ACTIVE until reconciled.
    """
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    BLOCKED = "BLOCKED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"

    TERMINAL = frozenset({PASS, FAIL, SKIP, BLOCKED, TIMEOUT, CANCELLED, ERROR})
    ACTIVE = frozenset({QUEUED, RUNNING})
    ALL = TERMINAL | ACTIVE


class ProfileName:
    """Profile names select suites (``SuiteSpec.profile``) and group history.

    ``STANDARD`` is the allowlist a manifest's ``profile`` must fall inside,
    append-only in intent: mutation before agent-eval is deliberate (the
    expensive, rarely-wanted eval stays last). The TUI picker uses its own
    ``default_profiles`` tuple, which currently omits MUTATION, so the allowlist
    and the picker are not identical.
    """
    UNIT = "unit"
    INTEGRATION = "integration"
    TERMINAL = "terminal"
    SECURITY = "security"
    REGRESSION = "regression"
    BENCHMARK = "benchmark"
    AGENT_EVAL = "agent-eval"
    MUTATION = "mutation"

    STANDARD = (UNIT, INTEGRATION, TERMINAL, SECURITY, REGRESSION, BENCHMARK, MUTATION, AGENT_EVAL)


class FailureKind:
    """Why a run or suite did not pass; persisted into ``runs.failure_kind``.

    These are closed-vocabulary strings consumed by the TUI, by JUnit
    derivation (``reporting.render_junit_xml`` maps several of them to
    ``<error>`` rather than ``<failure>``) and by ``testbench status``. Values
    are also written as free-form ``FailureRecord.kind`` by adapters, which is
    why the report layer compares against literals instead of this class in
    places — keep the string values stable; adding an alias silently changes
    the JUnit shape.
    """
    ASSERTION = "assertion"
    HARNESS_ERROR = "harness_error"
    SPAWN_ERROR = "spawn_error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BLOCKED_POLICY = "blocked_policy"
    ABANDONED_WORKER = "abandoned_worker"
    UNTRUSTED_MANIFEST = "untrusted_manifest"
    SANDBOX_DENIED = "sandbox_denied"


@dataclass
class SuiteSpec:
    """A discovered, not-yet-executed suite: the unit of selection and gating.

    ``custom_from_manifest`` records provenance, and provenance is what the
    trust gate reads: the engine only enforces the ``allow_network`` /
    ``allow_repo_write`` veto when the spec came from the project's own
    ``.testbench.toml`` (``discovery._suite_from_manifest`` passes
    ``custom_from_manifest=True``). Suites synthesised by zero-config
    detection, or by an operator-supplied external manifest, are marked
    ``False`` and are therefore never subject to that veto — they were
    authored by the caller, not by the repository under test.

    ``required_paths``/``required_binaries`` are run-time facts, not discovery
    facts: a missing one becomes SKIP with a named reason instead of a
    discovery-time crash, so a machine without ``cobc`` still yields a full
    profile listing.
    """
    name: str
    adapter: str
    profile: str
    command: List[str] = field(default_factory=list)
    cwd: str = "."
    timeout_s: float = 120.0
    needs_tmux: bool = False
    needs_pty: bool = False
    allow_network: bool = False
    allow_repo_write: bool = False
    billable: bool = False
    custom_from_manifest: bool = False
    env_overrides: Dict[str, str] = field(default_factory=dict)
    required_paths: List[str] = field(default_factory=list)
    required_binaries: List[str] = field(default_factory=list)
    description: str = ""
    warmup_runs: int = 1
    repeat_runs: int = 5
    # Mutation testing: minimum acceptable ``caught / (caught + survived)``.
    # A gate declared by a manifest is only honoured when that manifest is trusted;
    # ``mutation_gate_from_cli`` marks an operator-supplied gate (always honoured).
    mutation_score_gate: Optional[float] = None
    mutation_gate_from_cli: bool = False


@dataclass
class ProjectSpec:
    name: str
    root: str
    languages: List[str] = field(default_factory=list)
    suites: List[SuiteSpec] = field(default_factory=list)
    manifest_path: Optional[str] = None
    manifest_sha256: Optional[str] = None
    trusted: bool = False
    revision: str = ""
    branch: str = ""


@dataclass
class FailureRecord:
    suite: str
    case_name: str
    kind: str = "assertion"
    message: str = ""
    file: Optional[str] = None
    line: Optional[int] = None
    excerpt: str = ""


@dataclass
class CaseResult:
    name: str
    suite: str
    status: str
    duration_ms: float = 0.0
    message: str = ""
    file: Optional[str] = None
    line: Optional[int] = None


@dataclass
class SuiteResult:
    name: str
    adapter: str
    profile: str
    command: List[str]
    cwd: str
    status: str
    exit_code: Optional[int] = None
    signal_num: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: float = 0.0
    total_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    skip_count: int = 0
    blocked_count: int = 0
    error_count: int = 0
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    sanitized_path: Optional[str] = None
    summary_excerpt: str = ""
    repro_command: str = ""
    failures: List[FailureRecord] = field(default_factory=list)
    cases: List[CaseResult] = field(default_factory=list)
    isolation_meta: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    """One run: the canonical record, and the sole input to every artifact.

    The store persists this two ways in the same transaction: a set of
    denormalised columns (``run_id``, ``status``, ``created_at``, counts,
    ``pinned`` ... ) used for filtering, ordering and reconciliation, and
    ``payload_json`` = ``render_run_json(self)`` which is the *only* thing
    ``get_run``/``list_runs`` read back. Columns are the index; the payload is
    the record. Any new field must reach both, or it will be silently lost on
    read-back while still appearing in ``runs.status``-style queries.

    ``created_at`` is immutable after insert (the upsert in ``store`` never
    rewrites it) because it is the ordering key for history, retention and
    "latest run per project". ``run_id`` and the ``*_path`` fields are filled
    by ``store._prepare_run_paths``; the paths are derived pointers into
    ``evidence_dir`` and carry no independent truth.

    ``run_nonce`` is declared but unpopulated anywhere in the package today —
    treat it as reserved, not as data.
    """
    run_id: str
    project: str
    project_root: str
    revision: str
    profile: str
    status: str
    branch: str = ""
    failure_kind: Optional[str] = None
    commands: List[List[str]] = field(default_factory=list)
    created_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[float] = None
    exit_code: Optional[int] = None
    total_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    skip_count: int = 0
    blocked_count: int = 0
    error_count: int = 0
    environment: Dict[str, Any] = field(default_factory=dict)
    evidence_dir: str = ""
    run_json_path: Optional[str] = None
    junit_xml_path: Optional[str] = None
    repro_path: Optional[str] = None
    summary: str = ""
    first_failure: str = ""
    repro_instructions: str = ""
    suites: List[SuiteResult] = field(default_factory=list)
    model_usage: Optional[Dict[str, Any]] = None
    delegation_id: Optional[str] = None
    worker_pid: Optional[int] = None
    worker_pgid: Optional[int] = None
    worker_start_time: Optional[float] = None
    run_nonce: str = ""
    pinned: bool = False


_CLOCK_LOCK = threading.Lock()
_LAST_UTC_US = 0
_LAST_RID_STAMP = ""
_LAST_RID_SEQ = -1


def utc_now_iso() -> str:
    """UTC timestamp, strictly increasing within a process, ``...Z`` suffixed.

    Ordering keys throughout the store are these strings compared
    lexicographically, so a clock that steps backwards would silently reorder
    history (and with it "latest run per project" and retention order). The
    guard bumps by one microsecond instead of returning a smaller value. It is
    per-process only: two processes racing the same second can still emit
    equal timestamps, which is why every ordering query has ``rowid`` /
    ``run_id`` tiebreakers behind it.
    """
    global _LAST_UTC_US
    with _CLOCK_LOCK:
        now = datetime.now(timezone.utc)
        us = int(now.timestamp() * 1_000_000)
        if us <= _LAST_UTC_US:
            us = _LAST_UTC_US + 1
            now = datetime.fromtimestamp(0, tz=timezone.utc) + timedelta(microseconds=us)
        _LAST_UTC_US = us
    return now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_run_id() -> str:
    """``tb_<UTCyyyymmdd_HHMMSS>_<4 hex seq><4 hex random>``.

    The stamp and the seq are per-process monotonic (seq is derived from the
    microsecond field, then bumped on collision), so IDs sort in creation
    order and two runs started in the same second still order correctly —
    history order is a product requirement, not cosmetics
    (``tests/test_evidence_quality_fixes`` pins the same-second case). Two
    *processes* share no state, so cross-process uniqueness rests on the
    16-bit seq bucket plus two random bytes; the store does not detect a
    collision, it upserts over it. Treat uniqueness as probabilistic.
    """
    global _LAST_RID_STAMP, _LAST_RID_SEQ
    with _CLOCK_LOCK:
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%d_%H%M%S")
        seq = min(0xFFFF, (now.microsecond * 65536) // 1_000_000)
        if _LAST_RID_STAMP and (stamp, seq) <= (_LAST_RID_STAMP, _LAST_RID_SEQ):
            stamp = _LAST_RID_STAMP
            if _LAST_RID_SEQ < 0xFFFF:
                seq = _LAST_RID_SEQ + 1
            else:
                dt = datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc) + timedelta(seconds=1)
                stamp = dt.strftime("%Y%m%d_%H%M%S")
                seq = 0
        _LAST_RID_STAMP = stamp
        _LAST_RID_SEQ = seq
    return f"tb_{stamp}_{seq:04x}{os.urandom(2).hex()}"


def run_to_dict(run: RunRecord) -> Dict[str, Any]:
    return asdict(run)


def _suite_from_dict(raw: Dict[str, Any]) -> SuiteResult:
    """Rebuild one suite, re-hydrating its two nested record lists.

    Nested failures/cases are plain dicts in JSON, so they need explicit
    construction; anything else is a field-for-field copy. Unknown keys raise
    here too, for the same reason as ``run_from_dict``.
    """
    failures = [FailureRecord(**f) for f in (raw.get("failures") or [])]
    cases = [CaseResult(**c) for c in (raw.get("cases") or [])]
    copy = dict(raw)
    copy["failures"] = failures
    copy["cases"] = cases
    return SuiteResult(**copy)


def run_from_dict(data: Dict[str, Any]) -> RunRecord:
    """Inverse of ``run_to_dict``; the only read path out of ``state.db``.

    Deliberately strict (``RunRecord(**raw)``): an unknown key is a TypeError,
    not a silent drop. That is safe only because ``TestBenchStore`` refuses to
    open a database written by a different ``SCHEMA_VERSION`` — payload shape
    and on-disk version move together. If a migration ever lands, it must
    rewrite ``payload_json``; loosening this function instead would let
    half-understood records into history.
    """
    raw = dict(data)
    raw["suites"] = [_suite_from_dict(s) for s in (raw.get("suites") or [])]
    if raw.get("model_usage") is None:
        raw["model_usage"] = None
    return RunRecord(**raw)
