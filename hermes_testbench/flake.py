"""Pure flake classification over repeated TestBench attempts.

Zero I/O, zero runner dependencies, zero formatting (ADR-0004): the same rules
drive a live ``flake run`` and ``flake analyze`` over runs already persisted in
the store. An attempt is one complete ``TestBenchEngine.run_sync`` execution
with its own independent ``run_id`` and preserved ``runs/RUN_ID/`` evidence.

Verdicts (evaluated over ``N`` attempts, 1-based ``attempt``):

* ``always_pass``  - all ``N`` attempts ended ``PASS`` (deterministic success)
* ``always_fail``  - all ``N`` attempts ended ``FAIL`` (deterministic failure,
  which is *not* flakiness)
* ``mixed``        - at least one ``PASS`` *and* at least one ``FAIL``: the
  definition of a flaky target. Both id sets are preserved verbatim so the
  failing evidence directory and the later passing retry stay inspectable.
* ``inconclusive`` - fewer than two attempts, or any attempt ended in a
  non-decisive status (``TIMEOUT``/``CANCELLED``/``ERROR``/``BLOCKED``/
  ``SKIP``) without both a pass and a fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import RunRecord, RunStatus

FLAKE_ENV_KEY = "flake"
MISSING_STATUS = "MISSING"
MIN_ATTEMPTS = 2


class FlakeVerdict:
    ALWAYS_PASS = "always_pass"
    ALWAYS_FAIL = "always_fail"
    MIXED = "mixed"
    INCONCLUSIVE = "inconclusive"

    ALL = (ALWAYS_PASS, ALWAYS_FAIL, MIXED, INCONCLUSIVE)
    DECISIVE = frozenset({ALWAYS_PASS, ALWAYS_FAIL, MIXED})


class TargetLevel:
    SUITE = "suite"
    CASE = "case"


def flake_metadata(run: RunRecord) -> Optional[Dict[str, Any]]:
    """Return validated ``environment['flake']`` metadata, or ``None``.

    A run is only a flake attempt when the metadata names a group and carries a
    1-based attempt index plus the intended repeat count.
    """
    env = run.environment or {}
    raw = env.get(FLAKE_ENV_KEY)
    if not isinstance(raw, dict):
        return None
    group_id = str(raw.get("group_id") or "").strip()
    if not group_id:
        return None
    try:
        attempt = int(raw.get("attempt") or 0)
        repeats = int(raw.get("repeats") or 0)
    except (TypeError, ValueError):
        return None
    if attempt < 1 or repeats < 1:
        return None
    return {"group_id": group_id, "attempt": attempt, "repeats": repeats}


def _order_key(run: RunRecord) -> Tuple[int, str, str]:
    meta = flake_metadata(run)
    attempt = meta["attempt"] if meta else 0
    return (attempt, run.created_at or "", run.run_id)


def order_attempts(runs: Sequence[RunRecord]) -> List[RunRecord]:
    """Order attempts by their recorded 1-based index (stable on ties)."""
    ordered: List[RunRecord] = []
    seen: set = set()
    for run in sorted(runs, key=_order_key):
        if run.run_id in seen:
            continue
        seen.add(run.run_id)
        ordered.append(run)
    return ordered


def classify_statuses(statuses: Sequence[str]) -> str:
    """Classify one target's per-attempt statuses into a :class:`FlakeVerdict`."""
    total = len(statuses)
    passes = sum(1 for s in statuses if s == RunStatus.PASS)
    fails = sum(1 for s in statuses if s == RunStatus.FAIL)
    if total >= MIN_ATTEMPTS:
        if passes == total:
            return FlakeVerdict.ALWAYS_PASS
        if fails == total:
            return FlakeVerdict.ALWAYS_FAIL
        if passes >= 1 and fails >= 1:
            return FlakeVerdict.MIXED
    return FlakeVerdict.INCONCLUSIVE


def aggregate_verdict(verdicts: Sequence[str]) -> str:
    """Worst verdict across targets; ``always_pass`` only if every target is."""
    vals = list(verdicts)
    if not vals:
        return FlakeVerdict.INCONCLUSIVE
    if all(v == FlakeVerdict.ALWAYS_PASS for v in vals):
        return FlakeVerdict.ALWAYS_PASS
    if any(v == FlakeVerdict.MIXED for v in vals):
        return FlakeVerdict.MIXED
    if any(v == FlakeVerdict.ALWAYS_FAIL for v in vals):
        return FlakeVerdict.ALWAYS_FAIL
    return FlakeVerdict.INCONCLUSIVE


@dataclass
class TargetClassification:
    """One classified target (a suite or, when structured, a single case)."""

    target: str
    level: str
    suite: str
    case: str
    verdict: str
    attempts: int
    pass_count: int
    fail_count: int
    other_count: int
    statuses: List[str] = field(default_factory=list)
    passing_run_ids: List[str] = field(default_factory=list)
    failing_run_ids: List[str] = field(default_factory=list)
    other_run_ids: List[str] = field(default_factory=list)
    passing_evidence_dirs: List[str] = field(default_factory=list)
    failing_evidence_dirs: List[str] = field(default_factory=list)
    other_evidence_dirs: List[str] = field(default_factory=list)
    missing_attempts: List[int] = field(default_factory=list)

    @property
    def is_flaky(self) -> bool:
        return self.verdict == FlakeVerdict.MIXED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "level": self.level,
            "suite": self.suite,
            "case": self.case,
            "verdict": self.verdict,
            "attempts": self.attempts,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "other_count": self.other_count,
            "statuses": list(self.statuses),
            "passing_run_ids": list(self.passing_run_ids),
            "failing_run_ids": list(self.failing_run_ids),
            "other_run_ids": list(self.other_run_ids),
            "passing_evidence_dirs": list(self.passing_evidence_dirs),
            "failing_evidence_dirs": list(self.failing_evidence_dirs),
            "other_evidence_dirs": list(self.other_evidence_dirs),
            "missing_attempts": list(self.missing_attempts),
        }


def _classify_target(
    target: str,
    level: str,
    suite: str,
    case: str,
    pairs: Sequence[Tuple[RunRecord, str]],
) -> TargetClassification:
    statuses = [status for _run, status in pairs]
    tc = TargetClassification(
        target=target,
        level=level,
        suite=suite,
        case=case,
        verdict=classify_statuses(statuses),
        attempts=len(statuses),
        pass_count=sum(1 for s in statuses if s == RunStatus.PASS),
        fail_count=sum(1 for s in statuses if s == RunStatus.FAIL),
        other_count=sum(1 for s in statuses if s not in (RunStatus.PASS, RunStatus.FAIL)),
        statuses=statuses,
    )
    for (run, status), attempt_no in zip(pairs, range(1, len(pairs) + 1)):
        evidence = run.evidence_dir or ""
        if status == RunStatus.PASS:
            tc.passing_run_ids.append(run.run_id)
            tc.passing_evidence_dirs.append(evidence)
        elif status == RunStatus.FAIL:
            tc.failing_run_ids.append(run.run_id)
            tc.failing_evidence_dirs.append(evidence)
        else:
            tc.other_run_ids.append(run.run_id)
            tc.other_evidence_dirs.append(evidence)
            if status == MISSING_STATUS:
                tc.missing_attempts.append(attempt_no)
    return tc


def _suite_map(run: RunRecord) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for suite in run.suites or ():
        out.setdefault(suite.name, suite)
    return out


def _target_names(runs: Sequence[RunRecord], suite_names: Optional[Sequence[str]]) -> List[str]:
    if suite_names:
        return list(dict.fromkeys(str(n) for n in suite_names))
    names: List[str] = []
    for run in runs:
        for suite in run.suites or ():
            if suite.name not in names:
                names.append(suite.name)
    return names


def _case_names(suite_name: str, runs: Sequence[RunRecord]) -> List[str]:
    names: List[str] = []
    for run in runs:
        suite = _suite_map(run).get(suite_name)
        if suite is None:
            continue
        for case in suite.cases or ():
            if case.name not in names:
                names.append(case.name)
    return names


def _suite_has_structured_cases(suite_name: str, runs: Sequence[RunRecord]) -> bool:
    for run in runs:
        suite = _suite_map(run).get(suite_name)
        if suite is not None and suite.cases:
            return True
    return False


@dataclass
class FlakeReport:
    """Classification of one flake group (all attempts sharing ``group_id``)."""

    group_id: str
    project: str
    revision: str
    profile: str
    verdict: str
    attempts: int
    expected_repeats: int
    targets: List[TargetClassification] = field(default_factory=list)
    run_ids: List[str] = field(default_factory=list)
    statuses: List[str] = field(default_factory=list)
    evidence_dirs: List[str] = field(default_factory=list)
    missing_attempts: List[int] = field(default_factory=list)
    created_at: str = ""
    last_activity: str = ""

    @property
    def suite_targets(self) -> List[TargetClassification]:
        return [t for t in self.targets if t.level == TargetLevel.SUITE]

    @property
    def case_targets(self) -> List[TargetClassification]:
        return [t for t in self.targets if t.level == TargetLevel.CASE]

    @property
    def has_cases(self) -> bool:
        return bool(self.case_targets)

    @property
    def flaky_targets(self) -> List[TargetClassification]:
        return [t for t in self.targets if t.verdict == FlakeVerdict.MIXED]

    @property
    def failed_targets(self) -> List[TargetClassification]:
        return [t for t in self.targets if t.verdict == FlakeVerdict.ALWAYS_FAIL]

    @property
    def inconclusive_targets(self) -> List[TargetClassification]:
        return [t for t in self.targets if t.verdict == FlakeVerdict.INCONCLUSIVE]

    def failing_run_ids(self) -> List[str]:
        out: List[str] = []
        for status, rid in zip(self.statuses, self.run_ids):
            if status in (RunStatus.FAIL, RunStatus.TIMEOUT, RunStatus.ERROR, RunStatus.CANCELLED):
                out.append(rid)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "project": self.project,
            "revision": self.revision,
            "profile": self.profile,
            "verdict": self.verdict,
            "attempts": self.attempts,
            "expected_repeats": self.expected_repeats,
            "run_ids": list(self.run_ids),
            "statuses": list(self.statuses),
            "evidence_dirs": list(self.evidence_dirs),
            "missing_attempts": list(self.missing_attempts),
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "targets": [t.to_dict() for t in self.targets],
        }


def build_report(
    runs: Sequence[RunRecord],
    *,
    expected_repeats: Optional[int] = None,
    suite_names: Optional[Sequence[str]] = None,
) -> FlakeReport:
    """Classify an ordered set of attempt runs into suite-level and case-level targets."""
    ordered = order_attempts(runs)
    metas = [flake_metadata(r) for r in ordered]
    group_id = next((m["group_id"] for m in metas if m), "")
    repeats = expected_repeats or max((m["repeats"] for m in metas if m), default=0) or len(ordered)

    names = _target_names(ordered, suite_names)
    targets: List[TargetClassification] = []
    for name in names:
        pairs: List[Tuple[RunRecord, str]] = []
        for run in ordered:
            suite = _suite_map(run).get(name)
            pairs.append((run, suite.status if suite is not None else MISSING_STATUS))
        targets.append(_classify_target(name, TargetLevel.SUITE, name, "", pairs))

        if not _suite_has_structured_cases(name, ordered):
            continue
        for case_name in _case_names(name, ordered):
            case_pairs: List[Tuple[RunRecord, str]] = []
            for run in ordered:
                suite = _suite_map(run).get(name)
                status = MISSING_STATUS
                if suite is not None:
                    for case in suite.cases or ():
                        if case.name == case_name:
                            status = case.status
                            break
                case_pairs.append((run, status))
            targets.append(
                _classify_target(
                    f"{name}::{case_name}", TargetLevel.CASE, name, case_name, case_pairs
                )
            )

    statuses = [run.status for run in ordered]
    report = FlakeReport(
        group_id=group_id,
        project=ordered[0].project if ordered else "",
        revision=ordered[0].revision if ordered else "",
        profile=ordered[0].profile if ordered else "",
        verdict=aggregate_verdict([t.verdict for t in targets]),
        attempts=len(ordered),
        expected_repeats=max(int(repeats or 0), len(ordered)),
        targets=targets,
        run_ids=[run.run_id for run in ordered],
        statuses=statuses,
        evidence_dirs=[run.evidence_dir or "" for run in ordered],
        created_at=(ordered[0].created_at if ordered else ""),
        last_activity=(ordered[-1].created_at if ordered else ""),
    )
    seen_attempts = {
        m["attempt"] for m in metas if m is not None
    }
    if seen_attempts:
        report.missing_attempts = [
            i for i in range(1, report.expected_repeats + 1) if i not in seen_attempts
        ]
    return report


@dataclass
class FlakeHistory:
    """Historical analysis result grouped by flake ``group_id``."""

    groups: List[FlakeReport] = field(default_factory=list)

    @property
    def flaky_groups(self) -> List[FlakeReport]:
        return [g for g in self.groups if g.verdict == FlakeVerdict.MIXED]

    @property
    def always_fail_groups(self) -> List[FlakeReport]:
        return [g for g in self.groups if g.verdict == FlakeVerdict.ALWAYS_FAIL]

    @property
    def always_pass_groups(self) -> List[FlakeReport]:
        return [g for g in self.groups if g.verdict == FlakeVerdict.ALWAYS_PASS]

    @property
    def inconclusive_groups(self) -> List[FlakeReport]:
        return [g for g in self.groups if g.verdict == FlakeVerdict.INCONCLUSIVE]

    def verdict_counts(self) -> Dict[str, int]:
        counts = {v: 0 for v in FlakeVerdict.ALL}
        for group in self.groups:
            counts[group.verdict] = counts.get(group.verdict, 0) + 1
        return counts

    @property
    def has_flakes(self) -> bool:
        return any(g.verdict != FlakeVerdict.ALWAYS_PASS for g in self.groups)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "verdict_counts": self.verdict_counts(),
            "flaky_groups": len(self.flaky_groups),
            "always_fail_groups": len(self.always_fail_groups),
            "always_pass_groups": len(self.always_pass_groups),
            "inconclusive_groups": len(self.inconclusive_groups),
        }


def analyze_flake_history(
    runs: Sequence[RunRecord],
    *,
    project: Optional[str] = None,
    revision: Optional[str] = None,
    group_id: Optional[str] = None,
) -> FlakeHistory:
    """Group stored runs by flake group and classify each without re-running.

    Only runs carrying ``environment['flake']`` metadata participate; runs from
    ordinary ``testbench run`` invocations are ignored. Filters are applied
    before grouping so ``project``/``revision`` scope mirrors the store query.
    """
    buckets: Dict[str, List[RunRecord]] = {}
    for run in order_attempts(runs):
        meta = flake_metadata(run)
        if meta is None:
            continue
        if project and run.project != project:
            continue
        if revision and run.revision != revision:
            continue
        if group_id and meta["group_id"] != group_id:
            continue
        buckets.setdefault(meta["group_id"], []).append(run)

    groups: List[FlakeReport] = []
    for gid, members in buckets.items():
        groups.append(build_report(members))
    groups.sort(key=lambda g: (g.last_activity or "", g.group_id), reverse=True)
    return FlakeHistory(groups=groups)
