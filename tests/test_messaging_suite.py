"""TDD tests for the offline ``agent-messaging`` verification suite.

The suite is exercised against a synthetic source tree and synthetic artifact so the
tests stay hermetic: no live LLM, no gateway, no dependency on this host's Hermes
checkout. A final integration check against the real artifact lives outside unittest.

The fixture source files (``BOT_MODE_DM`` and friends) are *text*, never imported:
the contract cases are substring greps over a known-good upstream vocabulary, so a
fixture that is not valid Python is still a valid fixture. What the fixtures must
contain, exactly, is the string each case looks for — that is why they are excerpts
with comments about intent rather than runnable modules.

Layout of this file mirrors the suite under test: ``ContractSuiteTests`` (the greps
and evidence receipts), ``WiringSuiteTests`` (the only part that executes artifact
code), ``PytestSuiteTests`` (opt-in upstream run) and ``SettlementFallbackTests``
(this package's own notification→receipt fallback). Negative controls edit the
fixture and re-run, so each important case is shown to fail when its behavior is
removed, not merely to pass on the fixture.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import textwrap
import unittest
from pathlib import Path

from hermes_testbench.messaging_suite import (
    delivery_receipt_path,
    run_agent_messaging_verification,
    settlement_from_notification,
)
from hermes_testbench.schema import RunStatus

BOT_MODE_DM = '''
"""Bot DM transport (fixture excerpt)."""
import json
import os
import time
from pathlib import Path


def _err(message, *, roster=None, peers=None):
    payload = {"error": message, "reason": "target_scope_unresolved"}
    if roster is not None:
        payload["teammates"] = roster
    if peers is not None:
        payload["peers"] = peers
    return json.dumps(payload)


def _ack(label, delivery_id, proc_id):
    detail = ("this acknowledges the hand-off to a background delivery process, "
              "not a delivery receipt")
    return json.dumps({
        "status": "queued",
        "delivery_id": delivery_id,
        "to": label,
        "reply_delivery": "notification",
        "detail": detail,
        "process_id": proc_id,
        "queued_at": int(time.time()),
    })


def _author(me, handle):
    signature = f"Message from 🤖 {name} (@{handle}): "
    author = {"id": f"bot:{me}", "name": handle, "is_bot": True}
    peer_author = {**author, "id": bot_author_id(me, local_origin())}
    return signature, author, peer_author


def _admit_live_dm(profile_home, dm_file, author=None):
    """Pin intent before admission; retries may inspect, never change transport."""
    intent_path = Path(dm_file + ".live.json")
    fd = os.open(intent_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.fsync(stream.fileno())
    fsync_directory(intent_path.parent)
    return json.dumps({"status": "ambiguous", "delivery_id": delivery_id,
                       "error": f"Live admission outcome unknown: {cause}. Do not resend."})


def _resolve(raw_target, roster):
    match = resolve_remote_target(raw_target, roster) if roster else None
    if match == "ambiguous":
        return _err(f"'{raw_target}' exists on several connected machines — "
                    f"disambiguate with one of: {forms}.")


def _spawn(command, proc_id):
    raw = terminal_tool(command, background=True, notify_on_complete=True)
    return raw


def _persist_reply_when_done(proc_id, agent):
    """Append the runner outcome to the sender transcript."""
    display_kind = "process_complete"
    return True
'''

FAILURE_REASONS = '''
"""Closed bot failure vocabulary and retry policy."""
RUNTIME_OFFLINE = "runtime_offline"
QUEUED_EXPIRED = "queued_expired"
DELIVERY_TIMEOUT = "delivery_timeout"
AGENT_BLOCKED = "agent_blocked"
CANCELLED = "cancelled"
PROVIDER_AUTH_OR_ACCESS = "provider_auth_or_access"
PROVIDER_QUOTA_LIMIT = "provider_quota_limit"
PROVIDER_RATE_LIMIT = "provider_rate_limit"
PROVIDER_SERVER_ERROR = "provider_server_error"
CONTEXT_OVERFLOW = "context_overflow"
MISSING_CONFIG = "missing_config"
MODEL_UNAVAILABLE = "model_unavailable"
TARGET_SCOPE_UNRESOLVED = "target_scope_unresolved"
UNKNOWN = "unknown"

ALL_REASONS = frozenset({
    RUNTIME_OFFLINE, QUEUED_EXPIRED, DELIVERY_TIMEOUT, AGENT_BLOCKED, CANCELLED,
    PROVIDER_AUTH_OR_ACCESS, PROVIDER_QUOTA_LIMIT, PROVIDER_RATE_LIMIT,
    PROVIDER_SERVER_ERROR, CONTEXT_OVERFLOW, MISSING_CONFIG, MODEL_UNAVAILABLE,
    TARGET_SCOPE_UNRESOLVED, UNKNOWN,
})

HELD_NOT_SENT = "runtime_coordination_required"

AUTO_RETRYABLE = frozenset({RUNTIME_OFFLINE, DELIVERY_TIMEOUT, PROVIDER_RATE_LIMIT, PROVIDER_SERVER_ERROR})

RETRY_RESUME = "resume"
RETRY_COMPRESS_THEN_RESUME = "compress_then_resume"
RETRY_NONE = "none"

def retry_action(reason):
    if reason in AUTO_RETRYABLE:
        return RETRY_RESUME
    if reason == CONTEXT_OVERFLOW:
        return RETRY_COMPRESS_THEN_RESUME
    return RETRY_NONE
'''

BOT_LIVE_DELIVERY = '''
"""Durable Bot Chat delivery receipts."""
import json
import re
from pathlib import Path

DELIVERY_DIR_NAME = "bot_live_delivery"
_TERMINAL = frozenset({"settled", "failed", "cancelled", "ambiguous"})


def _delivery_id(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32,64}", value) is None:
        raise ValueError("delivery id must be 32 to 64 lowercase hex characters")
    return value


def _root(home):
    return Path(home).resolve() / "runtime" / DELIVERY_DIR_NAME


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))
'''

BOT_CHAIN = '''
"""Per-session bot chain loop guard."""
import hashlib

def admit_send(agent, recipients, body):
    settings = _settings()
    digest = hashlib.sha256(" ".join(str(body).split()).encode()).hexdigest()
    bot_turn = state["hops"] > 0
    hop = state["sent"] + 1 if bot_turn else 1
    if hop > settings.max_hops:
        return Admission(refusal=("This bot-to-bot exchange already reached its limit"))
    for recipient in recipients:
        ask = (recipient, digest)
        if bot_turn and ask in state["asks"]:
            admission.refused[recipient] = "You already sent this exact message"
        else:
            state["asks"].add(ask)
    return admission
'''

ARTIFACT_README = '''
# agent-messaging

| Behaviour | Status |
|---|---|
| Duplicate delivery | Known gap: identical sends in one human-originated turn are both delivered (dedupe is bot-turn only) |
| Gateway restart mid-flight | Not verified: blocked by the in-gateway restart guard; procedure in the ADR |
'''

SETTLED_RECEIPT = {
    "admission_id": "f8a881dfa956431d8b81d91ed72e6c50",
    "author": {"id": "bot:default", "is_bot": True, "name": "hermes"},
    "delivery_id": "ed42a3e228655eaa1c811c3cbada4bfa90eb700a2614aa5ceb849a3c18354d1d",
    "message": "Message from 🤖 hermes (@hermes): question",
    "profile_home": "/tmp/does-not-matter/profiles/opus",
    "reply": "Record delivery_id and the question text.",
    "session_id": "local-c8d1",
    "status": "settled",
}

FAILED_RECEIPT = {
    "admission_id": "62340f2c2b8242e5b650c17378be64b2",
    "author": {"id": "bot:default", "is_bot": True, "name": "hermes"},
    "delivery_id": "c96f918e8e8d2bb71ed2254aaa23069dee66fb723b8ed8de27bb404da41c2dd5",
    "error": "HTTP 402: This request would exceed your available credits.",
    "reason": "provider_quota_limit",
    "reply": "Billing or credits exhausted.",
    "status": "failed",
}

CHECK_WIRING = '''
#!/usr/bin/env python3
"""Minimal stand-in for the artifact's read-only wiring check."""
import argparse
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--hermes-home", default=str(Path.home() / ".hermes"))
args = ap.parse_args()
home = Path(args.hermes_home).expanduser()
failures = []
if not (home / "profiles" / "opus" / "config.yaml").is_file():
    failures.append("1. advisor profile missing")
if not any(p.is_file() for p in [home / "profile.yaml"]):
    failures.append("2. no profile carries ui_meta.hermes-bots: message_agent stays unavailable")
print(f"install      : {home}")
print(f"bot-mode flag: {', '.join(['default']) if not failures else 'none'}")
if failures:
    print("\\nFAILED:")
    for line in failures:
        print("  " + line)
    sys.exit(1)
print("\\nwiring OK")
'''


class _FixtureCase(unittest.TestCase):
    """Build the temp tree, source fixtures and artifact once per test.

    Both trees are rewritten from the constants on every ``setUp``, so a test
    that mutates a fixture file (the negative controls) cannot leak state into
    the next one. ``self.evidence`` is the run's own evidence dir: the contract
    suite reads receipts from it as well as from the artifact.
    """

    def setUp(self) -> None:
        scratch = Path.home() / ".hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(dir=str(scratch), prefix="tb-msg-")
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.evidence = self.root / "evidence"
        self.src = self.root / "hermes-agent"
        self.artifact = self.root / "artifact"
        self._write_source(self.src)
        self._write_artifact(self.artifact)

    @staticmethod
    def _write_source(root: Path) -> None:
        (root / "tools").mkdir(parents=True)
        (root / "agent").mkdir(parents=True)
        (root / "tests" / "tools").mkdir(parents=True)
        (root / "tools" / "bot_mode_dm.py").write_text(BOT_MODE_DM, encoding="utf-8")
        (root / "tools" / "bot_failure_reasons.py").write_text(FAILURE_REASONS, encoding="utf-8")
        (root / "tools" / "bot_live_delivery.py").write_text(BOT_LIVE_DELIVERY, encoding="utf-8")
        (root / "agent" / "bot_chain.py").write_text(BOT_CHAIN, encoding="utf-8")
        (root / "tests" / "tools" / "test_bot_mode_dm.py").write_text(
            "def test_live_dm_runner_retry_never_reexecutes_failed_admission():\n    pass\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_artifact(root: Path) -> None:
        (root / "evidence").mkdir(parents=True)
        (root / "scripts").mkdir(parents=True)
        (root / "README.md").write_text(ARTIFACT_README, encoding="utf-8")
        (root / "evidence" / "receipt-settled.json").write_text(
            json.dumps(SETTLED_RECEIPT), encoding="utf-8"
        )
        (root / "evidence" / "receipt-failed-quota.json").write_text(
            json.dumps(FAILED_RECEIPT), encoding="utf-8"
        )
        script = root / "scripts" / "check-wiring.py"
        script.write_text(CHECK_WIRING, encoding="utf-8")
        script.chmod(0o700)

    def _suites(self, **kwargs):
        """Run the verification against this test's fixture trees; extra kwargs pass through."""
        kwargs.setdefault("artifact_root", self.artifact)
        kwargs.setdefault("hermes_agent_root", self.src)
        return run_agent_messaging_verification(self.evidence, **kwargs)


class ContractSuiteTests(_FixtureCase):
    """Controls on the contract checklist: full green, two negative controls, one absent input."""

    def test_contract_suite_verifies_all_seven_behaviors_and_records_restart_skip(self) -> None:
        """A green contract run must contain every required case, including the skip.

        ``required`` names the 26 behaviors explicitly, so a case silently
        dropped from ``_REQUIRED_CASES`` (or renamed) fails here instead of
        shrinking the suite. ``total_count == len(cases)`` pins ``_suite``'s
        derivation; the one SKIP is the mid-flight restart case — it is part of
        the contract ("this was not verified"), so a green run has
        ``pass_count == total_count - 1`` and never ``total_count``.
        """
        suites = self._suites()
        by_name = {s.name: s for s in suites}
        self.assertIn("agent-messaging:contract", by_name)
        contract = by_name["agent-messaging:contract"]
        self.assertEqual(contract.status, RunStatus.PASS, [c.message for c in contract.cases if c.status != RunStatus.PASS])
        self.assertEqual(contract.fail_count, 0)
        self.assertEqual(contract.error_count, 0)
        self.assertEqual(contract.total_count, len(contract.cases))
        self.assertEqual(contract.adapter, "contract")
        self.assertEqual(contract.profile, "integration")

        names = {c.name for c in contract.cases}
        required = {
            "a1_ack_is_queued_handoff_with_delivery_id",
            "a2_ack_documented_as_handoff_not_receipt",
            "a3_terminal_status_vocabulary_includes_settled",
            "a4_evidence_settled_receipt_carries_reply",
            "b1_server_side_attribution_prefix",
            "b2_author_envelope_is_bot_typed",
            "b3_peer_author_envelope_host_qualified",
            "b4_evidence_author_envelope",
            "c1_closed_failure_reason_vocabulary",
            "c2_provider_quota_limit_is_not_auto_retryable",
            "c3_rate_limit_and_runtime_offline_are_retryable",
            "c4_ambiguous_means_do_not_resend",
            "c5_held_not_sent_outside_all_reasons",
            "c6_evidence_quota_failure_is_typed",
            "d1_structured_refusal_payload_with_live_roster",
            "d2_ambiguous_target_produces_disambiguation_forms",
            "e1_bot_turn_duplicate_send_is_refused",
            "e2_hop_limit_enforced",
            "e3_human_turn_gap_documented",
            "f1_pinned_live_intent_file",
            "f2_retry_inspects_and_never_retransports",
            "f3_restart_regression_test_exists",
            "g1_completion_notification_is_the_settle_channel",
            "g2_durable_process_complete_transcript_row",
            "g3_durable_receipt_directory",
            "g4_truncated_notification_falls_back_to_receipt_file",
        }
        missing = required - names
        self.assertEqual(missing, set(), f"missing cases: {sorted(missing)}")
        self.assertEqual(contract.skip_count, 1)
        skip_case = [c for c in contract.cases if c.status == RunStatus.SKIP][0]
        self.assertIn("gateway_restart", skip_case.name)
        self.assertIn("restart guard", skip_case.message)
        self.assertEqual(contract.pass_count, contract.total_count - 1)

    def test_quota_retry_exclusion_regression_fails_its_case_and_the_suite(self) -> None:
        """Negative control: making quota errors retryable must turn c2 red.

        The fixture is edited on disk, not mocked, so this goes through the same
        regex parse a real source change would. Asserting the case *and* the
        suite status catches a regression where a failing case stops propagating
        into ``_status_for``; the message check keeps the failure attributed to
        the quota reason rather than to some unrelated parse change.
        """
        path = self.src / "tools" / "bot_failure_reasons.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "AUTO_RETRYABLE = frozenset({RUNTIME_OFFLINE, DELIVERY_TIMEOUT, PROVIDER_RATE_LIMIT",
                "AUTO_RETRYABLE = frozenset({RUNTIME_OFFLINE, DELIVERY_TIMEOUT, PROVIDER_QUOTA_LIMIT, PROVIDER_RATE_LIMIT",
            ),
            encoding="utf-8",
        )
        contract = {s.name: s for s in self._suites()}["agent-messaging:contract"]
        self.assertEqual(contract.status, RunStatus.FAIL)
        failed = {c.name: c for c in contract.cases if c.status == RunStatus.FAIL}
        self.assertIn("c2_provider_quota_limit_is_not_auto_retryable", failed)
        self.assertIn("provider_quota_limit", failed["c2_provider_quota_limit_is_not_auto_retryable"].message)
        self.assertGreaterEqual(contract.fail_count, 1)

    def test_removed_duplicate_guard_fails_the_dedupe_case(self) -> None:
        """Negative control: dropping the ``bot_turn`` scope breaks exactly e1.

        The case greps for the full guarded expression, so removing the bot-turn
        condition fails it; that is what distinguishes "a duplicate guard exists"
        from "the guard applies to bot turns only", which is the documented gap
        for human-originated turns.
        """
        path = self.src / "agent" / "bot_chain.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "if bot_turn and ask in state[\"asks\"]:", "if ask in state[\"asks\"]:"
            ),
            encoding="utf-8",
        )
        contract = {s.name: s for s in self._suites()}["agent-messaging:contract"]
        failed = {c.name for c in contract.cases if c.status != RunStatus.PASS}
        self.assertIn("e1_bot_turn_duplicate_send_is_refused", failed)

    def test_missing_source_tree_marks_every_contract_case_skipped_not_passed(self) -> None:
        """An unconfigured host must not look verified: absent tree → all SKIP.

        ``pass_count == 0`` is asserted in addition to the statuses because a
        caller that only aggregates statuses can still read SKIP as green; the
        message check keeps the reason "source" rather than a generic skip.
        """
        contract = {
            s.name: s
            for s in run_agent_messaging_verification(
                self.evidence,
                artifact_root=self.artifact,
                hermes_agent_root=self.root / "absent",
            )
        }["agent-messaging:contract"]
        self.assertEqual(contract.status, RunStatus.SKIP)
        self.assertEqual(contract.pass_count, 0)
        self.assertTrue(all(c.status == RunStatus.SKIP for c in contract.cases))
        self.assertIn("source", contract.cases[0].message.lower())


class WiringSuiteTests(_FixtureCase):
    """The only suite that executes artifact code, so it is tested in both directions."""

    def test_wiring_suite_passes_on_a_synthetic_wired_home_and_flags_an_unwired_one(self) -> None:
        """A wired home passes and an empty one is flagged — in the same run.

        A checker that always exits 0 would satisfy the first case alone, which
        is why w2 exists: requiring the unwired fixture to fail is what makes w1
        evidence. Case statuses and the suite status are both asserted so a
        wiring regression is visible at either level.
        """
        suites = {s.name: s for s in self._suites()}
        wiring = suites["agent-messaging:wiring"]
        cases = {c.name: c for c in wiring.cases}
        self.assertIn("w1_synthetic_wired_home_passes", cases)
        self.assertIn("w2_missing_bot_mode_marker_fails", cases)
        self.assertEqual(cases["w1_synthetic_wired_home_passes"].status, RunStatus.PASS)
        self.assertEqual(cases["w2_missing_bot_mode_marker_fails"].status, RunStatus.PASS)
        self.assertEqual(wiring.status, RunStatus.PASS)
        self.assertIn("wiring", wiring.summary_excerpt.lower())

    def test_missing_artifact_root_skips_wiring_suite(self) -> None:
        """No artifact root drops the suite rather than reporting it red.

        Absence of the thing under test is not a failure — the contract suite
        still runs and SKIPs its evidence cases — so this asserts only that the
        wiring suite is absent from the result list.
        """
        suites = {
            s.name: s
            for s in run_agent_messaging_verification(
                self.evidence, artifact_root=None, hermes_agent_root=self.src
            )
        }
        self.assertNotIn("agent-messaging:wiring", suites)


class PytestSuiteTests(_FixtureCase):
    """The upstream pytest run is opt-in and must announce a skip when it cannot run."""

    def test_no_pytest_suite_without_opt_in(self) -> None:
        """The default run must never spawn the upstream interpreter."""
        names = {s.name for s in self._suites()}
        self.assertNotIn("agent-messaging:pytest", names)

    def test_opt_in_without_a_venv_records_an_explicit_skip(self) -> None:
        """Opting in without ``venv/bin/python`` is a recorded SKIP, not an omission.

        ``skip_count == 1`` is asserted because the suite exists specifically to
        say "this could not run"; a missing suite would be indistinguishable from
        a passing one in any aggregate that only counts suites.
        """
        suites = {s.name: s for s in self._suites(run_hermes_pytest=True)}
        self.assertIn("agent-messaging:pytest", suites)
        self.assertEqual(suites["agent-messaging:pytest"].status, RunStatus.SKIP)
        self.assertEqual(suites["agent-messaging:pytest"].skip_count, 1)


class SettlementFallbackTests(_FixtureCase):
    """Receipt-path validation and the notification→file→missing fallback chain.

    These exercise ``hermes_testbench.messaging_suite``'s own helpers, not the
    artifact's delivery code: the fallback order asserted here is the one this
    module documents, and the contract suite's g4 case re-runs the same logic.
    """

    def _write_receipt(self, profile_home: Path, delivery_id: str, status: str = "settled") -> Path:
        """Write a minimal durable receipt at the canonical path for ``delivery_id``."""
        path = delivery_receipt_path(profile_home, delivery_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": status,
                    "delivery_id": delivery_id,
                    "reply": "durable reply text",
                    "reply_delivery": "notification",
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_receipt_path_is_the_runtime_bot_live_delivery_file(self) -> None:
        """The path shape is what the durable-receipt cases and ``g4`` rely on.

        ``profile_home`` is resolved before joining, so the returned path is
        always absolute regardless of how the caller spelled the root.
        """
        delivery_id = "ab" * 32
        path = delivery_receipt_path(self.root / "profile", delivery_id)
        self.assertEqual(
            path,
            (self.root / "profile").resolve()
            / "runtime"
            / "bot_live_delivery"
            / f"{delivery_id}.json",
        )

    def test_traversal_and_short_ids_are_rejected(self) -> None:
        """The id is used as a filename, so validation *is* the traversal defence.

        ``"../../etc/passwd"`` is the traversal case; ``"abc"`` and ``""`` are
        the too-short and empty cases; ``"ZZ" * 20`` is the right length but not
        lowercase hex, which is what makes ``fullmatch`` (not a length check) the
        thing under test.
        """
        for bad in ("../../etc/passwd", "abc", "ZZ" * 20, ""):
            with self.assertRaises(ValueError):
                delivery_receipt_path(self.root / "profile", bad)

    def test_truncated_notification_falls_back_to_the_durable_receipt_file(self) -> None:
        """A truncated notification has no JSON line, so the file is the only source.

        ``source == "receipt_file"`` discriminates the fallback from the
        notification path, and the reply text proves the payload was read from
        the file rather than synthesized from the delivery id.
        """
        profile = self.root / "profile"
        delivery_id = "cd" * 32
        self._write_receipt(profile, delivery_id)
        result = settlement_from_notification(
            "runner exited; [output truncated at 4096 chars]",
            profile_home=profile,
            delivery_id=delivery_id,
        )
        self.assertEqual(result["source"], "receipt_file")
        self.assertEqual(result["status"], "settled")
        self.assertEqual(result["reply"], "durable reply text")
        self.assertEqual(result["delivery_id"], delivery_id)

    def test_complete_notification_json_is_preferred_over_the_file(self) -> None:
        """Precedence: a matching notification line wins even when a file exists.

        The file says ``settled`` and the notification says ``failed``, so a
        result of ``failed``/``notification`` proves precedence rather than
        agreement between the two channels.
        """
        profile = self.root / "profile"
        delivery_id = "ef" * 32
        self._write_receipt(profile, delivery_id)
        payload = json.dumps(
            {"status": "failed", "delivery_id": delivery_id, "error": "provider_quota_limit"}
        )
        result = settlement_from_notification(
            f"runner finished\n{payload}\n", profile_home=profile, delivery_id=delivery_id
        )
        self.assertEqual(result["source"], "notification")
        self.assertEqual(result["status"], "failed")

    def test_unknown_delivery_without_either_channel_is_reported_as_missing(self) -> None:
        """Neither channel has data, so the result is ``source == "missing"`` and
        ``status == "ambiguous"``; it never guesses settled.

        ``ambiguous`` is the safe answer because the caller's documented retry
        policy treats it as "do not resend" rather than as a failure to replay.
        """
        result = settlement_from_notification(
            "truncated", profile_home=self.root / "profile", delivery_id="12" * 32
        )
        self.assertEqual(result["source"], "missing")
        self.assertEqual(result["status"], "ambiguous")


if __name__ == "__main__":
    unittest.main()
