"""Secret redaction and terminal-text sanitization at the export and disk boundary.

Two policies are pinned here, both from ADR-0004:

* *Sanitization* (``sanitize_terminal_text``) — bytes that must never reach a
  terminal, a report, or an agent: escape sequences, C0/C1 controls, bidi
  overrides and isolates, and Plane-14 tag characters. It removes rather than
  escapes, so the expected outputs are exact strings, not patterns.
* *Redaction* (``redact_secrets`` / ``StreamRedactor``) — things that must never
  reach disk. Pattern-based by design (PEM blocks, URL userinfo, ``Bearer``,
  a closed list of vendor prefixes, credential-shaped ``NAME=value``); an
  unlabelled high-entropy secret passes through, which is why the primary
  defence is credential env scrubbing plus Seatbelt read-denial rather than
  these regexes.

The assertions are all *negative* (a literal is absent) except where a positive
check is needed to prove the sanitizer did not simply delete everything: the
first test's exact-equality and the ``[REDACTED]``/``done``/framing checks are
those controls. Without them, a function returning ``""`` would satisfy every
``assertNotIn``.
"""

from __future__ import annotations

import unittest

from hermes_testbench.sanitize import (
    StreamRedactor,
    redact_secrets,
    sanitize_for_agent,
    sanitize_terminal_text,
)


class SanitizeTests(unittest.TestCase):
    """One test per layer, from byte classes up to the agent export composition."""

    def test_sanitize_strips_osc52_osc8_csi_dcs_c1_bidi_tags_and_nul(self) -> None:
        """Every forbidden class is removed and the surviving text is unchanged.

        The exact-equality at the end is the discriminating assertion: an
        absence-only check passes for a sanitizer that drops the payload
        wholesale, while this one fails immediately if a legitimate character
        disappears or the sequence boundaries mis-slice the text. The hostile
        input also covers the two string-sequence terminators (BEL for OSC 52,
        ST for the OSC 8 hyperlink and the DCS reply), so the state machine is
        exercised on both, plus C1 CSI (0x9B) and a bidi override.
        """
        hostile = (
            "ok\x1b]52;c;ZXZpbA==\x07"
            "mid\x1b]8;;https://evil.example\x1b\\link\x1b]8;;\x1b\\"
            "\x1b[2J\x1b[31mred\x1b[0m"
            "\x9b2J"
            "\x1bP1$r\x1b\\"
            "bidi\u202eevil\u202c"
            "tag\U000e0041\U000e007f"
            "nul\x00end\n\tkeep"
        )
        clean = sanitize_terminal_text(hostile)
        self.assertNotIn("\x1b", clean)
        self.assertNotIn("\x9b", clean)
        self.assertNotIn("52;c;", clean)
        self.assertNotIn("\u202e", clean)
        self.assertNotIn("\U000e0041", clean)
        self.assertNotIn("\x00", clean)
        self.assertEqual(clean, "okmidlinkredbidieviltagnulend\n\tkeep")

    def test_redact_secrets_masks_tokens_env_vars_urls_and_pem_keys(self) -> None:
        """Each of the five pattern families replaces the secret, keeps the label.

        The loop asserts only that the original secret text is gone, so it pins
        confidentiality, not formatting: ``NAME=[REDACTED]`` and ``NAME=`` would
        both pass the loop. The single ``[REDACTED]`` check is what rules out
        deletion of the whole line. Whole-string ``redact_secrets`` is the
        at-rest path; chunked streams are covered by the next test.
        """
        raw = (
            "Authorization: Bearer super-secret-jwt-token\n"
            "OPENAI_API_KEY=sk-proj-1234567890abcdef\n"
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345\n"
            "FASTMAIL_TOKEN=fmu1-secret-value\n"
            "https://user:hunter2password@api.example.com/v1\n"
            "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC\n-----END PRIVATE KEY-----\n"
        )
        red = redact_secrets(raw)
        for secret in (
            "super-secret-jwt-token",
            "sk-proj-1234567890abcdef",
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
            "fmu1-secret-value",
            "hunter2password",
            "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC",
        ):
            self.assertNotIn(secret, red)
        self.assertIn("[REDACTED]", red)

    def test_stream_redactor_catches_token_split_across_read_chunks_before_disk_write(self) -> None:
        """A token split across three ``feed`` calls never appears in the emitted stream.

        The key has no newline until the third chunk, so a redactor that scanned
        each chunk independently would emit the prefix on chunk 1 and the
        remainder on chunk 2 — both unredacted. The assertions are therefore on
        the *concatenation* of all ``feed`` returns plus ``flush``, which is
        exactly what the caller writes to disk; that is the property, not the
        return value of any single call. ``done`` is the positive control that
        the tail was released rather than swallowed at EOF.
        """
        sr = StreamRedactor()
        # Split an sk-... key across three tiny chunks without newline until chunk 3
        c1 = sr.feed("booting with key sk-proj-")
        c2 = sr.feed("9876543210fedcba")
        c3 = sr.feed("abcdef012345\ndone")
        c4 = sr.flush()
        persisted = c1 + c2 + c3 + c4
        self.assertNotIn("9876543210fedcba", persisted)
        self.assertNotIn("sk-proj-9876543210fedcbaabcdef012345", persisted)
        self.assertIn("[REDACTED]", persisted)
        self.assertIn("done", persisted)

    def test_sanitize_for_agent_frames_untrusted_bounds_length_and_redacts(self) -> None:
        """The export composition redacts, sanitizes, labels, and bounds — in that order.

        ``max_chars=400`` truncates the *cleaned* body and then appends a marker,
        so the total is allowed to exceed 400; the ``<= 520`` bound is therefore
        a runaway guard on the framing overhead (prefix + suffix), not a hard
        byte budget. The framing assertions are load-bearing: ADR-0004 requires
        that an agent reading the excerpt can tell it came from a child process,
        and the ``truncated`` marker is what tells it the excerpt is partial.
        The two ``assertNotIn`` checks prove both passes ran, but not their
        order — either order removes this payload's escape and secret, so this
        file does not pin the composition order.
        """
        payload = "err\x1b[31m " + ("A" * 5000) + " API_SECRET=topsecretvalue"
        out = sanitize_for_agent(payload, max_chars=400)
        self.assertLessEqual(len(out), 520)
        self.assertNotIn("\x1b", out)
        self.assertNotIn("topsecretvalue", out)
        self.assertIn("[untrusted test output]", out)
        self.assertIn("truncated", out)


if __name__ == "__main__":
    unittest.main()
