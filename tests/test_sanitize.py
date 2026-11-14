"""TDD tests for untrusted text sanitization and chunk-safe in-stream secret redaction."""

from __future__ import annotations

import unittest

from hermes_testbench.sanitize import (
    StreamRedactor,
    redact_secrets,
    sanitize_for_agent,
    sanitize_terminal_text,
)


class SanitizeTests(unittest.TestCase):
    def test_sanitize_strips_osc52_osc8_csi_dcs_c1_bidi_tags_and_nul(self) -> None:
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
        payload = "err\x1b[31m " + ("A" * 5000) + " API_SECRET=topsecretvalue"
        out = sanitize_for_agent(payload, max_chars=400)
        self.assertLessEqual(len(out), 520)
        self.assertNotIn("\x1b", out)
        self.assertNotIn("topsecretvalue", out)
        self.assertIn("[untrusted test output]", out)
        self.assertIn("truncated", out)


if __name__ == "__main__":
    unittest.main()
