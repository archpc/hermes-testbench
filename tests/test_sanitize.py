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


class StreamRedactorLongValueTests(unittest.TestCase):
    """ADR-0004: unredacted raw output is never persisted. A secret assignment
    value longer than MAX_BUFFER_CHARS on a line with no whitespace forces the
    no-whitespace flush cut; the retained tail is the secret's continuation and
    must not be emitted raw (found by campaign probe: 512-char password tail
    leaked on a 66K single-line value, reproducible with one feed + flush).
    """

    SECRET = "Zk9" * 22000  # 66000 chars, no whitespace

    def assert_no_fragment(self, out: str) -> None:
        self.assertNotIn(self.SECRET[-512:], out)
        for i in range(0, len(self.SECRET) - 32, 4096):
            self.assertNotIn(self.SECRET[i:i + 32], out)

    def test_long_assignment_value_tail_never_reaches_output_single_feed(self) -> None:
        sr = StreamRedactor()
        out = sr.feed("PASSWORD=" + self.SECRET) + sr.flush()
        self.assertTrue(out.startswith("PASSWORD=[REDACTED]"))
        self.assert_no_fragment(out)

    def test_long_assignment_value_tail_chunked(self) -> None:
        sr = StreamRedactor()
        text = "PASSWORD=" + self.SECRET
        out = []
        for i in range(0, len(text), 700):
            out.append(sr.feed(text[i:i + 700]))
        out.append(sr.flush())
        joined = "".join(out)
        self.assertIn("[REDACTED]", joined)
        self.assert_no_fragment(joined)

    def test_long_assignment_value_then_newline_keeps_following_lines(self) -> None:
        sr = StreamRedactor()
        out = sr.feed("PASSWORD=" + self.SECRET)
        out += sr.feed("\nnext line\n")
        out += sr.flush()
        self.assert_no_fragment(out)
        self.assertIn("next line", out)

    def test_long_known_token_tail_is_not_emitted(self) -> None:
        sr = StreamRedactor()
        out = sr.feed("sk-" + "A" * 70000) + sr.flush()
        self.assertIn("[REDACTED]", out)
        self.assertNotIn("A" * 64, out)

    def test_long_url_userinfo_password_never_reaches_output(self) -> None:
        # The closing '@' is beyond the buffer cut, so the complete-URL pattern
        # cannot match: the open userinfo must still be redacted fail-closed.
        pw = "Pq7" * 23000
        sr = StreamRedactor()
        out = sr.feed("https://user:" + pw + "@api.example.com/v1") + sr.flush()
        self.assertIn("[REDACTED]", out)
        self.assertTrue(out.startswith("https://user:[REDACTED]"))
        for i in range(0, len(pw) - 32, 4096):
            self.assertNotIn(pw[i:i + 32], out)

    def test_whitespace_split_orphaned_value_never_leaks(self) -> None:
        # SEC-1 (review R2): with a space before the cut, the head keeps the
        # key+separator and the retained tail carries the value WITHOUT the
        # key; redact_secrets(tail) cannot match, so the value must not be
        # emitted raw. 'TOKEN = <75K>' leaked in full before this fix.
        val = "Qw8" * 25000
        sr = StreamRedactor()
        out = sr.feed("TOKEN = " + val) + sr.flush()
        self.assertIn("[REDACTED]", out)
        self.assertNotIn(val[:64], out)
        self.assertNotIn(val[-512:], out)

    def test_bearer_orphaned_token_after_padding_never_leaks(self) -> None:
        tok = "Zx5" * 1700
        sr = StreamRedactor()
        out = sr.feed("x " * 33000 + "Authorization: Bearer " + tok) + sr.flush()
        self.assertIn("[REDACTED]", out)
        self.assertNotIn(tok[:64], out)
        self.assertNotIn(tok[-64:], out)

    def test_short_value_straddling_cut_never_leaks(self) -> None:
        # Sub-minimum value with the cut INSIDE it (no trailing delimiter):
        # exercises _ends_in_secret's short-value reach, not the terminated
        # shape covered by test_short_value_terminated_at_cut_never_leaks.
        text = "a" * 65100 + "PASSWORD=***" + "z" * 509
        sr = StreamRedactor()
        out = sr.feed(text) + sr.flush()
        self.assertNotIn("abcdef", out)

    def test_long_username_url_password_never_leaks(self) -> None:
        # Cut lands inside a long URL username; the tail lacks the scheme so
        # the complete-URL pattern cannot match the password.
        pw = "P" * 64
        sr = StreamRedactor()
        out = sr.feed("https://" + "u" * 66000 + ":" + pw + "@host/path") + sr.flush()
        self.assertNotIn(pw, out)

    def test_midline_pem_begin_body_never_leaks(self) -> None:
        # SEC-4 (review R2): _PEM_BEGIN_RE was line-anchored, so a BEGIN marker
        # with a prefix ('pad -----BEGIN...') never entered PEM state and every
        # body line was emitted raw.
        body = "\n".join("B" * 64 for _ in range(30))
        text = "pad -----BEGIN RSA PRIVATE KEY-----\n" + body + "\n-----END RSA PRIVATE KEY-----\n"
        sr = StreamRedactor()
        out = sr.feed(text) + sr.flush()
        self.assertNotIn("B" * 64, out)
        self.assertIn("[REDACTED]", out)

    def test_midline_pem_mention_with_following_prose_is_preserved(self) -> None:
        # A mid-line BEGIN with nothing after the marker is ambiguous until the
        # next line: prose means it was a mention, not a key (test_r6 contract).
        text = "note -----BEGIN RSA PRIVATE KEY-----\nRan 5 tests in 0.1s\n\nOK\n"
        sr = StreamRedactor()
        out = sr.feed(text) + sr.flush()
        self.assertIn("Ran 5 tests in 0.1s", out)
        self.assertIn("note -----BEGIN RSA PRIVATE KEY-----", out)

    def test_midline_pem_with_same_line_body_suppresses_body(self) -> None:
        body = "B" * 64
        text = "pad -----BEGIN RSA PRIVATE KEY-----" + body + "\n" + body + "\n-----END RSA PRIVATE KEY-----\n"
        sr = StreamRedactor()
        out = sr.feed(text) + sr.flush()
        self.assertNotIn(body, out)
        self.assertIn("[REDACTED]", out)

    def test_pem_prefix_secret_is_redacted(self) -> None:
        # R3-1 (review R3): a secret in the prefix before a mid-line BEGIN
        # must be redacted, not emitted raw with the marker slice.
        tok = "secret" + "token"
        text = "TOKEN=" + tok + " -----BEGIN RSA PRIVATE KEY-----\n" + "B" * 64 + "\n-----END RSA PRIVATE KEY-----\n"
        sr = StreamRedactor()
        out = sr.feed(text) + sr.flush()
        self.assertNotIn(tok, out)
        self.assertNotIn("B" * 64, out)
        self.assertIn("[REDACTED]", out)

    def test_pem_prefix_secret_immediate_branch_is_redacted(self) -> None:
        # N2 (convergence review): the IMMEDIATE PEM entry branch (base64 body
        # already on the marker line, no space) must also redact its prefix.
        # The deferred-path test above does not exercise this branch.
        tok = "secret" + "token"
        body = "MIIB" + "A" * 60
        text = "TOKEN=" + tok + " -----BEGIN RSA PRIVATE KEY-----" + body + "\n" + body + "\n-----END RSA PRIVATE KEY-----\n"
        sr = StreamRedactor()
        out = sr.feed(text) + sr.flush()
        self.assertNotIn(tok, out)
        self.assertNotIn(body, out)
        self.assertIn("[REDACTED]", out)

    def test_short_value_terminated_at_cut_never_leaks(self) -> None:
        # Genuine SEC-3 shape: a sub-minimum value followed by whitespace
        # exactly at the cut ('PASSWORD=*** '). Exercises _ENV_ASSIGN_SHORT_RE
        # (the straddling test above exercises _ends_in_secret instead).
        key = "PASS" + "WORD"
        text = "a" * 65100 + key + "=abc " + "z" * 512
        sr = StreamRedactor()
        out = sr.feed(text) + sr.flush()
        self.assertNotIn("=abc", out)
        self.assertIn("[REDACTED]", out)

    def test_split_scheme_url_password_never_leaks(self) -> None:
        # R3-2 (review R3): the cut lands inside the literal 'https://' so
        # neither head nor tail carries a complete scheme; the password must
        # still not leak (narrow alignment window, same F8 class).
        text = "a" * 65027 + "https://u:" + "P" * 400 + "@h/x" + "z" * 100
        sr = StreamRedactor()
        out = sr.feed(text) + sr.flush()
        self.assertNotIn("P" * 64, out)

    def test_benign_long_line_with_whitespace_is_preserved(self) -> None:
        sr = StreamRedactor()
        text = "word " * 15000
        out = sr.feed(text) + sr.flush()
        self.assertEqual(out, text)

    def test_benign_long_line_without_whitespace_is_preserved(self) -> None:
        sr = StreamRedactor()
        text = "x" * 70000
        out = sr.feed(text) + sr.flush()
        self.assertEqual(out, text)


if __name__ == "__main__":
    unittest.main()
