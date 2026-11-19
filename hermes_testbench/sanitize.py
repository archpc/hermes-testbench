"""Untrusted text sanitization and in-stream secret redaction (ADR-0004).

Two responsibilities, one policy. ``redact_secrets``/``StreamRedactor`` remove
things that must never reach disk; ``strip_terminal_escapes`` and
``sanitize_terminal_text`` remove bytes that must never reach a terminal, a
report, or an agent. ``sanitize_for_agent`` is the composition used at the
export boundary (MCP, JSON excerpts).

Pattern-based, and that is a stated limitation rather than an oversight: the
matchers cover PEM private-key blocks, userinfo in URLs, bearer tokens, a set
of known provider prefixes, and ``NAME=value`` assignments whose *name*
looks like a credential. An unlabelled high-entropy secret printed with no
context passes straight through. The primary defence is therefore upstream
(credential env scrubbing plus Seatbelt read-denial before the child ever
starts), with these patterns catching what a buggy test echoes back.

``StreamRedactor`` is the disk-write gate: it is fed arbitrary ``read()``
chunks, holds back the unterminated tail, and only emits complete lines (or a
whitespace-bounded prefix past the buffer cap) that have already been
redacted. Secrets straddling a chunk boundary therefore never appear in the
emitted stream. Its one structural blind spot: a PEM *begin* line is only
recognised at the start of a line, so a key introduced mid-line—``json:
-----BEGIN RSA PRIVATE KEY-----``—is never entered as PEM state and its body is
emitted line by line (each line individually matching nothing). Anchoring was
chosen over the alternative, where any prose mentioning a BEGIN marker
swallows the rest of the log.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Each pattern is anchored on the credential's own syntax, never on a bare
# high-entropy shape: false positives here rewrite real test output.
_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN ([A-Z0-9 ]*PRIVATE KEY)-----[\s\S]*?-----END \1-----"
)
# Line-anchored BEGIN: the only place a PEM header can start. See the module
# docstring for the mid-line trade.
_PEM_BEGIN_RE = re.compile(r"^\s*-----BEGIN ([A-Z0-9 ]*PRIVATE KEY)-----\s*$")
_PEM_END_RE = re.compile(r"-----END ([A-Z0-9 ]*PRIVATE KEY)-----")
# scheme://user:password@host — keeps scheme and '@', replaces the password.
_URL_USERINFO_RE = re.compile(r"(https?://[^/\s:@]+:)([^@\s/]+)(@)")
# "Bearer <token>" with a 6-char floor so the literal word is not redacted.
_BEARER_RE = re.compile(r"(?i)\b(Bearer\s+)([^\s\"'`,;]{6,})")
# Known vendor prefixes: OpenAI/Anthropic-style sk-, GitHub gh*_/github_pat_,
# Fastmail fmu1-, Slack xox*-, AWS AKIA. Deliberately a closed list.
_KNOWN_TOKEN_RE = re.compile(
    r"\b(?:"
    r"sk-[A-Za-z0-9_-]{10,}"
    r"|gh[pousr]_[A-Za-z0-9_]{16,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|fmu1-[A-Za-z0-9_-]{8,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r")"
)
# NAME=value / NAME: value where NAME looks like a credential. The 4-char
# value floor keeps "TOKEN=" and "PASSWORD: none" readable.
_ENV_ASSIGN_RE = re.compile(
    r"(?i)(\b[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|AUTH_KEY)"
    r"[A-Z0-9_]*\s*[:=]\s*['\"]?)([^\s'\"`,;]{4,})"
)


def redact_secrets(text: str) -> str:
    """Apply all five patterns in a fixed order, on a whole string.

    Order is not arbitrary: PEM first (a key body would otherwise have its
    lines partially matched by the token patterns), then URL userinfo, bearer,
    known prefixes, and finally the generic ``NAME=value`` rule — the greediest
    last so it cannot pre-empt a more specific match. Whole-string use is for
    records at rest and agent exports; streams go through ``StreamRedactor``,
    which applies the same patterns per complete line and adds the PEM state
    machine that a per-line call cannot express.
    """
    if not text:
        return ""
    out = _PEM_BLOCK_RE.sub(
        r"-----BEGIN \1-----\n[REDACTED]\n-----END \1-----", text
    )
    out = _URL_USERINFO_RE.sub(r"\1[REDACTED]\3", out)
    out = _BEARER_RE.sub(r"\1[REDACTED]", out)
    out = _KNOWN_TOKEN_RE.sub("[REDACTED]", out)
    out = _ENV_ASSIGN_RE.sub(r"\1[REDACTED]", out)
    return out


class StreamRedactor:
    """Line-buffered chunk redactor so secrets split across read() boundaries never hit disk.

    Contract for callers in ``isolation``: feed every chunk in order, write
    exactly what ``feed`` returns, and call ``flush``/``finish`` once at EOF
    and write that too. Nothing may be written from the raw chunk — the
    redactor's whole value is that no byte reaches disk before it has been
    examined in a complete-token context.

    Buffering rules:

    * Only text up to the last newline is released, so a token split mid-line
      is redacted once its line completes. A stream with no newlines (progress
      bars, a hung prompt) accumulates.
    * Past ``MAX_BUFFER_CHARS`` the head is flushed at a whitespace boundary
      and the tail is retained, so a token that straddles the flush point is
      still seen whole. Without a whitespace to cut on, the cut is hard — a
      secret spanning it can be split into two under-length halves and missed.
      That is the price of not growing without bound on a hostile producer.
    * Inside an unclosed PEM block the buffer is trimmed to the overlap tail and
      a single ``[REDACTED]`` is emitted, so a key that never terminates still
      cannot stream out verbatim.

    ``OVERLAP_TAIL`` is 512 chars. OPERATING-GUIDE says 256; the constant is
    the truth and the guide is stale.
    """

    MAX_BUFFER_CHARS = 65536
    OVERLAP_TAIL = 512

    def __init__(self) -> None:
        self._buf: str = ""
        self._in_pem: Optional[str] = None
        self._pem_redacted_emitted: bool = False

    def _redact_complete_lines(self, block: str) -> str:
        """Redact a block of whole lines, carrying PEM state across calls.

        The begin/end markers are passed through verbatim while the *body* is
        replaced by one ``[REDACTED]`` line, so the log still shows that a key
        was printed (and which type) without its bytes. A block whose END never
        arrives leaves ``_in_pem`` set for the next call; ``flush`` is
        responsible for closing that out.
        """
        out_lines: List[str] = []
        for line in block.splitlines(keepends=True):
            if self._in_pem is not None:
                m_end = _PEM_END_RE.search(line)
                if m_end:
                    if not self._pem_redacted_emitted:
                        out_lines.append("[REDACTED]\n")
                    out_lines.append(line)
                    self._in_pem = None
                    self._pem_redacted_emitted = False
                continue
            m_begin = _PEM_BEGIN_RE.search(line)
            if m_begin and not _PEM_END_RE.search(line):
                self._in_pem = m_begin.group(1)
                self._pem_redacted_emitted = False
                out_lines.append(line)
                continue
            out_lines.append(redact_secrets(line))
        return "".join(out_lines)

    def feed(self, chunk: str) -> str:
        """Append ``chunk``, return the redacted prefix that is safe to write now.

        Empty return is normal and does not mean "nothing happened": the text is
        buffered until a line completes or the cap is exceeded. Callers must
        treat the concatenation of all ``feed`` returns plus ``flush`` as the
        stream, never the raw chunks.
        """
        if not chunk:
            return ""
        self._buf += chunk
        last_nl = self._buf.rfind("\n")
        if last_nl >= 0:
            complete = self._buf[: last_nl + 1]
            self._buf = self._buf[last_nl + 1 :]
            return self._redact_complete_lines(complete)
        if len(self._buf) > self.MAX_BUFFER_CHARS:
            if self._in_pem is not None:
                # Cap memory inside an unclosed PEM stream while retaining tail for split END marker
                self._buf = self._buf[-self.OVERLAP_TAIL :]
                if not self._pem_redacted_emitted:
                    self._pem_redacted_emitted = True
                    return "[REDACTED]\n"
                return ""
            # Flush safely past a whitespace boundary, retaining tail for split tokens
            cut = len(self._buf) - self.OVERLAP_TAIL
            ws = self._buf.rfind(" ", 0, cut)
            split_at = ws + 1 if ws > 0 else cut
            head = self._buf[:split_at]
            self._buf = self._buf[split_at:]
            return redact_secrets(head)
        return ""

    def flush(self) -> str:
        """Release the unterminated tail at EOF; close an open PEM block.

        An unterminated PEM body emits one ``[REDACTED]`` (not the retained
        bytes), so a truncated key is not "handled" by leaking whatever fit in
        the buffer. Calling twice is safe: the second call returns "".
        """
        if not self._buf and self._in_pem is None:
            return ""
        rem = self._buf
        self._buf = ""
        if self._in_pem is not None:
            emitted = self._pem_redacted_emitted
            self._in_pem = None
            self._pem_redacted_emitted = False
            return "" if emitted else "[REDACTED]\n"
        return redact_secrets(rem)

    finish = flush


def _is_forbidden_codepoint(cp: int) -> bool:
    """Codepoints removed outright, not escaped.

    Tab and newline survive because they are the only whitespace a log needs.
    Everything else in C0/DEL/C1 goes, plus the invisible-character classes
    that make output lie: soft hyphen and friends, zero-widths, bidi overrides
    and isolates (which can visually reorder a PASS into a FAIL), interlinear
    annotation controls, and Plane-14 tag characters (invisible ASCII
    smuggling). This is a denylist of *categories*, so a new exotic
    invisible is invisible here too — the honest characterisation is "the
    known spoofing alphabet", not "all non-printing codepoints".
    """
    if cp == 0x09 or cp == 0x0A:  # \t, \n allowed
        return False
    if cp < 0x20 or cp == 0x7F:  # C0 + DEL
        return True
    if 0x80 <= cp <= 0x9F:  # 8-bit C1
        return True
    if cp in (0x00AD, 0x034F, 0x061C, 0x180E, 0xFEFF):
        return True
    if 0x200B <= cp <= 0x200F:
        return True
    if 0x202A <= cp <= 0x202E:  # bidi overrides
        return True
    if 0x2060 <= cp <= 0x206F:  # word joiners + bidi isolates
        return True
    if 0xFFF9 <= cp <= 0xFFFB:
        return True
    if 0xE0000 <= cp <= 0xE007F:  # Plane-14 tag chars
        return True
    return False


def strip_terminal_escapes(text: str) -> str:
    """State-machine stripper for CSI, OSC (52/8/...), DCS/SOS/PM/APC, and C1 escapes.

    A hand-rolled scanner, not a regex, because these sequences nest and
    terminate in ways regexes handle badly: ``ESC [`` params then a final byte
    in 0x40–0x7E; string sequences (OSC/DCS/SOS/PM/APC) running until BEL, ST
    (``ESC \\``) or a newline; 8-bit CSI/DCS equivalents where a single byte
    starts the sequence.

    Two deliberate choices. A newline inside a string sequence is *kept* (and
    terminates it), so a runaway OSC title cannot swallow the rest of a log —
    the alternative loses every line after a malformed escape. And an
    unterminated sequence is dropped to end-of-input rather than emitted,
    because emitting it would hand a live escape to whatever renders next.

    Only escapes are removed here — control characters and bidi are
    ``_is_forbidden_codepoint``'s job, and redaction is not applied at all.
    """
    if not text:
        return ""
    out: List[str] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        cp = ord(ch)
        if cp == 0x1B:  # ESC
            if i + 1 >= n:
                i += 1
                continue
            nxt = text[i + 1]
            ncp = ord(nxt)
            if nxt == "[":  # CSI: ESC [ params final(0x40..0x7E)
                i += 2
                while i < n and 0x20 <= ord(text[i]) <= 0x3F:
                    i += 1
                if i < n and 0x40 <= ord(text[i]) <= 0x7E:
                    i += 1
                continue
            if nxt in ("]", "P", "X", "^", "_"):  # OSC / DCS / SOS / PM / APC until BEL, ST, or NL
                i += 2
                while i < n:
                    c = text[i]
                    if c == "\x07" or c == "\n" or c == "\x9c":
                        if c == "\n":
                            out.append("\n")
                        i += 1
                        break
                    if c == "\x1b" and i + 1 < n and text[i + 1] == "\\":
                        i += 2
                        break
                    i += 1
                continue
            # 2-byte ESC sequence (e.g. ESC c, ESC 7, ESC ( B)
            if 0x20 <= ncp <= 0x2F:
                i += 2
                while i < n and 0x20 <= ord(text[i]) <= 0x2F:
                    i += 1
                if i < n and 0x30 <= ord(text[i]) <= 0x7E:
                    i += 1
                continue
            i += 2
            continue
        if cp == 0x9B:  # 8-bit CSI
            i += 1
            while i < n and 0x20 <= ord(text[i]) <= 0x3F:
                i += 1
            if i < n and 0x40 <= ord(text[i]) <= 0x7E:
                i += 1
            continue
        if cp in (0x90, 0x98, 0x9D, 0x9E, 0x9F):  # 8-bit DCS/SOS/OSC/PM/APC
            i += 1
            while i < n:
                c = text[i]
                if c in ("\x07", "\x9c", "\n"):
                    if c == "\n":
                        out.append("\n")
                    i += 1
                    break
                if c == "\x1b" and i + 1 < n and text[i + 1] == "\\":
                    i += 2
                    break
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def sanitize_terminal_text(text: str, max_chars: Optional[int] = None) -> str:
    """Make untrusted text safe to store, print, and embed in XML/JSON.

    Three passes, in order: CRLF→LF (so a stored log has one newline
    convention), escape stripping, then codepoint filtering. The order matters
    — filtering first would leave escape *bodies* behind as literal text.

    ``max_chars`` truncates **after** cleaning and appends a marker, so the
    result can exceed ``max_chars`` by the marker's length. That is the
    documented behaviour (``tests/test_sanitize`` bounds the total, not the
    body) and is why callers use it for excerpts rather than for hard byte
    budgets.

    Sanitization is not redaction: a secret that survives ``redact_secrets``
    survives this too. Callers that need both must compose them, as
    ``reporting._clean_field`` and ``store._upsert_run_row`` do.
    """
    if not text:
        return ""
    norm = text.replace("\r\n", "\n")
    stripped = strip_terminal_escapes(norm)
    cleaned = "".join(ch for ch in stripped if not _is_forbidden_codepoint(ord(ch)))
    if max_chars is not None and max_chars > 0 and len(cleaned) > max_chars:
        return cleaned[:max_chars] + "\n... [truncated]"
    return cleaned


def sanitize_for_agent(text: str, max_chars: int = 2000) -> str:
    """Export boundary: redact, strip, truncate, and label as untrusted.

    The prefix is not decoration — ADR-0004 requires that anything an agent
    sees be marked as untrusted test output, so a model reading a failure
    message knows the text came from a child process rather than from
    TestBench. The suffix names ``sanitized.log`` as the way to see more: the
    agent gets a bounded excerpt and a pointer, never an unbounded dump and
    never ``stdout.log``/``stderr.log``.

    ``max_chars <= 0`` disables truncation but keeps the framing — no caller
    does that today, and it is the one way to get an unbounded export.
    """
    redacted = redact_secrets(text or "")
    cleaned = sanitize_terminal_text(redacted)
    prefix = "[untrusted test output]\n"
    suffix = "\n... [truncated; inspect sanitized.log]"
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + suffix
    return prefix + cleaned
