"""Untrusted text sanitization and in-stream secret redaction (ADR-0004)."""

from __future__ import annotations

import re
from typing import List, Optional

_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN ([A-Z0-9 ]*PRIVATE KEY)-----[\s\S]*?-----END \1-----"
)
_PEM_BEGIN_RE = re.compile(r"-----BEGIN ([A-Z0-9 ]*PRIVATE KEY)-----")
_PEM_END_RE = re.compile(r"-----END ([A-Z0-9 ]*PRIVATE KEY)-----")
_URL_USERINFO_RE = re.compile(r"(https?://[^/\s:@]+:)([^@\s/]+)(@)")
_BEARER_RE = re.compile(r"(?i)\b(Bearer\s+)([^\s\"'`,;]{6,})")
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
_ENV_ASSIGN_RE = re.compile(
    r"(?i)(\b[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|AUTH_KEY)"
    r"[A-Z0-9_]*\s*[:=]\s*['\"]?)([^\s'\"`,;]{4,})"
)
# URL userinfo whose closing '@' has not arrived yet (buffer cut mid-password).
# Only used on the fail-closed over-cap stream path, never in redact_secrets,
# so a line-final host:port in ordinary text is not over-redacted.
_URL_USERINFO_OPEN_RE = re.compile(r"(https?://[^/\s:@]+:)[^@\s/]+$")
# Head ends inside a URL authority before any '@' or '/': a userinfo password
# may follow beyond the cut (possibly with a long username).
_URL_AUTHORITY_OPEN_RE = re.compile(r"https?://[^/\s@]*$")
# Head ends with a secret key+separator (or Bearer) expecting a value: after a
# whitespace split the retained tail would carry the value WITHOUT its key.
_EXPECTS_VALUE_RE = re.compile(
    r"(?i)(?:\b[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|AUTH_KEY)"
    r"[A-Z0-9_]*\s*[:=]\s*['\"]?|\bBearer\s+)$"
)
# Head ends with a key+separator and a value shorter than the 4-char minimum
# of _ENV_ASSIGN_RE: the complete-pattern check cannot fire, but the value is
# still a secret continuation across the cut.
_ENV_ASSIGN_SHORT_RE = re.compile(
    r"(?i)(\b[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|AUTH_KEY)"
    r"[A-Z0-9_]*\s*[:=]\s*['\"]?)[^\s'\"`,;]{0,3}$"
)


def redact_secrets(text: str) -> str:
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


def _ends_in_secret(head: str) -> bool:
    """True when a secret pattern's match runs to the end of ``head``.

    A buffer cut there leaves the retained tail INSIDE a secret value (an
    assignment value, a long known token, a bearer token, or URL userinfo),
    so the tail must be treated as secret continuation, not emitted raw.
    """
    for rx in (_ENV_ASSIGN_RE, _KNOWN_TOKEN_RE, _BEARER_RE, _URL_USERINFO_RE):
        for m in rx.finditer(head):
            if m.end() == len(head):
                return True
    return False


class StreamRedactor:
    """Line-buffered chunk redactor so secrets split across read() boundaries never hit disk."""

    MAX_BUFFER_CHARS = 65536
    OVERLAP_TAIL = 512

    def __init__(self) -> None:
        self._buf: str = ""
        self._in_pem: Optional[str] = None
        self._pem_redacted_emitted: bool = False
        self._secret_tail: bool = False

    def _redact_complete_lines(self, block: str) -> str:
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
                # Keep any benign prefix and the marker itself; anything after
                # the marker on this line is already key body — suppress it.
                out_lines.append(line[: m_begin.end()] + "\n")
                continue
            out_lines.append(redact_secrets(line))
        return "".join(out_lines)

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        prefix = ""
        if self._secret_tail:
            # Continuation of a secret value cut mid-stream: fail closed and
            # discard up to the next newline (the head already shows [REDACTED]).
            nl = self._buf.find("\n")
            if nl < 0:
                self._buf = ""
                return ""
            self._buf = self._buf[nl + 1 :]
            self._secret_tail = False
            prefix = "\n"
        last_nl = self._buf.rfind("\n")
        if last_nl >= 0:
            complete = self._buf[: last_nl + 1]
            self._buf = self._buf[last_nl + 1 :]
            return prefix + self._redact_complete_lines(complete)
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
            tail = self._buf[split_at:]
            if (
                _ends_in_secret(head)
                or _EXPECTS_VALUE_RE.search(head)
                or _ENV_ASSIGN_SHORT_RE.search(head)
                or _URL_USERINFO_OPEN_RE.search(head)
                or _URL_AUTHORITY_OPEN_RE.search(head)
                or _URL_USERINFO_OPEN_RE.search(tail)
            ):
                # The cut landed inside a secret value, or separated a secret
                # from its key (whitespace split: head keeps 'TOKEN = ', tail
                # carries the value with no key to match on). The retained tail
                # would be raw secret continuation: fail closed — redact what
                # the head already exposes, drop the tail, suppress to newline.
                expects_value = bool(_EXPECTS_VALUE_RE.search(head))
                head = _URL_USERINFO_OPEN_RE.sub(r"\1[REDACTED]", head)
                head = _ENV_ASSIGN_SHORT_RE.sub(r"\1[REDACTED]", head)
                redacted = redact_secrets(head)
                if expects_value and not redacted.endswith("[REDACTED]"):
                    # The value was dropped with its key left behind: mark the
                    # suppression so the log does not look merely truncated.
                    redacted += "[REDACTED]"
                self._buf = ""
                self._secret_tail = True
                return prefix + redacted
            self._buf = tail
            return prefix + redact_secrets(head)
        return prefix

    def flush(self) -> str:
        if self._secret_tail:
            self._buf = ""
            self._secret_tail = False
            return ""
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
    """State-machine stripper for CSI, OSC (52/8/...), DCS/SOS/PM/APC, and C1 escapes."""
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
    if not text:
        return ""
    norm = text.replace("\r\n", "\n")
    stripped = strip_terminal_escapes(norm)
    cleaned = "".join(ch for ch in stripped if not _is_forbidden_codepoint(ord(ch)))
    if max_chars is not None and max_chars > 0 and len(cleaned) > max_chars:
        return cleaned[:max_chars] + "\n... [truncated]"
    return cleaned


def sanitize_for_agent(text: str, max_chars: int = 2000) -> str:
    redacted = redact_secrets(text or "")
    cleaned = sanitize_terminal_text(redacted)
    prefix = "[untrusted test output]\n"
    suffix = "\n... [truncated; inspect sanitized.log]"
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + suffix
    return prefix + cleaned
