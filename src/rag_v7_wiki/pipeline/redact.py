"""PII / secrets redaction перед чанкингом.

Regex-based замена известных секретных паттернов на маркеры
вида `[REDACTED:<kind>:<n>]`. Без LLM, без БД — чистая функция.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Подбор паттернов: высокая специфичность, low false-positive rate.
# Каждый паттерн возвращает «знак секрета» — заведомо узнаваемый префикс/формат.
_BASE_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"),
    "github_pat": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]+?\.[A-Za-z0-9_\-]+?\.[A-Za-z0-9_\-]+\b"),
    "private_key_block": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
        r"[\s\S]+?"
        r"-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
    ),
    "basic_auth_url": re.compile(r"\b[a-z]+://[^\s/@:]+:[^\s/@]+@[^\s]+", re.IGNORECASE),
}

_STRICT_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b\+?\d[\d\s().\-]{8,}\d\b"),
}


@dataclass(slots=True, frozen=True)
class RedactionMatch:
    kind: str
    start: int
    end: int
    sample: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "sample": self.sample,
        }


def redact(text: str, strict: bool = False) -> tuple[str, list[RedactionMatch]]:
    """Возвращает (redacted_text, matches).

    Каждое совпадение заменяется на `[REDACTED:<kind>:<n>]`. n — порядковый
    номер этого вида в документе (полезно при сверке цитат).
    """
    if not text:
        return text, []

    matches: list[RedactionMatch] = []
    patterns = dict(_BASE_PATTERNS)
    if strict:
        patterns.update(_STRICT_PATTERNS)

    for kind, pattern in patterns.items():
        for m in pattern.finditer(text):
            sample = m.group(0)
            matches.append(
                RedactionMatch(
                    kind=kind,
                    start=m.start(),
                    end=m.end(),
                    sample=sample[:8] + ("…" if len(sample) > 8 else ""),
                )
            )

    if not matches:
        return text, []

    # Сортируем по позиции, разрешаем перекрытия в пользу более раннего.
    matches.sort(key=lambda x: (x.start, -x.end))
    deduped: list[RedactionMatch] = []
    last_end = -1
    for m in matches:
        if m.start >= last_end:
            deduped.append(m)
            last_end = m.end

    counters: dict[str, int] = {}
    out_parts: list[str] = []
    cursor = 0
    for m in deduped:
        out_parts.append(text[cursor : m.start])
        counters[m.kind] = counters.get(m.kind, 0) + 1
        out_parts.append(f"[REDACTED:{m.kind}:{counters[m.kind]}]")
        cursor = m.end
    out_parts.append(text[cursor:])

    return "".join(out_parts), deduped
