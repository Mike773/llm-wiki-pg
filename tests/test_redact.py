from __future__ import annotations

from rag_v7_wiki.pipeline.redact import redact


def test_redact_no_secrets_passthrough() -> None:
    text = "Просто обычный текст без секретов."
    out, matches = redact(text)
    assert out == text
    assert matches == []


def test_redact_openai_key() -> None:
    text = "use sk-ABCDEFGHIJ1234567890XYZ now"
    out, matches = redact(text)
    assert "sk-ABCDEFGHIJ1234567890XYZ" not in out
    assert "[REDACTED:openai_key:1]" in out
    assert any(m.kind == "openai_key" for m in matches)


def test_redact_github_pat_and_aws() -> None:
    text = (
        "creds: ghp_aaaaaaaaaaaaaaaaaaaa1234, AKIA1234567890ABCDEF, "
        "AKIAEXAMPLEKEY12345"
    )
    out, matches = redact(text)
    assert "ghp_aaaaaaaaaaaaaaaaaaaa1234" not in out
    assert "AKIA1234567890ABCDEF" not in out
    kinds = {m.kind for m in matches}
    assert "github_pat" in kinds
    assert "aws_access_key" in kinds


def test_redact_basic_auth_url() -> None:
    text = "fetch https://user:pass@example.com/path"
    out, matches = redact(text)
    assert "user:pass" not in out
    assert any(m.kind == "basic_auth_url" for m in matches)


def test_redact_jwt() -> None:
    jwt = "eyJhbGciOi.eyJzdWIiOi.SflKxw"
    text = f"token={jwt}"
    out, matches = redact(text)
    assert jwt not in out
    assert any(m.kind == "jwt" for m in matches)


def test_redact_strict_mode_email_phone() -> None:
    text = "ping me at user@example.com or +1 (555) 123-4567"
    relaxed_out, relaxed = redact(text, strict=False)
    assert "user@example.com" in relaxed_out
    assert relaxed == []

    strict_out, strict_matches = redact(text, strict=True)
    assert "user@example.com" not in strict_out
    kinds = {m.kind for m in strict_matches}
    assert "email" in kinds
    assert "phone" in kinds


def test_redact_counter_per_kind() -> None:
    text = "first sk-AAAAAAAAAAAAAAAAAAAA1, second sk-BBBBBBBBBBBBBBBBBBBB2"
    out, matches = redact(text)
    assert "[REDACTED:openai_key:1]" in out
    assert "[REDACTED:openai_key:2]" in out
    assert sum(1 for m in matches if m.kind == "openai_key") == 2
