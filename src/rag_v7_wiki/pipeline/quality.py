"""Quality pass над сгенерированной страницей.

После synthesize_pages берём страницу, спрашиваем LLM:
- хорошо ли структурирована?
- ссылается ли на claim-ы / wikilinks?
- нет ли явных противоречий с собой?
Если нужно — один retry с issues/suggestions в user-prompt.
"""

from __future__ import annotations

from typing import Any, Callable

from rag_v7_wiki.config import WikiConfig
from rag_v7_wiki.protocols import LLM
from rag_v7_wiki.schemas import PageQualityResponse


QUALITY_SYSTEM = (
    "Ты оцениваешь качество wiki-страницы. Учти:\n"
    "- структурирована ли страница (заголовок, секции, списки)?\n"
    "- соответствует ли содержание перечисленным фактам?\n"
    "- есть ли [[wikilinks]] на канонические имена связанных сущностей?\n"
    "- нет ли явных самопротиворечий?\n"
    "Верни общий score (0..1), список issues и suggestions, "
    "и needs_resynthesis=true если оценка ниже хорошей."
)


def _build_user_prompt(
    title: str,
    content_md: str,
    facts_block: str,
) -> str:
    return (
        f"Заголовок: {title}\n\n"
        f"Факты, на которых построена страница:\n{facts_block}\n\n"
        f"Текущий контент:\n```markdown\n{content_md}\n```"
    )


def score_page(
    title: str,
    content_md: str,
    facts_block: str,
    llm: LLM,
) -> PageQualityResponse:
    user = _build_user_prompt(title, content_md, facts_block)
    return llm.structured(QUALITY_SYSTEM, user, PageQualityResponse)


def quality_pass(
    title: str,
    content_md: str,
    facts_block: str,
    llm: LLM,
    config: WikiConfig,
    resynthesize: Callable[[str, str], tuple[str, str]] | None = None,
) -> tuple[str, str, PageQualityResponse]:
    """Возвращает (final_title, final_content_md, last_quality_response).

    Если score < threshold и `resynthesize` передан — делаем до
    `quality_resynthesis_max_attempts` попыток. Каждая попытка получает
    issues и suggestions предыдущего score-pass-а как фидбэк.
    """
    final_title = title
    final_content = content_md

    response = score_page(final_title, final_content, facts_block, llm)
    attempts = 0
    while (
        resynthesize is not None
        and (
            response.needs_resynthesis
            or response.score < config.quality_score_threshold
        )
        and attempts < config.quality_resynthesis_max_attempts
    ):
        feedback_lines: list[str] = []
        if response.issues:
            feedback_lines.append("Проблемы:")
            feedback_lines.extend(f"- {x}" for x in response.issues)
        if response.suggestions:
            feedback_lines.append("Предложения:")
            feedback_lines.extend(f"- {x}" for x in response.suggestions)
        feedback = "\n".join(feedback_lines) or "Нужна более качественная версия."

        new_title, new_content = resynthesize(feedback, final_content)
        if new_content and new_content.strip():
            final_title = new_title or final_title
            final_content = new_content.strip()
            response = score_page(final_title, final_content, facts_block, llm)
        attempts += 1

    return final_title, final_content, response


def build_facts_block(claims: list[dict[str, Any]]) -> str:
    """Утилита: компактное представление фактов для score-prompt-а."""
    if not claims:
        return "(нет фактов)"
    lines: list[str] = []
    for c in claims:
        obj = (
            c.get("object_canonical_name")
            or c.get("object_text")
            or "—"
        )
        lines.append(
            f"- id={c['id']}: [{c['predicate']}] → {obj} :: {c['claim_text']} "
            f"(×{c.get('times_confirmed', 1)}, conf={c.get('confidence', 0):.2f})"
        )
    return "\n".join(lines)
