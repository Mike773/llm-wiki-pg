from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# STEP 1b — document summary (для needs_chunking=true)
# ============================================================


class DocumentSummaryResponse(BaseModel):
    summary: str = Field(
        description="Сжатое 200–500 слов изложение содержания документа."
    )


# ============================================================
# STEP 3 — Entity extraction
# ============================================================


class ExtractedEntity(BaseModel):
    entity_type: str = Field(description="Тип сущности (Person, Project, ...).")
    canonical_name: str = Field(description="Каноническое имя.")
    aliases: list[str] = Field(default_factory=list)
    salient_attrs: dict[str, str] = Field(default_factory=dict)
    supporting_chunk_indices: list[int] = Field(
        default_factory=list,
        description="Индексы чанков (по порядку, как они переданы), где сущность упомянута.",
    )


class EntitiesResponse(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)


# ============================================================
# STEP 4 — Entity resolution arbiter
# ============================================================


class ResolutionDecision(BaseModel):
    decision: Literal["same", "new"] = Field(
        description="'same' — это та же сущность что и одна из кандидатов; 'new' — действительно новая."
    )
    matched_candidate_index: int | None = Field(
        default=None,
        description="Индекс кандидата (если decision='same'). null если 'new'.",
    )
    reason: str = ""


# ============================================================
# STEP 5 — Claim extraction
# ============================================================


class ExtractedClaim(BaseModel):
    subject_canonical_name: str = Field(
        description="Каноническое имя субъекта. Должно совпадать с одной из извлечённых entities."
    )
    predicate: str = Field(description="Предикат в свободной форме.")
    object_kind: Literal["entity", "literal"]
    object_canonical_name: str | None = Field(
        default=None,
        description="Если object_kind='entity', каноническое имя объекта.",
    )
    object_text: str | None = Field(
        default=None,
        description="Если object_kind='literal', буквальный текст объекта.",
    )
    claim_text: str = Field(description="Атомарное утверждение в естественной форме.")
    citation_chunk_indices: list[int] = Field(default_factory=list)
    confidence_hint: float = Field(default=0.5, ge=0.0, le=1.0)


class ClaimsResponse(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


# ============================================================
# STEP 6 — Claim supersession arbiter
# ============================================================


class SupersessionDecision(BaseModel):
    decision: Literal["same", "supersedes_old", "contradiction", "orthogonal"] = Field(
        description=(
            "'same' — новый claim повторяет существующий; "
            "'supersedes_old' — новый отменяет старый; "
            "'contradiction' — противоречие, оба остаются с флагом; "
            "'orthogonal' — независимый, оба остаются."
        )
    )
    reason: str = ""


# ============================================================
# STEP 7 — Wiki page synthesis
# ============================================================


class PageSynthesisResponse(BaseModel):
    title: str = Field(description="Заголовок страницы.")
    content_md: str = Field(
        description=(
            "Полный markdown-контент страницы. Можно использовать [[wikilinks]] на "
            "канонические имена других сущностей этого направления."
        )
    )


# ============================================================
# STEP 5b — Predicate normalization arbiter
# ============================================================


class PredicateResolutionDecision(BaseModel):
    decision: Literal["same", "new"] = Field(
        description="'same' — это тот же предикат что и один из кандидатов; 'new' — действительно новый."
    )
    matched_canonical: str | None = Field(
        default=None,
        description="Каноническое имя совпавшего предиката (если decision='same').",
    )
    proposed_canonical: str | None = Field(
        default=None,
        description="Каноническое имя для нового предиката (если decision='new').",
    )
    reason: str = ""


# ============================================================
# STEP 6c — Contradiction auto-resolution arbiter
# ============================================================


class ContradictionResolutionDecision(BaseModel):
    winner: Literal["a", "b", "unresolved"] = Field(
        description=(
            "'a' — claim A корректнее, B уходит в superseded; "
            "'b' — наоборот; 'unresolved' — оба остаются flagged_contradiction."
        )
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Уверенность арбитра в решении (0..1).",
    )
    reason: str = ""


# ============================================================
# STEP 7a — Source page synthesis
# ============================================================


class SourcePageResponse(BaseModel):
    title: str = Field(description="Заголовок страницы-источника.")
    abstract: str = Field(description="Краткий пересказ источника, 3–5 предложений.")
    key_takeaways: list[str] = Field(
        default_factory=list,
        description="Ключевые тезисы источника, по одному на пункт.",
    )
    top_entities: list[str] = Field(
        default_factory=list,
        description=(
            "Канонические имена ключевых сущностей источника — для wikilinks."
        ),
    )


# ============================================================
# STEP 7d — Page quality pass
# ============================================================


class PageQualityResponse(BaseModel):
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Общий quality score: структурированность, цитирования, согласованность."
        ),
    )
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    needs_resynthesis: bool = Field(
        default=False,
        description="True если страницу следует пересинтезировать с учётом issues/suggestions.",
    )
