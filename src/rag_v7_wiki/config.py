from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WikiConfig:
    expected_embedding_dim: int = 2560

    # Чанкинг — character-based (semantic-text-splitter без tokenizer'а).
    # ~4000 символов ≈ 1000 английских токенов; для русского — ~1500 токенов.
    chunk_size_chars: int = 4000
    chunk_overlap_chars: int = 400

    entity_resolution_pool: int = 8
    entity_resolution_threshold: float = 0.85

    claim_supersession_pool: int = 8
    claim_supersession_threshold: float = 0.82

    predicate_normalization_pool: int = 8
    predicate_normalization_threshold: float = 0.85

    contradiction_auto_resolve_min_confidence: float = 0.7

    quality_score_threshold: float = 0.6
    quality_resynthesis_max_attempts: int = 1

    tier_promotion_episodic_min_confirmations: int = 2
    tier_promotion_semantic_min_confirmations: int = 3
    tier_promotion_semantic_min_age_days: int = 7

    log_page_recent_entries: int = 200
    pii_strict_mode: bool = False

    embed_batch_size: int = 64
