"""Runtime configuration.

Everything is env-overridable so the same code runs offline (CI, laptop demo)
and against a real LLM provider without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class RetrievalConfig:
    """Knobs for the hybrid retrieval stage."""

    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 4))
    candidate_k: int = field(default_factory=lambda: _env_int("RAG_CANDIDATE_K", 12))
    # Reciprocal Rank Fusion constant. 60 is the value from the original
    # Cormack et al. paper; small values over-weight the top rank.
    rrf_k: int = field(default_factory=lambda: _env_int("RAG_RRF_K", 60))
    use_hyde: bool = field(default_factory=lambda: _env_bool("RAG_USE_HYDE", True))
    use_rerank: bool = field(default_factory=lambda: _env_bool("RAG_USE_RERANK", True))
    # Below this fused score we treat the corpus as "no answer here" and
    # refuse instead of letting the LLM hallucinate.
    min_score: float = field(default_factory=lambda: _env_float("RAG_MIN_SCORE", 0.02))


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 320))
    overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 64))
    strategy: str = field(default_factory=lambda: os.getenv("RAG_CHUNK_STRATEGY", "semantic"))


@dataclass(frozen=True)
class Settings:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen-plus"))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    )
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "hashing")
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 512))
    corpus_dir: str = field(default_factory=lambda: os.getenv("RAG_CORPUS_DIR", "data/corpus"))


def load_settings() -> Settings:
    return Settings()
