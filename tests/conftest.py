from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fin_rag import MockLLM, RagPipeline, load_settings  # noqa: E402
from fin_rag.ingest import build_from_corpus  # noqa: E402


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    return ROOT / "data" / "corpus"


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture()
def pipeline(corpus_dir, settings):
    retriever, _ = build_from_corpus(corpus_dir, settings)
    return RagPipeline(retriever=retriever, llm=MockLLM(), settings=settings)
