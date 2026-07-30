"""Financial knowledge-base RAG: LangGraph workflow, hybrid retrieval, compliance guardrails."""

from .chunking import Chunk, chunk_document
from .compliance import Verdict, check_query
from .config import Settings, load_settings
from .embeddings import HashingEmbedder, build_embedder
from .evaluation import EvalCase, evaluate
from .graph import RagAnswer, RagPipeline
from .ingest import build_from_corpus, build_retriever, documents_to_chunks, load_documents
from .llm import MockLLM, build_llm
from .retrieval import BM25, HybridRetriever

__version__ = "0.1.0"

__all__ = [
    "BM25",
    "Chunk",
    "EvalCase",
    "HashingEmbedder",
    "HybridRetriever",
    "MockLLM",
    "RagAnswer",
    "RagPipeline",
    "Settings",
    "Verdict",
    "build_embedder",
    "build_from_corpus",
    "build_llm",
    "build_retriever",
    "check_query",
    "chunk_document",
    "documents_to_chunks",
    "evaluate",
    "load_documents",
    "load_settings",
]
