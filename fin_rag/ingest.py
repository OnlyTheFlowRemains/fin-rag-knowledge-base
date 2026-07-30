"""Corpus loading and indexing.

Markdown/text files are split on `##` headings first so a chunk never spans
two unrelated sections — section titles then become citation anchors, which is
what makes answers auditable ("where did this number come from").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .chunking import Chunk, chunk_document
from .config import Settings, load_settings
from .embeddings import build_embedder
from .retrieval import HybridRetriever

_HEADING = re.compile(r"^##\s+(.*)$", re.MULTILINE)
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, str]


def load_documents(corpus_dir: str | Path) -> list[Document]:
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        first_line = text.splitlines()[0].lstrip("# ").strip()
        docs.append(
            Document(
                doc_id=path.stem,
                title=first_line or path.stem,
                text=text,
                metadata={"path": str(path.relative_to(root)), "title": first_line or path.stem},
            )
        )
    return docs


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (section_title, body) pairs; body before the first `##` is 'intro'."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    intro = text[: matches[0].start()].strip()
    if intro:
        sections.append(("", intro))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((match.group(1).strip(), body))
    return sections


def documents_to_chunks(docs: Iterable[Document], settings: Settings) -> list[Chunk]:
    cfg = settings.chunk
    chunks: list[Chunk] = []
    for doc in docs:
        for sec_idx, (section, body) in enumerate(split_sections(doc.text)):
            metadata = {**doc.metadata}
            if section:
                metadata["section"] = section
            produced = chunk_document(
                # "::" is reserved as the doc/section separator. Using "-s{i}"
                # here silently corrupted doc ids that already contained "-s"
                # (e.g. "fund-bond-stable"), which broke evaluation attribution.
                doc_id=f"{doc.doc_id}::s{sec_idx}",
                text=body,
                strategy=cfg.strategy,
                size=cfg.chunk_size,
                overlap=cfg.overlap,
                metadata=metadata,
            )
            chunks.extend(produced)
    return chunks


def build_retriever(chunks: Sequence[Chunk], settings: Settings | None = None) -> HybridRetriever:
    settings = settings or load_settings()
    embedder = build_embedder(settings.embedding_provider, dim=settings.embedding_dim)
    retriever = HybridRetriever(embedder, rrf_k=settings.retrieval.rrf_k)
    retriever.index(chunks)
    return retriever


def build_from_corpus(
    corpus_dir: str | Path | None = None, settings: Settings | None = None
) -> tuple[HybridRetriever, list[Chunk]]:
    settings = settings or load_settings()
    docs = load_documents(corpus_dir or settings.corpus_dir)
    chunks = documents_to_chunks(docs, settings)
    return build_retriever(chunks, settings), chunks
