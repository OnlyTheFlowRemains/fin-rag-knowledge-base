"""Chunking strategies.

Chunking is the highest-leverage knob in a RAG system: retrieval can only be
as good as the unit it indexes. Three strategies are implemented so the
trade-off is measurable rather than folklore.

- ``fixed``        sliding window over characters. Cheap, splits mid-sentence.
- ``semantic``     pack whole sentences up to the budget. Default.
- ``small_to_big`` index small (precise match) but return the parent window
                   (enough context for the LLM). Best recall/precision mix.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

_SENT_SPLIT = re.compile(r"(?<=[。！？；])|(?<=[.!?])\s+|\n+")


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    # What the LLM sees. For small_to_big this is wider than `text`, which is
    # what gets embedded/indexed.
    context: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.context:
            self.context = self.text

    @property
    def citation(self) -> str:
        source = self.metadata.get("title") or self.doc_id
        section = self.metadata.get("section")
        return f"{source} §{section}" if section else source

    @property
    def index_text(self) -> str:
        """Text actually fed to the embedder and BM25 — the chunk body prefixed
        with a contextual header.

        Chunks lose document-level context: a "费用结构" section says "管理费
        年费率为 0.50%" without ever repeating which fund it belongs to, so a
        query naming the fund would rank the sections that *do* mention it and
        miss the one holding the answer. Prefixing title + section restores
        that context at index time. Cheap, and the single largest retrieval
        win in this pipeline (see docs/03-踩坑记录.md).
        """
        header_parts = [self.metadata.get("title", ""), self.metadata.get("section", "")]
        header = " · ".join(p for p in header_parts if p)
        return f"{header}\n{self.text}" if header else self.text


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def fixed_window(text: str, *, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    out: list[str] = []
    for start in range(0, max(len(text), 1), step):
        piece = text[start : start + size].strip()
        if piece:
            out.append(piece)
        if start + size >= len(text):
            break
    return out


def semantic_window(text: str, *, size: int) -> list[str]:
    """Greedily pack sentences, never splitting one in half."""
    chunks: list[str] = []
    buffer: list[str] = []
    length = 0
    for sentence in split_sentences(text):
        if length + len(sentence) > size and buffer:
            chunks.append("".join(buffer).strip())
            buffer, length = [], 0
        buffer.append(sentence)
        length += len(sentence)
        # A single sentence longer than the budget still has to be emitted.
        if length > size:
            chunks.append("".join(buffer).strip())
            buffer, length = [], 0
    if buffer:
        chunks.append("".join(buffer).strip())
    return [c for c in chunks if c]


def chunk_document(
    *,
    doc_id: str,
    text: str,
    strategy: str = "semantic",
    size: int = 320,
    overlap: int = 64,
    metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    metadata = dict(metadata or {})
    if strategy == "fixed":
        pieces = fixed_window(text, size=size, overlap=overlap)
        return [
            Chunk(doc_id=doc_id, chunk_id=f"{doc_id}#{i}", text=piece, metadata=metadata)
            for i, piece in enumerate(pieces)
        ]
    if strategy == "semantic":
        pieces = semantic_window(text, size=size)
        return [
            Chunk(doc_id=doc_id, chunk_id=f"{doc_id}#{i}", text=piece, metadata=metadata)
            for i, piece in enumerate(pieces)
        ]
    if strategy == "small_to_big":
        return _small_to_big(doc_id=doc_id, text=text, size=size, metadata=metadata)
    raise ValueError(f"unknown chunk strategy: {strategy!r}")


def _small_to_big(
    *, doc_id: str, text: str, size: int, metadata: dict[str, str]
) -> list[Chunk]:
    parents = semantic_window(text, size=size)
    chunks: list[Chunk] = []
    for p_idx, parent in enumerate(parents):
        children = split_sentences(parent) or [parent]
        for c_idx, child in enumerate(children):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{p_idx}.{c_idx}",
                    text=child,
                    context=parent,
                    metadata={**metadata, "parent": f"{doc_id}#{p_idx}"},
                )
            )
    return chunks


def dedupe_context(chunks: Sequence[Chunk]) -> list[str]:
    """small_to_big children share a parent — emit each parent window once."""
    seen: set[str] = set()
    out: list[str] = []
    for chunk in chunks:
        if chunk.context in seen:
            continue
        seen.add(chunk.context)
        out.append(chunk.context)
    return out
