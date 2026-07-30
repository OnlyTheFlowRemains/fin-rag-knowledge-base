from __future__ import annotations

import pytest

from fin_rag.chunking import chunk_document, dedupe_context, fixed_window, semantic_window


def test_fixed_window_overlaps():
    text = "abcdefghij" * 5  # 50 chars
    pieces = fixed_window(text, size=20, overlap=5)
    assert len(pieces) >= 3
    # consecutive pieces must share the overlap region
    assert pieces[0][-5:] == pieces[1][:5]


def test_fixed_window_rejects_bad_overlap():
    with pytest.raises(ValueError):
        fixed_window("abc", size=10, overlap=10)


def test_semantic_window_keeps_sentences_intact():
    text = "第一句话很短。第二句话也不长。第三句话是最后一句。"
    pieces = semantic_window(text, size=14)
    assert len(pieces) > 1
    # no piece may start mid-sentence, i.e. every piece ends on a terminator
    assert all(p.endswith("。") for p in pieces)


def test_semantic_window_emits_oversized_sentence():
    text = "这是一个非常非常非常非常长的句子没有任何标点直到最后才结束。"
    pieces = semantic_window(text, size=5)
    assert "".join(pieces) == text


def test_small_to_big_children_share_parent_context():
    text = "甲句内容在这里。乙句内容在这里。丙句内容在这里。"
    chunks = chunk_document(doc_id="d1", text=text, strategy="small_to_big", size=200)
    assert len(chunks) == 3
    # indexed unit is the sentence, but the context handed to the LLM is wider
    assert all(len(c.context) >= len(c.text) for c in chunks)
    assert len({c.context for c in chunks}) == 1
    assert all(c.metadata["parent"] == "d1#0" for c in chunks)


def test_dedupe_context_collapses_shared_parents():
    text = "甲句内容在这里。乙句内容在这里。"
    chunks = chunk_document(doc_id="d1", text=text, strategy="small_to_big", size=200)
    assert len(dedupe_context(chunks)) == 1


def test_chunk_ids_are_unique():
    text = "一。二。三。四。五。六。"
    for strategy in ("fixed", "semantic", "small_to_big"):
        chunks = chunk_document(doc_id="d", text=text, strategy=strategy, size=6, overlap=2)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), strategy


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown chunk strategy"):
        chunk_document(doc_id="d", text="x", strategy="nope")


def test_index_text_prepends_title_and_section():
    chunks = chunk_document(
        doc_id="d",
        text="管理费年费率为 0.50%。",
        metadata={"title": "某基金年报", "section": "费用结构"},
    )
    indexed = chunks[0].index_text
    assert indexed.startswith("某基金年报 · 费用结构")
    assert "0.50%" in indexed
    # the body handed to the LLM stays clean, header is an indexing artefact
    assert chunks[0].context == "管理费年费率为 0.50%。"


def test_index_text_without_metadata_is_just_the_body():
    chunks = chunk_document(doc_id="d", text="没有元数据。")
    assert chunks[0].index_text == "没有元数据。"


def test_citation_prefers_title_and_section():
    chunks = chunk_document(
        doc_id="d",
        text="内容。",
        metadata={"title": "年报", "section": "费用结构"},
    )
    assert chunks[0].citation == "年报 §费用结构"
