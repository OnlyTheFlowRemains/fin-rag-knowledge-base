from __future__ import annotations

import numpy as np

from fin_rag.chunking import Chunk
from fin_rag.embeddings import HashingEmbedder, build_embedder, cosine_scores
from fin_rag.retrieval import BM25, HybridRetriever


def _chunks() -> list[Chunk]:
    texts = [
        "泓远沪深300增强ETF 基金代码 519301，2024 年净值增长率为 12.47%。",
        "泓远稳健纯债债券型基金 代码 519502，2024 年最大回撤为 1.12%。",
        "管理费年费率为 0.50%，托管费年费率为 0.10%。",
        "投资者风险承受能力分为 C1 至 C5 五级，产品风险等级分为 R1 至 R5。",
    ]
    return [Chunk(doc_id=f"d{i}", chunk_id=f"d{i}#0", text=t) for i, t in enumerate(texts)]


def test_embeddings_are_l2_normalised():
    emb = HashingEmbedder(dim=128)
    matrix = emb.encode(["测试文本内容", "another piece of text"])
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embeddings_are_deterministic():
    a = HashingEmbedder(dim=64).encode(["同样的输入"])
    b = HashingEmbedder(dim=64).encode(["同样的输入"])
    assert np.array_equal(a, b)


def test_cosine_self_similarity_is_highest():
    emb = HashingEmbedder(dim=256)
    texts = ["基金管理费率是多少", "宏观经济数据概览", "债券久期管理策略"]
    matrix = emb.encode(texts)
    scores = cosine_scores(matrix[0], matrix)
    assert scores.argmax() == 0


def test_empty_matrix_returns_empty_scores():
    emb = HashingEmbedder(dim=32)
    scores = cosine_scores(emb.encode(["q"])[0], np.zeros((0, 32), dtype=np.float32))
    assert scores.shape == (0,)


def test_bm25_ranks_exact_term_match_first():
    corpus = [["基金", "代码", "519301"], ["基金", "代码", "519502"], ["管理", "费率"]]
    bm25 = BM25(corpus)
    scores = bm25.scores(["519502"])
    assert scores.argmax() == 1


def test_bm25_handles_empty_corpus():
    assert BM25([]).scores(["x"]).shape == (0,)


def test_bm25_idf_downweights_ubiquitous_terms():
    corpus = [["基金", "甲"], ["基金", "乙"], ["基金", "丙"]]
    bm25 = BM25(corpus)
    # a term in every document carries no discriminative power
    common = bm25.scores(["基金"])
    rare = bm25.scores(["乙"])
    assert rare.max() > common.max()


def test_hybrid_retriever_finds_fund_code_via_sparse_channel():
    retriever = HybridRetriever(build_embedder("hashing", dim=256))
    retriever.index(_chunks())
    hits = retriever.search("519502 的最大回撤", top_k=2)
    assert hits
    assert "519502" in hits[0].chunk.text


def test_hybrid_retriever_empty_index_returns_nothing():
    retriever = HybridRetriever(build_embedder("hashing", dim=64))
    retriever.index([])
    assert retriever.search("任何问题") == []
    assert len(retriever) == 0


def test_rrf_downweights_expansion_queries():
    retriever = HybridRetriever(build_embedder("hashing", dim=256))
    retriever.index(_chunks())
    base = retriever.search("管理费年费率", top_k=1)
    with_extra = retriever.search("管理费年费率", top_k=1, extra_queries=["完全无关的宏观经济内容"])
    # an off-topic expansion must not displace the correct top hit
    assert base[0].chunk.chunk_id == with_extra[0].chunk.chunk_id


def test_search_records_which_channel_matched():
    retriever = HybridRetriever(build_embedder("hashing", dim=256))
    retriever.index(_chunks())
    hits = retriever.search("风险承受能力 C1 C5", top_k=3)
    assert any(h.dense_rank is not None or h.sparse_rank is not None for h in hits)
