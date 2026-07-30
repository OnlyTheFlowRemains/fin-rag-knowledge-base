# Financial Knowledge Base — RAG with Compliance Guardrails

A retrieval-augmented Q&A service over financial product documents, orchestrated
as an explicit **LangGraph** state machine. Hybrid retrieval (dense + BM25 fused
with RRF), contextual chunk headers, cross-encoder-style reranking, and a
compliance layer that refuses trade instructions and strips guaranteed-return
language.

**Runs fully offline.** The default provider is a deterministic mock LLM and a
dependency-free hashing embedder, so `pytest` and the demo need no API key and no
model download. Point it at Qwen / DeepSeek / vLLM by changing two env vars.

Data note: the compliance document summarises real regulation (see its sources
line). Fund product figures are illustrative and internally consistent — the fund
names and codes are not real products, so don't read the numbers as market data.

## Why this design

| Decision | Reason |
|---|---|
| LangGraph, not a linear chain | The flow genuinely branches: a compliance refusal must skip retrieval; a low-confidence retrieval must skip generation. Each path is independently testable. |
| Hybrid dense + sparse | Embeddings blur exact identifiers (fund code `519301`, `2024Q3`); BM25 misses paraphrase (`回撤` vs `下跌幅度`). Finance queries contain both. |
| RRF fusion, not score addition | BM25 scores are unbounded, cosine sits in `[-1, 1]`. Fusing by rank keeps one channel from dominating for arbitrary reasons. |
| Contextual chunk headers | The single biggest quality win here — see below. |
| Compliance as code, not prompt | A prompt is a request; a regex post-filter and a routing node are controls. |
| Refuse on low retrieval score | An unanswerable question should return "not in the corpus", not a fluent guess. |

## Measured result: contextual chunk headers

A fee section says *"管理费年费率为 0.50%"* without repeating which fund it belongs
to. So a query naming the fund ranks the sections that *do* mention the name, and
misses the one holding the answer. Prefixing `title · section` to the **indexed**
text (not the text shown to the model) fixes it.

On the 3 fee queries in `tests/test_evaluation.py`:

| Variant | Correct section ranked #1 | Answer contains the right number |
|---|---|---|
| With contextual header | 3 / 3 | 3 / 3 |
| Bare chunk body | 1 / 3 | 1 / 3 |

The interesting failure is not the miss — it is that one ablated query returns the
**other fund's** fee section. A confident answer about the wrong product is worse
than no answer. Both variants are pinned by tests so this table cannot rot.

Across the full 11-case evaluation set, same corpus, only the header ablated:

| Variant | hit_rate | MRR | ctx_precision | groundedness |
|---|---|---|---|---|
| With contextual header | 1.000 | 0.944 | **0.861** | 1.000 |
| Bare chunk body | 1.000 | 0.944 | **0.750** | 1.000 |

`hit_rate` and MRR are unchanged because they score attribution at *document*
level — any chunk from the right document counts. The header's gain shows up in
section-level precision, which is a reminder that the wrong metric hides a real
improvement.

**Honest caveat:** with the mock provider, HyDE shows *no* measurable gain — the
mock's hypothetical answer is drawn from the same corpus, so it adds no new query
signal. HyDE's benefit needs a real LLM to appear. `python scripts/eval.py --ablate`
prints the comparison rather than asking you to take a claim on faith.

## Architecture

```
                 ┌──────────────┐
   question ────►│  compliance  │──(regulated action)──────────────┐
                 └──────┬───────┘                                  │
                        │ allow / allow-with-disclaimer            │
                        ▼                                          │
                 ┌──────────────┐                                  │
                 │ expand(HyDE) │                                  │
                 └──────┬───────┘                                  ▼
                        ▼                                   ┌────────────┐
                 ┌──────────────┐   dense ─┐                │  refuse    │
                 │   retrieve   │          ├─ RRF fusion    └─────┬──────┘
                 └──────┬───────┘   BM25 ──┘                      │
                        ▼                                          │
                 ┌──────────────┐                                  │
                 │    rerank    │──(no hit above min_score)────────┤
                 └──────┬───────┘                                  │
                        ▼                                          │
                 ┌──────────────┐                                  │
                 │   generate   │──(provider error → sources only) │
                 └──────┬───────┘                                  │
                        └──────────────────┬───────────────────────┘
                                           ▼
                            answer + citations + trace
```

Every node appends to a `trace` list that ships in the response, so you can see
which path a request took without attaching a debugger.

## Quick start

```bash
pip install -e ".[dev]"

# ask a question (offline, no key needed)
python scripts/ask.py "泓远沪深300增强ETF 的管理费年费率是多少"

# retrieval quality report + ablation
python scripts/eval.py --verbose
python scripts/eval.py --ablate

# tests
pytest -q          # 66 tests

# HTTP service
uvicorn fin_rag.api:app --reload
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"投资者风险承受能力如何分级"}' | python -m json.tool
```

### Using a real model

```bash
export LLM_PROVIDER=qwen
export LLM_MODEL=qwen-plus
export LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export LLM_API_KEY=...            # keep this out of git
export EMBEDDING_PROVIDER=sentence-transformers   # needs pip install ".[semantic]"
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + indexed chunk count |
| `POST /ask` | JSON answer with citations, verdict, trace |
| `POST /ask/stream` | SSE. **Citations are emitted before the first token** so the UI can render sources while the answer streams. |

`POST /ask` response shape:

```json
{
  "question": "泓远沪深300增强ETF 的托管费年费率是多少",
  "answer": "托管费年费率为 0.10%。...",
  "citations": [{"index": 1, "source": "...§费用结构", "score": 0.478, "dense_rank": 0, "sparse_rank": 1}],
  "verdict": "allow",
  "reasons": [],
  "disclaimer": "...仅供信息参考，不构成投资建议。",
  "trace": ["compliance:allow", "expand:hyde(102chars)", "retrieve:12", "rerank:12->4", "generate:118chars"]
}
```

## Compliance behaviour

| Input | Verdict | Effect |
|---|---|---|
| `帮我下单买入 519301` | `refuse` | Routed straight to refusal; retrieval never runs |
| `这只基金该不该买` | `allow_with_disclaimer` | Answers with sourced facts + disclaimer, no recommendation |
| `明天会涨吗` | `allow_with_disclaimer` | Same; prediction framing is not honoured |
| `管理费年费率是多少` | `allow` | Plain factual lookup |

Generated text also passes a regex sanitiser that strips `保证收益` / `稳赚不赔` /
`建议买入`-style phrasing even if a model produces it.

## Failure modes handled

- **Provider down** — generation degrades to "here are your sources" with the
  citations intact, and records `generation-error:<Type>` in `reasons`. It does not
  return a 500.
- **HyDE failure** — falls back to the raw question.
- **Empty index / nothing above threshold** — refuses instead of generating.
- **Duplicate context** — `small_to_big` children sharing a parent window are
  deduplicated before the prompt is built.

## Known limitations

- The hashing embedder is a lexical approximation, not a semantic model. It handles
  paraphrase far worse than BGE; install the `semantic` extra for real embeddings.
- `LexicalOverlapReranker` is not a cross-encoder. `TorchReranker` is the real one
  and is not exercised in CI (it needs a model download).
- No authentication, rate limiting, or per-tenant isolation. Put it behind a gateway.
- Retriever state is in-memory: the index is rebuilt at startup and there is no
  incremental update path. Swap in a persistent vector store for anything real.
- Corpus loading covers `.md` / `.txt` only. PDF/OCR ingestion is out of scope here.

## Layout

```
fin_rag/
  config.py       env-driven settings
  llm.py          LLM protocol, MockLLM, OpenAI-compatible client
  embeddings.py   hashing embedder (default) + sentence-transformers option
  chunking.py     fixed / semantic / small_to_big + contextual headers
  retrieval.py    BM25 (from scratch) + dense + RRF fusion
  rerank.py       lexical reranker, cross-encoder option, HyDE expansion
  compliance.py   query classification + answer sanitisation
  graph.py        the LangGraph state machine
  evaluation.py   hit_rate / MRR / context precision / groundedness
  api.py          FastAPI: /ask, /ask/stream (SSE)
```

## License

MIT
