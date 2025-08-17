# Babel

## Project Overview

Babel is a multilingual semantic search and RAG platform. It crawls a corpus from
the open web (or a local document tree), filters it for language and quality,
chunks it semantically with overlap, and embeds it with Cohere's multilingual
embeddings into a hybrid store. At query time it rewrites and expands the query,
runs dense + BM25 retrieval fused with Reciprocal Rank Fusion, reranks the
candidate set with a Cohere cross-encoder, and produces a grounded answer with
inline citations from Cohere Command — all inside a LangGraph agent that
self-corrects when its own answer fails a groundedness check.

Key goals:
- Cross-lingual retrieval: ask in one language, retrieve relevant passages in another
- High top-k precision via a two-stage retrieve-then-rerank funnel
- Grounded, citable answers that never invent sources
- Self-correction to recover from weak first-pass retrieval
- Full retrieval + faithfulness observability across LangFuse (online) and LangSmith (offline)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI + Uvicorn |
| Embeddings | Cohere `embed-multilingual-v3.0` (1024-dim) |
| Reranking | Cohere `rerank-multilingual-v3.0` (cross-encoder) |
| Generation | Cohere `command-r-plus` (grounded chat + citations) |
| Query ops | Cohere `command-r` (rewrite + multi-query expansion) |
| Lexical search | rank-bm25 (BM25Okapi) |
| Dense search | NumPy cosine over L2-normalised vectors |
| Orchestration | LangGraph (stateful self-correction graph) |
| Crawl / parse | httpx + BeautifulSoup |
| Language ID | langdetect |
| Observability | LangFuse (online tracing/scoring), LangSmith (offline datasets) |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions (compile-check, pytest, docker build) |

---

## Architecture Overview

```
Ingestion pipeline
  crawl_web / crawl_directory → filter_documents → chunk_documents
      → Cohere Embed (search_document) → HybridStore (dense .npy + BM25 + records)

Query-time (LangGraph AgentState)
  retrieve_node:
      rewrite_query → expand_query (multi-query)
      per query: dense cosine + BM25 → RRF
      union of variants → RRF → Cohere Rerank → top-n
  generate_node:  grounded command-r-plus with documents → text + citations
  grade_node:     command-r judges PASS/FAIL on groundedness
  conditional:    FAIL & iterations < max → rewrite_node → retrieve ; else END

Observability
  LangFuse @observe traces each run; scores faithfulness + relevance
  LangSmith dataset runs with faithfulness + answer-relevance evaluators
```

---

## Phase 1 — Scaffolding & Cohere Client

**Goal:** repo structure, dependency pinning, config, and a single wrapper around
the Cohere SDK so the rest of the code never touches the raw client.

- Directory layout under `src/` (ingestion, embeddings, vectorstore, retrieval,
  generation, agents, evaluation), `api/`, `configs/`, `tests/`.
- `requirements.txt`, `.env.example`, `configs/config.yaml`.
- `src/clients.py`: cached `cohere.Client`, `embed()` helper that normalises the
  v5 embedding response shape; key from `COHERE_API_KEY`.
- `Dockerfile` (`python:3.11-slim`, uvicorn CMD) and `docker-compose.yml`.

## Phase 2 — Ingestion: Crawl, Filter, Chunk

**Goal:** turn raw sources into a clean, multilingual set of overlapping chunks.

- `crawler.py`: BFS `crawl_web` (host-scoped, depth/page caps, polite delay,
  boilerplate stripping) and `crawl_directory` for local html/txt/md.
- `filters.py`: `detect_language`, min-word gate, exact (sha256) + near-duplicate
  (k-shingle Jaccard) removal; tags each doc with `lang`; optional allow-list.
- `chunker.py`: sentence splitter (incl. CJK punctuation), greedy token-bounded
  packing with a carried overlap tail; propagates metadata + `chunk_index/total`.
- `pipeline.py`: `build_corpus` = filter → chunk → embed → `store.add`.
- Tests: `test_chunker.py`, `test_filters.py`.

## Phase 3 — Embeddings & Hybrid Store

**Goal:** multilingual vectors plus a co-located lexical index.

- `cohere_embed.py`: `embed_documents` (batched, `search_document`) /
  `embed_query` (`search_query`).
- `store.py`: `HybridStore` holding L2-normalised NumPy matrix + `BM25Okapi` +
  records, with `dense_search`, `sparse_search`, and `save`/`load` that keep the
  two indices aligned.
- Tests: `test_hybrid.py` (cosine ordering, BM25 match, save/load round-trip).

## Phase 4 — Retrieval Funnel

**Goal:** maximise recall first, then precision.

- `query_expansion.py`: `rewrite_query` (self-contained query) and `expand_query`
  (N paraphrases/sub-questions, JSON-parsed) via `command-r`.
- `hybrid.py`: `reciprocal_rank_fusion`, `hybrid_search` (single query), and
  `retrieve` (rewrite → multi-query → per-query hybrid → RRF union → rerank).
- `rerank.py`: Cohere Rerank over candidate passages → top-n `Ranked`.

## Phase 5 — Grounded Generation & Self-Correcting Agent

**Goal:** citable answers, with a loop that fixes weak retrievals.

- `grounded.py`: pass reranked passages as `documents` to `command-r-plus`;
  capture citation spans; `with_inline_markers()` renders `[n]` markers.
- `rag_agent.py`: LangGraph `retrieve → generate → grade → rewrite?` with a
  `max_iterations` budget; `run_query` returns a `GroundedAnswer`.

## Phase 6 — API, Evaluation & CI/CD

**Goal:** ship it and keep quality measurable.

- `api/main.py`: `/ingest` (background), `/query`, `/health`; load store at startup.
- `langfuse_eval.py`: `@observe` tracing, `retrieval_recall`, faithfulness /
  relevance LLM judges, `evaluate_batch`.
- `langsmith_eval.py`: dataset creation, agent target, faithfulness + relevance
  evaluators, `run_evaluation`.
- `.github/workflows/ci.yml`: compile-check → pytest → docker build.
- Smoke test:
  ```bash
  curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
    -d '{"question": "Wie funktioniert das Reranking?"}'
  ```
