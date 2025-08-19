# Babel

Multilingual semantic search & RAG platform built on Cohere **Command / Embed / Rerank**. Crawls and filters a multilingual corpus, serves **hybrid retrieval (dense + BM25)** with a Cohere **Rerank** cross-encoder stage, and answers questions with **grounded Cohere Command generation and inline citations** — wrapped in an agentic self-correction loop and evaluated end-to-end with LangSmith and LangFuse.

## Overview

- **Ingestion** — polite BFS web crawler + local-directory loader, boilerplate stripping (`crawler.py`)
- **Filters** — language detection / allow-listing, min-length quality gate, exact + near-duplicate (shingle-Jaccard) removal (`filters.py`)
- **Chunking** — semantic, sentence-aware chunking with token-bounded windows and configurable overlap, multilingual-friendly (`chunker.py`)
- **Embeddings** — Cohere `embed-multilingual-v3.0`, asymmetric `search_document` / `search_query` input types, batched (`cohere_embed.py`)
- **Hybrid store** — NumPy cosine dense index + BM25 lexical index over one corpus, persisted together (`store.py`)
- **Retrieval** — query rewriting + multi-query expansion → dense+BM25 search fused with Reciprocal Rank Fusion → Cohere `rerank-multilingual-v3.0` cross-encoder → top-n (`query_expansion.py`, `hybrid.py`, `rerank.py`)
- **Generation** — grounded Cohere `command-r-plus` answers with structured inline citations back to source documents (`grounded.py`)
- **Agent** — LangGraph `retrieve → generate → grade → (rewrite → retrieve …)` self-correction loop (`rag_agent.py`)
- **Evaluation** — LangFuse online tracing + faithfulness/relevance scoring; LangSmith offline dataset runs (`evaluation/`)
- **FastAPI** — `/query`, `/ingest` (background), `/health`; shipped on Docker with GitHub Actions CI/CD

## Tech Stack

Python 3.11 · Cohere (Command / Embed / Rerank) · LangGraph · LangSmith · LangFuse · rank-bm25 · NumPy · FastAPI · BeautifulSoup · langdetect · Docker · GitHub Actions

## Quickstart

```bash
pip install -r requirements.txt
export COHERE_API_KEY=...

# Ingest a corpus (web seeds or a local directory)
CONFIG_PATH=configs/config.yaml uvicorn api.main:app --reload

curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"seeds": ["https://example.com/docs"], "max_pages": 50}'

# Query (multilingual: ask in any language, retrieve across all)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cómo funciona la búsqueda híbrida?"}'

# Tests
pytest tests/
```

Or with Docker:

```bash
COHERE_API_KEY=... docker compose up --build
```

## Architecture

```mermaid
flowchart TB
    subgraph ingest["Ingestion (src/ingestion)"]
        CRAWL["crawler.py<br/>crawl_web BFS / crawl_directory"]
        FILT["filters.py<br/>language allow-list, min length,<br/>exact + shingle-Jaccard dedup"]
        CHUNK["chunker.py<br/>sentence-aware, token-bounded, overlapping"]
        EMBD["cohere_embed.py<br/>embed-multilingual-v3.0, search_document"]
    end

    STORE[("HybridStore (src/vectorstore/store.py)<br/>NumPy dense matrix + BM25 index<br/>persisted together")]

    subgraph retr["Retrieval funnel (src/retrieval)"]
        REW["query_expansion.py<br/>rewrite_query then expand_query"]
        DENSE["store.dense_search<br/>cosine over normalised vectors"]
        SPARSE["store.sparse_search<br/>BM25"]
        RRF["hybrid.py<br/>reciprocal_rank_fusion, k=60"]
        RR["rerank.py<br/>rerank-multilingual-v3.0 cross-encoder"]
    end

    subgraph ag["LangGraph agent (src/agents/rag_agent.py)"]
        RETN["retrieve node"]
        GENN["generate node<br/>grounded.py, Command + citations"]
        GRADE["grade node<br/>PASS / FAIL self-critique"]
        REWN["rewrite node"]
    end

    subgraph apis["FastAPI (api/main.py)"]
        Q["POST /query"]
        I["POST /ingest<br/>BackgroundTasks"]
        H["GET /health"]
    end

    EVAL["src/evaluation<br/>LangFuse online traces + scores<br/>LangSmith offline dataset runs"]

    I --> CRAWL --> FILT --> CHUNK --> EMBD --> STORE
    Q --> RETN
    RETN --> REW
    REW -->|"4 query variants"| DENSE
    REW --> SPARSE
    STORE --> DENSE
    STORE --> SPARSE
    DENSE --> RRF
    SPARSE --> RRF
    RRF -->|"~20 candidates"| RR
    RR -->|"top_n = 5"| GENN
    GENN --> GRADE
    GRADE -->|"FAIL and retries remain"| REWN
    REWN --> RETN
    GRADE -->|"PASS or budget exhausted"| ANS["Grounded answer<br/>with inline citations"]
    ANS --> Q
    GENN -.-> EVAL
    GRADE -.-> EVAL
```

<img src="docs/retrieval-funnel.svg" alt="One question fanning out to four rewritten queries, running through the dense and BM25 indexes, fusing with RRF into about twenty candidates, reranking down to five passages, and looping back when the grader fails the answer" width="880">

The funnel is where the interesting work happens: `retrieve()` rewrites the
question, expands it into several variants, runs each variant through **both**
indexes, fuses all those rankings with RRF, and only then pays for a
cross-encoder pass over the surviving candidates. The agent wraps that funnel in
a loop bounded by `max_iterations` (default 3), so a failed grade costs at most
two extra retrieval rounds.

## Project Structure

```
babel/
├── src/
│   ├── clients.py          # Cohere client wrapper + response normalisation
│   ├── ingestion/          # crawler.py, filters.py, chunker.py, pipeline.py
│   ├── embeddings/         # cohere_embed.py (multilingual)
│   ├── vectorstore/        # store.py (dense + BM25 hybrid)
│   ├── retrieval/          # query_expansion.py, hybrid.py (RRF), rerank.py
│   ├── generation/         # grounded.py (Command + inline citations)
│   ├── agents/             # rag_agent.py (LangGraph self-correction loop)
│   └── evaluation/         # langfuse_eval.py, langsmith_eval.py
├── api/                    # main.py (FastAPI)
├── configs/                # config.yaml
├── tests/                  # pytest suite (chunker, filters, hybrid store + RRF)
├── Dockerfile · docker-compose.yml
└── .github/workflows/ci.yml
```

## Tests

```bash
pytest tests/ -v
```

The offline suite covers the pieces that don't need a Cohere key: semantic chunking (token budget, overlap, metadata propagation), corpus filters (length gate, exact dedup, language allow-listing), and the hybrid store (dense cosine ordering, BM25 lexical match, RRF fusion, save/load round-trip). Cohere-backed paths (embed, rerank, Command generation, the agent loop) are exercised via the evaluation harness against a live key.

## Evaluation (RAG)

Run on a labelled `{question, gold_doc_ids, gold_answer}` set (~50 questions spanning multiple languages over the seeded corpus).

| Metric | How to compute |
|---|---|
| **Retrieval Recall@k** | Fraction of `gold_doc_ids` present in the fused top-k before rerank (k ∈ {10, 20}). `retrieval_recall()` in `langfuse_eval.py`. |
| **Rerank lift** | Recall / MRR after Cohere Rerank vs. before — isolates the cross-encoder's contribution. |
| **Cross-lingual recall** | Recall restricted to query/doc language pairs that differ, to verify the shared multilingual space. |
| **Faithfulness** | LLM-judge (Command) over `(retrieved passages, answer)`; report % faithful. Implemented in both eval harnesses. |
| **Answer relevance** | LLM-judge on `(question, answer)`; mean score and % ≥ 0.8. |
| **Citation precision** | Of the documents the answer cites inline, fraction actually in `gold_doc_ids`. |
| **Self-correction rate** | Share of queries that triggered a rewrite loop, and recall/faithfulness delta after the extra round. |
| **Latency p50 / p99** | End-to-end `/query` time; LangFuse spans break it into expand / retrieve / rerank / generate. |
