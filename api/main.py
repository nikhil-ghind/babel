"""FastAPI surface for Babel: ingest a corpus, query it through the self-correcting
agentic RAG pipeline, and report health. The store is loaded once at startup and
reused across requests; ingestion runs as a background task."""
from __future__ import annotations
import os
from typing import List, Optional

import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.rag_agent import build_rag_agent, run_query
from src.ingestion.pipeline import ingest_directory, ingest_web
from src.vectorstore.store import HybridStore

app = FastAPI(title="Babel — Multilingual Semantic Search & RAG")

_cfg: dict = {}
_store: Optional[HybridStore] = None
_agent = None


def _load_config() -> dict:
    with open(os.getenv("CONFIG_PATH", "configs/config.yaml")) as f:
        return yaml.safe_load(f)


@app.on_event("startup")
def startup() -> None:
    global _cfg, _store, _agent
    _cfg = _load_config()
    path = _cfg.get("store_path", "data/store")
    if os.path.isdir(path):
        _store = HybridStore.load(path)
        _agent = build_rag_agent(_store, max_iterations=_cfg.get("max_iterations", 3),
                                 top_n=_cfg.get("rerank_top_n", 5))


class QueryRequest(BaseModel):
    question: str
    top_n: Optional[int] = None


class Source(BaseModel):
    id: str
    title: str
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]


class IngestRequest(BaseModel):
    seeds: Optional[List[str]] = None
    directory: Optional[str] = None
    allowed_languages: Optional[List[str]] = None
    max_pages: int = 100


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if _agent is None:
        raise HTTPException(503, "Corpus not loaded — ingest documents first")
    answer = run_query(_agent, req.question)
    sources = [Source(id=d["id"], title=d["title"], snippet=d["snippet"]) for d in answer.documents]
    return QueryResponse(question=req.question, answer=answer.with_inline_markers(), sources=sources)


@app.post("/ingest")
def ingest(req: IngestRequest, background_tasks: BackgroundTasks) -> dict:
    if not req.seeds and not req.directory:
        raise HTTPException(400, "Provide either 'seeds' or 'directory'")
    path = _cfg.get("store_path", "data/store")

    def _run() -> None:
        global _store, _agent
        if req.directory:
            _store = ingest_directory(req.directory, persist_path=path,
                                      allowed_languages=req.allowed_languages)
        else:
            _store = ingest_web(req.seeds, persist_path=path, max_pages=req.max_pages,
                                allowed_languages=req.allowed_languages)
        _agent = build_rag_agent(_store, max_iterations=_cfg.get("max_iterations", 3),
                                 top_n=_cfg.get("rerank_top_n", 5))

    background_tasks.add_task(_run)
    return {"status": "ingestion started"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "corpus_loaded": _store is not None,
            "documents": len(_store) if _store else 0}
