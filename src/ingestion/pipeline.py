"""End-to-end ingestion pipeline: crawl → filter → semantic chunk → Cohere Embed
→ persist to the hybrid store. Stitches the ingestion modules into one callable so
the API and CLI share the same path to a built corpus."""
from __future__ import annotations
from typing import Iterable, List, Optional

import cohere

from src.embeddings.cohere_embed import embed_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.crawler import RawDoc, crawl_directory, crawl_web
from src.ingestion.filters import filter_documents
from src.vectorstore.store import HybridStore


def build_corpus(raw_docs: Iterable[RawDoc], *, store: Optional[HybridStore] = None,
                 min_words: int = 30, allowed_languages: Optional[Iterable[str]] = None,
                 max_tokens: int = 256, overlap_tokens: int = 48,
                 client: Optional[cohere.Client] = None) -> HybridStore:
    """Filter → chunk → embed → add to store. Returns the (new or extended) store."""
    store = store or HybridStore()
    clean = list(filter_documents(raw_docs, min_words=min_words,
                                  allowed_languages=allowed_languages))
    chunks = chunk_documents(clean, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    if chunks:
        vectors = embed_documents([c.text for c in chunks], client=client)
        store.add(chunks, vectors)
    return store


def ingest_web(seeds: List[str], persist_path: Optional[str] = None,
               max_pages: int = 100, **kwargs) -> HybridStore:
    store = build_corpus(crawl_web(seeds, max_pages=max_pages), **kwargs)
    if persist_path:
        store.save(persist_path)
    return store


def ingest_directory(path: str, persist_path: Optional[str] = None, **kwargs) -> HybridStore:
    store = build_corpus(crawl_directory(path), **kwargs)
    if persist_path:
        store.save(persist_path)
    return store
