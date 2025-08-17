"""Cross-encoder reranking via Cohere Rerank. The hybrid stage casts a wide net
(many candidates from dense + BM25); Rerank then scores each (query, passage)
pair jointly with a cross-encoder, which is far more precise than the
bi-encoder cosine similarity used for first-stage retrieval. We keep the top-n."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cohere

from src.clients import get_client
from src.vectorstore.store import Record

DEFAULT_MODEL = "rerank-multilingual-v3.0"


@dataclass
class Ranked:
    record: Record
    score: float


def rerank(query: str, candidates: Sequence[Record], top_n: int = 5,
           model: str = DEFAULT_MODEL,
           client: Optional[cohere.Client] = None) -> List[Ranked]:
    """Re-order `candidates` by cross-encoder relevance, returning the top_n."""
    if not candidates:
        return []
    client = client or get_client()
    resp = client.rerank(
        model=model,
        query=query,
        documents=[c.text for c in candidates],
        top_n=min(top_n, len(candidates)),
    )
    return [Ranked(record=candidates[r.index], score=r.relevance_score) for r in resp.results]
