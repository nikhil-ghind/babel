"""Hybrid retrieval: run dense (Cohere Embed cosine) and sparse (BM25) search
over the same store, fuse the two rankings with Reciprocal Rank Fusion, optionally
across several expanded query variants, then hand the candidate set to Cohere
Rerank for cross-encoder precision. This is the full first-stage → rerank funnel."""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence

import cohere

from src.embeddings.cohere_embed import embed_query
from src.retrieval.query_expansion import expand_query, rewrite_query
from src.retrieval.rerank import Ranked, rerank
from src.vectorstore.store import HybridStore, Record


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> List[int]:
    """Fuse multiple ranked lists of record indices into one. Each list votes
    with 1/(k + rank); higher RRF score ranks first. `k` damps the tail."""
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_search(store: HybridStore, query: str, k: int = 20,
                  client: Optional[cohere.Client] = None) -> List[Record]:
    """Single-query dense+BM25 retrieval fused with RRF. Returns up to `k` records."""
    qvec = embed_query(query, client=client)
    dense = [idx for idx, _ in store.dense_search(qvec, k=k)]
    sparse = [idx for idx, _ in store.sparse_search(query, k=k)]
    fused = reciprocal_rank_fusion([dense, sparse])[:k]
    return [store.get(i) for i in fused]


def retrieve(store: HybridStore, query: str, *, expand: bool = True,
             n_expansions: int = 3, candidate_k: int = 20, top_n: int = 5,
             client: Optional[cohere.Client] = None) -> List[Ranked]:
    """End-to-end retrieval funnel: rewrite → (multi-query) hybrid search → RRF
    union of candidates → Cohere Rerank → top_n. With `expand=False` it skips
    rewriting/expansion (useful for ablations and tests)."""
    queries = [query]
    if expand:
        queries = expand_query(rewrite_query(query, client=client),
                               n=n_expansions, client=client)

    rankings: List[List[int]] = []
    id_to_record: Dict[int, Record] = {}
    for q in queries:
        records = hybrid_search(store, q, k=candidate_k, client=client)
        ranking = [int(r.id) for r in records]
        rankings.append(ranking)
        for r in records:
            id_to_record[int(r.id)] = r

    fused_ids = reciprocal_rank_fusion(rankings)[:candidate_k]
    candidates = [id_to_record[i] for i in fused_ids]
    return rerank(query, candidates, top_n=top_n, client=client)
