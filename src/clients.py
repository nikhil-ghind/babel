"""Thin wrapper around the Cohere SDK so the rest of the codebase never touches
the raw client directly. Keeps API-key handling and response-shape normalisation
in one place, and makes the Cohere calls easy to mock in tests."""
from __future__ import annotations
import os
from functools import lru_cache
from typing import List, Sequence

import cohere


@lru_cache(maxsize=1)
def get_client() -> cohere.Client:
    """Return a process-wide Cohere client built from COHERE_API_KEY."""
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is not set")
    return cohere.Client(api_key=api_key)


def _normalise_embeddings(response) -> List[List[float]]:
    """Cohere v5 returns embeddings as either a bare list or an object with a
    `.float` attribute depending on `embedding_types`. Collapse both to a list."""
    emb = response.embeddings
    if hasattr(emb, "float"):          # EmbedByTypeResponse
        return list(emb.float)
    return list(emb)                   # plain list[list[float]]


def embed(texts: Sequence[str], model: str, input_type: str,
          client: cohere.Client | None = None) -> List[List[float]]:
    """Embed a batch of texts. `input_type` must be 'search_document' for the
    corpus and 'search_query' for queries (asymmetric multilingual embeddings)."""
    client = client or get_client()
    response = client.embed(
        texts=list(texts),
        model=model,
        input_type=input_type,
    )
    return _normalise_embeddings(response)
