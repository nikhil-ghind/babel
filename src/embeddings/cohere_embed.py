"""Multilingual embeddings via Cohere Embed. Uses asymmetric input types
(`search_document` for the corpus, `search_query` at query time) and batches
calls to respect rate limits. `embed-multilingual-v3.0` covers 100+ languages
in a shared vector space, which is what lets Babel match a query in one language
against documents in another."""
from __future__ import annotations
from typing import List, Sequence

import cohere

from src.clients import embed

DEFAULT_MODEL = "embed-multilingual-v3.0"
EMBED_DIM = 1024  # embed-multilingual-v3.0 output dimensionality


def embed_documents(texts: Sequence[str], model: str = DEFAULT_MODEL,
                    batch_size: int = 96,
                    client: cohere.Client | None = None) -> List[List[float]]:
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors.extend(embed(batch, model=model, input_type="search_document", client=client))
    return vectors


def embed_query(text: str, model: str = DEFAULT_MODEL,
                client: cohere.Client | None = None) -> List[float]:
    return embed([text], model=model, input_type="search_query", client=client)[0]
