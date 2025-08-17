"""Query rewriting and multi-query expansion via Cohere Command. Rewriting
normalises a raw user question into a clean search query; expansion generates
several paraphrases / sub-questions so hybrid retrieval can cover more of the
relevant corpus before reranking trims it back down."""
from __future__ import annotations
import json
from typing import List, Optional

import cohere

from src.clients import get_client

DEFAULT_MODEL = "command-r"

_REWRITE_PREAMBLE = (
    "You rewrite a user's question into a single, self-contained search query. "
    "Resolve pronouns and context, keep it in the original language, and return "
    "ONLY the rewritten query with no preamble."
)

_EXPAND_PREAMBLE = (
    "You expand a search query into diverse alternative phrasings and sub-questions "
    "that would retrieve relevant passages. Return a JSON array of strings only."
)


def rewrite_query(query: str, model: str = DEFAULT_MODEL,
                  client: Optional[cohere.Client] = None) -> str:
    client = client or get_client()
    resp = client.chat(model=model, preamble=_REWRITE_PREAMBLE, message=query, temperature=0.0)
    rewritten = resp.text.strip()
    return rewritten or query


def expand_query(query: str, n: int = 3, model: str = DEFAULT_MODEL,
                 client: Optional[cohere.Client] = None) -> List[str]:
    """Return the original query plus up to `n` alternative formulations."""
    client = client or get_client()
    resp = client.chat(
        model=model,
        preamble=_EXPAND_PREAMBLE,
        message=f"Query: {query}\nReturn {n} alternatives as a JSON array.",
        temperature=0.3,
    )
    variants: List[str] = []
    try:
        parsed = json.loads(resp.text)
        if isinstance(parsed, list):
            variants = [str(v).strip() for v in parsed if str(v).strip()]
    except json.JSONDecodeError:
        variants = [line.strip("-• ").strip() for line in resp.text.splitlines() if line.strip()]
    # De-duplicate while keeping the original first.
    seen, out = set(), []
    for q in [query, *variants]:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[: n + 1]
