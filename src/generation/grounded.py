"""Grounded answer generation with inline citations via Cohere Command. Passes
reranked passages as `documents` to `co.chat`, which returns both the answer and
structured citation spans linking answer text back to specific source documents.
We render those spans into inline [n] markers plus a source list."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cohere

from src.clients import get_client
from src.retrieval.rerank import Ranked

DEFAULT_MODEL = "command-r-plus"

_PREAMBLE = (
    "You are Babel, a multilingual research assistant. Answer the user's question "
    "using ONLY the supplied documents. Cite the documents you rely on. If the "
    "documents do not contain the answer, say so. Reply in the language of the question."
)


@dataclass
class Citation:
    start: int
    end: int
    text: str
    document_ids: List[str] = field(default_factory=list)


@dataclass
class GroundedAnswer:
    text: str
    citations: List[Citation]
    documents: List[Dict[str, str]]

    def with_inline_markers(self) -> str:
        """Splice [n] markers into the answer text at each citation span."""
        if not self.citations:
            return self.text
        doc_order: Dict[str, int] = {}
        for c in self.citations:
            for did in c.document_ids:
                doc_order.setdefault(did, len(doc_order) + 1)
        out, cursor = [], 0
        for c in sorted(self.citations, key=lambda x: x.start):
            out.append(self.text[cursor:c.end])
            markers = "".join(f"[{doc_order[d]}]" for d in c.document_ids)
            out.append(markers)
            cursor = c.end
        out.append(self.text[cursor:])
        return "".join(out)


def _to_documents(passages: Sequence[Ranked]) -> List[Dict[str, str]]:
    docs = []
    for i, p in enumerate(passages):
        docs.append({
            "id": p.record.id or str(i),
            "title": str(p.record.metadata.get("source", f"doc-{i}")),
            "snippet": p.record.text,
        })
    return docs


def generate(query: str, passages: Sequence[Ranked], model: str = DEFAULT_MODEL,
             client: Optional[cohere.Client] = None) -> GroundedAnswer:
    client = client or get_client()
    documents = _to_documents(passages)
    resp = client.chat(
        model=model,
        preamble=_PREAMBLE,
        message=query,
        documents=documents,
        temperature=0.2,
    )
    citations = [
        Citation(start=c.start, end=c.end, text=c.text, document_ids=list(c.document_ids))
        for c in (resp.citations or [])
    ]
    return GroundedAnswer(text=resp.text, citations=citations, documents=documents)
