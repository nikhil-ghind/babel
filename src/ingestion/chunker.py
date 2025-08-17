"""Semantic chunking with overlap. Splits documents on natural boundaries
(paragraphs / sentences) and packs them into token-bounded windows that carry a
configurable overlap, so retrieval units keep enough surrounding context. Works
across languages by falling back to character estimates when needed."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from src.ingestion.crawler import RawDoc

# Sentence boundary that also handles CJK full-stops and common punctuation.
_SENT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")
# Rough multilingual token estimate: ~4 chars/token is a reasonable default.
_CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, object] = field(default_factory=dict)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def chunk_document(doc: RawDoc, max_tokens: int = 256, overlap_tokens: int = 48) -> List[Chunk]:
    """Greedily pack sentences into windows of up to `max_tokens`, then carry the
    last `overlap_tokens` worth of sentences into the next window."""
    sentences = split_sentences(doc.text)
    chunks: List[Chunk] = []
    window: List[str] = []
    window_tokens = 0

    def flush(idx: int) -> None:
        if window:
            meta = dict(doc.metadata)
            meta["chunk_index"] = idx
            chunks.append(Chunk(text=" ".join(window), metadata=meta))

    for sent in sentences:
        t = _estimate_tokens(sent)
        if window and window_tokens + t > max_tokens:
            flush(len(chunks))
            # Build overlap tail from the end of the previous window.
            tail, tail_tokens = [], 0
            for s in reversed(window):
                st = _estimate_tokens(s)
                if tail_tokens + st > overlap_tokens:
                    break
                tail.insert(0, s)
                tail_tokens += st
            window, window_tokens = tail, tail_tokens
        window.append(sent)
        window_tokens += t

    flush(len(chunks))
    for i, c in enumerate(chunks):
        c.metadata["chunk_total"] = len(chunks)
        c.metadata["chunk_index"] = i
    return chunks


def chunk_documents(docs: Iterable[RawDoc], max_tokens: int = 256,
                    overlap_tokens: int = 48) -> List[Chunk]:
    out: List[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, max_tokens, overlap_tokens))
    return out
