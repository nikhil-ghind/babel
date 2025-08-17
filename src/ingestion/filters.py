"""Corpus quality filters applied between crawling and chunking: language
detection / allow-listing, minimum-length and quality heuristics, and exact +
near-duplicate removal. Keeps the multilingual corpus clean before embedding."""
from __future__ import annotations
import hashlib
import re
from typing import Iterable, Iterator, List, Optional, Set

from langdetect import LangDetectException, detect

from src.ingestion.crawler import RawDoc

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def detect_language(text: str) -> Optional[str]:
    """Best-effort ISO-639-1 language code, or None if detection fails."""
    try:
        return detect(text)
    except LangDetectException:
        return None


def _shingle_hash(text: str, k: int = 5) -> Set[int]:
    """k-word shingle set used for near-duplicate (Jaccard) detection."""
    words = _WORD_RE.findall(text.lower())
    return {hash(" ".join(words[i:i + k])) for i in range(max(len(words) - k + 1, 1))}


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def filter_documents(docs: Iterable[RawDoc], min_words: int = 30,
                     allowed_languages: Optional[Iterable[str]] = None,
                     near_dup_threshold: float = 0.85) -> Iterator[RawDoc]:
    """Stream docs through the filter chain.

    - drops docs shorter than `min_words`
    - tags each doc with `metadata['lang']`; drops langs outside `allowed_languages`
      (None = keep every detected language)
    - drops exact duplicates (content hash) and near-duplicates (shingle Jaccard
      above `near_dup_threshold`)
    """
    allow = set(allowed_languages) if allowed_languages else None
    seen_hashes: Set[str] = set()
    seen_shingles: List[Set[int]] = []

    for doc in docs:
        words = _WORD_RE.findall(doc.text)
        if len(words) < min_words:
            continue

        lang = detect_language(doc.text)
        if allow is not None and lang not in allow:
            continue

        content_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            continue

        shingles = _shingle_hash(doc.text)
        if any(_jaccard(shingles, prev) >= near_dup_threshold for prev in seen_shingles):
            continue

        seen_hashes.add(content_hash)
        seen_shingles.append(shingles)
        doc.metadata["lang"] = lang
        yield doc
