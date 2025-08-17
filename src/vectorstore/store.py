"""Self-contained hybrid store: dense vectors (NumPy, cosine) alongside a BM25
lexical index over the same corpus, persisted to disk together so the two views
never drift. Keeping both in one store is what lets the retriever fuse dense and
sparse rankings without a second service to operate."""
from __future__ import annotations
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk


@dataclass
class Record:
    id: str
    text: str
    metadata: Dict[str, object] = field(default_factory=dict)


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class HybridStore:
    """Holds aligned dense vectors + BM25 index + records, indexed by position."""

    def __init__(self) -> None:
        self._records: List[Record] = []
        self._matrix: Optional[np.ndarray] = None  # (n, dim), L2-normalised
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_tokens: List[List[str]] = []

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> List[Record]:
        return self._records

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        start = len(self._records)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            self._records.append(Record(id=str(start + i), text=chunk.text,
                                        metadata=chunk.metadata))
            self._corpus_tokens.append(_tokenize(chunk.text))
        new = _l2_normalise(np.asarray(vectors, dtype=np.float32))
        self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def dense_search(self, query_vector: Sequence[float], k: int = 10) -> List[Tuple[int, float]]:
        """Return (record_index, cosine_score) for the top-k nearest vectors."""
        if self._matrix is None or len(self._records) == 0:
            return []
        q = _l2_normalise(np.asarray([query_vector], dtype=np.float32))[0]
        scores = self._matrix @ q
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]

    def sparse_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Return (record_index, bm25_score) for the top-k lexical matches."""
        if self._bm25 is None or len(self._records) == 0:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]

    def get(self, idx: int) -> Record:
        return self._records[idx]

    # --- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self._matrix if self._matrix is not None else np.empty((0, 0)))
        (d / "records.json").write_text(
            json.dumps([asdict(r) for r in self._records], ensure_ascii=False))
        with open(d / "bm25.pkl", "wb") as f:
            pickle.dump(self._corpus_tokens, f)

    @classmethod
    def load(cls, path: str) -> "HybridStore":
        d = Path(path)
        store = cls()
        records = json.loads((d / "records.json").read_text())
        store._records = [Record(**r) for r in records]
        matrix = np.load(d / "vectors.npy")
        store._matrix = matrix if matrix.size else None
        with open(d / "bm25.pkl", "rb") as f:
            store._corpus_tokens = pickle.load(f)
        if store._corpus_tokens:
            store._bm25 = BM25Okapi(store._corpus_tokens)
        return store


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
