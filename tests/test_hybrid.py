import numpy as np

from src.ingestion.chunker import Chunk
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.vectorstore.store import HybridStore


def build_store():
    chunks = [
        Chunk(text="cats and dogs are common household pets", metadata={"source": "a"}),
        Chunk(text="distributed systems use consensus protocols like raft", metadata={"source": "b"}),
        Chunk(text="multilingual embeddings map languages into one vector space", metadata={"source": "c"}),
    ]
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    store = HybridStore()
    store.add(chunks, vectors)
    return store


def test_dense_search_orders_by_cosine():
    store = build_store()
    results = store.dense_search([0.0, 0.0, 1.0], k=3)
    assert results[0][0] == 2  # the multilingual-embeddings vector is closest


def test_sparse_search_finds_lexical_match():
    store = build_store()
    results = store.sparse_search("raft consensus", k=3)
    assert store.get(results[0][0]).metadata["source"] == "b"


def test_rrf_rewards_agreement():
    # doc 2 ranks high in both lists -> should win the fusion
    fused = reciprocal_rank_fusion([[2, 0, 1], [2, 1, 0]])
    assert fused[0] == 2


def test_save_and_load_roundtrip(tmp_path):
    store = build_store()
    store.save(str(tmp_path / "store"))
    loaded = HybridStore.load(str(tmp_path / "store"))
    assert len(loaded) == 3
    assert loaded.sparse_search("pets", k=1)[0][0] == 0
    assert loaded.dense_search([1.0, 0.0, 0.0], k=1)[0][0] == 0
