from src.ingestion.chunker import chunk_document, split_sentences
from src.ingestion.crawler import RawDoc


def make_doc(n_sentences=40):
    text = " ".join(f"This is sentence number {i} about multilingual retrieval." for i in range(n_sentences))
    return RawDoc(text=text, metadata={"source": "doc0", "lang": "en"})


def test_chunks_respect_token_budget():
    chunks = chunk_document(make_doc(), max_tokens=64, overlap_tokens=16)
    assert len(chunks) > 1
    # ~4 chars/token estimate, allow slack for the final sentence pushing over.
    assert all(len(c.text) <= 64 * 4 + 80 for c in chunks)


def test_overlap_carries_context():
    chunks = chunk_document(make_doc(60), max_tokens=64, overlap_tokens=16)
    # consecutive windows should share at least one sentence due to overlap
    first_sents = set(split_sentences(chunks[0].text))
    second_sents = set(split_sentences(chunks[1].text))
    assert first_sents & second_sents


def test_metadata_propagates_and_indexes():
    chunks = chunk_document(make_doc(), max_tokens=64, overlap_tokens=16)
    for i, c in enumerate(chunks):
        assert c.metadata["source"] == "doc0"
        assert c.metadata["lang"] == "en"
        assert c.metadata["chunk_index"] == i
        assert c.metadata["chunk_total"] == len(chunks)
