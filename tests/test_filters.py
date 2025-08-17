from src.ingestion.crawler import RawDoc
from src.ingestion.filters import detect_language, filter_documents


def test_drops_short_documents():
    docs = [RawDoc(text="too short", metadata={}),
            RawDoc(text="word " * 50, metadata={"source": "a"})]
    kept = list(filter_documents(docs, min_words=30))
    assert len(kept) == 1
    assert kept[0].metadata["source"] == "a"


def test_exact_duplicates_removed():
    body = "Babel performs multilingual semantic search across many languages. " * 5
    docs = [RawDoc(text=body, metadata={"source": "a"}),
            RawDoc(text=body, metadata={"source": "b"})]
    kept = list(filter_documents(docs, min_words=10))
    assert len(kept) == 1


def test_language_allowlist():
    en = RawDoc(text="The quick brown fox jumps over the lazy dog near the river bank today. " * 3,
                metadata={})
    fr = RawDoc(text="Le chat noir dort paisiblement sur le canapé du salon pendant la journée. " * 3,
                metadata={})
    kept = list(filter_documents([en, fr], min_words=10, allowed_languages=["en"]))
    assert len(kept) == 1
    assert kept[0].metadata["lang"] == "en"


def test_language_tagged_when_no_allowlist():
    docs = [RawDoc(text="word " * 50, metadata={})]
    kept = list(filter_documents(docs, min_words=10))
    assert "lang" in kept[0].metadata
