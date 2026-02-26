from src.models import Chunk
from src.retrieval import HashEmbedder, Retriever


def test_hash_embedder_is_deterministic() -> None:
    embedder = HashEmbedder(dim=128)
    a = embedder.encode(["alpha beta gamma"], convert_to_numpy=True)
    b = embedder.encode(["alpha beta gamma"], convert_to_numpy=True)
    assert len(a) == 1
    assert len(a[0]) == 128
    assert a == b


def test_retriever_search_returns_ranked_chunks(monkeypatch) -> None:
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("EMBED_PROVIDER", "hash")

    chunks = [
        Chunk(
            chunk_id="CHUNK-0001",
            doc_id="DOC-01",
            source_path="doc1.txt",
            text="carbon accounting baseline emissions",
            start_word=0,
            end_word=5,
        ),
        Chunk(
            chunk_id="CHUNK-0002",
            doc_id="DOC-02",
            source_path="doc2.txt",
            text="biodiversity protection and habitat restoration",
            start_word=0,
            end_word=6,
        ),
    ]
    retriever = Retriever(chunks)
    results = retriever.search("carbon emissions baseline", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.doc_id in {"DOC-01", "DOC-02"}
