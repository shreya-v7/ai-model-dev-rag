import json

from src.ingest import chunk_documents, load_documents
from src.pipeline import answer_question, build_index_from_paths, run_rag


def test_load_documents_and_chunking(tmp_path) -> None:
    p = tmp_path / "doc.txt"
    p.write_text("alpha beta gamma " * 500, encoding="utf-8")
    docs = load_documents([str(p)])
    chunks = chunk_documents(docs)
    assert docs[0].doc_id == "DOC-01"
    assert len(chunks) >= 1
    assert chunks[0].doc_id == "DOC-01"


def test_run_rag_offline_generates_valid_output(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    p = tmp_path / "doc1.txt"
    p.write_text(
        "Carbon accounting assumptions are documented and supported by source evidence.",
        encoding="utf-8",
    )
    out = tmp_path / "result.json"
    result = run_rag("Summarize assumptions", [str(p)], output_json_path=str(out))
    assert result.topic_summary
    assert out.exists()
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert "claims" in parsed


def test_answer_question_offline(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    p = tmp_path / "doc2.txt"
    p.write_text(
        "Emission factors are explained in section 4 with explicit assumptions.",
        encoding="utf-8",
    )
    index = build_index_from_paths([str(p)])
    answer = answer_question(index, "What assumptions are listed?")
    assert answer.answer
    assert answer.question
