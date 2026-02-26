from src.llm_client import MockOfflineClient, get_llm_client


def test_get_llm_client_returns_mock_in_offline_mode(monkeypatch) -> None:
    monkeypatch.setenv("OFFLINE_MODE", "1")
    client = get_llm_client()
    assert isinstance(client, MockOfflineClient)


def test_mock_client_generates_synthesis_payload() -> None:
    prompt = """
    RETRIEVED_CONTEXTS:
    [1] doc_id=DOC-01 chunk_id=CHUNK-0001 score=0.99
    source=doc1.txt
    text=Carbon accounting assumptions are documented in detail.

    JSON_SCHEMA:
    {"topic_summary":"string","claims":[]}
    """
    client = MockOfflineClient()
    payload = client.generate_json(prompt=prompt)
    assert "topic_summary" in payload
    assert payload["claims"][0]["evidences"][0]["references"][0]["doc_id"] == "DOC-01"


def test_mock_client_generates_qa_payload() -> None:
    prompt = """
    RETRIEVED_CONTEXTS:
    [1] doc_id=DOC-02 chunk_id=CHUNK-0002 score=0.91
    source=doc2.txt
    text=Offsets should be reported separately from reductions.

    JSON_SCHEMA:
    {"question":"string","answer":"string","references":[],"uncertainty":"string"}
    """
    client = MockOfflineClient()
    payload = client.generate_json(prompt=prompt)
    assert payload["references"][0]["chunk_id"] == "CHUNK-0002"
    assert "uncertainty" in payload
