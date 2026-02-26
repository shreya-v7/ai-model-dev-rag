from src.grounding import filter_references_with_real_quotes
from src.models import ClaimItem, EvidenceItem, QAAnswer, Reference, SynthesisResult
from src.prompts import SYSTEM_PROMPT, build_qa_prompt, build_synthesis_prompt


def test_prompt_templates_include_context() -> None:
    s = build_synthesis_prompt("topic", "ctx")
    q = build_qa_prompt("question", "ctx")
    assert "RETRIEVED_CONTEXTS" in s
    assert "RETRIEVED_CONTEXTS" in q
    assert "untrusted" in SYSTEM_PROMPT.lower()


def test_grounding_filters_invalid_references() -> None:
    chunks = [
        type("ChunkObj", (), {"chunk_id": "CHUNK-1", "text": "valid quote is here"})(),
    ]
    qa = QAAnswer(
        question="q",
        answer="a",
        references=[
            Reference(doc_id="DOC-1", chunk_id="CHUNK-1", quote="valid quote"),
            Reference(doc_id="DOC-1", chunk_id="CHUNK-1", quote="not present"),
        ],
        uncertainty="u",
    )
    filter_references_with_real_quotes(qa, chunks)  # type: ignore[arg-type]
    assert len(qa.references) == 1


def test_grounding_filters_synthesis_refs() -> None:
    chunks = [
        type("ChunkObj", (), {"chunk_id": "CHUNK-1", "text": "this quote exists"})(),
    ]
    result = SynthesisResult(
        topic_summary="s",
        claims=[
            ClaimItem(
                claim="c",
                confidence="low",
                evidences=[
                    EvidenceItem(
                        statement="e",
                        references=[
                            Reference(doc_id="DOC-1", chunk_id="CHUNK-1", quote="this quote"),
                            Reference(doc_id="DOC-1", chunk_id="CHUNK-1", quote="invalid"),
                        ],
                    )
                ],
            )
        ],
        unresolved_questions=[],
    )
    filter_references_with_real_quotes(result, chunks)  # type: ignore[arg-type]
    assert len(result.claims[0].evidences[0].references) == 1
