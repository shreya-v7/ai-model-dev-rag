"""End-to-end RAG pipeline for synthesis and grounded Q&A."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.grounding import filter_references_with_real_quotes
from src.ingest import chunk_documents, load_documents, load_uploaded_documents
from src.llm_client import get_llm_client
from src.models import Chunk, Document, QAAnswer, SynthesisResult
from src.prompts import SYSTEM_PROMPT, build_qa_prompt, build_synthesis_prompt
from src.retrieval import Retriever


def _render_context_blocks(retrieved_chunks) -> str:
    blocks: list[str] = []
    for idx, item in enumerate(retrieved_chunks, start=1):
        chunk = item.chunk
        block = (
            f"[{idx}] doc_id={chunk.doc_id} chunk_id={chunk.chunk_id} score={item.score:.4f}\n"
            f"source={chunk.source_path}\n"
            f"text={chunk.text}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


@dataclass
class CorpusIndex:
    documents: list[Document]
    chunks: list[Chunk]
    retriever: Retriever
    injection_lines_filtered: int = 0


def _build_index(documents: list[Document], injection_lines_filtered: int = 0) -> CorpusIndex:
    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("No text chunks were created from the provided documents.")
    return CorpusIndex(
        documents=documents,
        chunks=chunks,
        retriever=Retriever(chunks=chunks),
        injection_lines_filtered=injection_lines_filtered,
    )


class RagService:
    def __init__(self) -> None:
        self._llm = get_llm_client()

    def synthesize_topic(
        self,
        index: CorpusIndex,
        topic: str,
        output_json_path: str | None = None,
    ) -> SynthesisResult:
        retrieved = index.retriever.search(topic)
        contexts = _render_context_blocks(retrieved)
        prompt = build_synthesis_prompt(topic=topic, contexts=contexts)
        payload = self._llm.generate_json(prompt=prompt, system=SYSTEM_PROMPT, max_tokens=3500)
        result = SynthesisResult.model_validate(payload)
        filter_references_with_real_quotes(result, index.chunks)

        if output_json_path:
            out_path = Path(output_json_path).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(result.model_dump(), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
        return result

    def answer_question(self, index: CorpusIndex, question: str) -> QAAnswer:
        retrieved = index.retriever.search(question)
        contexts = _render_context_blocks(retrieved)
        prompt = build_qa_prompt(question=question, contexts=contexts)
        payload = self._llm.generate_json(prompt=prompt, system=SYSTEM_PROMPT, max_tokens=1200)
        answer = QAAnswer.model_validate(payload)
        filter_references_with_real_quotes(answer, index.chunks)
        if not answer.references:
            answer.uncertainty = (
                "Insufficient verifiable quotes in retrieved chunks. "
                "Please refine the question."
            )
        return answer


_DEFAULT_RAG_SERVICE = RagService()


def build_index_from_paths(
    document_paths: list[str],
    max_documents: int | None = None,
) -> CorpusIndex:
    documents = load_documents(
        document_paths,
        max_documents=max_documents if max_documents is not None else len(document_paths),
    )
    return _build_index(documents)


def build_index_from_uploads(files, max_documents: int) -> CorpusIndex:
    documents, filtered_lines = load_uploaded_documents(files=files, max_documents=max_documents)
    return _build_index(documents, injection_lines_filtered=filtered_lines)


def synthesize_topic(
    index: CorpusIndex,
    topic: str,
    output_json_path: str | None = None,
) -> SynthesisResult:
    return _DEFAULT_RAG_SERVICE.synthesize_topic(
        index=index,
        topic=topic,
        output_json_path=output_json_path,
    )


def answer_question(index: CorpusIndex, question: str) -> QAAnswer:
    return _DEFAULT_RAG_SERVICE.answer_question(index=index, question=question)


def run_rag(
    topic: str,
    document_paths: list[str],
    output_json_path: str | None = None,
) -> SynthesisResult:
    index = build_index_from_paths(document_paths)
    return synthesize_topic(index=index, topic=topic, output_json_path=output_json_path)
