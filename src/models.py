"""Shared data models for the RAG skeleton."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    path: str
    text: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    start_word: int
    end_word: int


class Reference(BaseModel):
    doc_id: str = Field(..., description="Document id backing the claim/evidence.")
    chunk_id: str = Field(..., description="Chunk id backing the claim/evidence.")
    quote: str = Field(..., description="Verbatim excerpt from the source chunk.")


class EvidenceItem(BaseModel):
    statement: str
    references: list[Reference]


class ClaimItem(BaseModel):
    claim: str
    confidence: str = Field(..., description="low/medium/high")
    evidences: list[EvidenceItem]


class SynthesisResult(BaseModel):
    topic_summary: str
    claims: list[ClaimItem]
    unresolved_questions: list[str]


class QAAnswer(BaseModel):
    question: str
    answer: str
    references: list[Reference]
    uncertainty: str = Field(
        ...,
        description="Explain gaps or ambiguity if evidence is incomplete.",
    )
