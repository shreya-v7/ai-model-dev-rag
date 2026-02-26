"""Reference grounding and validation helpers."""

from __future__ import annotations

from src.models import Chunk, QAAnswer, SynthesisResult
from src.security import quote_supported_by_chunk


def chunk_lookup(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {chunk.chunk_id: chunk for chunk in chunks}


def filter_references_with_real_quotes(
    result: SynthesisResult | QAAnswer,
    chunks: list[Chunk],
) -> None:
    lookup = chunk_lookup(chunks)

    if isinstance(result, QAAnswer):
        result.references = [
            ref
            for ref in result.references
            if (chunk := lookup.get(ref.chunk_id))
            and quote_supported_by_chunk(ref.quote, chunk.text)
        ]
        return

    for claim in result.claims:
        for evidence in claim.evidences:
            evidence.references = [
                ref
                for ref in evidence.references
                if (chunk := lookup.get(ref.chunk_id))
                and quote_supported_by_chunk(ref.quote, chunk.text)
            ]
