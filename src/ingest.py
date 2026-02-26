"""Document loading and chunking utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import fitz

from src.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, MAX_DOCUMENTS, WORDS_PER_TOKEN
from src.models import Chunk, Document
from src.security import sanitize_text, validate_extension, validate_upload_size


class UploadedDoc(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def _read_pdf(path: Path) -> str:
    with fitz.open(path) as pdf:
        return "\n".join(page.get_text("text") for page in pdf)


def _read_pdf_bytes(raw: bytes) -> str:
    with fitz.open(stream=raw, filetype="pdf") as pdf:
        return "\n".join(page.get_text("text") for page in pdf)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_text_for_extension(
    ext: str,
    *,
    path: Path | None = None,
    raw: bytes | None = None,
) -> str:
    if ext == ".pdf":
        if path is not None:
            return _read_pdf(path)
        if raw is not None:
            return _read_pdf_bytes(raw)
        raise ValueError("Expected PDF path or bytes.")
    if ext in {".txt", ".md", ".rst", ".json", ".csv"}:
        if path is not None:
            return _read_text(path)
        if raw is not None:
            return raw.decode("utf-8", errors="ignore")
        raise ValueError("Expected text path or bytes.")
    raise ValueError(f"Unsupported extension: {ext}")


def load_documents(paths: list[str]) -> list[Document]:
    if len(paths) > MAX_DOCUMENTS:
        raise ValueError(f"Expected at most {MAX_DOCUMENTS} documents, got {len(paths)}.")

    docs: list[Document] = []
    for idx, raw_path in enumerate(paths, start=1):
        path = Path(raw_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        ext = validate_extension(path.name)
        text = _extract_text_for_extension(ext, path=path)

        doc_id = f"DOC-{idx:02d}"
        cleaned, _ = sanitize_text(text)
        docs.append(Document(doc_id=doc_id, path=str(path), text=cleaned))
    return docs


def load_uploaded_documents(
    files: list[UploadedDoc],
    max_documents: int = MAX_DOCUMENTS,
) -> tuple[list[Document], int]:
    if len(files) > max_documents:
        raise ValueError(f"Expected at most {max_documents} documents, got {len(files)}.")

    docs: list[Document] = []
    filtered_lines_total = 0
    for idx, uploaded in enumerate(files, start=1):
        ext = validate_extension(uploaded.name)
        raw = uploaded.getvalue()
        validate_upload_size(len(raw))
        text = _extract_text_for_extension(ext, raw=raw)
        cleaned, filtered = sanitize_text(text)
        filtered_lines_total += filtered
        docs.append(Document(doc_id=f"DOC-{idx:02d}", path=uploaded.name, text=cleaned))
    return docs, filtered_lines_total


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    words_per_chunk = max(100, int(CHUNK_SIZE_TOKENS * WORDS_PER_TOKEN))
    overlap_words = max(20, int(CHUNK_OVERLAP_TOKENS * WORDS_PER_TOKEN))
    step = max(1, words_per_chunk - overlap_words)

    all_chunks: list[Chunk] = []
    chunk_counter = 1
    for doc in documents:
        words = doc.text.split()
        if not words:
            continue

        for start in range(0, len(words), step):
            end = min(len(words), start + words_per_chunk)
            if start >= end:
                continue
            chunk_text = " ".join(words[start:end]).strip()
            if not chunk_text:
                continue
            chunk_id = f"CHUNK-{chunk_counter:04d}"
            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source_path=doc.path,
                    text=chunk_text,
                    start_word=start,
                    end_word=end,
                )
            )
            chunk_counter += 1
            if end == len(words):
                break
    return all_chunks
