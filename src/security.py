"""Security helpers for safer RAG ingestion and prompting."""

from __future__ import annotations

import re
from pathlib import Path

from src.config import MAX_UPLOAD_FILE_MB

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".rst", ".json", ".csv"}

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+the\s+above", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"developer\s+message", re.IGNORECASE),
    re.compile(r"you\s+are\s+chatgpt", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?instructions", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow", re.IGNORECASE),
    re.compile(r"tool\s+call", re.IGNORECASE),
]


def validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension '{ext}'. Allowed: {allowed}.")
    return ext


def validate_upload_size(size_bytes: int) -> None:
    max_bytes = MAX_UPLOAD_FILE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise ValueError(f"File exceeds size limit ({MAX_UPLOAD_FILE_MB} MB).")


def sanitize_text(text: str) -> tuple[str, int]:
    """
    Returns sanitized text and number of filtered suspicious lines.
    """
    clean_lines: list[str] = []
    filtered = 0
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in _INJECTION_PATTERNS):
            filtered += 1
            continue
        clean_lines.append(line)
    sanitized = "\n".join(clean_lines).replace("\x00", " ").strip()
    return sanitized, filtered


def quote_supported_by_chunk(quote: str, chunk_text: str) -> bool:
    q = " ".join(quote.lower().split())
    c = " ".join(chunk_text.lower().split())
    if not q:
        return False
    return q in c
