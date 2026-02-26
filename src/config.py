"""Centralized configuration for the RAG skeleton."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
for path in (DATA_DIR, OUTPUTS_DIR):
    path.mkdir(parents=True, exist_ok=True)

# LLM provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "grok")

GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_ENDPOINT = os.getenv(
    "GROK_ENDPOINT",
    "https://cmu-llm-api-resource.services.ai.azure.com/openai/v1/",
)
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3")

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_MODEL = os.getenv("AZURE_MODEL", "o4-mini")

# Generation and reliability
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))
LLM_BACKOFF_BASE_S = float(os.getenv("LLM_BACKOFF_BASE_S", "1.0"))
LLM_BACKOFF_MAX_S = float(os.getenv("LLM_BACKOFF_MAX_S", "15.0"))
LLM_MIN_CALL_INTERVAL_S = float(os.getenv("LLM_MIN_CALL_INTERVAL_S", "0.5"))

# Retrieval configuration
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "450"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
WORDS_PER_TOKEN = float(os.getenv("WORDS_PER_TOKEN", "0.75"))
TOP_K_PER_CLAIM = int(os.getenv("TOP_K_PER_CLAIM", "6"))
MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "10"))

# UI and upload policy
UI_MIN_DOCS = int(os.getenv("UI_MIN_DOCS", "10"))
UI_MAX_DOCS = int(os.getenv("UI_MAX_DOCS", "20"))
MAX_UPLOAD_FILE_MB = int(os.getenv("MAX_UPLOAD_FILE_MB", "25"))
