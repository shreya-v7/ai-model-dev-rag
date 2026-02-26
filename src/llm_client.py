"""Provider-agnostic LLM client abstraction reused across projects."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


class LLMServiceError(RuntimeError):
    """Raised when an LLM call fails after retries."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ContextRow:
    doc_id: str
    chunk_id: str
    text: str


class LLMClient:
    """Abstract base class for LLM providers."""

    provider: str = "base"
    _last_call_ts: float = 0.0

    def _throttle(self) -> None:
        from src.config import LLM_MIN_CALL_INTERVAL_S

        if LLM_MIN_CALL_INTERVAL_S <= 0:
            return
        elapsed = time.time() - self._last_call_ts
        if elapsed < LLM_MIN_CALL_INTERVAL_S:
            time.sleep(LLM_MIN_CALL_INTERVAL_S - elapsed)

    def _sleep_backoff(self, attempt: int) -> None:
        from src.config import LLM_BACKOFF_BASE_S, LLM_BACKOFF_MAX_S

        base = max(0.1, LLM_BACKOFF_BASE_S)
        max_wait = max(base, LLM_BACKOFF_MAX_S)
        wait = min(max_wait, base * (2**attempt))
        jitter = random.uniform(0.0, base)  # nosec B311
        time.sleep(wait + jitter)

    def _is_retryable_error(self, exc: Exception) -> tuple[bool, str]:
        name = exc.__class__.__name__
        status_code = getattr(exc, "status_code", None)
        body = str(exc).lower()
        retryable_status = {408, 409, 429, 500, 502, 503, 504}
        retryable_name_markers = (
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
        )

        if status_code in retryable_status:
            return True, f"status={status_code}"
        if any(marker in name for marker in retryable_name_markers):
            return True, name
        if "rate limit" in body or "too many requests" in body or "timeout" in body:
            return True, name
        return False, name

    def _chat_completion_with_retry(self, client, kwargs: dict):
        from src.config import LLM_MAX_RETRIES

        attempts = max(1, LLM_MAX_RETRIES + 1)
        for attempt in range(attempts):
            try:
                self._throttle()
                resp = client.chat.completions.create(**kwargs)
                self._last_call_ts = time.time()
                return resp
            except Exception as exc:
                retryable, reason = self._is_retryable_error(exc)
                is_last = attempt == attempts - 1
                if not retryable or is_last:
                    msg = (
                        f"{self.__class__.__name__} failed after "
                        f"{attempt + 1}/{attempts} attempts: {exc}"
                    )
                    raise LLMServiceError(
                        msg
                    ) from exc
                log.warning(
                    "%s transient error (attempt %d/%d, reason=%s). Retrying...",
                    self.__class__.__name__,
                    attempt + 1,
                    attempts,
                    reason,
                )
                self._sleep_backoff(attempt)

        raise LLMServiceError(f"{self.__class__.__name__} failed unexpectedly.")

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        response = self.generate(
            prompt=prompt,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.text.strip()
        # Handles JSON wrapped in markdown fences.
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
        return json.loads(text)


class GrokClient(LLMClient):
    provider = "grok"

    def __init__(self) -> None:
        from openai import OpenAI

        from src.config import GROK_API_KEY, GROK_ENDPOINT

        if not GROK_API_KEY:
            raise ValueError("GROK_API_KEY is required when LLM_PROVIDER=grok.")
        self._client = OpenAI(base_url=GROK_ENDPOINT, api_key=GROK_API_KEY)

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from src.config import GENERATION_TEMPERATURE, GROK_MODEL

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": model or GROK_MODEL,
            "messages": messages,
            "temperature": temperature if temperature is not None else GENERATION_TEMPERATURE,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        resp = self._chat_completion_with_retry(self._client, kwargs)
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )


class AzureOpenAIClient(LLMClient):
    provider = "azure_openai"

    def __init__(self) -> None:
        from openai import AzureOpenAI

        from src.config import AZURE_API_KEY, AZURE_API_VERSION, AZURE_ENDPOINT

        if not AZURE_API_KEY:
            raise ValueError("AZURE_API_KEY is required when LLM_PROVIDER=azure_openai.")
        if not AZURE_ENDPOINT:
            raise ValueError("AZURE_ENDPOINT is required when LLM_PROVIDER=azure_openai.")

        self._client = AzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from src.config import AZURE_MODEL, GENERATION_TEMPERATURE

        deploy = model or AZURE_MODEL
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {"model": deploy, "messages": messages}
        if deploy.startswith("o"):
            if max_tokens is not None:
                kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = (
                temperature if temperature is not None else GENERATION_TEMPERATURE
            )
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

        resp = self._chat_completion_with_retry(self._client, kwargs)
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )


class MockOfflineClient(LLMClient):
    provider = "mock"

    def _extract_context_rows(self, prompt: str) -> list[ContextRow]:
        pattern = re.compile(
            r"\s*\[\d+\]\s+doc_id=(?P<doc_id>\S+)\s+chunk_id=(?P<chunk_id>\S+)"
            r".*?\n\s*source=.*?\n\s*text=(?P<text>.*?)(?=\n\s*\n\s*\[\d+\]|\Z)",
            re.DOTALL,
        )
        rows: list[ContextRow] = []
        for match in pattern.finditer(prompt):
            text = " ".join(match.group("text").split())
            rows.append(
                ContextRow(
                    doc_id=match.group("doc_id"),
                    chunk_id=match.group("chunk_id"),
                    text=text,
                )
            )
        return rows

    def _quote(self, text: str) -> str:
        words = text.split()
        return " ".join(words[:24]).strip() or "No quote available."

    def _build_payload(self, prompt: str) -> dict:
        rows = self._extract_context_rows(prompt)
        first = (
            rows[0]
            if rows
            else ContextRow(
                doc_id="DOC-00",
                chunk_id="CHUNK-0000",
                text="No context provided.",
            )
        )
        quote = self._quote(first.text)

        if '"topic_summary"' in prompt and '"claims"' in prompt:
            return {
                "topic_summary": (
                    f"Offline deterministic synthesis over {len(rows)} retrieved chunks."
                ),
                "claims": [
                    {
                        "claim": (
                            "The uploaded corpus contains extractable grounded evidence "
                            "for the requested topic."
                        ),
                        "confidence": "medium",
                        "evidences": [
                            {
                                "statement": (
                                    "A representative quote was extracted "
                                    "from the top retrieved chunk."
                                ),
                                "references": [
                                    {
                                        "doc_id": first.doc_id,
                                        "chunk_id": first.chunk_id,
                                        "quote": quote,
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "unresolved_questions": [
                    "Would additional documents change confidence and coverage of this synthesis?"
                ],
            }

        return {
            "question": "Offline question",
            "answer": "Offline deterministic answer generated from retrieved context only.",
            "references": [
                {
                    "doc_id": first.doc_id,
                    "chunk_id": first.chunk_id,
                    "quote": quote,
                }
            ],
            "uncertainty": (
                "This is a mock offline answer intended "
                "for reproducible demonstrations."
            ),
        }

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del system, model, temperature, max_tokens
        payload = self._build_payload(prompt)
        return LLMResponse(text=json.dumps(payload), input_tokens=0, output_tokens=0)


def get_llm_client() -> LLMClient:
    from src.config import LLM_PROVIDER, OFFLINE_MODE

    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    offline = os.getenv("OFFLINE_MODE", "1" if OFFLINE_MODE else "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if offline:
        return MockOfflineClient()

    if provider == "grok":
        from src.config import GROK_API_KEY

        if not GROK_API_KEY:
            raise ValueError("GROK_API_KEY is required when LLM_PROVIDER=grok.")
        return GrokClient()
    if provider == "azure_openai":
        return AzureOpenAIClient()
    if provider == "mock":
        return MockOfflineClient()
    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}")
