"""Provider-agnostic LLM client abstraction reused across projects."""

from __future__ import annotations

import json
import logging
import random
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
        jitter = random.uniform(0.0, base)
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
                    raise LLMServiceError(
                        f"{self.__class__.__name__} failed after {attempt + 1}/{attempts} attempts: {exc}"
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


def get_llm_client() -> LLMClient:
    from src.config import LLM_PROVIDER

    if LLM_PROVIDER == "grok":
        return GrokClient()
    if LLM_PROVIDER == "azure_openai":
        return AzureOpenAIClient()
    raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}")
