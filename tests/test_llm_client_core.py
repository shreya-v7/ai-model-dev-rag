import json

from src.llm_client import LLMClient, LLMServiceError, get_llm_client


class DummyLLM(LLMClient):
    def generate(self, *args, **kwargs):
        raise NotImplementedError


class _RespUsage:
    prompt_tokens = 10
    completion_tokens = 20


class _RespMessage:
    content = '{"ok": true}'


class _RespChoice:
    message = _RespMessage()


class _Resp:
    usage = _RespUsage()
    choices = [_RespChoice()]


class _ChatCompletions:
    def __init__(self, fail_first: bool = False):
        self._fail_first = fail_first
        self._called = 0

    def create(self, **kwargs):
        del kwargs
        self._called += 1
        if self._fail_first and self._called == 1:
            raise RuntimeError("timeout")
        return _Resp()


class _Client:
    def __init__(self, fail_first: bool = False):
        self.chat = type("Chat", (), {"completions": _ChatCompletions(fail_first=fail_first)})()


def test_generate_json_parses_fenced_json() -> None:
    class FenceLLM(DummyLLM):
        def generate(self, *args, **kwargs):
            del args, kwargs
            return type(
                "LLMResponseLike",
                (),
                {"text": "```json\n{\"x\":1}\n```", "input_tokens": 0, "output_tokens": 0},
            )()

    payload = FenceLLM().generate_json("prompt")
    assert payload == {"x": 1}


def test_retry_flow_succeeds_on_second_attempt(monkeypatch) -> None:
    llm = DummyLLM()
    monkeypatch.setattr(llm, "_sleep_backoff", lambda attempt: None)
    monkeypatch.setattr(llm, "_throttle", lambda: None)
    resp = llm._chat_completion_with_retry(_Client(fail_first=True), {"model": "x"})
    assert json.loads(resp.choices[0].message.content)["ok"] is True


def test_retry_flow_raises_on_non_retryable(monkeypatch) -> None:
    llm = DummyLLM()
    monkeypatch.setattr(llm, "_sleep_backoff", lambda attempt: None)
    monkeypatch.setattr(llm, "_throttle", lambda: None)

    class BadClient:
        completions = type(
            "Completions",
            (),
            {
                "create": staticmethod(
                    lambda **kwargs: (_ for _ in ()).throw(ValueError("bad"))
                )
            },
        )
        chat = type(
            "Chat",
            (),
            {"completions": completions},
        )()

    try:
        llm._chat_completion_with_retry(BadClient(), {"model": "x"})
        raise AssertionError("Expected LLMServiceError.")
    except LLMServiceError:
        pass


def test_get_llm_client_unknown_provider(monkeypatch) -> None:
    monkeypatch.setenv("OFFLINE_MODE", "0")
    monkeypatch.setenv("LLM_PROVIDER", "unknown_provider")
    try:
        get_llm_client()
        raise AssertionError("Expected ValueError for unknown provider.")
    except ValueError as exc:
        assert "Unknown LLM_PROVIDER" in str(exc)


def test_get_llm_client_routes_to_grok(monkeypatch) -> None:
    monkeypatch.setenv("OFFLINE_MODE", "0")
    monkeypatch.setenv("LLM_PROVIDER", "grok")
    monkeypatch.setattr("src.llm_client.GrokClient", lambda: "grok-client")
    assert get_llm_client() == "grok-client"


def test_get_llm_client_routes_to_azure(monkeypatch) -> None:
    monkeypatch.setenv("OFFLINE_MODE", "0")
    monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
    monkeypatch.setattr("src.llm_client.AzureOpenAIClient", lambda: "azure-client")
    assert get_llm_client() == "azure-client"
