import httpx
import pytest

from app.llm import LLMClient, LLMError


def ok(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def client_with(handler):
    return LLMClient("key", "test-model", "https://api.groq.com/openai/v1", transport=httpx.MockTransport(handler))


def test_returns_reply_text():
    assert client_with(lambda req: ok("  Hello  ")).chat("sys", "hi") == "Hello"


def test_json_mode_is_retried_without_response_format_if_rejected():
    bodies = []

    def handler(req):
        body = req.read().decode()
        bodies.append(body)
        return httpx.Response(400) if "response_format" in body else ok('{"route": "none", "heroes": []}')

    assert client_with(handler).chat("sys", "q", json_mode=True) == '{"route": "none", "heroes": []}'
    assert len(bodies) == 2 and "response_format" not in bodies[1]


@pytest.mark.parametrize("response", [httpx.Response(401), httpx.Response(429), httpx.Response(200, json={}), ok("   ")])
def test_failures_become_llm_error(response):
    with pytest.raises(LLMError):
        client_with(lambda req: response).chat("sys", "q")


def test_reasoning_effort_only_sent_to_gpt_oss_models():
    sent = []

    def handler(req):
        sent.append(req.read().decode())
        return ok("hi")

    def client(model):
        return LLMClient("key", model, "https://api.groq.com/openai/v1",
                         transport=httpx.MockTransport(handler), reasoning_effort="low")

    client("openai/gpt-oss-120b").chat("sys", "q")
    client("llama-3.1-8b-instant").chat("sys", "q")
    assert '"reasoning_effort"' in sent[0] and '"reasoning_effort"' not in sent[1]
