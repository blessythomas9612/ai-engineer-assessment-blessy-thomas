"""Minimal client for Groq's OpenAI-compatible chat completions endpoint."""
import logging

import httpx

log = logging.getLogger(__name__)


class LLMError(Exception):
    """The LLM call failed or returned something unusable."""


class LLMClient:
    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 15,
                 transport: httpx.BaseTransport | None = None, reasoning_effort: str | None = None):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self.model = model
        # Only reasoning models (GPT-OSS) accept this parameter; others would reject it.
        self.reasoning_effort = reasoning_effort if model.startswith("openai/gpt-oss") else None
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def chat(self, system: str, user: str, json_mode: bool = False, temperature: float = 0.2) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.post("/chat/completions", json=body)
            if json_mode and resp.status_code == 400:
                # Some models reject JSON mode; the prompt already asks for JSON, so retry without it.
                log.warning("JSON mode rejected by %s; retrying without it", self.model)
                body.pop("response_format")
                resp = self._client.post("/chat/completions", json=body)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except httpx.TimeoutException as exc:
            raise LLMError("LLM request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM returned HTTP {exc.response.status_code}") from exc
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"LLM request failed ({type(exc).__name__})") from exc
        if not content or not content.strip():
            raise LLMError("LLM returned an empty answer")
        return content.strip()
