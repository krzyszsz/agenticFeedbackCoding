from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import ModelConfig


class OpenAICompatClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int | None = None, temperature: float | None = None) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.cfg.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
            method="POST",
        )
        request_timeout = None if self.cfg.request_timeout_seconds <= 0 else self.cfg.request_timeout_seconds
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"model request failed: {exc}") from exc
        if body.get("error"):
            raise RuntimeError(f"model returned error: {body['error']}")
        msg = body["choices"][0]["message"]
        return str(msg.get("content") or msg.get("reasoning_content") or "")
