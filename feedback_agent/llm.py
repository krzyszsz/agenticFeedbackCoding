from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import TextIO

from .config import ModelConfig


class ModelRequestRetrier:
    """Retry transient model-server failures around one HTTP request.

    Long agentic runs are expensive. A single dropped TCP connection, overloaded
    model server, or temporary HTTP 5xx should not discard hours of useful
    work. The retrier is intentionally small and independent so tests can inject
    a zero-delay sleeper and prove retry behavior without talking to a real LLM.
    """

    def __init__(
        self,
        *,
        attempts: int,
        sleep_seconds: int,
        sleep: Callable[[float], None] = time.sleep,
        stream: TextIO = sys.stderr,
    ):
        self.attempts = max(1, attempts)
        self.sleep_seconds = max(0, sleep_seconds)
        self.sleep = sleep
        self.stream = stream

    def run(self, operation: Callable[[], str]) -> str:
        last_error: BaseException | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                return operation()
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                OSError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                RuntimeError,
            ) as exc:
                last_error = exc
                remaining = self.attempts - attempt
                if remaining <= 0:
                    break
                self._log_retry(attempt, remaining, exc)
                if self.sleep_seconds:
                    self.sleep(self.sleep_seconds)
        raise RuntimeError(f"model request failed after {self.attempts} attempts: {last_error}") from last_error

    def _log_retry(self, attempt: int, remaining: int, exc: BaseException) -> None:
        print(
            f"[model-retry] request failed on attempt {attempt}/{self.attempts}: "
            f"{exc!r}. Retrying in {self.sleep_seconds}s; {remaining} attempts left.",
            file=self.stream,
            flush=True,
        )


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

        def send_once() -> str:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if body.get("error"):
                raise RuntimeError(f"model returned error: {body['error']}")
            msg = body["choices"][0]["message"]
            return str(msg.get("content") or msg.get("reasoning_content") or "")

        return ModelRequestRetrier(
            attempts=self.cfg.retry_attempts,
            sleep_seconds=self.cfg.retry_sleep_seconds,
        ).run(send_once)
