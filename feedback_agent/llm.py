from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import TextIO

from .config import ModelConfig


MIN_MODEL_RESPONSE_BYTES = 1_000_000
MAX_MODEL_RESPONSE_BYTES = 64 * 1024 * 1024
JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}


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
                if not self._is_retryable(exc):
                    raise
                remaining = self.attempts - attempt
                if remaining <= 0:
                    break
                self._log_retry(attempt, remaining, exc)
                if self.sleep_seconds:
                    self.sleep(self.sleep_seconds)
        raise RuntimeError(f"model request failed after {self.attempts} attempts: {last_error}") from last_error

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """Return whether repeating the same request can plausibly succeed."""
        if not isinstance(exc, urllib.error.HTTPError):
            return True
        return exc.code in {408, 409, 425, 429} or 500 <= exc.code < 600

    def _log_retry(self, attempt: int, remaining: int, exc: BaseException) -> None:
        print(
            f"[model-retry] request failed on attempt {attempt}/{self.attempts}: "
            f"{exc!r}. Retrying in {self.sleep_seconds}s; {remaining} attempts left.",
            file=self.stream,
            flush=True,
        )


class ModelRequestHeartbeat:
    """Print terminal-only progress while one model request is still in flight.

    These messages deliberately go to the human-facing stream only. They are
    not appended to the shared conversation transcript, because heartbeat noise
    would make later model turns worse rather than wiser.
    """

    def __init__(
        self,
        *,
        interval_seconds: float,
        stream: TextIO = sys.stderr,
        clock: Callable[[], float] = time.monotonic,
        health_check: Callable[[], str] | None = None,
    ):
        self.interval_seconds = max(0, interval_seconds)
        self.stream = stream
        self.clock = clock
        self.health_check = health_check

    def run(self, label: str, operation: Callable[[], str]) -> str:
        if self.interval_seconds <= 0:
            return operation()

        stop = threading.Event()
        start = self.clock()

        def emit_progress() -> None:
            while not stop.wait(self.interval_seconds):
                elapsed = int(self.clock() - start)
                health = self._health_status()
                print(
                    f"[model-call] still waiting for {label}: {elapsed}s elapsed; {health}.",
                    file=self.stream,
                    flush=True,
                )

        thread = threading.Thread(target=emit_progress, daemon=True)
        thread.start()
        try:
            return operation()
        finally:
            stop.set()
            thread.join(timeout=0.2)

    def _health_status(self) -> str:
        if not self.health_check:
            return "health=not-configured"
        try:
            return self.health_check()
        except Exception as exc:  # pragma: no cover - defensive guard for user-facing progress only.
            return f"health-check-error={_short_error(exc)}"


class OpenAICompatClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int | None = None, temperature: float | None = None) -> str:
        return self._chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            request_label=self.cfg.name,
            reasoning_budget_tokens=self.cfg.reasoning_budget_tokens,
            request_json_object=self.cfg.request_json_object,
        )

    def chat_labeled(
        self,
        messages: list[dict[str, str]],
        *,
        request_label: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Run a normal request with a phase label in terminal-only progress."""
        return self._chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            request_label=f"{self.cfg.name}/{request_label}",
            reasoning_budget_tokens=self.cfg.reasoning_budget_tokens,
            request_json_object=self.cfg.request_json_object,
        )

    def chat_labeled_with_reasoning_budget(
        self,
        messages: list[dict[str, str]],
        *,
        request_label: str,
        reasoning_budget_tokens: int | None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Run a labeled request with a phase-selected reasoning allowance."""
        return self._chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            request_label=f"{self.cfg.name}/{request_label}",
            reasoning_budget_tokens=reasoning_budget_tokens,
            request_json_object=self.cfg.request_json_object,
        )

    def chat_for_compaction(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Summarize context without spending the full task reasoning budget."""
        reasoning_budget = self.cfg.reasoning_budget_tokens
        if reasoning_budget is not None:
            reasoning_budget = min(reasoning_budget, 512)
        return self._chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            request_label=f"{self.cfg.name}/context-compaction",
            reasoning_budget_tokens=reasoning_budget,
            request_json_object=False,
        )

    def chat_for_progress_review(
        self,
        messages: list[dict[str, str]],
        *,
        request_label: str,
        request_timeout_seconds: int,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Run one bounded review without delaying a completed tool call for hours.

        A progress review is advisory monitoring around a still-running process,
        not the task itself. If it cannot answer within the configured review
        cadence, the process runner conservatively keeps the approved command
        running and can ask again later.
        """
        reasoning_budget = self.cfg.reasoning_budget_tokens
        if reasoning_budget is not None:
            reasoning_budget = min(reasoning_budget, 512)
        configured_timeout = self.cfg.request_timeout_seconds
        review_timeout = max(1, request_timeout_seconds)
        if configured_timeout > 0:
            review_timeout = min(review_timeout, configured_timeout)
        return self._chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            request_label=f"{self.cfg.name}/{request_label}",
            reasoning_budget_tokens=reasoning_budget,
            request_timeout_seconds=review_timeout,
            retry_attempts=1,
            request_json_object=self.cfg.request_json_object,
        )

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None,
        temperature: float | None,
        request_label: str,
        reasoning_budget_tokens: int | None,
        request_timeout_seconds: int | None = None,
        retry_attempts: int | None = None,
        request_json_object: bool,
    ) -> str:
        response_tokens = self.cfg.max_tokens if max_tokens is None else max_tokens
        model_messages = _messages_for_model(
            messages,
            system_prompt_as_user=self.cfg.system_prompt_as_user,
        )
        payload = {
            "model": self.cfg.model,
            "messages": model_messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": response_tokens,
        }
        if self.cfg.top_p is not None:
            payload["top_p"] = self.cfg.top_p
        if self.cfg.top_k is not None:
            payload["top_k"] = self.cfg.top_k
        if self.cfg.min_p is not None:
            payload["min_p"] = self.cfg.min_p
        if self.cfg.presence_penalty is not None:
            payload["presence_penalty"] = self.cfg.presence_penalty
        if self.cfg.repeat_penalty is not None:
            payload["repeat_penalty"] = self.cfg.repeat_penalty
        if self.cfg.send_reasoning_budget and reasoning_budget_tokens is not None:
            payload["reasoning_budget"] = reasoning_budget_tokens
        if request_json_object:
            payload["response_format"] = JSON_OBJECT_RESPONSE_FORMAT
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
        effective_timeout = (
            self.cfg.request_timeout_seconds
            if request_timeout_seconds is None
            else request_timeout_seconds
        )
        request_timeout = None if effective_timeout <= 0 else effective_timeout
        response_byte_limit = min(
            MAX_MODEL_RESPONSE_BYTES,
            max(MIN_MODEL_RESPONSE_BYTES, response_tokens * 32),
        )

        def send_once() -> str:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                raw_body = resp.read(response_byte_limit + 1)
            if len(raw_body) > response_byte_limit:
                raise ValueError(
                    "model response exceeded the bounded HTTP response limit "
                    f"of {response_byte_limit} bytes"
                )
            body = json.loads(raw_body.decode("utf-8"))
            if body.get("error"):
                raise RuntimeError(f"model returned error: {body['error']}")
            msg = body["choices"][0]["message"]
            return format_assistant_message(msg, preserve_reasoning=self.cfg.preserve_reasoning)

        def send_with_heartbeat() -> str:
            input_tokens = sum(max(1, len(str(message.get("content") or "")) // 4) for message in model_messages)
            print(
                f"[model-call] starting {request_label}: input~{input_tokens} tokens; "
                f"max_output={response_tokens}; reasoning_budget={reasoning_budget_tokens}.",
                file=sys.stderr,
                flush=True,
            )
            return ModelRequestHeartbeat(
                interval_seconds=self.cfg.request_heartbeat_seconds,
                health_check=self.health_status,
            ).run(request_label, send_once)

        return ModelRequestRetrier(
            attempts=self.cfg.retry_attempts if retry_attempts is None else retry_attempts,
            sleep_seconds=self.cfg.retry_sleep_seconds,
        ).run(send_with_heartbeat)

    def health_status(self) -> str:
        """Probe the OpenAI-compatible REST endpoint for terminal progress.

        The agent uses `/chat/completions` for real work. The heartbeat probes
        `/models` because it is cheap, standard for OpenAI-compatible servers,
        and does not disturb the in-flight generation request.
        """
        probe_timeout = 5
        if self.cfg.request_timeout_seconds > 0:
            probe_timeout = max(1, min(5, self.cfg.request_timeout_seconds))
        req = urllib.request.Request(
            f"{self.cfg.base_url}/models",
            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=probe_timeout) as resp:
                return f"health=ok http={getattr(resp, 'status', 'unknown')}"
        except urllib.error.HTTPError as exc:
            return f"health=http-{exc.code}"
        except urllib.error.URLError as exc:
            return f"health=unreachable {_short_error(exc.reason)}"
        except TimeoutError as exc:
            return f"health=timeout {_short_error(exc)}"
        except OSError as exc:
            return f"health=unreachable {_short_error(exc)}"


def format_assistant_message(msg: dict, *, preserve_reasoning: bool) -> str:
    """Return assistant text while optionally preserving separate thinking.

    llama.cpp can expose reasoning as `message.reasoning_content` when started
    with `--reasoning-format deepseek`. The orchestration code still expects a
    single text transcript, so thinking is represented as a normal `<think>`
    block before the final content. This lets the current phase parse the final
    JSON object from the same text; the agent removes visible scratch reasoning
    before storing the response in durable chat memory.
    """
    content = str(msg.get("content") or "")
    reasoning = _message_reasoning_content(msg)
    if not preserve_reasoning:
        return content or reasoning
    if not reasoning:
        return content
    if content.lstrip().lower().startswith("<think"):
        return content
    reasoning_block = f"<think>\n{reasoning.strip()}\n</think>"
    if content:
        return f"{reasoning_block}\n{content}"
    return reasoning_block


def _messages_for_model(
    messages: list[dict[str, str]],
    *,
    system_prompt_as_user: bool,
) -> list[dict[str, str]]:
    """Apply role constraints and produce a template-safe turn sequence.

    Recipient filtering and compaction can legitimately leave adjacent messages
    with the same role. Coalescing those messages preserves their full labelled
    content while avoiding model-template assumptions that roles alternate or
    that only one initial system turn exists.
    """
    converted: list[dict[str, str]] = []
    for message in messages:
        original_role = str(message.get("role") or "user")
        role = "user" if system_prompt_as_user and original_role == "system" else original_role
        content = str(message.get("content") or "")
        if system_prompt_as_user and original_role == "system":
            content = "Harness instructions:\n" + content
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"] += "\n\n" + content
        else:
            converted.append({"role": role, "content": content})
    return converted


def _message_reasoning_content(msg: dict) -> str:
    """Handle the small field-name differences used by OpenAI-compatible servers."""
    reasoning = msg.get("reasoning_content")
    if reasoning is None:
        reasoning = msg.get("reasoning")
    if isinstance(reasoning, dict):
        reasoning = reasoning.get("content") or reasoning.get("text")
    return str(reasoning or "")


def _short_error(exc: object, *, limit: int = 120) -> str:
    text = str(exc).replace("\n", " ").strip() or exc.__class__.__name__
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
