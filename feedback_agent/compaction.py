from __future__ import annotations

import re

from .config import AgentConfig
from .conversation import Conversation


def maybe_compact(
    conversation: Conversation,
    config: AgentConfig,
    client,
    *,
    context_window: int | None = None,
    incoming_tokens: int = 0,
    force: bool = False,
) -> bool:
    cfg = config.context_compaction
    if not cfg.enabled:
        return False
    limit = int((context_window or config.implementation_model.context_window) * cfg.threshold_ratio)
    if not force and conversation.estimated_tokens() + max(0, incoming_tokens) < limit:
        return False
    old_turns = conversation.turns[:-cfg.keep_recent_turns]
    source = "\n\n".join(f"{t.role}: {t.content}" for t in old_turns)
    prompt = (
        "Summarize this coding-agent conversation into durable memory for a later model turn. "
        "Preserve requirements, decisions, failed attempts, accepted evidence, open risks, and next steps. "
        "Use plain prose or bullets, not JSON. Do not include <think> text or trivia.\n\n"
        + source[-120000:]
    )
    try:
        memory = client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=cfg.summary_max_tokens,
            temperature=0.1,
        )
    except Exception:
        memory = deterministic_compact(source)
    cleaned = _clean_compaction_memory(memory)
    if _compaction_memory_is_too_weak(cleaned):
        cleaned = deterministic_compact(source)
    conversation.replace_with_memory(cleaned, cfg.keep_recent_turns)
    return True


def _clean_compaction_memory(memory: str) -> str:
    """Remove reasoning wrappers that some local models leak into summaries."""
    cleaned = re.sub(r"<think>.*?</think>", "", memory, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\|channel\>[^<]*<channel\|>", "", cleaned)
    cleaned = re.sub(r"<\|[^>]+?\|>", "", cleaned).strip()
    cleaned = "\n".join(
        line for line in cleaned.splitlines()
        if not line.strip().startswith(("<|channel>", "<channel|>", "<tool_call>", "<|tool_call>"))
    ).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned or "Compaction produced no usable memory; rely on the recent verbatim turns."


def _compaction_memory_is_too_weak(memory: str) -> bool:
    """Reject local-model summaries that would erase useful context.

    Some small or stressed local models occasionally return one token, a fake
    channel label, or a generic sentence during compaction. Keeping that would
    make the next turn worse than deterministic truncation, so the harness falls
    back to a mechanical head/tail summary instead.
    """
    text = memory.strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered in {"fallible_thought", "thought", "summary", "ok"}:
        return True
    if len(text) < 120 and not any(marker in lowered for marker in ("require", "step", "file", "test", "review", "plan")):
        return True
    return False


def deterministic_compact(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    head = lines[:40]
    tail = lines[-80:]
    return "\n".join([
        "Deterministic fallback compaction was used because model compaction failed.",
        "Important early context:",
        *head,
        "Recent older context:",
        *tail,
    ])
