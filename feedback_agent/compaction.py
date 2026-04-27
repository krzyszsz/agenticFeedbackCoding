from __future__ import annotations

from .config import AgentConfig
from .conversation import Conversation


def maybe_compact(conversation: Conversation, config: AgentConfig, client, *, context_window: int | None = None) -> bool:
    cfg = config.context_compaction
    if not cfg.enabled:
        return False
    limit = int((context_window or config.implementation_model.context_window) * cfg.threshold_ratio)
    if conversation.estimated_tokens() < limit:
        return False
    old_turns = conversation.turns[:-cfg.keep_recent_turns]
    source = "\n\n".join(f"{t.role}: {t.content}" for t in old_turns)
    prompt = (
        "Compact this coding-agent conversation into durable memory. Preserve requirements, "
        "decisions, failed attempts, open risks, and next steps. Do not include trivia.\n\n"
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
    conversation.replace_with_memory(memory, cfg.keep_recent_turns)
    return True


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
