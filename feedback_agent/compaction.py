from __future__ import annotations

import re

from .config import AgentConfig
from .conversation import Conversation, Turn


def maybe_compact(
    conversation: Conversation,
    config: AgentConfig,
    client,
    *,
    context_window: int | None = None,
    incoming_tokens: int = 0,
    pinned_context: str | None = None,
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
        "Do not mark a step complete unless the newest reviewer decision accepted it. "
        "If a later NEXT_IMPLEMENTATION_DIRECTIVE says needs_rework, pending, needs_plan_change, "
        "or needs_requirements_change, preserve that unresolved state exactly. "
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
    control_state = latest_control_state(conversation.turns)
    if control_state:
        cleaned = f"{cleaned}\n\n{control_state}"
    if pinned_context:
        cleaned = f"{cleaned}\n\nPINNED_WORKFLOW_STATE:\n{pinned_context}"
    conversation.replace_with_memory(cleaned, cfg.keep_recent_turns)
    return True


def _clean_compaction_memory(memory: str) -> str:
    """Remove reasoning wrappers that some local models leak into summaries."""
    cleaned = re.sub(r"<think>.*?</think>", "", memory, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
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
    early = lowered[:3000]
    if (
        '"files"' in early
        and '"commands"' in early
        and any(marker in early for marker in ('"plan_note"', '"test_evidence"', '"resolution_request"'))
    ):
        return True
    return False


def deterministic_compact(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    head = [_clip_compaction_line(line) for line in lines[:40]]
    tail = [_clip_compaction_line(line) for line in lines[-80:]]
    return "\n".join([
        "Deterministic fallback compaction was used because model compaction failed.",
        "Important early context:",
        *head,
        "Recent older context:",
        *tail,
    ])


def _clip_compaction_line(line: str, limit: int = 1200) -> str:
    if len(line) <= limit:
        return line
    return line[:limit].rstrip() + " ... [truncated long compaction line]"


def latest_control_state(turns: list[Turn]) -> str:
    """Return an authoritative workflow-state guard for compacted memory.

    Local summarizers can be over-eager and turn "the implementation claims the
    step is complete but feedback rejected it" into "the step is complete".
    That is poisonous in a long chat. This deterministic guard is appended after
    model-generated memory and explicitly wins over older prose summaries.
    """
    last_request = _last_matching_turn_with_index(turns, "IMPLEMENTATION_AGENT_REQUEST:")
    last_directive = _last_matching_turn_with_index(turns, "NEXT_IMPLEMENTATION_DIRECTIVE:")
    last_feedback = _last_matching_turn_with_index(turns, "FEEDBACK_AGENT_RESPONSE:")
    last_final_request = _last_turn_containing_with_index(turns, "FINAL_PROJECT_REVIEW_PHASE")

    final_review_is_latest = (
        last_final_request is not None
        and last_feedback is not None
        and last_feedback[0] > last_final_request[0]
        and (last_request is None or last_final_request[0] > last_request[0])
    )

    lines: list[str] = []
    if final_review_is_latest:
        _append_reviewer_state(lines, "Final project review", last_feedback[1].content)
        if not lines:
            lines.append("- Final project review response is present in the recent transcript.")
    elif last_request:
        request_turn = last_request[1]
        step = _extract_first_group(
            r"IMPLEMENT_PLAN_STEP_PHASE\s+step_id=([A-Za-z0-9_.-]+)\s+attempt=([0-9]+)",
            request_turn.content,
        )
        if step:
            lines.append(f"- Current implementation request: step_id={step[0]} attempt={step[1]}.")
        else:
            lines.append("- Current implementation request is present in the recent transcript.")

        directive_source = _latest_indexed_turn(last_directive, last_feedback)
        if directive_source:
            marker = "Last reviewer directive" if directive_source == last_directive else "Last reviewer response"
            _append_reviewer_state(lines, marker, directive_source[1].content)

    if not lines:
        return ""
    return "\n".join([
        "AUTHORITATIVE_RECENT_CONTROL_STATE:",
        "This deterministic block overrides any older compacted prose above.",
        "If it says a step is pending, needs_rework, needs_plan_change, or needs_requirements_change, "
        "do not treat that step as accepted just because an older summary says it is complete.",
        *lines,
    ])


def _append_reviewer_state(lines: list[str], marker: str, content: str) -> None:
    status = _extract_jsonish_value("status", content)
    needs_rework = _extract_jsonish_value("needs_rework", content)
    summary = _extract_jsonish_value("summary", content)
    state_bits = []
    if status:
        state_bits.append(f"status={status}")
    if needs_rework:
        state_bits.append(f"needs_rework={needs_rework}")
    if state_bits:
        lines.append(f"- {marker}: " + ", ".join(state_bits) + ".")
    else:
        lines.append(f"- {marker} is present in the recent transcript.")
    if summary:
        lines.append(f"- Reviewer summary: {_clip(summary, 500)}")


def _latest_indexed_turn(
    first: tuple[int, Turn] | None,
    second: tuple[int, Turn] | None,
) -> tuple[int, Turn] | None:
    if first is None:
        return second
    if second is None:
        return first
    return first if first[0] > second[0] else second


def _last_matching_turn_with_index(turns: list[Turn], prefix: str) -> tuple[int, Turn] | None:
    for index, turn in reversed(list(enumerate(turns))):
        if turn.content.startswith(prefix):
            return index, turn
    return None


def _last_turn_containing_with_index(turns: list[Turn], needle: str) -> tuple[int, Turn] | None:
    for index, turn in reversed(list(enumerate(turns))):
        if needle in turn.content:
            return index, turn
    return None


def _extract_first_group(pattern: str, text: str) -> tuple[str, ...] | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return tuple(group for group in match.groups() if group is not None)


def _extract_jsonish_value(key: str, text: str) -> str:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*("(?:\\.|[^"\\])*"|true|false|null)', text, flags=re.IGNORECASE)
    if not match:
        return ""
    raw = match.group(1)
    if raw.startswith('"'):
        try:
            import json

            return str(json.loads(raw))
        except Exception:
            return raw.strip('"')
    return raw.lower()


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit - 40)
    return text[:keep].rstrip() + " ... [truncated control-state text]"
