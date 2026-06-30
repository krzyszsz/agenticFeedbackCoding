from __future__ import annotations

import re

from .bounds import estimate_tokens
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
    if cfg.max_uncompacted_tokens > 0:
        limit = min(limit, cfg.max_uncompacted_tokens)
    if not force and conversation.estimated_tokens() + max(0, incoming_tokens) < limit:
        return False
    keep_recent_turns = _bounded_recent_turn_count(
        conversation.turns,
        cfg.keep_recent_turns,
        cfg.recent_turns_max_tokens,
    )
    old_turns = conversation.turns[:-keep_recent_turns] if keep_recent_turns else conversation.turns
    source = _source_for_compaction(old_turns)
    initial_context = initial_request_context(conversation.turns)
    prompt = (
        "Summarize this coding-agent conversation into durable memory for a later model turn. "
        "Preserve the initial user request, requirements, decisions, facts discovered during analysis, "
        "failed attempts, accepted evidence, open risks, and next steps. Prioritize information needed "
        "for future repair or verification over dead-end detail. Mention dead ends only by outcome unless "
        "their exact evidence is still needed. "
        "Do not mark a step complete unless the newest reviewer decision accepted it. "
        "Do not use words like confirmed, verified, resolved, or passed for a result unless the newest "
        "reviewer-owned validation or reviewer decision actually accepted it. If validation failed, "
        "timed out, was blocked, or exposed a mismatch, preserve that failure state and label any "
        "implementation-side success statement as a claim. "
        "If a later NEXT_IMPLEMENTATION_DIRECTIVE says needs_rework, pending, needs_plan_change, "
        "or needs_requirements_change, preserve that unresolved state exactly. "
        "Use plain prose or bullets, not JSON. Do not include <think> text or trivia.\n\n"
        "Pinned initial request/context to preserve:\n"
        + initial_context
        + "\n\nOlder transcript to summarize:\n"
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
    control_state = latest_control_state(conversation.turns)
    if _compaction_memory_is_too_weak(cleaned):
        cleaned = deterministic_compact(source)
    elif _compaction_memory_conflicts_with_control_state(cleaned, control_state):
        cleaned = deterministic_compact(source)
    cleaned = f"COMPACTED_WORKFLOW_MEMORY:\n{cleaned}"
    if initial_context:
        cleaned = f"INITIAL_REQUEST_CONTEXT:\n{initial_context}\n\n{cleaned}"
    if control_state:
        cleaned = f"{cleaned}\n\n{control_state}"
    if pinned_context:
        cleaned = f"{cleaned}\n\nPINNED_WORKFLOW_STATE:\n{pinned_context}"
    conversation.replace_with_memory(cleaned, keep_recent_turns)
    return True


def _bounded_recent_turn_count(turns: list[Turn], max_turns: int, max_tokens: int) -> int:
    """Keep recent verbatim context useful without carrying huge tool payloads forever."""
    if max_turns <= 0 or not turns:
        return 0
    if max_tokens <= 0:
        return min(max_turns, len(turns))
    total = 0
    keep = 0
    for turn in reversed(turns):
        turn_tokens = estimate_tokens(turn.content)
        if keep >= max_turns:
            break
        if keep > 0 and total + turn_tokens > max_tokens:
            break
        total += turn_tokens
        keep += 1
        if total >= max_tokens:
            break
    return keep


def _source_for_compaction(turns: list[Turn]) -> str:
    return "\n\n".join(_turn_for_compaction(turn) for turn in turns)


def _turn_for_compaction(turn: Turn) -> str:
    if turn.role == "system" and _is_compacted_memory_turn(turn.content):
        return (
            "system: Previous compacted-memory block omitted; "
            "fresh initial context, control state, and pinned workflow state are appended separately."
        )
    if turn.role == "system":
        return "system: [base system prompt omitted from compaction source]"
    if turn.role == "user" and turn.content.startswith(("IMPLEMENTATION_AGENT_REQUEST:", "FEEDBACK_AGENT_REQUEST:")):
        return _summarize_generated_prompt_turn(turn)
    return f"{turn.role}: {turn.content}"


def _summarize_generated_prompt_turn(turn: Turn) -> str:
    marker, _, rest = turn.content.partition("\n")
    phase = ""
    for raw_line in rest.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{") or line.startswith("["):
            break
        phase = line
        break
    suffix = f" {phase}" if phase else ""
    return f"{turn.role}: {marker}{suffix}\n[generated harness prompt omitted from compaction source]"


def initial_request_context(turns: list[Turn], *, limit: int = 8000) -> str:
    """Return deterministic initial context that model summaries must not erase."""
    selected: list[str] = []
    for turn in turns:
        if turn.role == "system":
            if _is_compacted_memory_turn(turn.content):
                preserved = _initial_request_context_from_memory(turn.content)
                if preserved:
                    selected.append(preserved)
            continue
        if turn.role == "user" and turn.content.startswith("PROJECT DESIGN:"):
            selected.append(f"user: {_clip_compaction_line(turn.content, 5000)}")
        if len(selected) >= 3:
            break
    text = "\n\n".join(selected)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ... [truncated initial request context]"


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


def _compaction_memory_conflicts_with_control_state(memory: str, control_state: str) -> bool:
    """Reject summaries that erase a newer unresolved reviewer state.

    This is deliberately narrow. A summary may mention accepted older work while
    the current step is pending. The unsafe case is a blanket success summary
    that does not also preserve the pending/rework/failure signal from the
    authoritative control state appended below it.
    """
    if not control_state:
        return False
    state = control_state.lower()
    unresolved_markers = (
        "needs_rework",
        "needs_plan_change",
        "needs_requirements_change",
        "needs_analysis_change",
        "pending",
        "timed out",
        "timeout",
        "failed",
        "failure",
        "returned 1",
        "returned 2",
        "mismatch",
        "non-zero",
        "not accepted",
        "rejected",
    )
    if not any(marker in state for marker in unresolved_markers):
        return False

    text = memory.lower()
    acknowledges_unresolved = (
        "needs_rework",
        "needs rework",
        "needs_plan_change",
        "needs plan change",
        "needs_requirements_change",
        "needs requirements change",
        "needs_analysis_change",
        "needs analysis change",
        "pending",
        "not accepted",
        "rejected",
        "failed",
        "failure",
        "mismatch",
        "timed out",
        "timeout",
        "non-zero",
        "unresolved",
        "claim",
        "claimed",
    )
    if any(marker in text for marker in acknowledges_unresolved):
        return False

    success_patterns = (
        r"\ball validation (?:passed|succeeded)\b",
        r"\b(?:everything|all checks|all tests) (?:passed|succeeded|works?)\b",
        r"\b(?:step\s+)?[a-z0-9_.-]+\s+is complete\b",
        r"\b(?:successfully|correctly)\s+(?:implemented|completed|resolved|verified|validated)\b",
        r"\b(?:confirmed|verified|validated)\b.{0,80}\b(?:correct|passed|successful|success|expected)\b",
        r"\b(?:resolved|accepted|complete|completed)\b.{0,80}\b(?:with no|without)\b.{0,40}\b(?:issues|failures|risks)\b",
        r"\bno (?:remaining )?(?:issues|failures|risks|problems)\b",
    )
    return any(re.search(pattern, text) for pattern in success_patterns)


def deterministic_compact(text: str) -> str:
    text = _redact_generated_prompt_turns(text)
    summaries = _deterministic_turn_summaries(text)
    if summaries:
        head = summaries[:12]
        tail_source = summaries[12:] if len(summaries) <= 36 else summaries[-24:]
        tail = tail_source
    else:
        lines = [_clip_compaction_line(line.strip()) for line in text.splitlines() if _keep_compaction_line(line)]
        head = [_clip_compaction_line(line) for line in lines[:24]]
        tail_source = lines[24:] if len(lines) <= 72 else lines[-48:]
        tail = [_clip_compaction_line(line) for line in tail_source]
    parts = [
        "Deterministic fallback compaction was used because model compaction failed.",
        "Older context summarized mechanically:",
        *head,
    ]
    if tail:
        parts.extend(["Recent older context summaries:", *tail])
    return "\n".join(parts)


def _deterministic_turn_summaries(text: str) -> list[str]:
    """Summarize old turns without pasting stale raw JSON/code back to the model.

    Deterministic fallback compaction is intentionally lossy. The authoritative
    control state and pinned workflow state are appended after it, so this block
    should preserve old outcomes and evidence at a high level instead of copying
    rejected implementation payloads or command bodies that small models may
    follow as current instructions.
    """
    summaries: list[str] = []
    pattern = re.compile(r"(?ms)^(?P<role>system|user|assistant): (?P<body>.*?)(?=^(?:system|user|assistant): |\Z)")
    for match in pattern.finditer(text):
        summary = _summarize_turn_for_deterministic_memory(match.group("role"), match.group("body"))
        if summary:
            summaries.append(_clip_compaction_line(summary, 1200))
    return _dedupe_preserving_order(summaries)


def _summarize_turn_for_deterministic_memory(role: str, body: str) -> str:
    stripped = body.strip()
    if not stripped:
        return ""
    if role == "system" and (
        stripped.startswith("Previous compacted-memory block omitted")
        or stripped.startswith("[base system prompt omitted")
    ):
        return ""
    if stripped.startswith("PROJECT DESIGN:"):
        return f"{role}: {_clip_compaction_line(stripped, 900)}"
    if "[generated harness prompt omitted" in stripped:
        first = stripped.splitlines()[0].strip()
        return f"{role}: {first} [generated harness prompt omitted]"
    if stripped.startswith("FEEDBACK_AGENT_RESPONSE:"):
        return _jsonish_turn_summary("Feedback response", stripped)
    if stripped.startswith("TOOL_CALL_VERIFICATION_RESULT:"):
        return _jsonish_turn_summary("Tool-call verification result", stripped)
    if stripped.startswith(("NEXT_IMPLEMENTATION_DIRECTIVE:", "REQUIREMENTS_REWORK_DIRECTIVE:", "PLAN_REWORK_DIRECTIVE:", "ANALYSIS_REWORK_DIRECTIVE:")):
        label = stripped.split(":", 1)[0].replace("_", " ").title()
        return _jsonish_turn_summary(label, stripped)
    if stripped.startswith("IMPLEMENTATION_AGENT_RESPONSE:"):
        return _implementation_response_summary(stripped)
    if stripped.startswith(("WEB_RESEARCH_TOOL_RESULT:", "TOOL_PROGRESS_REVIEW_RESULT:")):
        label = stripped.split(":", 1)[0].replace("_", " ").title()
        return _jsonish_turn_summary(label, stripped)
    if len(stripped) <= 240 and not _looks_like_json_or_code_line(stripped):
        return f"{role}: {stripped}"
    return ""


def _jsonish_turn_summary(label: str, text: str) -> str:
    status = _extract_jsonish_value("status", text)
    decision = _extract_jsonish_value("decision", text)
    needs_rework = _extract_jsonish_value("needs_rework", text)
    summary = _extract_jsonish_value("summary", text)
    fields: list[str] = []
    if status:
        fields.append(f"status={status}")
    if decision and decision != status:
        fields.append(f"decision={decision}")
    if needs_rework:
        fields.append(f"needs_rework={needs_rework}")
    if summary:
        fields.append(f"summary={_clip(summary, 500)}")
    if not fields:
        return f"{label}: present; details omitted from deterministic fallback memory."
    return f"{label}: " + "; ".join(fields)


def _implementation_response_summary(text: str) -> str:
    plan_note = _extract_jsonish_value("plan_note", text)
    resolution = _extract_jsonish_value("resolution_request", text)
    paths = re.findall(r'"path"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    decoded_paths = [_decode_jsonish_string(path) for path in paths[:8]]
    fields: list[str] = []
    if plan_note:
        fields.append(f"plan_note={_clip(plan_note, 500)}")
    if decoded_paths:
        fields.append("files=" + ", ".join(decoded_paths))
    if resolution:
        fields.append(f"resolution_request={resolution}")
    if not fields:
        return "Implementation response: present; raw payload omitted from deterministic fallback memory."
    return "Implementation response: " + "; ".join(fields)


def _looks_like_json_or_code_line(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith(("{", "}", "[", "]", '"')) or stripped.endswith((",", "{", "[")):
        return True
    return any(marker in stripped for marker in ("def ", "class ", "return ", "import ", "python -c", "bash -lc"))


def _decode_jsonish_string(text: str) -> str:
    try:
        import json

        return str(json.loads(f'"{text}"'))
    except Exception:
        return text


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _is_compacted_memory_turn(content: str) -> bool:
    stripped = content.lstrip()
    return (
        stripped.startswith("Compacted durable memory from earlier turns.")
        or stripped.startswith("ACTIVE_CONTEXT_COMPACTED:")
        or "INITIAL_REQUEST_CONTEXT:" in stripped[:1000]
    )


def _initial_request_context_from_memory(content: str) -> str:
    start = re.search(r"(?m)^user: PROJECT DESIGN:", content)
    if not start:
        return ""
    body = content[start.start():]
    stop = re.search(
        r"(?ms)\n\n(?="
        r"(?:system|user|assistant): "
        r"|AUTHORITATIVE_RECENT_CONTROL_STATE:"
        r"|PINNED_WORKFLOW_STATE:"
        r"|COMPACTED_WORKFLOW_MEMORY:"
        r"|Deterministic fallback compaction was used because model compaction failed\."
        r"|Important early context:"
        r"|Recent older context:"
        r"|\*\*(?:Initial User Request|Requirements|Decisions|Assumptions|Open Questions|Plan|Implementation|Validation)\*\*"
        r"|#{1,6}\s+(?:Initial User Request|Requirements|Decisions|Assumptions|Open Questions|Plan|Implementation|Validation)\b"
        r")",
        body,
    )
    if stop:
        body = body[:stop.start()]
    return _clip_compaction_line(body.strip(), 5000)


def _redact_generated_prompt_turns(text: str) -> str:
    """Remove generated prompt contracts from deterministic fallback memory.

    Fallback compaction should preserve the project request, phase names, model
    decisions, and recent evidence. It should not recursively carry full harness
    prompt schemas or earlier compacted-memory blocks into later model turns.
    """

    def redact_compacted(match: re.Match[str]) -> str:
        return (
            "system: Previous compacted-memory block omitted; "
            "fresh initial context, control state, and pinned workflow state are appended separately.\n"
        )

    def redact_prompt(match: re.Match[str]) -> str:
        role = match.group("role")
        marker = match.group("marker")
        body = match.group("body")
        phase = ""
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("{") or line.startswith("["):
                break
            phase = line
            break
        suffix = f" {phase}" if phase else ""
        return f"{role}: {marker}{suffix}\n[generated harness prompt omitted from deterministic compaction]\n"

    text = re.sub(
        r"(?ms)^system: Compacted durable memory from earlier turns\..*?(?=^(?:system|user|assistant): |\Z)",
        redact_compacted,
        text,
    )
    text = re.sub(
        r"(?ms)^(?P<role>user): (?P<marker>IMPLEMENTATION_AGENT_REQUEST:|FEEDBACK_AGENT_REQUEST:)\n(?P<body>.*?)(?=^(?:system|user|assistant): |\Z)",
        redact_prompt,
        text,
    )
    return text


def _keep_compaction_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    skip_prefixes = (
        "initial_request_context:",
        "important early context:",
        "older context summarized mechanically:",
        "recent older context:",
        "recent older context summaries:",
        "deterministic fallback compaction was used because model compaction failed.",
        "compacted_workflow_memory:",
        "pinned_workflow_state:",
        "authoritative_recent_control_state:",
        "system: previous compacted-memory block omitted",
        "system: [base system prompt omitted",
        "[live transcript turn truncated:",
    )
    if any(lowered.startswith(prefix) for prefix in skip_prefixes):
        return False
    return True


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
    last_directive = _last_turn_with_prefixes(
        turns,
        (
            "NEXT_IMPLEMENTATION_DIRECTIVE:",
            "REQUIREMENTS_REWORK_DIRECTIVE:",
            "PLAN_REWORK_DIRECTIVE:",
            "ANALYSIS_REWORK_DIRECTIVE:",
        ),
    )
    last_feedback = _last_matching_turn_with_index(turns, "FEEDBACK_AGENT_RESPONSE:")
    last_final_request = _last_turn_with_prefix_containing_with_index(
        turns,
        "FEEDBACK_AGENT_REQUEST:",
        "FINAL_PROJECT_REVIEW_PHASE",
    )

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
            marker = _control_state_marker_for_turn(directive_source[1]) if directive_source == last_directive else "Last reviewer response"
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


def _last_turn_with_prefixes(turns: list[Turn], prefixes: tuple[str, ...]) -> tuple[int, Turn] | None:
    for index, turn in reversed(list(enumerate(turns))):
        if turn.content.startswith(prefixes):
            return index, turn
    return None


def _last_turn_containing_with_index(turns: list[Turn], needle: str) -> tuple[int, Turn] | None:
    for index, turn in reversed(list(enumerate(turns))):
        if needle in turn.content:
            return index, turn
    return None


def _last_turn_with_prefix_containing_with_index(
    turns: list[Turn],
    prefix: str,
    needle: str,
) -> tuple[int, Turn] | None:
    for index, turn in reversed(list(enumerate(turns))):
        if turn.content.startswith(prefix) and needle in turn.content:
            return index, turn
    return None


def _control_state_marker_for_turn(turn: Turn) -> str:
    content = turn.content
    if content.startswith("REQUIREMENTS_REWORK_DIRECTIVE:"):
        return "Last requirements rework directive"
    if content.startswith("PLAN_REWORK_DIRECTIVE:"):
        return "Last plan rework directive"
    if content.startswith("ANALYSIS_REWORK_DIRECTIVE:"):
        return "Last analysis rework directive"
    return "Last implementation directive"


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
