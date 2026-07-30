"""Finite workflow transport tokens; this module contains no task semantics."""

from __future__ import annotations

import json
from typing import Any


HARNESS_EFFECTIVE_REVIEW_MARKER = "HARNESS_EFFECTIVE_REVIEW:"
HARNESS_RESPONSE_OMISSION_MARKER = "HARNESS_RESPONSE_OMISSION:"
HARNESS_PROTOCOL_ERROR_STATUS = "protocol_error"
VALIDATED_FEEDBACK_DECISION_MARKER = "VALIDATED_FEEDBACK_DECISION:"

FEEDBACK_REPAIR_PHASE_SUFFIXES = ("_JSON_REPAIR", "_MINIMAL_JSON_REPAIR")


COMMAND_RESPONSE_PHASES = frozenset({
    "REQUIREMENTS_REFINEMENT_PHASE",
    "PLAN_REFINEMENT_PHASE",
    "IMPLEMENT_PLAN_STEP_PHASE",
    "FINAL_PROJECT_CORRECTION_PHASE",
})


FILE_RESPONSE_PHASES = frozenset({
    "IMPLEMENT_PLAN_STEP_PHASE",
    "FINAL_PROJECT_CORRECTION_PHASE",
})


PLAN_RESPONSE_PHASES = frozenset({
    "REQUIREMENTS_REFINEMENT_PHASE",
    "PLAN_REFINEMENT_PHASE",
})


CONTROL_PROTOCOL_TURN_PREFIXES = (
    "FEEDBACK_AGENT_RESPONSE:",
    VALIDATED_FEEDBACK_DECISION_MARKER,
    HARNESS_EFFECTIVE_REVIEW_MARKER,
    "TOOL_CALL_VERIFICATION_RESULT:",
    "TOOL_PROGRESS_REVIEW_RESULT:",
    "NEXT_IMPLEMENTATION_DIRECTIVE:",
    "REQUIREMENTS_REWORK_DIRECTIVE:",
    "PLAN_REWORK_DIRECTIVE:",
    "ANALYSIS_REWORK_DIRECTIVE:",
)


PHASE_STATUS_VALUES: dict[str, frozenset[str]] = {
    "PROBLEM_ANALYSIS_REVIEW_PHASE": frozenset({"resolved", "needs_rework", "cannot_resolve"}),
    "REQUIREMENTS_REVIEW_PHASE": frozenset({
        "resolved",
        "needs_rework",
        "needs_requirements_change",
        "cannot_resolve",
        "skipped_with_note",
    }),
    "PLAN_VALIDATION_PHASE": frozenset({
        "resolved",
        "needs_plan_change",
        "needs_requirements_change",
        "cannot_resolve",
    }),
    "STEP_REVIEW_PHASE": frozenset({
        "resolved",
        "needs_rework",
        "cannot_resolve",
        "needs_requirements_change",
        "needs_plan_change",
        "skipped_with_note",
        "resolved_with_compromise",
    }),
    "FINAL_PROJECT_REVIEW_PHASE": frozenset({
        "resolved",
        "needs_rework",
        "cannot_resolve",
        "needs_requirements_change",
        "needs_plan_change",
        "skipped_with_note",
        "resolved_with_compromise",
    }),
    "APPROACH_REVIEW_PHASE": frozenset({
        "resolved",
        "try_another_approach",
        "cannot_resolve",
    }),
    "TOOL_CALL_VERIFICATION_PHASE": frozenset({"approved", "blocked"}),
    "TOOL_PROGRESS_REVIEW_PHASE": frozenset({"continue", "terminate"}),
}


WORKFLOW_REVIEW_PHASES = frozenset({
    "PROBLEM_ANALYSIS_REVIEW_PHASE",
    "REQUIREMENTS_REVIEW_PHASE",
    "PLAN_VALIDATION_PHASE",
    "STEP_REVIEW_PHASE",
    "FINAL_PROJECT_REVIEW_PHASE",
    "APPROACH_REVIEW_PHASE",
})


PHASE_DECISION_VALUES: dict[str, frozenset[str]] = {
    "APPROACH_REVIEW_PHASE": frozenset({"keep_result", "retry_with_new_approach", "stop_unresolved"}),
    "PLAN_VALIDATION_LIFECYCLE_PHASE": frozenset({"valid", "needs_plan_change"}),
    "TOOL_PROGRESS_REVIEW_PHASE": frozenset({"continue", "terminate"}),
}


FEEDBACK_PHASES = frozenset(PHASE_STATUS_VALUES) | frozenset(PHASE_DECISION_VALUES)


REVIEW_STATUSES = frozenset().union(*(
    PHASE_STATUS_VALUES[phase]
    for phase in WORKFLOW_REVIEW_PHASES
))


CONTROL_STATUS_VALUES = (
    frozenset().union(*PHASE_STATUS_VALUES.values())
    | frozenset({HARNESS_PROTOCOL_ERROR_STATUS})
)


def review_directive_text(marker: str, instruction: str, review: dict[str, Any]) -> str:
    """Encode a harness-owned repair directive as one stable JSON envelope."""
    return marker + ":\n" + json.dumps(
        {
            "instruction": instruction.strip(),
            "review": review,
        },
        indent=2,
        ensure_ascii=False,
    )


def protocol_payload_from_turn(content: str) -> dict[str, Any] | None:
    """Parse a protocol turn only when its body is exactly one JSON object."""
    _marker, separator, body = content.partition("\n")
    if not separator:
        return None
    try:
        payload = json.loads(body.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def review_payload_from_protocol_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a directive's nested review or an ordinary review payload."""
    review = payload.get("review")
    return review if isinstance(review, dict) else payload
