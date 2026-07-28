from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    context_window: int
    max_tokens: int
    temperature: float
    top_p: float | None
    top_k: int | None
    request_timeout_seconds: int
    retry_attempts: int
    retry_sleep_seconds: int
    request_heartbeat_seconds: int
    preserve_reasoning: bool
    reasoning_budget_tokens: int | None
    critical_reasoning_budget_tokens: int | None
    send_reasoning_budget: bool
    request_json_object: bool


@dataclass(frozen=True)
class ToolConfig:
    terminal: bool
    web_scraping: bool
    web_interaction: bool


@dataclass(frozen=True)
class RuntimeConfig:
    docker_isolation: bool
    docker_image: str
    docker_user: str
    workspace: Path
    plan_file: str
    requirements_file: str
    research_file: str
    command_timeout_seconds: int
    max_command_timeout_seconds: int
    command_progress_review_interval_seconds: int
    command_progress_review_min_interval_seconds: int
    command_progress_review_max_interval_seconds: int
    command_progress_review_request_timeout_seconds: int
    print_transcript: bool
    live_turn_max_chars: int
    color_transcript: bool
    final_summary: str
    feedback_response_max_tokens: int


@dataclass(frozen=True)
class CompactionConfig:
    enabled: bool
    threshold_ratio: float
    keep_recent_turns: int
    summary_max_tokens: int
    tool_output_max_chars: int = 4000
    workspace_file_max_bytes: int = 20000
    workspace_snapshot_max_files: int = 1000
    workspace_snapshot_max_chars: int = 2_000_000
    git_diff_max_chars: int = 20000
    transcript_review_max_chars: int = 24000
    max_uncompacted_tokens: int = 24000
    recent_turns_max_tokens: int = 12000
    model_summary_min_new_tokens: int = 2048


@dataclass(frozen=True)
class LoopConfig:
    max_approach_reattempts: int


@dataclass(frozen=True)
class PhaseLoopConfig:
    max_iterations: int


@dataclass(frozen=True)
class PhaseConfig:
    analysis: PhaseLoopConfig
    requirements_refinement: PhaseLoopConfig
    plan_validation: PhaseLoopConfig
    implementation: PhaseLoopConfig


@dataclass(frozen=True)
class ResolutionPolicy:
    max_same_error_repeats: int
    allow_skip_with_note: bool
    stop_on_cannot_resolve: bool


@dataclass(frozen=True)
class QualityPolicy:
    assume_code_quality_when_unspecified: bool


@dataclass(frozen=True)
class ReviewPolicy:
    hard_pushback_iterations: int
    compromise_iterations: int
    final_review_iterations: int


@dataclass(frozen=True)
class WebResearchConfig:
    enabled: bool
    allow_private_network: bool
    max_search_results: int
    max_pages: int
    timeout_seconds: int
    max_page_bytes: int
    excerpt_chars: int
    user_agent: str


@dataclass(frozen=True)
class GitPolicy:
    enabled: bool
    commit_completed_steps: bool
    require_step_diff: bool
    leave_final_changes_uncommitted: bool
    final_reset_mode: str
    commit_user_name: str
    commit_user_email: str


@dataclass(frozen=True)
class ProjectDesign:
    title: str
    prompt: str


@dataclass(frozen=True)
class AgentConfig:
    implementation_model: ModelConfig
    feedback_model: ModelConfig | None
    mcp_tools: ToolConfig
    runtime: RuntimeConfig
    context_compaction: CompactionConfig
    loop: LoopConfig
    phases: PhaseConfig
    resolution_policy: ResolutionPolicy
    quality_policy: QualityPolicy
    review_policy: ReviewPolicy
    web_research: WebResearchConfig
    git_policy: GitPolicy
    project_design: ProjectDesign


DEFAULT_CONFIG: dict[str, Any] = {
    "implementation_model": {
        "name": "gemma4-26b-a4b-qat-mtp",
        "base_url": "http://127.0.0.1:8161/v1",
        "api_key": "not-needed",
        "model": "local-gguf",
        "context_window": 131072,
        "max_tokens": 32768,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "request_timeout_seconds": 21_600,
        "retry_attempts": 20,
        "retry_sleep_seconds": 30,
        "request_heartbeat_seconds": 30,
        "preserve_reasoning": True,
        "reasoning_budget_tokens": 4096,
        "critical_reasoning_budget_tokens": None,
        "send_reasoning_budget": True,
        "request_json_object": True,
    },
    "feedback_model": None,
    "mcp_tools": {
        "terminal": True,
        "web_scraping": False,
        "web_interaction": True,
    },
    "runtime": {
        "docker_isolation": True,
        "docker_image": "agentic-feedback-coding:local",
        "docker_user": "host",
        "workspace": "workspaces/my-agentic-project",
        "plan_file": "PLAN.md",
        "requirements_file": "REQUIREMENTS.md",
        "research_file": "RESEARCH.md",
        "command_timeout_seconds": 0,
        "max_command_timeout_seconds": 21_600,
        "command_progress_review_interval_seconds": 300,
        "command_progress_review_min_interval_seconds": 30,
        "command_progress_review_max_interval_seconds": 3600,
        "command_progress_review_request_timeout_seconds": 120,
        "print_transcript": True,
        "live_turn_max_chars": 0,
        "color_transcript": True,
        "final_summary": "compact",
        "feedback_response_max_tokens": 2048,
    },
    "context_compaction": {
        "enabled": True,
        "threshold_ratio": 0.25,
        "keep_recent_turns": 8,
        "summary_max_tokens": 1024,
        "tool_output_max_chars": 4000,
        "workspace_file_max_bytes": 12000,
        "workspace_snapshot_max_files": 1000,
        "workspace_snapshot_max_chars": 2_000_000,
        "git_diff_max_chars": 12000,
        "transcript_review_max_chars": 12000,
        "max_uncompacted_tokens": 24000,
        "recent_turns_max_tokens": 12000,
        "model_summary_min_new_tokens": 2048,
    },
    "loop": {
        "max_approach_reattempts": 5,
    },
    "phases": {
        "analysis": {"max_iterations": 2},
        "requirements_refinement": {"max_iterations": 2},
        "plan_validation": {"max_iterations": 2},
        "implementation": {"max_iterations": 7},
    },
    "resolution_policy": {
        "max_same_error_repeats": 2,
        "allow_skip_with_note": True,
        "stop_on_cannot_resolve": False,
    },
    "quality_policy": {
        "assume_code_quality_when_unspecified": True,
    },
    "review_policy": {
        "hard_pushback_iterations": 3,
        "compromise_iterations": 4,
        "final_review_iterations": 1,
    },
    "web_research": {
        "enabled": False,
        "allow_private_network": False,
        "max_search_results": 3,
        "max_pages": 3,
        "timeout_seconds": 15,
        "max_page_bytes": 1_000_000,
        "excerpt_chars": 3000,
        "user_agent": "agenticFeedbackCoding/0.1 (+local research harness)",
    },
    "git_policy": {
        "enabled": True,
        "commit_completed_steps": True,
        "require_step_diff": True,
        "leave_final_changes_uncommitted": False,
        "final_reset_mode": "soft",
        "commit_user_name": "agenticFeedbackCoding",
        "commit_user_email": "agentic-feedback@example.local",
    },
    "project_design": {
        "title": "Agentic project",
        "prompt": "Build a small, well-tested project and document how to run it.",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def derive_critical_reasoning_budget(
    reasoning_budget_tokens: int | None,
    max_tokens: int,
    configured_critical_budget_tokens: int | None = None,
) -> int | None:
    """Return an explicit critical budget or a bounded four-times default.

    The response ceiling includes reasoning and final output for the local
    models used here. Automatic escalation therefore leaves a proportional
    answer reserve when a small custom ``max_tokens`` value cannot fit the full
    four-times budget. Normal publication settings fit the full 4x default.
    """
    if configured_critical_budget_tokens is not None:
        return int(configured_critical_budget_tokens)
    if reasoning_budget_tokens is None:
        return None
    normal = int(reasoning_budget_tokens)
    if normal <= 0:
        return normal
    answer_reserve = min(2048, max(1, int(max_tokens) // 4))
    largest_usable_budget = max(0, int(max_tokens) - answer_reserve)
    return max(normal, min(normal * 4, largest_usable_budget))


def _model(data: dict[str, Any], *, base_url_override: str | None = None) -> ModelConfig:
    max_tokens = int(data.get("max_tokens", 32768))
    reasoning_budget = (
        None
        if data.get("reasoning_budget_tokens") is None
        else int(data.get("reasoning_budget_tokens", 4096))
    )
    critical_reasoning_budget = derive_critical_reasoning_budget(
        reasoning_budget,
        max_tokens,
        (
            None
            if data.get("critical_reasoning_budget_tokens") is None
            else int(data["critical_reasoning_budget_tokens"])
        ),
    )
    return ModelConfig(
        name=str(data["name"]),
        base_url=str(base_url_override or data["base_url"]).rstrip("/"),
        api_key=str(data.get("api_key") or "not-needed"),
        model=str(data.get("model") or "local-gguf"),
        context_window=int(data["context_window"]),
        max_tokens=max_tokens,
        temperature=float(data.get("temperature", 1.0)),
        top_p=None if data.get("top_p") is None else float(data["top_p"]),
        top_k=None if data.get("top_k") is None else int(data["top_k"]),
        request_timeout_seconds=int(data.get("request_timeout_seconds", 21_600)),
        retry_attempts=int(data.get("retry_attempts", 20)),
        retry_sleep_seconds=int(data.get("retry_sleep_seconds", 30)),
        request_heartbeat_seconds=int(data.get("request_heartbeat_seconds", 30)),
        preserve_reasoning=bool(data.get("preserve_reasoning", True)),
        reasoning_budget_tokens=reasoning_budget,
        critical_reasoning_budget_tokens=critical_reasoning_budget,
        send_reasoning_budget=bool(data.get("send_reasoning_budget", False)),
        request_json_object=bool(data.get("request_json_object", True)),
    )


def _phase_loop(data: dict[str, Any], key: str, default: int) -> PhaseLoopConfig:
    value = data.get(key, {})
    return PhaseLoopConfig(max_iterations=int(value.get("max_iterations", default)))


def _phases(data: dict[str, Any]) -> PhaseConfig:
    phase_data = data.get("phases", {})
    return PhaseConfig(
        analysis=_phase_loop(phase_data, "analysis", 2),
        requirements_refinement=_phase_loop(phase_data, "requirements_refinement", 2),
        plan_validation=_phase_loop(phase_data, "plan_validation", 2),
        implementation=_phase_loop(phase_data, "implementation", 7),
    )


def _resolution_policy(data: dict[str, Any]) -> ResolutionPolicy:
    policy = data.get("resolution_policy", {})
    return ResolutionPolicy(
        max_same_error_repeats=int(policy.get("max_same_error_repeats", 2)),
        allow_skip_with_note=bool(policy.get("allow_skip_with_note", True)),
        stop_on_cannot_resolve=bool(policy.get("stop_on_cannot_resolve", False)),
    )


def _quality_policy(data: dict[str, Any]) -> QualityPolicy:
    policy = data.get("quality_policy", {})
    return QualityPolicy(
        assume_code_quality_when_unspecified=bool(policy.get("assume_code_quality_when_unspecified", True)),
    )


def _review_policy(data: dict[str, Any]) -> ReviewPolicy:
    policy = data.get("review_policy", {})
    return ReviewPolicy(
        hard_pushback_iterations=int(policy.get("hard_pushback_iterations", 3)),
        compromise_iterations=int(policy.get("compromise_iterations", 4)),
        final_review_iterations=int(policy.get("final_review_iterations", 1)),
    )


def _web_research(data: dict[str, Any]) -> WebResearchConfig:
    policy = data.get("web_research", {})
    return WebResearchConfig(
        enabled=bool(policy.get("enabled", True)),
        allow_private_network=bool(policy.get("allow_private_network", False)),
        max_search_results=int(policy.get("max_search_results", 3)),
        max_pages=int(policy.get("max_pages", 3)),
        timeout_seconds=int(policy.get("timeout_seconds", 15)),
        max_page_bytes=int(policy.get("max_page_bytes", 1_000_000)),
        excerpt_chars=int(policy.get("excerpt_chars", 3000)),
        user_agent=str(policy.get("user_agent", "agenticFeedbackCoding/0.1 (+local research harness)")),
    )


def _git_policy(data: dict[str, Any]) -> GitPolicy:
    policy = data.get("git_policy", {})
    return GitPolicy(
        enabled=bool(policy.get("enabled", True)),
        commit_completed_steps=bool(policy.get("commit_completed_steps", True)),
        require_step_diff=bool(policy.get("require_step_diff", True)),
        leave_final_changes_uncommitted=bool(policy.get("leave_final_changes_uncommitted", False)),
        final_reset_mode=str(policy.get("final_reset_mode", "soft")),
        commit_user_name=str(policy.get("commit_user_name", "agenticFeedbackCoding")),
        commit_user_email=str(policy.get("commit_user_email", "agentic-feedback@example.local")),
    )


def _validate_workflow_filename(name: str, field: str) -> None:
    path = Path(name)
    if not name.strip() or path.is_absolute() or len(path.parts) != 1 or path.name in {".", ".."}:
        raise ValueError(f"{field} must be a root-level relative filename, got {name!r}")
    if path.name in {".git", ".agent_state"}:
        raise ValueError(f"{field} must not use reserved control-state name {name!r}")


def _raw_config_key_errors(data: Any, schema: Any, *, path: str = "") -> list[str]:
    """Report unknown keys and malformed sections before constructing dataclasses."""
    if not isinstance(schema, dict):
        return []
    if not isinstance(data, dict):
        return [f"{path or 'config'} must be an object"]
    errors: list[str] = []
    for key, value in data.items():
        field = f"{path}.{key}" if path else key
        if key not in schema:
            errors.append(f"unknown configuration field: {field}")
            continue
        child_schema = schema[key]
        if key == "feedback_model" and value is not None:
            child_schema = DEFAULT_CONFIG["implementation_model"]
        if isinstance(child_schema, dict):
            errors.extend(_raw_config_key_errors(value, child_schema, path=field))
    return errors


_BOOLEAN_CONFIG_PATHS = (
    "implementation_model.preserve_reasoning",
    "implementation_model.send_reasoning_budget",
    "implementation_model.request_json_object",
    "mcp_tools.terminal",
    "mcp_tools.web_scraping",
    "mcp_tools.web_interaction",
    "runtime.docker_isolation",
    "runtime.print_transcript",
    "runtime.color_transcript",
    "context_compaction.enabled",
    "resolution_policy.allow_skip_with_note",
    "resolution_policy.stop_on_cannot_resolve",
    "quality_policy.assume_code_quality_when_unspecified",
    "web_research.enabled",
    "web_research.allow_private_network",
    "git_policy.enabled",
    "git_policy.commit_completed_steps",
    "git_policy.require_step_diff",
    "git_policy.leave_final_changes_uncommitted",
)


def _raw_boolean_type_errors(data: dict[str, Any]) -> list[str]:
    """Reject truthy strings where a safety or capability switch needs JSON bool."""
    errors: list[str] = []
    paths = list(_BOOLEAN_CONFIG_PATHS)
    if data.get("feedback_model") is not None:
        paths.extend((
            "feedback_model.preserve_reasoning",
            "feedback_model.send_reasoning_budget",
            "feedback_model.request_json_object",
        ))
    for path in paths:
        value: Any = data
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None and not isinstance(value, bool):
            errors.append(f"{path} must be a JSON boolean")
    return errors


def validate_config(config: AgentConfig) -> None:
    """Validate a fully constructed config, including runtime overrides."""
    errors: list[str] = []

    for label, model in (
        ("implementation_model", config.implementation_model),
        ("feedback_model", config.feedback_model),
    ):
        if model is None:
            continue
        if model.context_window <= 0:
            errors.append(f"{label}.context_window must be greater than zero")
        if model.max_tokens <= 0:
            errors.append(f"{label}.max_tokens must be greater than zero")
        if model.context_window > 0 and model.max_tokens >= model.context_window:
            errors.append(f"{label}.max_tokens must be smaller than context_window")
        if model.temperature < 0:
            errors.append(f"{label}.temperature must be zero or greater")
        if model.top_p is not None and not 0 < model.top_p <= 1:
            errors.append(f"{label}.top_p must be greater than zero and at most one")
        if model.top_k is not None and model.top_k <= 0:
            errors.append(f"{label}.top_k must be greater than zero")
        if model.request_timeout_seconds <= 0:
            errors.append(f"{label}.request_timeout_seconds must be greater than zero")
        if model.retry_attempts <= 0:
            errors.append(f"{label}.retry_attempts must be greater than zero")
        if model.retry_sleep_seconds < 0:
            errors.append(f"{label}.retry_sleep_seconds must be zero or greater")
        if model.request_heartbeat_seconds < 0:
            errors.append(f"{label}.request_heartbeat_seconds must be zero or greater")
        if model.reasoning_budget_tokens is not None and model.reasoning_budget_tokens < 0:
            errors.append(f"{label}.reasoning_budget_tokens must be zero or greater")
        if (
            model.reasoning_budget_tokens is not None
            and model.max_tokens > 0
            and model.reasoning_budget_tokens >= model.max_tokens
        ):
            errors.append(f"{label}.reasoning_budget_tokens must be smaller than max_tokens")
        if model.critical_reasoning_budget_tokens is not None and model.critical_reasoning_budget_tokens < 0:
            errors.append(f"{label}.critical_reasoning_budget_tokens must be zero or greater")
        if (
            model.critical_reasoning_budget_tokens is not None
            and model.max_tokens > 0
            and model.critical_reasoning_budget_tokens >= model.max_tokens
        ):
            errors.append(f"{label}.critical_reasoning_budget_tokens must be smaller than max_tokens")
        if (
            model.reasoning_budget_tokens is not None
            and model.critical_reasoning_budget_tokens is not None
            and model.critical_reasoning_budget_tokens < model.reasoning_budget_tokens
        ):
            errors.append(
                f"{label}.critical_reasoning_budget_tokens must be at least reasoning_budget_tokens"
            )

    runtime = config.runtime
    try:
        resolved_workspace = runtime.workspace.resolve(strict=False)
    except OSError as exc:
        errors.append(f"runtime.workspace could not be resolved: {exc}")
        resolved_workspace = runtime.workspace
    if resolved_workspace == Path(resolved_workspace.anchor):
        errors.append("runtime.workspace must not be the filesystem root")
    for field in ("plan_file", "requirements_file", "research_file"):
        try:
            _validate_workflow_filename(str(getattr(runtime, field)), f"runtime.{field}")
        except ValueError as exc:
            errors.append(str(exc))
    workflow_names = [runtime.plan_file, runtime.requirements_file, runtime.research_file]
    if len(set(workflow_names)) != len(workflow_names):
        errors.append("runtime plan, requirements, and research filenames must be distinct")
    if runtime.command_timeout_seconds < 0:
        errors.append("runtime.command_timeout_seconds must be zero or greater")
    if runtime.max_command_timeout_seconds < 0:
        errors.append("runtime.max_command_timeout_seconds must be zero or greater")
    if runtime.command_progress_review_interval_seconds < 0:
        errors.append("runtime.command_progress_review_interval_seconds must be zero or greater")
    if runtime.command_progress_review_min_interval_seconds <= 0:
        errors.append("runtime.command_progress_review_min_interval_seconds must be greater than zero")
    if runtime.command_progress_review_max_interval_seconds < 0:
        errors.append("runtime.command_progress_review_max_interval_seconds must be zero or greater")
    if runtime.command_progress_review_request_timeout_seconds <= 0:
        errors.append("runtime.command_progress_review_request_timeout_seconds must be greater than zero")
    if (
        runtime.command_progress_review_max_interval_seconds > 0
        and runtime.command_progress_review_max_interval_seconds
        < runtime.command_progress_review_min_interval_seconds
    ):
        errors.append(
            "runtime.command_progress_review_max_interval_seconds must be zero or at least "
            "command_progress_review_min_interval_seconds"
        )
    if runtime.live_turn_max_chars < 0:
        errors.append("runtime.live_turn_max_chars must be zero or greater")
    if runtime.feedback_response_max_tokens < 0:
        errors.append("runtime.feedback_response_max_tokens must be zero or greater")
    if runtime.final_summary not in {"compact", "full", "none"}:
        errors.append("runtime.final_summary must be one of: compact, full, none")

    compaction = config.context_compaction
    if not 0 < compaction.threshold_ratio <= 1:
        errors.append("context_compaction.threshold_ratio must be greater than zero and at most one")
    if compaction.keep_recent_turns < 0:
        errors.append("context_compaction.keep_recent_turns must be zero or greater")
    for field in (
        "summary_max_tokens",
        "tool_output_max_chars",
        "workspace_file_max_bytes",
        "workspace_snapshot_max_files",
        "workspace_snapshot_max_chars",
        "git_diff_max_chars",
        "transcript_review_max_chars",
        "recent_turns_max_tokens",
    ):
        if int(getattr(compaction, field)) <= 0:
            errors.append(f"context_compaction.{field} must be greater than zero")
    if compaction.transcript_review_max_chars < 512:
        errors.append("context_compaction.transcript_review_max_chars must be at least 512")
    if compaction.max_uncompacted_tokens < 0:
        errors.append("context_compaction.max_uncompacted_tokens must be zero or greater")
    if compaction.model_summary_min_new_tokens < 0:
        errors.append("context_compaction.model_summary_min_new_tokens must be zero or greater")
    if compaction.workspace_snapshot_max_chars < compaction.workspace_file_max_bytes:
        errors.append(
            "context_compaction.workspace_snapshot_max_chars must be at least workspace_file_max_bytes"
        )

    if config.loop.max_approach_reattempts <= 0:
        errors.append("loop.max_approach_reattempts must be greater than zero")
    for field in ("analysis", "requirements_refinement", "plan_validation", "implementation"):
        if getattr(config.phases, field).max_iterations <= 0:
            errors.append(f"phases.{field}.max_iterations must be greater than zero")
    if config.resolution_policy.max_same_error_repeats <= 0:
        errors.append("resolution_policy.max_same_error_repeats must be greater than zero")
    if config.review_policy.hard_pushback_iterations < 0:
        errors.append("review_policy.hard_pushback_iterations must be zero or greater")
    if config.review_policy.compromise_iterations < 0:
        errors.append("review_policy.compromise_iterations must be zero or greater")
    if config.review_policy.hard_pushback_iterations + config.review_policy.compromise_iterations <= 0:
        errors.append("review policy must allocate at least one step-review iteration")
    if config.review_policy.final_review_iterations < 0:
        errors.append("review_policy.final_review_iterations must be zero or greater")

    research = config.web_research
    for field in ("max_search_results", "max_pages", "timeout_seconds", "max_page_bytes", "excerpt_chars"):
        if int(getattr(research, field)) <= 0:
            errors.append(f"web_research.{field} must be greater than zero")
    if config.git_policy.final_reset_mode not in {"soft", "mixed"}:
        errors.append("git_policy.final_reset_mode must be one of: soft, mixed")
    if not config.project_design.prompt.strip():
        errors.append("project_design.prompt must not be empty")

    if errors:
        raise ValueError("Invalid harness configuration:\n- " + "\n- ".join(errors))


def load_config(path: str | Path, repo_root: Path | None = None) -> AgentConfig:
    cfg_path = Path(path).resolve()
    raw_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError("Invalid harness configuration:\n- config must be an object")
    data = _deep_merge(DEFAULT_CONFIG, raw_data)
    raw_errors = [
        *_raw_config_key_errors(data, DEFAULT_CONFIG),
        *_raw_boolean_type_errors(data),
    ]
    if raw_errors:
        raise ValueError("Invalid harness configuration:\n- " + "\n- ".join(raw_errors))
    base = repo_root.resolve() if repo_root else cfg_path.parent

    feedback = data.get("feedback_model")
    if feedback:
        feedback = _deep_merge(data["implementation_model"], feedback)
    runtime_data = data["runtime"]
    workspace_override = os.getenv("AGENT_WORKSPACE")
    workspace = Path(workspace_override or runtime_data["workspace"])
    workspace = (workspace if workspace.is_absolute() else base / workspace).resolve(strict=False)

    config = AgentConfig(
        implementation_model=_model(
            data["implementation_model"],
            base_url_override=os.getenv("AGENT_IMPLEMENTATION_BASE_URL"),
        ),
        feedback_model=_model(
            feedback,
            base_url_override=os.getenv("AGENT_FEEDBACK_BASE_URL"),
        ) if feedback else None,
        mcp_tools=ToolConfig(**data["mcp_tools"]),
        runtime=RuntimeConfig(
            docker_isolation=bool(runtime_data.get("docker_isolation", True)),
            docker_image=str(runtime_data.get("docker_image", "agentic-feedback-coding:local")),
            docker_user=str(runtime_data.get("docker_user", "host")),
            workspace=workspace,
            plan_file=str(runtime_data.get("plan_file", "PLAN.md")),
            requirements_file=str(runtime_data.get("requirements_file", "REQUIREMENTS.md")),
            research_file=str(runtime_data.get("research_file", "RESEARCH.md")),
            command_timeout_seconds=int(runtime_data.get("command_timeout_seconds", 0)),
            max_command_timeout_seconds=int(runtime_data.get("max_command_timeout_seconds", 21_600)),
            command_progress_review_interval_seconds=int(
                runtime_data.get("command_progress_review_interval_seconds", 300)
            ),
            command_progress_review_min_interval_seconds=int(
                runtime_data.get("command_progress_review_min_interval_seconds", 30)
            ),
            command_progress_review_max_interval_seconds=int(
                runtime_data.get("command_progress_review_max_interval_seconds", 3600)
            ),
            command_progress_review_request_timeout_seconds=int(
                runtime_data.get("command_progress_review_request_timeout_seconds", 120)
            ),
            print_transcript=bool(runtime_data.get("print_transcript", True)),
            live_turn_max_chars=int(runtime_data.get("live_turn_max_chars", 0)),
            color_transcript=bool(runtime_data.get("color_transcript", True)),
            final_summary=str(runtime_data.get("final_summary", "compact")),
            feedback_response_max_tokens=int(runtime_data.get("feedback_response_max_tokens", 2048)),
        ),
        context_compaction=CompactionConfig(**data["context_compaction"]),
        loop=LoopConfig(
            max_approach_reattempts=int(data.get("loop", {}).get("max_approach_reattempts", 5)),
        ),
        phases=_phases(data),
        resolution_policy=_resolution_policy(data),
        quality_policy=_quality_policy(data),
        review_policy=_review_policy(data),
        web_research=_web_research(data),
        git_policy=_git_policy(data),
        project_design=ProjectDesign(**data["project_design"]),
    )
    validate_config(config)
    return config
