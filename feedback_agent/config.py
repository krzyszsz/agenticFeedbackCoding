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
    request_timeout_seconds: int
    retry_attempts: int
    retry_sleep_seconds: int
    request_heartbeat_seconds: int
    preserve_reasoning: bool
    reasoning_budget_tokens: int | None
    send_reasoning_budget: bool


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
    git_diff_max_chars: int = 20000
    transcript_review_max_chars: int = 24000
    max_uncompacted_tokens: int = 24000
    recent_turns_max_tokens: int = 12000


@dataclass(frozen=True)
class LoopConfig:
    max_iterations: int
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
    allow_requirement_dilution: bool
    allow_skip_with_note: bool
    stop_on_cannot_resolve: bool


@dataclass(frozen=True)
class QualityPolicy:
    assume_code_quality_when_unspecified: bool
    require_research_and_structure_step: bool
    deterministic_semantic_scope_checks: bool


@dataclass(frozen=True)
class ReviewPolicy:
    hard_pushback_iterations: int
    compromise_iterations: int
    final_review_iterations: int


@dataclass(frozen=True)
class WebResearchConfig:
    enabled: bool
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
        "temperature": 0.25,
        "request_timeout_seconds": 21_600,
        "retry_attempts": 20,
        "retry_sleep_seconds": 30,
        "request_heartbeat_seconds": 30,
        "preserve_reasoning": True,
        "reasoning_budget_tokens": 4096,
        "send_reasoning_budget": True,
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
        "command_timeout_seconds": 120,
        "max_command_timeout_seconds": 21_600,
        "command_progress_review_interval_seconds": 300,
        "command_progress_review_min_interval_seconds": 30,
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
        "git_diff_max_chars": 12000,
        "transcript_review_max_chars": 12000,
        "max_uncompacted_tokens": 24000,
        "recent_turns_max_tokens": 12000,
    },
    "loop": {
        "max_iterations": 3,
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
        "allow_requirement_dilution": True,
        "allow_skip_with_note": True,
        "stop_on_cannot_resolve": False,
    },
    "quality_policy": {
        "assume_code_quality_when_unspecified": True,
        "require_research_and_structure_step": True,
        "deterministic_semantic_scope_checks": False,
    },
    "review_policy": {
        "hard_pushback_iterations": 3,
        "compromise_iterations": 4,
        "final_review_iterations": 1,
    },
    "web_research": {
        "enabled": False,
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


def _model(data: dict[str, Any], *, base_url_override: str | None = None) -> ModelConfig:
    return ModelConfig(
        name=str(data["name"]),
        base_url=str(base_url_override or data["base_url"]).rstrip("/"),
        api_key=str(data.get("api_key") or "not-needed"),
        model=str(data.get("model") or "local-gguf"),
        context_window=int(data["context_window"]),
        max_tokens=int(data.get("max_tokens", 32768)),
        temperature=float(data.get("temperature", 0.25)),
        request_timeout_seconds=int(data.get("request_timeout_seconds", 21_600)),
        retry_attempts=max(1, int(data.get("retry_attempts", 20))),
        retry_sleep_seconds=max(0, int(data.get("retry_sleep_seconds", 30))),
        request_heartbeat_seconds=max(0, int(data.get("request_heartbeat_seconds", 30))),
        preserve_reasoning=bool(data.get("preserve_reasoning", True)),
        reasoning_budget_tokens=(
            None
            if data.get("reasoning_budget_tokens") is None
            else max(0, int(data.get("reasoning_budget_tokens", 4096)))
        ),
        send_reasoning_budget=bool(data.get("send_reasoning_budget", False)),
    )


def _phase_loop(data: dict[str, Any], key: str, default: int) -> PhaseLoopConfig:
    value = data.get(key, {})
    return PhaseLoopConfig(max_iterations=int(value.get("max_iterations", default)))


def _phases(data: dict[str, Any], loop_data: dict[str, Any]) -> PhaseConfig:
    phase_data = data.get("phases", {})
    old_loop_iterations = int(loop_data.get("max_iterations", 3))
    return PhaseConfig(
        analysis=_phase_loop(phase_data, "analysis", 2),
        requirements_refinement=_phase_loop(phase_data, "requirements_refinement", 2),
        plan_validation=_phase_loop(phase_data, "plan_validation", 2),
        implementation=_phase_loop(phase_data, "implementation", old_loop_iterations),
    )


def _resolution_policy(data: dict[str, Any]) -> ResolutionPolicy:
    policy = data.get("resolution_policy", {})
    return ResolutionPolicy(
        max_same_error_repeats=int(policy.get("max_same_error_repeats", 2)),
        allow_requirement_dilution=bool(policy.get("allow_requirement_dilution", True)),
        allow_skip_with_note=bool(policy.get("allow_skip_with_note", True)),
        stop_on_cannot_resolve=bool(policy.get("stop_on_cannot_resolve", False)),
    )


def _quality_policy(data: dict[str, Any]) -> QualityPolicy:
    policy = data.get("quality_policy", {})
    return QualityPolicy(
        assume_code_quality_when_unspecified=bool(policy.get("assume_code_quality_when_unspecified", True)),
        require_research_and_structure_step=bool(policy.get("require_research_and_structure_step", True)),
        deterministic_semantic_scope_checks=bool(policy.get("deterministic_semantic_scope_checks", False)),
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


def load_config(path: str | Path, repo_root: Path | None = None) -> AgentConfig:
    cfg_path = Path(path).resolve()
    data = _deep_merge(DEFAULT_CONFIG, json.loads(cfg_path.read_text(encoding="utf-8")))
    base = repo_root.resolve() if repo_root else cfg_path.parent

    feedback = data.get("feedback_model")
    runtime_data = data["runtime"]
    workspace_override = os.getenv("AGENT_WORKSPACE")
    workspace = Path(workspace_override or runtime_data["workspace"])
    if not workspace.is_absolute():
        workspace = (base / workspace).resolve()

    loop_data = data.get("loop", {})

    return AgentConfig(
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
            command_timeout_seconds=int(runtime_data.get("command_timeout_seconds", 120)),
            max_command_timeout_seconds=int(runtime_data.get("max_command_timeout_seconds", 21_600)),
            command_progress_review_interval_seconds=max(
                0,
                int(runtime_data.get("command_progress_review_interval_seconds", 300)),
            ),
            command_progress_review_min_interval_seconds=max(
                1,
                int(runtime_data.get("command_progress_review_min_interval_seconds", 30)),
            ),
            print_transcript=bool(runtime_data.get("print_transcript", True)),
            live_turn_max_chars=int(runtime_data.get("live_turn_max_chars", 0)),
            color_transcript=bool(runtime_data.get("color_transcript", True)),
            final_summary=str(runtime_data.get("final_summary", "compact")),
            feedback_response_max_tokens=int(runtime_data.get("feedback_response_max_tokens", 4096)),
        ),
        context_compaction=CompactionConfig(**data["context_compaction"]),
        loop=LoopConfig(
            max_iterations=int(loop_data.get("max_iterations", 3)),
            max_approach_reattempts=max(1, int(loop_data.get("max_approach_reattempts", 5))),
        ),
        phases=_phases(data, loop_data),
        resolution_policy=_resolution_policy(data),
        quality_policy=_quality_policy(data),
        review_policy=_review_policy(data),
        web_research=_web_research(data),
        git_policy=_git_policy(data),
        project_design=ProjectDesign(**data["project_design"]),
    )
