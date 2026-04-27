from __future__ import annotations

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


@dataclass(frozen=True)
class ToolConfig:
    terminal: bool
    web_scraping: bool
    web_interaction: bool


@dataclass(frozen=True)
class RuntimeConfig:
    docker_isolation: bool
    docker_image: str
    workspace: Path
    command_timeout_seconds: int
    max_command_timeout_seconds: int
    print_transcript: bool


@dataclass(frozen=True)
class CompactionConfig:
    enabled: bool
    threshold_ratio: float
    keep_recent_turns: int
    summary_max_tokens: int


@dataclass(frozen=True)
class LoopConfig:
    max_iterations: int
    stop_when_review_clean: bool


@dataclass(frozen=True)
class PhaseLoopConfig:
    max_iterations: int


@dataclass(frozen=True)
class PhaseConfig:
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


def _model(data: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        name=str(data["name"]),
        base_url=str(data["base_url"]).rstrip("/"),
        api_key=str(data.get("api_key") or "not-needed"),
        model=str(data.get("model") or "local-gguf"),
        context_window=int(data["context_window"]),
        max_tokens=int(data.get("max_tokens", 2048)),
        temperature=float(data.get("temperature", 0.25)),
        request_timeout_seconds=int(data.get("request_timeout_seconds", 21_600)),
    )


def _phase_loop(data: dict[str, Any], key: str, default: int) -> PhaseLoopConfig:
    value = data.get(key, {})
    return PhaseLoopConfig(max_iterations=int(value.get("max_iterations", default)))


def _phases(data: dict[str, Any], loop_data: dict[str, Any]) -> PhaseConfig:
    phase_data = data.get("phases", {})
    old_loop_iterations = int(loop_data.get("max_iterations", 3))
    return PhaseConfig(
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
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = repo_root or cfg_path.parents[2]

    feedback = data.get("feedback_model")
    runtime_data = data["runtime"]
    workspace_override = os.getenv("AGENT_WORKSPACE")
    workspace = Path(workspace_override or runtime_data["workspace"])
    if not workspace.is_absolute():
        workspace = (base / workspace).resolve()

    loop_data = data.get("loop", {})

    return AgentConfig(
        implementation_model=_model(data["implementation_model"]),
        feedback_model=_model(feedback) if feedback else None,
        mcp_tools=ToolConfig(**data["mcp_tools"]),
        runtime=RuntimeConfig(
            docker_isolation=bool(runtime_data.get("docker_isolation", True)),
            docker_image=str(runtime_data.get("docker_image", "agentic-feedback-coding:local")),
            workspace=workspace,
            command_timeout_seconds=int(runtime_data.get("command_timeout_seconds", 120)),
            max_command_timeout_seconds=int(runtime_data.get("max_command_timeout_seconds", 21_600)),
            print_transcript=bool(runtime_data.get("print_transcript", True)),
        ),
        context_compaction=CompactionConfig(**data["context_compaction"]),
        loop=LoopConfig(
            max_iterations=int(loop_data.get("max_iterations", 3)),
            stop_when_review_clean=bool(loop_data.get("stop_when_review_clean", True)),
        ),
        phases=_phases(data, loop_data),
        resolution_policy=_resolution_policy(data),
        quality_policy=_quality_policy(data),
        review_policy=_review_policy(data),
        web_research=_web_research(data),
        git_policy=_git_policy(data),
        project_design=ProjectDesign(**data["project_design"]),
    )
