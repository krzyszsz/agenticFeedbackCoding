from __future__ import annotations

import json
from typing import Any

from .compaction import maybe_compact
from .config import AgentConfig
from .conversation import Conversation
from .git_tools import commit_all, ensure_git_repo, git_evidence, reset_to_ref
from .llm import OpenAICompatClient, MockClient
from .web_research import compact_research_for_prompt, research_to_markdown, run_web_research
from .workspace import (
    append_plan_note,
    collect_workspace_files,
    ensure_plan,
    extract_json_object,
    normalize_plan_steps,
    run_commands,
    write_files,
    write_plan_doc,
    write_requirements_doc,
)


REQUIREMENTS_CONTRACT = """
Return strict JSON only:
{
  "project_summary": "one paragraph",
  "refined_requirements": ["clear requirement"],
  "assumptions": ["explicit assumption or gap resolution"],
  "open_questions": [{"question": "gap", "resolution_strategy": "ask|assume|dilute|skip", "decision": "chosen resolution"}],
  "planning_confirmation": {
    "is_feasible": true,
    "is_clear": true,
    "is_verifiable": true,
    "verification_strategy": "how the plan will be checked step by step",
    "remaining_risks": ["risk or limitation"]
  },
  "plan": [
    {
      "id": "S1",
      "title": "short task title",
      "description": "what this task changes",
      "depends_on": [],
      "acceptance_criteria": ["verifiable criterion"],
      "validation_commands": [["python", "-m", "unittest", "-v"]]
    }
  ]
}
The plan must be ordered, distinct, and executable one step at a time.
Commands may be argv lists or {"cmd": ["python", "script.py"], "timeout_seconds": 7200}
when one specific tool call legitimately needs longer than the default timeout.
"""


IMPLEMENTATION_CONTRACT = """
Return strict JSON only:
{
  "plan_note": "short note for PLAN.md",
  "files": [{"path": "relative/path", "content": "complete file content"}],
  "commands": [["python", "-m", "unittest", "-v"]],
  "test_evidence": ["short description of command/report/screenshot evidence produced"],
  "resolution_request": "none|needs_requirements_change|needs_plan_change|cannot_resolve"
}
Only write paths inside the project workspace. Prefer small validation commands that finish quickly.
Commands may be argv lists or {"cmd": ["python", "script.py"], "timeout_seconds": 7200}
when one specific tool call legitimately needs longer than the default timeout.
"""


REVIEW_STATUSES = {
    "resolved",
    "needs_rework",
    "cannot_resolve",
    "needs_requirements_change",
    "needs_plan_change",
    "skipped_with_note",
    "resolved_with_compromise",
}


FEEDBACK_SYSTEM_PROMPT = """
You are the feedback/review agent in a two-agent development loop.
Read the full transcript, including implementation attempts, prior feedback,
requirements decisions, plan updates, command results, screenshots/reports when
listed, git status/diffs, and unresolved risks. Always challenge the work.
Always inspect test evidence before accepting a claim. Always inspect the git
diff for the current step when git evidence is present. If the implementation
made no meaningful workspace changes for a plan step, explicitly request the
missing implementation work instead of accepting statements. Phrase feedback as
clear requests: "Please change X", "Please provide evidence Y", "Please rerun Z".
Be strict in hard-pushback mode and only compromise in compromise mode when
bounded retries are more valuable than perfect adherence.
Return strict JSON only.
"""


class FeedbackLoopAgent:
    """Orchestrates a phased two-agent workflow over one durable transcript.

    The implementation and feedback clients may point at the same local model or
    two different OpenAI-compatible endpoints. The orchestration layer owns the
    phase boundaries, file writes, validation commands, retry limits, and the
    transcript discipline that keeps long sessions coherent.
    """

    def __init__(self, config: AgentConfig, *, mock: bool = False):
        self.config = config
        self.workspace = config.runtime.workspace
        self.state_dir = self.workspace / ".agent_state"
        self.conversation = Conversation(
            self.state_dir / "conversation.jsonl",
            echo=config.runtime.print_transcript,
        )
        if mock:
            client = MockClient()
            self.impl_client = client
            self.feedback_client = client
        else:
            self.impl_client = OpenAICompatClient(config.implementation_model)
            self.feedback_client = OpenAICompatClient(config.feedback_model or config.implementation_model)
        self.requirements: dict[str, Any] = {}
        self.plan_steps: list[dict[str, Any]] = []
        self.plan_notes: list[str] = []
        self.web_research_result: dict[str, Any] = {
            "status": "not_run",
            "requested": False,
            "targets": [],
        }
        self.git_baseline_ref = ""

    def initialize(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        ensure_plan(self.workspace)
        if self.config.git_policy.enabled:
            ensure_git_repo(
                self.workspace,
                user_name=self.config.git_policy.commit_user_name,
                user_email=self.config.git_policy.commit_user_email,
            )
        if not self.conversation.turns:
            self.conversation.append(
                "system",
                (
                    "You are an agentic coding/workflow model. Work in explicit phases: "
                    "requirements refinement, plan validation, then one implementation feedback loop per plan step. "
                    "Maintain PLAN.md and REQUIREMENTS.md. Keep all work inside the project workspace. "
                    "The workspace is a git repository when git_policy is enabled; accepted plan steps are "
                    "committed only after feedback review agrees they are complete. "
                    "This transcript is durable chat memory: IMPLEMENTATION_AGENT_REQUEST/RESPONSE and "
                    "FEEDBACK_AGENT_REQUEST/RESPONSE blocks are cumulative context, not isolated prompts."
                ),
            )
            self.conversation.append(
                "user",
                f"PROJECT DESIGN: {self.config.project_design.title}\n\n{self.config.project_design.prompt}",
            )

    def _implementation_chat(self, prompt: str) -> str:
        """Append a request, call the implementation model, then persist its reply.

        Keeping the request itself in the transcript matters. Without it, later
        turns see model answers but not the exact task/review that caused them,
        which makes long local-model sessions drift quickly.
        """
        maybe_compact(
            self.conversation,
            self.config,
            self.impl_client,
            context_window=self.config.implementation_model.context_window,
        )
        self.conversation.append("user", "IMPLEMENTATION_AGENT_REQUEST:\n" + prompt)
        raw = self.impl_client.chat(self.conversation.messages())
        self.conversation.append("assistant", "IMPLEMENTATION_AGENT_RESPONSE:\n" + raw)
        return raw

    def _feedback_chat(self, prompt: str, *, temperature: float = 0.1) -> str:
        """Run the feedback model against the same durable transcript.

        Feedback replies are stored as user-visible transcript blocks so the
        implementation model treats them as external critique on the next turn.
        The feedback model still receives the entire history, including its own
        previous reviews, which gives it continuity across loops.
        """
        feedback_cfg = self.config.feedback_model or self.config.implementation_model
        maybe_compact(
            self.conversation,
            self.config,
            self.feedback_client,
            context_window=feedback_cfg.context_window,
        )
        self.conversation.append("user", "FEEDBACK_AGENT_REQUEST:\n" + prompt)
        raw = self.feedback_client.chat(
            [
                {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
                *self.conversation.messages(system_as_user=True),
            ],
            temperature=temperature,
        )
        self.conversation.append("user", "FEEDBACK_AGENT_RESPONSE:\n" + raw)
        return raw

    def run(self) -> dict:
        self.initialize()
        research_result = self._web_research_phase()
        req_result = self._requirements_refinement_phase()
        plan_result = self._plan_validation_phase()
        git_baseline = self._git_baseline_commit()
        step_results: list[dict[str, Any]] = []
        for step in self.plan_steps:
            step_results.append(self._implementation_loop_for_step(step))
            write_plan_doc(self.workspace, self.requirements, self.plan_steps, self.plan_notes)
            if step_results[-1]["status"] == "cannot_resolve" and self.config.resolution_policy.stop_on_cannot_resolve:
                break
        final_review = self._final_review_phase(step_results)
        git_finalize = self._git_finalize_policy()
        transcript_md = self.state_dir / "conversation.md"
        self.conversation.write_markdown(transcript_md)
        summary = {
            "workspace": str(self.workspace),
            "transcript_jsonl": ".agent_state/conversation.jsonl",
            "transcript_markdown": ".agent_state/conversation.md",
            "web_research": research_result,
            "requirements_refinement": req_result,
            "plan_validation": plan_result,
            "git": {
                "enabled": self.config.git_policy.enabled,
                "baseline": git_baseline,
                "baseline_ref": self.git_baseline_ref,
                "finalize": git_finalize,
            },
            "steps": step_results,
            "final_review": final_review,
            "final_status": self._final_status(step_results, final_review),
        }
        (self.state_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _web_research_phase(self) -> dict[str, Any]:
        """Fetch external research when the user explicitly asks for it.

        This is deliberately orchestration-owned rather than model-owned. Local
        models are good at using notes, but they are unreliable at proving they
        actually browsed. The harness therefore records fetched source evidence
        in RESEARCH.md, injects a compact version into later prompts, and lets
        review gates reject generated work that ignores the fetched sources.
        """
        if not self.config.mcp_tools.web_scraping:
            result = {
                "status": "skipped",
                "requested": False,
                "reason": "mcp_tools.web_scraping is disabled.",
                "targets": [],
            }
        else:
            result = run_web_research(self.config.project_design.prompt, self.config.web_research)
        self.web_research_result = result
        (self.workspace / "RESEARCH.md").write_text(research_to_markdown(result), encoding="utf-8")
        self.conversation.append("user", "WEB_RESEARCH_TOOL_RESULT:\n" + json.dumps(result, indent=2))
        if result.get("requested"):
            append_plan_note(self.workspace, f"[research] {result.get('status')}: web research evidence written to RESEARCH.md")
        return result

    def _requirements_refinement_phase(self, extra_context: str | None = None) -> dict:
        """Turn an underspecified project brief into requirements and a draft plan."""
        iterations: list[dict[str, Any]] = []
        latest: dict[str, Any] = {}
        review: dict[str, Any] = {}
        for index in range(1, self.config.phases.requirements_refinement.max_iterations + 1):
            prompt = (
                f"REQUIREMENTS_REFINEMENT_PHASE iteration={index}\n"
                "Refine the project requirements before implementation. Fill gaps, record assumptions, "
                "and create a first ordered plan. Do not write project files yet. "
                "Before returning, answer the planning_confirmation fields: is the plan feasible, clear, "
                "and verifiable, and what exact verification strategy will later be enforced?\n"
                f"{self._default_quality_instruction()}\n"
                f"Web research evidence: {compact_research_for_prompt(self.web_research_result)}\n"
                "If web research status is completed or partial, use those findings in the requirements and plan; "
                "the first research/structure step must cite the source URLs in generated project notes.\n"
                f"Extra context: {extra_context or 'none'}\n\n{REQUIREMENTS_CONTRACT}"
            )
            raw = self._implementation_chat(prompt)
            latest = extract_json_object(raw)
            self.requirements = latest
            self.plan_steps = normalize_plan_steps(latest.get("plan", []))
            for step in self.plan_steps:
                step.setdefault("status", "pending")
            write_requirements_doc(self.workspace, self.requirements)
            write_plan_doc(self.workspace, self.requirements, self.plan_steps, self.plan_notes)
            review = self._requirements_review(index, latest)
            iterations.append({"iteration": index, "requirements": latest, "review": review})
            if self._status(review) == "resolved":
                write_requirements_doc(self.workspace, self.requirements, review)
                append_plan_note(self.workspace, f"[requirements] resolved after iteration {index}: {review.get('summary', '')}")
                return {"status": "resolved", "iterations": iterations}
            self.conversation.append("user", "REQUIREMENTS_REWORK_DIRECTIVE:\nRevise requirements using this review:\n" + json.dumps(review, indent=2))
        fallback = self._fallback_resolution("requirements", review)
        self.requirements.setdefault("assumptions", []).append(fallback["note"])
        write_requirements_doc(self.workspace, self.requirements, review)
        return {"status": fallback["status"], "iterations": iterations, "resolution": fallback}

    def _requirements_review(self, index: int, requirements: dict[str, Any]) -> dict:
        """Ask the feedback agent whether requirements are actionable enough."""
        prompt = {
            "phase": "REQUIREMENTS_REVIEW_PHASE",
            "iteration": index,
            "project_design": self.config.project_design.prompt,
            "requirements": requirements,
            "web_research_evidence": self.web_research_result,
            "default_quality_policy": self._default_quality_policy_payload(),
            "expected_json": {
                "status": "resolved|needs_rework|needs_requirements_change|cannot_resolve|skipped_with_note",
                "needs_rework": True,
                "summary": "short review",
                "required_changes": ["specific change"],
                "cross_check_questions": ["requirement question the next pass must answer"],
            },
        }
        raw = self._feedback_chat(
            "REQUIREMENTS_REVIEW_PHASE\n"
            "Check whether the requirements are complete enough to support a distinct, verifiable plan. "
            "Reject vague requirements, missing gap decisions, and missing verification strategy.\n"
            "If default quality policy applies, reject requirements that omit project structure, tests, documentation, "
            "or the initial research/structure planning step.\n"
            "If WEB_RESEARCH_TOOL_RESULT has completed or partial sources, reject requirements that ignore those sources.\n"
            + json.dumps(prompt),
            temperature=0.1,
        )
        review = extract_json_object(raw)
        return self._normalize_review(review)

    def _plan_validation_phase(self) -> dict:
        """Block implementation until the ordered plan is executable and checkable."""
        iterations: list[dict[str, Any]] = []
        review: dict[str, Any] = {}
        for index in range(1, self.config.phases.plan_validation.max_iterations + 1):
            review = self._plan_validation_review(index)
            iterations.append({"iteration": index, "review": review, "plan": self.plan_steps})
            if self._status(review) == "resolved":
                append_plan_note(self.workspace, f"[plan] validated after iteration {index}: {review.get('summary', '')}")
                write_plan_doc(self.workspace, self.requirements, self.plan_steps, self.plan_notes)
                return {"status": "resolved", "iterations": iterations}
            refined = self._plan_refinement_pass(index, review)
            iterations[-1]["refinement"] = refined
        fallback = self._fallback_resolution("plan", review)
        self.plan_notes.append(fallback["note"])
        write_plan_doc(self.workspace, self.requirements, self.plan_steps, self.plan_notes)
        return {"status": fallback["status"], "iterations": iterations, "resolution": fallback}

    def _plan_validation_review(self, index: int) -> dict:
        """Combine deterministic plan checks with model-based plan critique."""
        structural_findings = self._plan_structural_findings()
        prompt = {
            "phase": "PLAN_VALIDATION_PHASE",
            "iteration": index,
            "requirements": self.requirements,
            "web_research_evidence": self.web_research_result,
            "plan": self.plan_steps,
            "deterministic_structural_findings": structural_findings,
            "checks": [
                "each step is distinct",
                "dependencies are explicit",
                "each step has acceptance criteria",
                "each step has validation commands or an explicit non-command validation method",
                "the sequence can be executed one step at a time",
                "planning_confirmation says the plan is feasible, clear, and verifiable",
                "the reviewer can name exactly how each step will be verified later",
                "when default quality policy applies, the first step researches needed patterns/knowledge and plans project structure before feature implementation",
                "when web research evidence exists, the plan requires generated notes to cite and apply researched source URLs",
            ],
            "expected_json": {
                "status": "resolved|needs_plan_change|needs_requirements_change|cannot_resolve",
                "needs_rework": True,
                "summary": "short review",
                "required_changes": ["specific change"],
                "planning_confirmation": {
                    "feasible": True,
                    "clear": True,
                    "verifiable": True,
                    "verification_matrix": [{"step_id": "S1", "how_verified": "command or explicit review method"}],
                },
            },
        }
        raw = self._feedback_chat(
            "PLAN_VALIDATION_PHASE\n"
            "Before implementation starts, explicitly confirm whether the plan is feasible, clear, "
            "and verifiable. If any step cannot be independently verified, return needs_plan_change.\n"
            + json.dumps(prompt),
            temperature=0.1,
        )
        return self._normalize_review(extract_json_object(raw))

    def _plan_refinement_pass(self, index: int, review: dict[str, Any]) -> dict:
        """Let the implementation model repair the plan while preserving context."""
        prompt = (
            f"PLAN_REFINEMENT_PHASE iteration={index}\n"
            "Revise only the ordered plan so every step is distinct, sequential, and verifiable. "
            "Keep requirements unless the review explicitly says they must change.\n"
            f"Current requirements: {json.dumps(self.requirements)}\n"
            f"Current plan: {json.dumps(self.plan_steps)}\n"
            f"Web research evidence: {compact_research_for_prompt(self.web_research_result)}\n"
            f"Review: {json.dumps(review)}\n\n{REQUIREMENTS_CONTRACT}"
        )
        raw = self._implementation_chat(prompt)
        payload = extract_json_object(raw)
        if payload.get("refined_requirements"):
            self.requirements = payload
        self.plan_steps = normalize_plan_steps(payload.get("plan", self.plan_steps))
        self.plan_notes.append(f"Plan refined after review iteration {index}.")
        write_plan_doc(self.workspace, self.requirements, self.plan_steps, self.plan_notes)
        return payload

    def _implementation_loop_for_step(self, step: dict[str, Any]) -> dict:
        """Run bounded implement/review attempts for one validated plan step."""
        attempts: list[dict[str, Any]] = []
        same_error_count = 0
        last_summary = ""
        max_attempts = max(
            self.config.phases.implementation.max_iterations,
            self.config.review_policy.hard_pushback_iterations + self.config.review_policy.compromise_iterations,
        )
        for attempt in range(1, max_attempts + 1):
            review_mode = self._review_mode(attempt)
            implementation = self._implementation_pass(step, attempt)
            review = self._step_review_pass(step, attempt, implementation, review_mode)
            attempts.append({"attempt": attempt, "implementation": implementation, "review": review})
            status = self._status(review)
            summary = str(review.get("summary", ""))
            same_error_count = same_error_count + 1 if summary == last_summary else 1
            last_summary = summary
            if status in {"resolved", "resolved_with_compromise", "skipped_with_note"}:
                step["status"] = "resolved"
                append_plan_note(self.workspace, f"[{step['id']}] resolved: {summary}")
                write_plan_doc(self.workspace, self.requirements, self.plan_steps, self.plan_notes)
                attempts[-1]["git_commit"] = self._git_commit_completed_step(step)
                return {"step_id": step["id"], "status": "resolved", "attempts": attempts}
            if status == "needs_plan_change":
                self._plan_refinement_pass(attempt, review)
            elif status == "needs_requirements_change":
                self._requirements_refinement_phase(extra_context=json.dumps(review))
            elif status == "cannot_resolve":
                step["status"] = "cannot_resolve"
                append_plan_note(self.workspace, f"[{step['id']}] cannot resolve: {summary}")
                return {"step_id": step["id"], "status": "cannot_resolve", "attempts": attempts}
            if same_error_count >= self.config.resolution_policy.max_same_error_repeats:
                append_plan_note(
                    self.workspace,
                    f"[{step['id']}] repeated review pattern in {review_mode} mode; continuing because retry budget is bounded.",
                )
            self.conversation.append(
                "user",
                "NEXT_IMPLEMENTATION_DIRECTIVE:\nApply this step review in the next attempt. "
                "Keep previous requirements, plan validation, and this step context in mind:\n"
                + json.dumps(review, indent=2),
            )
        resolution = self._fallback_resolution(f"step {step['id']}", attempts[-1]["review"] if attempts else {})
        step["status"] = resolution["status"]
        return {"step_id": step["id"], "status": resolution["status"], "attempts": attempts, "resolution": resolution}

    def _implementation_pass(self, step: dict[str, Any], attempt: int) -> dict:
        """Ask for complete-file edits and run the model-requested validations."""
        prompt = (
            f"IMPLEMENT_PLAN_STEP_PHASE step_id={step['id']} attempt={attempt}\n"
            "Work on this single plan step only. Do not silently jump ahead. If the step is impossible, "
            "use resolution_request and explain why. Cross-check your edits against this step's acceptance "
            "criteria and include validation commands that prove the step whenever terminal tools are enabled.\n"
            f"Refined requirements: {json.dumps(self.requirements)}\n"
            f"Full validated plan: {json.dumps(self.plan_steps)}\n"
            f"Current step: {json.dumps(step)}\n\n{IMPLEMENTATION_CONTRACT}"
        )
        if self._has_completed_research():
            prompt += (
                "\nWEB_RESEARCH_USAGE_REQUIREMENT:\n"
                f"Use this fetched research evidence and cite source URLs in ARCHITECTURE.md or the relevant deliverable: "
                f"{compact_research_for_prompt(self.web_research_result)}\n"
            )
        raw = self._implementation_chat(prompt)
        payload = extract_json_object(raw)
        written = write_files(self.workspace, payload.get("files", []))
        command_results = []
        if self.config.mcp_tools.terminal:
            command_results = run_commands(
                self.workspace,
                payload.get("commands", []),
                self.config.runtime.command_timeout_seconds,
                self.config.runtime.max_command_timeout_seconds,
            )
        note = payload.get("plan_note") or f"{step['id']} attempt {attempt} implementation pass completed."
        append_plan_note(self.workspace, f"[{step['id']} attempt {attempt}] {note}")
        return {"written": written, "commands": command_results, "raw": payload}

    def _step_review_pass(
        self,
        step: dict[str, Any],
        attempt: int,
        implementation: dict[str, Any],
        review_mode: str,
    ) -> dict:
        """Critique one step using reviewer-owned file and command evidence."""
        plan_text = (self.workspace / "PLAN.md").read_text(encoding="utf-8")
        feedback_tool_evidence = self._step_feedback_tool_evidence(step)
        evidence_findings = self._evidence_findings(step, implementation, feedback_tool_evidence)
        evidence_findings.extend(self._research_usage_findings(step, feedback_tool_evidence))
        prompt = {
            "phase": "STEP_REVIEW_PHASE",
            "step": step,
            "attempt": attempt,
            "review_mode": review_mode,
            "requirements": self.requirements,
            "web_research_evidence": self.web_research_result,
            "plan": self.plan_steps,
            "plan_file": plan_text,
            "implementation": implementation,
            "feedback_tool_evidence": feedback_tool_evidence,
            "deterministic_evidence_findings": evidence_findings,
            "review_policy": {
                "hard_pushback_iterations": self.config.review_policy.hard_pushback_iterations,
                "compromise_iterations": self.config.review_policy.compromise_iterations,
            },
            "review_instructions": [
                "Run-result failures, timeouts, missing files, broken UI hooks, and weak validation must be called out.",
                "Ask concrete cross-check questions against the refined requirements and the current step acceptance criteria.",
                "Use feedback_tool_evidence first: it is the reviewer-owned file snapshot and independent validation command run.",
                "Use feedback_tool_evidence.git.status_short/diff_stat/diff to review changes since the last accepted step commit.",
                "If git meaningful_changed_paths is empty for an implementation step, request the missing change and name the current plan requirement.",
                "Do not accept a step just because the implementation agent claims tests passed.",
                "If web_research_evidence has completed sources, confirm the generated work actually cites and applies those source URLs.",
                "If test evidence is absent in hard_pushback mode, return needs_rework.",
                "If evidence remains imperfect in compromise mode, either return needs_rework with a small bounded fix or resolved_with_compromise/skipped_with_note with an explicit diluted requirement note.",
                "For browser/game work, require Playwright-style interaction evidence and screenshot/report artifacts when configured.",
                "Return needs_plan_change if this step cannot be independently verified as written.",
                "Return needs_requirements_change if the requirements are contradictory or impossible.",
                "Return cannot_resolve only when bounded retries are unlikely to help.",
            ],
            "expected_json": {
                "status": "resolved|needs_rework|cannot_resolve|needs_requirements_change|needs_plan_change|skipped_with_note",
                "needs_rework": True,
                "summary": "short review",
                "required_changes": ["specific change"],
                "cross_check_questions": ["question answered by code/commands/files"],
                "verification_evidence": ["command/file/screenshot/report checked"],
                "compromise_note": "only when review_mode=compromise and perfection is not worth more retries",
            },
        }
        raw = self._feedback_chat(
            "STEP_REVIEW_PHASE\n"
            f"Review mode: {review_mode}. Critically verify exactly one plan step. "
            "Use the whole transcript to avoid repeating old mistakes, but judge only the current step "
            "against its acceptance criteria and test evidence.\n"
            + json.dumps(prompt),
            temperature=0.1,
        )
        review = self._normalize_review(extract_json_object(raw))
        review = self._enforce_evidence_policy(review, evidence_findings, review_mode)
        review["feedback_tool_evidence"] = feedback_tool_evidence
        review["deterministic_evidence_findings"] = evidence_findings
        append_plan_note(self.workspace, f"[{step['id']} attempt {attempt}] review: {review.get('summary', 'no summary')}")
        return review

    def _final_review_phase(self, step_results: list[dict[str, Any]]) -> dict:
        """Run whole-project review after individual plan steps complete."""
        iterations: list[dict[str, Any]] = []
        for attempt in range(1, self.config.review_policy.final_review_iterations + 1):
            review = self._final_project_review(attempt, step_results)
            item: dict[str, Any] = {"attempt": attempt, "review": review}
            if self._status(review) in {"resolved", "resolved_with_compromise", "skipped_with_note"}:
                append_plan_note(self.workspace, f"[final review] resolved: {review.get('summary', '')}")
                write_plan_doc(self.workspace, self.requirements, self.plan_steps, self.plan_notes)
                item["git_commit"] = self._git_commit_final_review()
                iterations.append(item)
                return {"status": self._status(review), "iterations": iterations}
            correction = self._final_correction_pass(attempt, review)
            item["correction"] = correction
            iterations.append(item)
        fallback = self._fallback_resolution("final review", iterations[-1]["review"] if iterations else {})
        append_plan_note(self.workspace, f"[final review] {fallback['status']}: {fallback['note']}")
        return {"status": fallback["status"], "iterations": iterations, "resolution": fallback}

    def _final_project_review(self, attempt: int, step_results: list[dict[str, Any]]) -> dict:
        feedback_tool_evidence = self._final_feedback_tool_evidence()
        evidence_findings = self._project_evidence_findings(step_results, feedback_tool_evidence)
        prompt = {
            "phase": "FINAL_PROJECT_REVIEW_PHASE",
            "attempt": attempt,
            "requirements": self.requirements,
            "plan": self.plan_steps,
            "step_results": step_results,
            "feedback_tool_evidence": feedback_tool_evidence,
            "deterministic_evidence_findings": evidence_findings,
            "expected_json": {
                "status": "resolved|needs_rework|cannot_resolve|needs_requirements_change|needs_plan_change|skipped_with_note|resolved_with_compromise",
                "needs_rework": True,
                "summary": "whole project review",
                "required_changes": ["specific final change"],
                "verification_evidence": ["evidence reviewed"],
            },
        }
        raw = self._feedback_chat(
            "FINAL_PROJECT_REVIEW_PHASE\n"
            "Review the whole project after all plan steps. Re-check original requirements, final files, "
            "and all test evidence. Push back if the project lacks proof or contradicts requirements.\n"
            + json.dumps(prompt),
            temperature=0.1,
        )
        review = self._normalize_review(extract_json_object(raw))
        if evidence_findings and self._status(review) == "resolved":
            review["status"] = "needs_rework"
            review["needs_rework"] = True
            review["summary"] = "Final review cannot resolve because deterministic evidence checks found gaps."
            review["required_changes"] = evidence_findings
        review["feedback_tool_evidence"] = feedback_tool_evidence
        review["deterministic_evidence_findings"] = evidence_findings
        return review

    def _final_correction_pass(self, attempt: int, review: dict[str, Any]) -> dict:
        prompt = (
            f"FINAL_PROJECT_CORRECTION_PHASE attempt={attempt}\n"
            "Apply only the final review changes needed to make the whole project consistent with requirements. "
            "Include validation commands and test evidence.\n"
            f"Review: {json.dumps(review)}\n\n{IMPLEMENTATION_CONTRACT}"
        )
        raw = self._implementation_chat(prompt)
        payload = extract_json_object(raw)
        written = write_files(self.workspace, payload.get("files", []))
        command_results = []
        if self.config.mcp_tools.terminal:
            command_results = run_commands(
                self.workspace,
                payload.get("commands", []),
                self.config.runtime.command_timeout_seconds,
                self.config.runtime.max_command_timeout_seconds,
            )
        append_plan_note(self.workspace, f"[final correction attempt {attempt}] {payload.get('plan_note', 'completed')}")
        return {"written": written, "commands": command_results, "raw": payload}

    def _plan_structural_findings(self) -> list[str]:
        """Cheap deterministic guardrails before the model-based plan review.

        The reviewer still makes the judgment call, but these findings prevent
        obvious misses such as empty validation commands or broken dependencies
        from slipping through just because a model review was too generous.
        """
        findings: list[str] = []
        if not self.plan_steps:
            return ["Plan has no steps."]
        seen_ids: set[str] = set()
        for step in self.plan_steps:
            step_id = str(step.get("id") or "<missing>")
            if step_id in seen_ids:
                findings.append(f"Duplicate step id: {step_id}.")
            seen_ids.add(step_id)
            if not step.get("acceptance_criteria"):
                findings.append(f"{step_id} has no acceptance criteria.")
            if not step.get("validation_commands"):
                findings.append(f"{step_id} has no validation commands or explicit validation method.")
            for dep in step.get("depends_on", []):
                if dep not in seen_ids:
                    findings.append(f"{step_id} depends on {dep}, which has not appeared earlier in the ordered plan.")
        if self._default_quality_policy_applies() and self.config.quality_policy.require_research_and_structure_step:
            first = self.plan_steps[0]
            first_text = " ".join([
                str(first.get("title", "")),
                str(first.get("description", "")),
                " ".join(first.get("acceptance_criteria", [])),
            ]).lower()
            if not (
                ("research" in first_text or "patterns" in first_text or "knowledge" in first_text)
                and ("structure" in first_text or "architecture" in first_text)
                and ("plan" in first_text or "order" in first_text)
            ):
                findings.append(
                    "First plan step must research needed patterns/knowledge, plan project structure/architecture, "
                    "and rewrite the remaining plan if structure changes task order."
                )
            if self._has_completed_research() and not any(marker in first_text for marker in ("source", "url", "cite", "citation")):
                findings.append(
                    "Web research evidence exists, so the first research/structure step must require citing and applying source URLs."
                )
        confirmation = self.requirements.get("planning_confirmation") if isinstance(self.requirements, dict) else None
        if not isinstance(confirmation, dict):
            findings.append("Requirements are missing planning_confirmation.")
        else:
            for key in ("is_feasible", "is_clear", "is_verifiable"):
                if confirmation.get(key) is not True:
                    findings.append(f"planning_confirmation.{key} is not true.")
            if not confirmation.get("verification_strategy"):
                findings.append("planning_confirmation.verification_strategy is empty.")
        return findings

    def _default_quality_policy_payload(self) -> dict[str, Any]:
        return {
            "applies": self._default_quality_policy_applies(),
            "assumed_requirement": (
                "Unless the user explicitly says otherwise, code should be well structured, well tested, "
                "well documented, and the first implementation step should research required patterns/knowledge, "
                "plan project structure, and update the remaining plan if structure changes the task order."
            ),
        }

    def _default_quality_instruction(self) -> str:
        if not self._default_quality_policy_applies():
            return "The user prompt appears to override the default code-quality policy; record that override explicitly."
        return (
            "Default quality policy applies unless the user explicitly says otherwise: add a requirement that the project "
            "is well structured, well tested, and well documented. The first implementation step in the plan must: "
            "A) research on the web or from available knowledge any required patterns/knowledge, and "
            "B) plan the project structure/architecture and rewrite the remaining plan if that structure changes task order."
        )

    def _default_quality_policy_applies(self) -> bool:
        if not self.config.quality_policy.assume_code_quality_when_unspecified:
            return False
        prompt = self.config.project_design.prompt.lower()
        overrides = [
            "skip tests",
            "no tests",
            "throwaway",
            "quick and dirty",
            "prototype only",
            "do not document",
            "no documentation",
            "ignore code quality",
        ]
        return not any(item in prompt for item in overrides)

    def _has_completed_research(self) -> bool:
        return self.web_research_result.get("status") in {"completed", "partial"} and bool(self._research_source_urls())

    def _research_source_urls(self) -> list[str]:
        urls: list[str] = []
        for item in self.web_research_result.get("targets", []):
            url = str(item.get("url") or "")
            if item.get("status") == "ok" and url:
                urls.append(url)
        return urls

    def _research_usage_findings(
        self,
        step: dict[str, Any],
        feedback_tool_evidence: dict[str, Any],
    ) -> list[str]:
        """Reject research/architecture work that ignores fetched sources.

        RESEARCH.md is generated by the harness, so it does not count as model
        usage. The generated deliverable must cite at least one researched URL
        in ARCHITECTURE.md or another project file for the research step to pass.
        """
        if not self._has_completed_research():
            return []
        step_text = " ".join([
            str(step.get("id", "")),
            str(step.get("title", "")),
            str(step.get("description", "")),
        ]).lower()
        if "research" not in step_text and "structure" not in step_text and str(step.get("id")) != "S1":
            return []
        generated_files = [
            item
            for item in feedback_tool_evidence.get("workspace_files", [])
            if item.get("path") not in {"RESEARCH.md", "PLAN.md", "REQUIREMENTS.md"}
        ]
        generated_text = "\n".join(str(item.get("content", "")) for item in generated_files)
        if any(url in generated_text for url in self._research_source_urls()):
            return []
        return [
            "Web research evidence exists but generated project work did not cite/use any researched source URL outside RESEARCH.md."
        ]

    def _git_diff_findings(self, step: dict[str, Any], feedback_tool_evidence: dict[str, Any]) -> list[str]:
        """Reject a step when the reviewer has no implementation diff to inspect."""
        if not (self.config.git_policy.enabled and self.config.git_policy.require_step_diff):
            return []
        git = feedback_tool_evidence.get("git") or {}
        changed_paths = git.get("meaningful_changed_paths") or []
        if changed_paths:
            return []
        step_id = step.get("id", "step")
        requirement = "; ".join(step.get("acceptance_criteria", [])[:2]) or step.get("title", "the planned work")
        return [
            (
                f"Git working tree has no implementation changes for {step_id}. "
                f"Please implement the plan requirement before review can accept it: {requirement}"
            )
        ]

    def _review_mode(self, attempt: int) -> str:
        if attempt <= self.config.review_policy.hard_pushback_iterations:
            return "hard_pushback"
        return "compromise"

    def _step_feedback_tool_evidence(self, step: dict[str, Any]) -> dict[str, Any]:
        """Collect the evidence the feedback agent can inspect directly.

        The feedback model is not trusted to merely believe the implementation
        model's report. Before each review, the harness takes a fresh workspace
        snapshot and re-runs the current plan step's validation commands. Those
        command results become the reviewer's evidence.
        """
        validation_commands = step.get("validation_commands", [])
        validation_results: list[dict[str, Any]] = []
        if self.config.mcp_tools.terminal and validation_commands:
            validation_results = run_commands(
                self.workspace,
                validation_commands,
                self.config.runtime.command_timeout_seconds,
                self.config.runtime.max_command_timeout_seconds,
            )
        return {
            "kind": "step_feedback_tools",
            "step_id": step.get("id"),
            "workspace_files": collect_workspace_files(self.workspace),
            "validation_commands": validation_commands,
            "validation_results": validation_results,
            "git": git_evidence(self.workspace) if self.config.git_policy.enabled else {"enabled": False},
        }

    def _final_feedback_tool_evidence(self) -> dict[str, Any]:
        """Re-run each plan step's validation commands for final project review."""
        step_validations: list[dict[str, Any]] = []
        for step in self.plan_steps:
            commands = step.get("validation_commands", [])
            results: list[dict[str, Any]] = []
            if self.config.mcp_tools.terminal and commands:
                results = run_commands(
                    self.workspace,
                    commands,
                    self.config.runtime.command_timeout_seconds,
                    self.config.runtime.max_command_timeout_seconds,
                )
            step_validations.append({
                "step_id": step.get("id"),
                "validation_commands": commands,
                "validation_results": results,
            })
        return {
            "kind": "final_feedback_tools",
            "workspace_files": collect_workspace_files(self.workspace),
            "step_validations": step_validations,
            "git": git_evidence(self.workspace) if self.config.git_policy.enabled else {"enabled": False},
        }

    def _evidence_findings(
        self,
        step: dict[str, Any],
        implementation: dict[str, Any],
        feedback_tool_evidence: dict[str, Any] | None = None,
    ) -> list[str]:
        findings: list[str] = []
        implementation_commands = implementation.get("commands", [])
        feedback_results = (feedback_tool_evidence or {}).get("validation_results", [])
        expected_validation = bool(step.get("validation_commands"))
        if expected_validation and not feedback_results:
            findings.append(f"{step.get('id', 'step')} has validation criteria but feedback tools produced no validation evidence.")
        for result in feedback_results:
            if result.get("timed_out"):
                findings.append(f"Feedback validation command timed out: {result.get('command')}")
            if int(result.get("returncode", 0)) != 0:
                findings.append(
                    f"Feedback validation command failed with return code {result.get('returncode')}: {result.get('command')}"
                )
        for result in implementation_commands:
            if result.get("timed_out"):
                findings.append(f"Implementation command timed out: {result.get('command')}")
            if int(result.get("returncode", 0)) != 0:
                findings.append(
                    f"Implementation command failed with return code {result.get('returncode')}: {result.get('command')}"
                )
        findings.extend(self._git_diff_findings(step, feedback_tool_evidence or {}))
        return findings

    def _project_evidence_findings(
        self,
        step_results: list[dict[str, Any]],
        feedback_tool_evidence: dict[str, Any] | None = None,
    ) -> list[str]:
        findings: list[str] = []
        final_validations = {
            str(item.get("step_id")): item
            for item in (feedback_tool_evidence or {}).get("step_validations", [])
        }
        for step_result in step_results:
            step_id = str(step_result.get("step_id"))
            if step_result.get("status") != "resolved":
                findings.append(f"Step {step_id} ended with status {step_result.get('status')}.")
            attempts = step_result.get("attempts", [])
            if not attempts:
                findings.append(f"Step {step_id} has no attempts.")
            validation = final_validations.get(step_id)
            if validation is not None:
                results = validation.get("validation_results", [])
                if validation.get("validation_commands") and not results:
                    findings.append(f"Step {step_id} final feedback validation produced no command evidence.")
                for result in results:
                    if result.get("timed_out"):
                        findings.append(f"Step {step_id} final feedback validation timed out: {result.get('command')}")
                    if int(result.get("returncode", 0)) != 0:
                        findings.append(
                            f"Step {step_id} final feedback validation failed: {result.get('command')}"
                        )
                continue
            if not attempts:
                continue
            implementation = attempts[-1].get("implementation", {})
            commands = implementation.get("commands", [])
            if not commands:
                findings.append(f"Step {step_id} final attempt has no command evidence.")
            for result in commands:
                if result.get("timed_out") or int(result.get("returncode", 0)) != 0:
                    findings.append(f"Step {step_id} final attempt has failing evidence: {result.get('command')}")
        return findings

    def _enforce_evidence_policy(
        self,
        review: dict[str, Any],
        evidence_findings: list[str],
        review_mode: str,
    ) -> dict[str, Any]:
        if not evidence_findings:
            return review
        review = dict(review)
        if self._status(review) not in {"resolved", "resolved_with_compromise", "skipped_with_note"}:
            existing = [str(item) for item in review.get("required_changes", [])]
            review["required_changes"] = existing + [item for item in evidence_findings if item not in existing]
            return review
        if review_mode == "hard_pushback":
            review["status"] = "needs_rework"
            review["needs_rework"] = True
            review["summary"] = "Please rework this step: hard-pushback evidence checks found missing, failing, or absent git-diff evidence."
            review["required_changes"] = evidence_findings
        else:
            review["status"] = "skipped_with_note"
            review["needs_rework"] = False
            review["summary"] = "Compromise mode accepted the step only with explicit evidence limitations recorded."
            review["required_changes"] = evidence_findings
            review["compromise_note"] = "Evidence remained imperfect after the hard-pushback budget."
        return review

    def _status(self, review: dict[str, Any]) -> str:
        status = str(review.get("status") or "").strip()
        if status in REVIEW_STATUSES:
            return status
        if review.get("needs_rework") is False:
            return "resolved"
        return "needs_rework"

    def _normalize_review(self, review: dict[str, Any]) -> dict[str, Any]:
        review = dict(review)
        status = self._status(review)
        review["status"] = status
        review["needs_rework"] = status not in {"resolved", "skipped_with_note", "resolved_with_compromise"}
        review.setdefault("summary", "no summary")
        review.setdefault("required_changes", [])
        return review

    def _fallback_resolution(self, scope: str, review: dict[str, Any]) -> dict[str, str]:
        """Choose a bounded outcome when retries stop making progress."""
        summary = review.get("summary", "No final review summary.") if review else "No review was produced."
        if self.config.resolution_policy.allow_skip_with_note:
            status = "skipped_with_note"
            note = f"Bounded retries exhausted for {scope}; skipped with note. Last review: {summary}"
        elif self.config.resolution_policy.allow_requirement_dilution:
            status = "needs_requirements_change"
            note = f"Bounded retries exhausted for {scope}; requirements must be diluted or clarified. Last review: {summary}"
        else:
            status = "cannot_resolve"
            note = f"Bounded retries exhausted for {scope}; cannot resolve. Last review: {summary}"
        return {"status": status, "note": note}

    def _git_baseline_commit(self) -> dict[str, Any]:
        if not self.config.git_policy.enabled:
            return {"enabled": False}
        result = commit_all(
            self.workspace,
            "harness baseline: requirements and validated plan",
            allow_empty=True,
        )
        self.git_baseline_ref = str(result.get("head_after") or "")
        return result

    def _git_commit_completed_step(self, step: dict[str, Any]) -> dict[str, Any]:
        if not (self.config.git_policy.enabled and self.config.git_policy.commit_completed_steps):
            return {"enabled": self.config.git_policy.enabled, "committed": False, "reason": "disabled"}
        step_id = str(step.get("id") or "step")
        title = str(step.get("title") or "completed plan step")
        return commit_all(self.workspace, f"{step_id}: {title}")

    def _git_commit_final_review(self) -> dict[str, Any]:
        if not (self.config.git_policy.enabled and self.config.git_policy.commit_completed_steps):
            return {"enabled": self.config.git_policy.enabled, "committed": False, "reason": "disabled"}
        return commit_all(self.workspace, "final review: accepted project state", allow_empty=True)

    def _git_finalize_policy(self) -> dict[str, Any]:
        if not self.config.git_policy.enabled:
            return {"enabled": False}
        if not self.config.git_policy.leave_final_changes_uncommitted:
            return {"enabled": True, "left_uncommitted": False, "git": git_evidence(self.workspace)}
        reset = reset_to_ref(
            self.workspace,
            self.git_baseline_ref,
            mode=self.config.git_policy.final_reset_mode,
        )
        return {"enabled": True, "left_uncommitted": True, "reset": reset, "git": git_evidence(self.workspace)}

    def _final_status(self, step_results: list[dict[str, Any]], final_review: dict[str, Any] | None = None) -> str:
        if not step_results:
            return "no_steps"
        statuses = {item["status"] for item in step_results}
        final_status = self._status(final_review or {})
        if statuses == {"resolved"} and final_status in {"resolved", "resolved_with_compromise", "skipped_with_note"}:
            return "resolved"
        if "cannot_resolve" in statuses:
            return "cannot_resolve"
        if "skipped_with_note" in statuses:
            return "resolved_with_skips"
        return "partial"
