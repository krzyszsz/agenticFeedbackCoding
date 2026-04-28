from __future__ import annotations

import json
import re
from typing import Any

from .bounds import clamp_text, estimate_tokens
from .compaction import maybe_compact
from .config import AgentConfig
from .conversation import Conversation
from .git_tools import commit_all, ensure_git_repo, git_evidence, reset_to_ref
from .llm import OpenAICompatClient
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
For large projects, group related work into 4-6 high-impact steps instead of
creating many tiny steps. Every step must remain independently verifiable.
Keep refined_requirements to at most 8 concise strings, assumptions to at most
5 concise strings, and open_questions to at most 3 entries. Keep step
descriptions and acceptance criteria short enough to fit in one local-model
response.
Commands may be argv lists or {"cmd": ["python", "script.py"], "timeout_seconds": 7200}
when one specific tool call legitimately needs longer than the default timeout.
Avoid embedding Python source, f-strings, braces, list comprehensions, or other
quote-heavy snippets inside JSON command strings during planning. Prefer simple
argv checks such as ["test", "-f", "README.md"]. For complex validation, add a
plan step that creates a validation script and then run ["python", "validate.py"].
For expected failure-path tests, use {"cmd": ["python", "-m", "app"], "expected_returncode": 2}
and assert the stderr/stdout message in a small wrapper command when possible.
"""


PLAN_REFINEMENT_CONTRACT = """
Return strict compact JSON only:
{
  "planning_confirmation": {
    "is_feasible": true,
    "is_clear": true,
    "is_verifiable": true,
    "verification_strategy": "short step-by-step validation strategy",
    "remaining_risks": ["risk"]
  },
  "plan": [
    {
      "id": "S1",
      "title": "short task title",
      "description": "short description",
      "depends_on": [],
      "acceptance_criteria": ["short verifiable criterion"],
      "validation_commands": [["python", "scripts/validate_step.py"]]
    }
  ]
}
Do not repeat the full requirements. Keep strings short. Validation commands
must terminate and assert behavior. Do not use `python -m http.server` by
itself; wrap any server startup inside a script that performs checks and exits.
Do not embed inline Python source in JSON command strings. Prefer simple argv
checks such as ["test", "-f", "index.html"], or run a generated validation
script such as ["python", "validate.py"].
"""


IMPLEMENTATION_CONTRACT = """
Return strict JSON only:
{
  "plan_note": "short note for the configured plan file",
  "files": [{"path": "relative/path", "content": "complete file content"}],
  "commands": [["python", "-m", "unittest", "-v"]],
  "test_evidence": ["short description of command/report/screenshot evidence produced"],
  "resolution_request": "none|needs_requirements_change|needs_plan_change|cannot_resolve"
}
Only write paths inside the project workspace. Prefer small validation commands that finish quickly.
Keep implementation responses compact. For non-trivial projects, write exactly
one meaningful file per attempt unless two tiny files are inseparable. Subsequent
feedback iterations can request the next file. Do not try to satisfy every
review request in one JSON object because local models may exceed the response
budget.
Commands may be argv lists or {"cmd": ["python", "script.py"], "timeout_seconds": 7200}
when one specific tool call legitimately needs longer than the default timeout.
Avoid quote-heavy inline Python in JSON command strings. If validation needs
logic, write a small validation file and run it as a command.
For expected failure-path tests, use {"cmd": ["python", "-m", "app"], "expected_returncode": 2}
or, better, a small assertion command that checks the non-zero return code and error text.
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
The harness, not the implementation agent, owns git staging and commits after a
step is accepted. Treat untracked meaningful files as reviewable pre-acceptance
diff evidence, and do not request `git add` or `git commit` from the
implementation agent.
The harness also writes configured plan-file notes and marks a plan step
resolved after feedback accepts it. During review, accept a current-step
plan-file marker that is still pending/in-progress when the implementation
evidence is otherwise complete; do not require the implementation agent to mark
a step completed before you have accepted it.
Reject shallow or tautological validation. Tests must exercise user-visible
behavior for the current requirement, not merely check that a file contains a
string or that a script exists. For browser work, prefer real interaction
evidence through Playwright, screenshots, and JSON reports when web interaction
tools are enabled. Do not ask generated validation scripts to install browsers
or packages at runtime; the harness container is responsible for browser/tool
availability. In compromise mode, accept a clearly-labelled fallback only when
browser launch fails under bounded timeouts and the fallback still proves the
user-visible behavior as directly as possible.
Expected failure paths are valid evidence when the command declares an
expected_returncode or uses a wrapper assertion to check the non-zero exit code
and error text. Do not reject such evidence just because the user-facing command
failed in the intended way.
Be strict in hard-pushback mode and only compromise in compromise mode when
bounded retries are more valuable than perfect adherence.
Do not turn implementation-response size guidance, such as "one meaningful file
per attempt", into a plan-step acceptance rule. That guidance exists to keep
local-model JSON outputs small; the plan itself may group related deliverables
when that is the smallest feasible way to satisfy the user's requirements.
When user constraints conflict, name the conflict, choose the smallest
verifiable compromise, and avoid looping forever over mutually impossible
constraints. Treat step-count limits as hard only when the user explicitly says
"hard", "strict", "exactly", or "must"; otherwise treat them as a planning
preference that can bend for evidence and quality.
Return strict JSON only.
"""


class FeedbackLoopAgent:
    """Orchestrates a phased two-agent workflow over one durable transcript.

    The implementation and feedback clients may point at the same local model or
    two different OpenAI-compatible endpoints. The orchestration layer owns the
    phase boundaries, file writes, validation commands, retry limits, and the
    transcript discipline that keeps long sessions coherent.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        implementation_client: Any | None = None,
        feedback_client: Any | None = None,
    ):
        self.config = config
        self.workspace = config.runtime.workspace
        self.state_dir = self.workspace / ".agent_state"
        self.conversation = Conversation(
            self.state_dir / "conversation.jsonl",
            full_path=self.state_dir / "conversation.full.jsonl",
            echo=config.runtime.print_transcript,
            echo_limit_chars=config.runtime.live_turn_max_chars,
            color=config.runtime.color_transcript,
        )
        self.impl_client = implementation_client or OpenAICompatClient(config.implementation_model)
        self.feedback_client = feedback_client or OpenAICompatClient(config.feedback_model or config.implementation_model)
        self.requirements: dict[str, Any] = {}
        self.plan_steps: list[dict[str, Any]] = []
        self.plan_notes: list[str] = []
        self.web_research_result: dict[str, Any] = {
            "status": "not_run",
            "requested": False,
            "targets": [],
        }
        self.git_baseline_ref = ""

    def _plan_path(self) -> Any:
        return self.workspace / self.config.runtime.plan_file

    def _requirements_path(self) -> Any:
        return self.workspace / self.config.runtime.requirements_file

    def _research_path(self) -> Any:
        return self.workspace / self.config.runtime.research_file

    def _harness_doc_names(self) -> set[str]:
        return {
            self.config.runtime.plan_file,
            self.config.runtime.requirements_file,
            self.config.runtime.research_file,
            "PLAN.md",
            "REQUIREMENTS.md",
            "RESEARCH.md",
            "AGENT_PLAN.md",
            "AGENT_REQUIREMENTS.md",
            "AGENT_RESEARCH.md",
        }

    def _ensure_plan(self) -> None:
        ensure_plan(self.workspace, self.config.runtime.plan_file)

    def _append_plan_note(self, note: str) -> None:
        append_plan_note(self.workspace, note, self.config.runtime.plan_file)

    def _write_plan_doc(self) -> None:
        write_plan_doc(
            self.workspace,
            self.requirements,
            self.plan_steps,
            self.plan_notes,
            self.config.runtime.plan_file,
        )

    def _write_requirements_doc(self, review: dict[str, Any] | None = None) -> None:
        write_requirements_doc(
            self.workspace,
            self.requirements,
            review,
            self.config.runtime.requirements_file,
        )

    def initialize(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_plan()
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
                    f"Maintain {self.config.runtime.plan_file} and {self.config.runtime.requirements_file}. "
                    "Keep all work inside the project workspace. "
                    "The workspace is a git repository when git_policy is enabled; accepted plan steps are "
                    "committed only by the harness after feedback review agrees they are complete. "
                    "Implementation turns may inspect git status and diffs, but must not run git add, "
                    "git commit, git reset, git checkout, or other repository-mutating git commands. "
                    "This transcript is durable chat memory: IMPLEMENTATION_AGENT_REQUEST/RESPONSE and "
                    "FEEDBACK_AGENT_REQUEST/RESPONSE blocks are cumulative context, not isolated prompts. "
                    f"Harness-owned state files are {self.config.runtime.plan_file}, "
                    f"{self.config.runtime.requirements_file}, and {self.config.runtime.research_file}."
                ),
            )
            self.conversation.append(
                "user",
                f"PROJECT DESIGN: {self.config.project_design.title}\n\n{self.config.project_design.prompt}",
            )

    def _implementation_chat(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Append a request, call the implementation model, then persist its reply.

        Keeping the request itself in the transcript matters. Without it, later
        turns see model answers but not the exact task/review that caused them,
        which makes long local-model sessions drift quickly.
        """
        content = "IMPLEMENTATION_AGENT_REQUEST:\n" + prompt
        expected_response_tokens = max_tokens or self.config.implementation_model.max_tokens
        maybe_compact(
            self.conversation,
            self.config,
            self.impl_client,
            context_window=self.config.implementation_model.context_window,
            incoming_tokens=estimate_tokens(content) + expected_response_tokens,
        )
        self.conversation.append("user", content)
        raw = self.impl_client.chat(self.conversation.messages(), max_tokens=max_tokens)
        self.conversation.append("assistant", "IMPLEMENTATION_AGENT_RESPONSE:\n" + raw)
        maybe_compact(
            self.conversation,
            self.config,
            self.impl_client,
            context_window=self.config.implementation_model.context_window,
        )
        return raw

    def _feedback_chat(self, prompt: str, *, temperature: float = 0.1) -> str:
        """Run the feedback model against the same durable transcript.

        Feedback replies are stored as user-visible transcript blocks so the
        implementation model treats them as external critique on the next turn.
        The feedback model still receives the entire history, including its own
        previous reviews, which gives it continuity across loops.
        """
        feedback_cfg = self.config.feedback_model or self.config.implementation_model
        response_tokens = self._feedback_response_tokens(feedback_cfg)
        content = "FEEDBACK_AGENT_REQUEST:\n" + prompt
        maybe_compact(
            self.conversation,
            self.config,
            self.feedback_client,
            context_window=feedback_cfg.context_window,
            incoming_tokens=estimate_tokens(content) + response_tokens,
        )
        self.conversation.append("user", content)
        raw = self.feedback_client.chat(
            [
                {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
                *self.conversation.messages(system_as_user=True),
            ],
            max_tokens=response_tokens,
            temperature=temperature,
        )
        self.conversation.append("user", "FEEDBACK_AGENT_RESPONSE:\n" + raw)
        maybe_compact(
            self.conversation,
            self.config,
            self.feedback_client,
            context_window=feedback_cfg.context_window,
        )
        return raw

    def _feedback_chat_with_compact_context(
        self,
        prompt: str,
        *,
        context_note: str,
        temperature: float = 0.1,
    ) -> str:
        """Run a feedback turn with compact model context but full transcript logging.

        Most feedback phases benefit from the full chat. Final project review is
        different: it already receives fresh reviewer-owned validation evidence,
        and the full transcript can grow beyond local server request limits. The
        request and response are still appended to the durable transcript, but
        the model call receives a compact summary plus the final-review payload.
        """
        feedback_cfg = self.config.feedback_model or self.config.implementation_model
        response_tokens = self._feedback_response_tokens(feedback_cfg)
        content = "FEEDBACK_AGENT_REQUEST:\n" + prompt
        maybe_compact(
            self.conversation,
            self.config,
            self.feedback_client,
            context_window=feedback_cfg.context_window,
            incoming_tokens=estimate_tokens(content) + response_tokens,
        )
        self.conversation.append("user", content)
        raw = self.feedback_client.chat(
            [
                {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
                {"role": "user", "content": "COMPACT_TRANSCRIPT_CONTEXT:\n" + context_note},
                {"role": "user", "content": "FEEDBACK_AGENT_REQUEST:\n" + prompt},
            ],
            max_tokens=response_tokens,
            temperature=temperature,
        )
        self.conversation.append("user", "FEEDBACK_AGENT_RESPONSE:\n" + raw)
        maybe_compact(
            self.conversation,
            self.config,
            self.feedback_client,
            context_window=feedback_cfg.context_window,
        )
        return raw

    def _feedback_response_tokens(self, feedback_cfg) -> int:
        """Keep reviewer JSON bounded even when implementation output can be large.

        The implementation model may need a high ceiling for generated files,
        but feedback turns should normally be compact JSON. Without a separate
        cap, a local model can spend many minutes filling the full implementation
        ceiling after it has already said enough for review.
        """
        configured = self.config.runtime.feedback_response_max_tokens
        if configured <= 0:
            return feedback_cfg.max_tokens
        return max(512, min(feedback_cfg.max_tokens, configured))

    def _extract_json_or_retry(
        self,
        raw: str,
        *,
        phase: str,
        contract: str,
        feedback: bool = False,
    ) -> dict[str, Any]:
        """Parse a model JSON response, with one repair turn for malformed output.

        Local models often produce useful content but wrap it in markdown,
        include thinking text, or run out of tokens midway through a long JSON
        object. Crashing loses the whole long run. A bounded repair turn keeps
        the transcript honest while asking the same agent to return a compact,
        machine-parseable object that matches the phase contract.
        """
        try:
            return extract_json_object(raw)
        except Exception as exc:
            tail = raw[-3000:]
            step_limit, limit_is_hard = self._configured_plan_step_limit()
            if step_limit and limit_is_hard:
                step_limit_text = f" Keep plans to the hard limit of at most {step_limit} steps."
            elif step_limit:
                step_limit_text = f" Prefer at most {step_limit} steps if that remains verifiable."
            else:
                step_limit_text = ""
            repair_prompt = (
                f"{phase}_JSON_REPAIR\n"
                f"The previous response could not be parsed as JSON: {exc}\n"
            "Return one valid compact JSON object only. Do not use markdown fences. "
            "Do not include analysis or <think> text. If the previous plan was too long, "
                "merge related tasks into the smallest independently verifiable set of steps. Keep refined_requirements "
                "to at most 8 short strings, assumptions to at most 5 short strings, and open_questions "
                "to at most 3 entries. If the previous "
                "implementation tried to write too many files, return exactly one small file in "
                "this repair; the feedback loop can request the rest later. Keep validation "
                "commands concise, runnable in the project workspace, terminating, and assertion-based. "
                "Do not use python -m http.server by itself as validation. Avoid inline python -c, "
                "f-strings, braces, list comprehensions, and quote-heavy snippets in JSON command strings; "
                "prefer simple argv checks or a generated validation script. Per-attempt file limits are "
                "not plan-step limits. For expected failure-path checks, use expected_returncode or a "
                "wrapper assertion that verifies the non-zero code and error text." + step_limit_text + "\n\n"
                f"Required contract:\n{contract}\n\n"
                f"Previous response tail for recovery:\n{tail}"
            )
            if feedback:
                repaired = self._feedback_chat(repair_prompt, temperature=0.0)
            else:
                repaired = self._implementation_chat(
                    repair_prompt,
                    max_tokens=max(self.config.implementation_model.max_tokens, 6144),
                )
            try:
                return extract_json_object(repaired)
            except Exception as repair_exc:
                if feedback:
                    return self._malformed_feedback_fallback(phase, exc, repair_exc)
                if "REQUIREMENTS" not in phase:
                    return self._malformed_implementation_fallback(phase, exc, repair_exc)
                last_chance_prompt = (
                    f"{phase}_MINIMAL_JSON_REPAIR\n"
                    f"The previous repair also failed: {repair_exc}\n"
                    "Return only one small JSON object. No markdown. No thinking text. "
                    "Use max 5 plan steps. Each step must have only: id, title, one-sentence description, "
                    "depends_on, two acceptance_criteria, and one validation_commands entry. "
                    "Use max 6 refined_requirements, max 3 assumptions, max 2 open_questions. "
                    "Use only simple validation commands such as [\"test\", \"-f\", \"index.html\"] "
                    "or [\"python\", \"validate.py\"]. Do not use inline python -c. "
                    "JSON starts with { and ends with }.\n\n"
                    f"Required contract:\n{contract}"
                )
                repaired_minimal = self._implementation_chat(last_chance_prompt, max_tokens=4096)
                return extract_json_object(repaired_minimal)

    def _malformed_feedback_fallback(
        self,
        phase: str,
        parse_error: Exception,
        repair_error: Exception,
    ) -> dict[str, Any]:
        """Convert an unparseable reviewer turn into bounded actionable feedback.

        Some local models occasionally drift into half-JSON or repeated analysis
        even during the repair turn. Crashing at that point loses a long run.
        This deterministic fallback keeps the transcript honest: it records that
        the reviewer failed to produce usable JSON and asks the implementation
        side for a smaller, more easily reviewed next attempt.
        """
        summary = f"{phase} reviewer response was malformed after JSON repair."
        return {
            "status": "needs_rework",
            "needs_rework": True,
            "summary": summary,
            "required_changes": [
                "Retry the current step with a smaller directly verifiable change.",
                "Provide concise test evidence so the next review does not depend on long free-form reasoning.",
            ],
            "verification_evidence": [
                "Harness parser could not extract valid reviewer JSON from the original or repair response."
            ],
            "parse_error": str(parse_error),
            "repair_error": str(repair_error),
        }

    def _malformed_implementation_fallback(
        self,
        phase: str,
        parse_error: Exception,
        repair_error: Exception,
    ) -> dict[str, Any]:
        """Return a no-op implementation payload when the model cannot emit JSON.

        The feedback loop can then reject the empty attempt through the normal
        git/evidence gates instead of terminating the whole run.
        """
        return {
            "plan_note": f"{phase} implementation response was malformed after JSON repair.",
            "files": [],
            "commands": [],
            "test_evidence": [],
            "resolution_request": "none",
            "parse_error": str(parse_error),
            "repair_error": str(repair_error),
        }

    def run(self) -> dict:
        self.initialize()
        research_result = self._web_research_phase()
        req_result = self._requirements_refinement_phase()
        plan_result = self._plan_validation_phase()
        git_baseline = self._git_baseline_commit()
        step_results: list[dict[str, Any]] = []
        for step in self.plan_steps:
            step_results.append(self._implementation_loop_for_step(step))
            self._write_plan_doc()
            if step_results[-1]["status"] == "cannot_resolve" and self.config.resolution_policy.stop_on_cannot_resolve:
                break
        final_review = self._final_review_phase(step_results)
        git_finalize = self._git_finalize_policy()
        active_transcript_md = self.state_dir / "conversation.md"
        full_transcript_md = self.state_dir / "conversation.full.md"
        self.conversation.write_markdown(active_transcript_md)
        self.conversation.write_markdown(full_transcript_md, full=True)
        summary = {
            "workspace": str(self.workspace),
            "transcript_jsonl": ".agent_state/conversation.full.jsonl",
            "transcript_markdown": ".agent_state/conversation.full.md",
            "active_transcript_jsonl": ".agent_state/conversation.jsonl",
            "active_transcript_markdown": ".agent_state/conversation.md",
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
        in the configured research file, injects a compact version into later prompts, and lets
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
        self._research_path().write_text(research_to_markdown(result), encoding="utf-8")
        self.conversation.append("user", "WEB_RESEARCH_TOOL_RESULT:\n" + json.dumps(result, indent=2))
        if result.get("requested"):
            self._append_plan_note(
                f"[research] {result.get('status')}: web research evidence written to {self.config.runtime.research_file}"
            )
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
                "Return compact JSON: short strings, no markdown, no <think> text. Validation commands "
                "must be terminating commands or scripts that assert behavior. Do not use python -m "
                "http.server by itself as a validation command; browser checks should be wrapped in a "
                "script that starts a server, interacts or inspects, writes evidence, and exits.\n"
                "If the user's requested step count conflicts with verifiable implementation, record "
                "that conflict as an assumption and choose the smallest feasible verifiable plan. "
                "Do not reinterpret per-attempt file-count guidance as a one-file-per-plan-step rule.\n"
                f"{self._default_quality_instruction()}\n"
                f"Web research evidence: {compact_research_for_prompt(self.web_research_result)}\n"
                "If web research status is completed or partial, use those findings in the requirements and plan; "
                "the first research/structure step must cite the source URLs in generated project notes. "
                "If web research is skipped or disabled, record available-knowledge notes instead and do not invent URLs.\n"
                f"Extra context: {extra_context or 'none'}\n\n{REQUIREMENTS_CONTRACT}"
            )
            raw = self._implementation_chat(prompt, max_tokens=max(self.config.implementation_model.max_tokens, 4096))
            try:
                latest = self._extract_json_or_retry(
                    raw,
                    phase="REQUIREMENTS_REFINEMENT_PHASE",
                    contract=REQUIREMENTS_CONTRACT,
                )
            except Exception as exc:
                latest = {
                    "project_summary": "Requirements refinement failed to return parseable JSON.",
                    "refined_requirements": [
                        "The implementation model must retry with smaller, valid JSON before implementation can start."
                    ],
                    "assumptions": [f"Requirements parse failure recorded: {exc}"],
                    "open_questions": [],
                    "planning_confirmation": {
                        "is_feasible": False,
                        "is_clear": False,
                        "is_verifiable": False,
                        "verification_strategy": "",
                        "remaining_risks": ["No validated requirements or plan yet."],
                    },
                    "plan": [],
                    "parse_error": str(exc),
                }
            self.requirements = latest
            self.plan_steps = normalize_plan_steps(latest.get("plan", []))
            for step in self.plan_steps:
                step.setdefault("status", "pending")
            self._write_requirements_doc()
            self._write_plan_doc()
            review = self._requirements_review(index, latest)
            iterations.append({"iteration": index, "requirements": latest, "review": review})
            if self._status(review) == "resolved":
                self._write_requirements_doc(review)
                self._append_plan_note(f"[requirements] resolved after iteration {index}: {review.get('summary', '')}")
                return {"status": "resolved", "iterations": iterations}
            self.conversation.append(
                "user",
                "REQUIREMENTS_REWORK_DIRECTIVE:\nRevise requirements using this review:\n"
                + json.dumps(self._compact_review_for_transcript(review), indent=2),
            )
        fallback = self._fallback_resolution("requirements", review)
        self.requirements.setdefault("assumptions", []).append(fallback["note"])
        self._write_requirements_doc(review)
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
            "If constraints conflict, request a clear compromise instead of repeatedly enforcing both sides "
            "of an impossible constraint. Per-attempt output-size guidance is not a plan-step limit.\n"
            "If default quality policy applies, reject requirements that omit project structure, tests, documentation, "
            "or the initial research/structure planning step.\n"
            "If WEB_RESEARCH_TOOL_RESULT has completed or partial sources, reject requirements that ignore those sources. "
            "If web research is skipped or disabled, do not require cited source URLs; request available-knowledge notes instead.\n"
            + json.dumps(prompt),
            temperature=0.1,
        )
        review = self._extract_json_or_retry(
            raw,
            phase="REQUIREMENTS_REVIEW_PHASE",
            contract='{"status":"resolved|needs_rework|needs_requirements_change|cannot_resolve|skipped_with_note","needs_rework":true,"summary":"short review","required_changes":["specific change"]}',
            feedback=True,
        )
        return self._normalize_review(review)

    def _plan_validation_phase(self) -> dict:
        """Block implementation until the ordered plan is executable and checkable."""
        iterations: list[dict[str, Any]] = []
        review: dict[str, Any] = {}
        for index in range(1, self.config.phases.plan_validation.max_iterations + 1):
            review = self._plan_validation_review(index)
            iterations.append({"iteration": index, "review": review, "plan": self.plan_steps})
            if self._status(review) == "resolved":
                self._append_plan_note(f"[plan] validated after iteration {index}: {review.get('summary', '')}")
                self._write_plan_doc()
                return {"status": "resolved", "iterations": iterations}
            refined = self._plan_refinement_pass(index, review)
            iterations[-1]["refinement"] = refined
        fallback = self._fallback_resolution("plan", review)
        self.plan_notes.append(fallback["note"])
        self._write_plan_doc()
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
                "validation commands terminate and assert behavior instead of starting a server forever",
                "browser/UI steps have executable browser evidence such as Playwright, screenshots, or a validation report when web interaction tools are enabled",
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
            "and verifiable. If any step cannot be independently verified, return needs_plan_change. "
            "A command that only starts an HTTP server is not validation. Treat step-count limits as "
            "hard only when the user explicitly says hard/strict/exactly/must; otherwise prefer the "
            "smallest feasible verifiable plan. Per-attempt file-count guidance is not a plan-step limit.\n"
            + json.dumps(prompt),
            temperature=0.1,
        )
        review = self._normalize_review(self._extract_json_or_retry(
            raw,
            phase="PLAN_VALIDATION_PHASE",
            contract='{"status":"resolved|needs_plan_change|needs_requirements_change|cannot_resolve","needs_rework":true,"summary":"short review","required_changes":["specific change"]}',
            feedback=True,
        ))
        if structural_findings:
            existing = [str(item) for item in review.get("required_changes", [])]
            review["required_changes"] = existing + [item for item in structural_findings if item not in existing]
            if self._status(review) == "resolved":
                review["status"] = "needs_plan_change"
                review["needs_rework"] = True
                review["summary"] = "Deterministic plan checks found unresolved structural validation issues."
        return review

    def _plan_refinement_pass(self, index: int, review: dict[str, Any]) -> dict:
        """Let the implementation model repair the plan while preserving context."""
        prompt = (
            f"PLAN_REFINEMENT_PHASE iteration={index}\n"
            "Revise only the ordered plan so every step is distinct, sequential, and verifiable. "
            "Keep requirements unless the review explicitly says they must change.\n"
            "Return only the compact plan/refined planning confirmation contract below; do not repeat "
            "the full requirements list. Validation commands must be scripts/commands that exit and "
            "assert behavior. Do not use python -m http.server by itself.\n"
            f"Requirements summary: {self._requirements_summary_for_prompt()}\n"
            f"Current plan: {json.dumps(self.plan_steps)}\n"
            f"Web research evidence: {compact_research_for_prompt(self.web_research_result)}\n"
            f"Review: {json.dumps(self._compact_review_for_transcript(review))}\n\n{PLAN_REFINEMENT_CONTRACT}"
        )
        raw = self._implementation_chat(prompt)
        try:
            payload = self._extract_json_or_retry(
                raw,
                phase="PLAN_REFINEMENT_PHASE",
                contract=PLAN_REFINEMENT_CONTRACT,
            )
        except Exception as exc:
            payload = {
                "plan": self.plan_steps,
                "parse_error": str(exc),
                "planning_confirmation": self.requirements.get("planning_confirmation", {}),
            }
            self.plan_notes.append(
                "Plan refinement output could not be parsed after repair; keeping previous plan for another review."
            )
        if payload.get("refined_requirements"):
            self.requirements = payload
        elif isinstance(payload.get("planning_confirmation"), dict):
            self.requirements["planning_confirmation"] = payload["planning_confirmation"]
        self.plan_steps = normalize_plan_steps(payload.get("plan", self.plan_steps))
        self.plan_notes.append(f"Plan refined after review iteration {index}.")
        self._write_plan_doc()
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
                self._append_plan_note(f"[{step['id']}] resolved: {summary}")
                self._write_plan_doc()
                attempts[-1]["git_commit"] = self._git_commit_completed_step(step)
                return {"step_id": step["id"], "status": "resolved", "attempts": attempts}
            if status == "needs_plan_change":
                self._plan_refinement_pass(attempt, review)
            elif status == "needs_requirements_change":
                self._requirements_refinement_phase(extra_context=json.dumps(self._compact_review_for_transcript(review)))
            elif status == "cannot_resolve":
                step["status"] = "cannot_resolve"
                self._append_plan_note(f"[{step['id']}] cannot resolve: {summary}")
                return {"step_id": step["id"], "status": "cannot_resolve", "attempts": attempts}
            if same_error_count >= self.config.resolution_policy.max_same_error_repeats:
                self._append_plan_note(
                    f"[{step['id']}] repeated review pattern in {review_mode} mode; continuing because retry budget is bounded.",
                )
            self.conversation.append(
                "user",
                "NEXT_IMPLEMENTATION_DIRECTIVE:\nApply this step review in the next attempt. "
                "Keep previous requirements, plan validation, and this step context in mind:\n"
                + json.dumps(self._compact_review_for_transcript(review), indent=2),
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
            "Do not stage or commit with git. The harness owns git add/commit after feedback accepts a step. "
            "You may run read-only git commands such as git status or git diff for your own evidence.\n"
            f"Do not rewrite {self.config.runtime.plan_file} just to mark the current step complete; put progress in plan_note. "
            f"The harness appends notes and marks resolved after feedback accepts the step. Only edit {self.config.runtime.plan_file} "
            "when the feedback request specifically requires substantive plan content changes.\n"
            "Keep this attempt small and parseable: write exactly one meaningful file, or two tiny files only if they "
            "are inseparable. If feedback requested several changes, choose the single most blocking file, note what "
            "remains, and let the next feedback iteration request the next file. Do not emit a full app dump.\n"
            f"Requirements summary: {self._requirements_summary_for_prompt()}\n"
            f"Validated plan step ids: {[step.get('id') for step in self.plan_steps]}\n"
            f"Current step: {json.dumps(step)}\n\n{IMPLEMENTATION_CONTRACT}"
        )
        if self._looks_like_browser_step(step):
            prompt += "\n" + self._browser_validation_guidance()
        if self._has_completed_research():
            prompt += (
                "\nWEB_RESEARCH_USAGE_REQUIREMENT:\n"
                f"Use this fetched research evidence and cite source URLs in ARCHITECTURE.md or the relevant deliverable: "
                f"{compact_research_for_prompt(self.web_research_result)}\n"
            )
        raw = self._implementation_chat(prompt)
        try:
            payload = self._extract_json_or_retry(
                raw,
                phase="IMPLEMENT_PLAN_STEP_PHASE",
                contract=IMPLEMENTATION_CONTRACT,
            )
        except Exception as exc:
            payload = {
                "plan_note": (
                    "Implementation output could not be parsed as JSON after repair. "
                    "No files were written; next attempt must return a much smaller valid JSON payload."
                ),
                "files": [],
                "commands": [],
                "test_evidence": [],
                "resolution_request": "none",
                "parse_error": str(exc),
                "raw_tail": raw[-2000:],
            }
        written = write_files(self.workspace, payload.get("files", []))
        command_results = []
        if self.config.mcp_tools.terminal:
            command_results = run_commands(
                self.workspace,
                payload.get("commands", []),
                self.config.runtime.command_timeout_seconds,
                self.config.runtime.max_command_timeout_seconds,
                output_limit_chars=self.config.context_compaction.tool_output_max_chars,
            )
        note = payload.get("plan_note") or f"{step['id']} attempt {attempt} implementation pass completed."
        self._append_plan_note(f"[{step['id']} attempt {attempt}] {note}")
        return {"written": written, "commands": command_results, "raw": payload}

    def _step_review_pass(
        self,
        step: dict[str, Any],
        attempt: int,
        implementation: dict[str, Any],
        review_mode: str,
    ) -> dict:
        """Critique one step using reviewer-owned file and command evidence."""
        plan_text = self._plan_path().read_text(encoding="utf-8")
        feedback_tool_evidence = self._step_feedback_tool_evidence(step)
        evidence_findings = self._evidence_findings(step, implementation, feedback_tool_evidence)
        evidence_findings.extend(self._research_usage_findings(step, feedback_tool_evidence))
        prompt = {
            "phase": "STEP_REVIEW_PHASE",
            "step": step,
            "attempt": attempt,
            "review_mode": review_mode,
            "requirements": self._requirements_summary_for_prompt(),
            "web_research_evidence": self.web_research_result,
            "plan": self._compact_plan_for_prompt(),
            "plan_file_excerpt": plan_text[-5000:],
            "implementation": self._compact_implementation_for_prompt(implementation),
            "feedback_tool_evidence": self._compact_step_evidence_for_prompt(feedback_tool_evidence),
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
                "Untracked meaningful paths are valid pre-acceptance implementation evidence; the harness will stage and commit after acceptance.",
                "If git meaningful_changed_paths is empty for an implementation step, request the missing change and name the current plan requirement.",
                "Validation-only steps may have no git diff when reviewer-owned validation commands pass; do not reject those solely for an empty working tree.",
                "Do not ask the implementation agent to run git add or git commit; repository mutation is harness-owned.",
                f"Do not require the implementation agent to pre-mark the current step completed in {self.config.runtime.plan_file}; the harness marks resolved after acceptance.",
                "Do not accept a step just because the implementation agent claims tests passed.",
                "Reject validation that is too shallow for the requirement; require evidence that exercises the feature from the user's perspective.",
                "For negative-path behavior, prefer wrapper commands that assert return code and error text, or commands with expected_returncode set.",
                "If web_research_evidence has completed sources, confirm the generated work actually cites and applies those source URLs.",
                "If test evidence is absent in hard_pushback mode, return needs_rework.",
                "If evidence remains imperfect in compromise mode, either return needs_rework with a small bounded fix or resolved_with_compromise/skipped_with_note with an explicit diluted requirement note.",
                "For browser/game work, prefer Playwright-style interaction evidence and screenshot/report artifacts when configured.",
                "Do not request package/browser installation inside generated validation scripts; request bounded browser launch timeouts and graceful fallback instead.",
                "In compromise mode, accept a clearly labelled non-browser fallback only when browser launch cannot be made reliable and the fallback still gives concrete evidence.",
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
        raw = self._feedback_chat_with_compact_context(
            "STEP_REVIEW_PHASE\n"
            f"Review mode: {review_mode}. Critically verify exactly one plan step. "
            "Use the whole transcript to avoid repeating old mistakes, but judge only the current step "
            "against its acceptance criteria and test evidence.\n"
            + json.dumps(prompt),
            context_note=(
                "The full multi-turn transcript is stored in .agent_state/conversation.full.jsonl. "
                "Use this compact step-review payload plus reviewer-owned validation reruns. "
                "If the compact evidence shows failed commands, missing files, or no meaningful git diff, "
                "request concrete implementation changes instead of accepting the step. Do not request git add/commit."
            ),
            temperature=0.1,
        )
        review = self._normalize_review(self._extract_json_or_retry(
            raw,
            phase="STEP_REVIEW_PHASE",
            contract='{"status":"resolved|needs_rework|cannot_resolve|needs_requirements_change|needs_plan_change|skipped_with_note","needs_rework":true,"summary":"short review","required_changes":["specific change"]}',
            feedback=True,
        ))
        review = self._enforce_evidence_policy(review, evidence_findings, review_mode)
        review["feedback_tool_evidence"] = feedback_tool_evidence
        review["deterministic_evidence_findings"] = evidence_findings
        self._append_plan_note(f"[{step['id']} attempt {attempt}] review: {review.get('summary', 'no summary')}")
        return review

    def _final_review_phase(self, step_results: list[dict[str, Any]]) -> dict:
        """Run whole-project review after individual plan steps complete."""
        iterations: list[dict[str, Any]] = []
        for attempt in range(1, self.config.review_policy.final_review_iterations + 1):
            review = self._final_project_review(attempt, step_results)
            item: dict[str, Any] = {"attempt": attempt, "review": review}
            if self._status(review) in {"resolved", "resolved_with_compromise", "skipped_with_note"}:
                self._append_plan_note(f"[final review] resolved: {review.get('summary', '')}")
                self._write_plan_doc()
                item["git_commit"] = self._git_commit_final_review()
                iterations.append(item)
                return {"status": self._status(review), "iterations": iterations}
            correction = self._final_correction_pass(attempt, review)
            item["correction"] = correction
            iterations.append(item)
        fallback = self._fallback_resolution("final review", iterations[-1]["review"] if iterations else {})
        self._append_plan_note(f"[final review] {fallback['status']}: {fallback['note']}")
        return {"status": fallback["status"], "iterations": iterations, "resolution": fallback}

    def _final_project_review(self, attempt: int, step_results: list[dict[str, Any]]) -> dict:
        feedback_tool_evidence = self._final_feedback_tool_evidence()
        evidence_findings = self._project_evidence_findings(step_results, feedback_tool_evidence)
        prompt = {
            "phase": "FINAL_PROJECT_REVIEW_PHASE",
            "attempt": attempt,
            "requirements": self._requirements_summary_for_prompt(),
            "plan": self._compact_plan_for_prompt(),
            "step_results": self._compact_step_results_for_prompt(step_results),
            "feedback_tool_evidence": self._compact_final_evidence_for_prompt(feedback_tool_evidence),
            "deterministic_evidence_findings": evidence_findings,
            "expected_json": {
                "status": "resolved|needs_rework|cannot_resolve|needs_requirements_change|needs_plan_change|skipped_with_note|resolved_with_compromise",
                "needs_rework": True,
                "summary": "whole project review",
                "required_changes": ["specific final change"],
                "verification_evidence": ["evidence reviewed"],
            },
        }
        raw = self._feedback_chat_with_compact_context(
            "FINAL_PROJECT_REVIEW_PHASE\n"
            "Review the whole project after all plan steps. Re-check original requirements, final files, "
            "and all test evidence. Push back if the project lacks proof or contradicts requirements.\n"
            + json.dumps(prompt),
            context_note=(
                "The full multi-turn transcript is stored in .agent_state/conversation.full.jsonl. "
                "Use this compact final-review payload plus reviewer-owned validation reruns to decide. "
                "All individual plan steps were reviewed before this final pass."
            ),
            temperature=0.1,
        )
        review = self._normalize_review(self._extract_json_or_retry(
            raw,
            phase="FINAL_PROJECT_REVIEW_PHASE",
            contract='{"status":"resolved|needs_rework|cannot_resolve|needs_requirements_change|needs_plan_change|skipped_with_note|resolved_with_compromise","needs_rework":true,"summary":"whole project review","required_changes":["specific final change"]}',
            feedback=True,
        ))
        if evidence_findings and self._status(review) == "resolved":
            review["status"] = "needs_rework"
            review["needs_rework"] = True
            review["summary"] = "Final review cannot resolve because deterministic evidence checks found gaps."
            review["required_changes"] = evidence_findings
        review["feedback_tool_evidence"] = feedback_tool_evidence
        review["deterministic_evidence_findings"] = evidence_findings
        return review

    def _compact_plan_for_prompt(self) -> list[dict[str, Any]]:
        """Summarize the plan without embedding large command/file payloads."""
        compact: list[dict[str, Any]] = []
        for step in self.plan_steps:
            compact.append({
                "id": step.get("id"),
                "title": step.get("title"),
                "status": step.get("status"),
                "acceptance_criteria": step.get("acceptance_criteria", [])[:4],
                "validation_command_count": len(step.get("validation_commands") or []),
            })
        return compact

    def _compact_step_results_for_prompt(self, step_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep final review context small enough for local-model endpoints.

        Full step results contain nested file contents, git diffs, and repeated
        transcripts. They are useful on disk, but sending all of that back to a
        local model can exceed context or HTTP limits. Final review only needs a
        terse trail of statuses and evidence summaries because reviewer-owned
        validations are re-run separately.
        """
        compact: list[dict[str, Any]] = []
        for result in step_results:
            attempts = result.get("attempts", [])
            last_attempt = attempts[-1] if attempts else {}
            review = last_attempt.get("review", {})
            implementation = last_attempt.get("implementation", {})
            compact.append({
                "step_id": result.get("step_id"),
                "status": result.get("status"),
                "attempt_count": len(attempts),
                "written_paths": implementation.get("written", []),
                "last_review_status": review.get("status"),
                "last_review_summary": review.get("summary"),
                "test_evidence": (implementation.get("raw") or {}).get("test_evidence", []),
            })
        return compact

    def _compact_command_results_for_prompt(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trim command results for prompts while preserving pass/fail signals."""
        compact: list[dict[str, Any]] = []
        limit = max(2000, min(self.config.context_compaction.tool_output_max_chars, 12000))
        for result in results:
            stdout = str(result.get("stdout", ""))
            stderr = str(result.get("stderr", ""))
            compact.append({
                "command": result.get("command"),
                "timeout_seconds": result.get("timeout_seconds"),
                "returncode": result.get("returncode"),
                "expected_returncode": result.get("expected_returncode"),
                "returncode_matches_expected": result.get("returncode_matches_expected"),
                "timed_out": result.get("timed_out"),
                "stdout": self._prompt_excerpt(stdout, limit),
                "stderr": self._prompt_excerpt(stderr, limit),
                "stdout_prompt_truncated": len(stdout) > limit,
                "stderr_prompt_truncated": len(stderr) > limit,
            })
        return compact

    def _compact_implementation_for_prompt(self, implementation: dict[str, Any]) -> dict[str, Any]:
        """Summarize one implementation attempt without echoing huge raw JSON."""
        raw = implementation.get("raw") or {}
        return {
            "written": implementation.get("written", []),
            "commands": self._compact_command_results_for_prompt(implementation.get("commands", [])),
            "plan_note": raw.get("plan_note"),
            "test_evidence": raw.get("test_evidence", []),
            "resolution_request": raw.get("resolution_request"),
            "parse_error": raw.get("parse_error"),
        }

    def _compact_step_evidence_for_prompt(self, evidence: dict[str, Any]) -> dict[str, Any]:
        """Summarize reviewer-owned step evidence for local-model context limits."""
        files = []
        for item in evidence.get("workspace_files", []):
            files.append(self._compact_file_for_prompt(item, default_limit=1800))
        git = evidence.get("git") or {}
        return {
            "kind": evidence.get("kind"),
            "step_id": evidence.get("step_id"),
            "workspace_files": files,
            "validation_commands": evidence.get("validation_commands", []),
            "validation_results": self._compact_command_results_for_prompt(evidence.get("validation_results", [])),
            "git": {
                "enabled": git.get("enabled"),
                "head": git.get("head"),
                "status_short": git.get("status_short"),
                "meaningful_changed_paths": git.get("meaningful_changed_paths"),
                "diff_stat": git.get("diff_stat"),
                "diff_excerpt": str(git.get("diff", ""))[:2000],
            },
        }

    def _compact_final_evidence_for_prompt(self, evidence: dict[str, Any]) -> dict[str, Any]:
        """Summarize final tool evidence without dumping full file contents."""
        files = []
        for item in evidence.get("workspace_files", []):
            files.append(self._compact_file_for_prompt(item, default_limit=1200))
        validations = []
        for validation in evidence.get("step_validations", []):
            results = []
            for result in validation.get("validation_results", []):
                results.append({
                    "command": result.get("command"),
                    "returncode": result.get("returncode"),
                    "expected_returncode": result.get("expected_returncode"),
                    "returncode_matches_expected": result.get("returncode_matches_expected"),
                    "timed_out": result.get("timed_out"),
                    "stdout": self._prompt_excerpt(str(result.get("stdout", "")), 2000),
                    "stderr": self._prompt_excerpt(str(result.get("stderr", "")), 2000),
                    "stdout_prompt_truncated": len(str(result.get("stdout", ""))) > 2000,
                    "stderr_prompt_truncated": len(str(result.get("stderr", ""))) > 2000,
                })
            validations.append({
                "step_id": validation.get("step_id"),
                "validation_commands": validation.get("validation_commands"),
                "validation_results": results,
            })
        git = evidence.get("git") or {}
        return {
            "kind": evidence.get("kind"),
            "workspace_files": files,
            "step_validations": validations,
            "git": {
                "enabled": git.get("enabled"),
                "head": git.get("head"),
                "status_short": git.get("status_short"),
                "meaningful_changed_paths": git.get("meaningful_changed_paths"),
                "diff_stat": git.get("diff_stat"),
            },
        }

    def _compact_file_for_prompt(self, item: dict[str, Any], *, default_limit: int) -> dict[str, Any]:
        """Prepare a workspace file for reviewer prompts without false truncation signals.

        The workspace snapshot records whether the source file itself was
        truncated. Reviewer prompts may need a smaller excerpt. Keep those two
        facts separate so the feedback model does not misread a prompt-budget
        excerpt as missing test evidence.
        """
        content = str(item.get("content", ""))
        path = str(item.get("path", ""))
        important = path.endswith((
            "validation_evidence.log",
            "results.json",
            "report.json",
            "playwright-report.json",
        ))
        limit = 12000 if important else default_limit
        prompt_truncated = len(content) > limit
        return {
            "path": path,
            "size": item.get("size", len(content.encode("utf-8"))),
            "source_truncated": item.get("truncated", False),
            "prompt_truncated": prompt_truncated,
            "content": self._prompt_excerpt(content, limit),
        }

    def _prompt_excerpt(self, text: str, limit: int) -> str:
        """Return text for a model prompt with an explicit middle truncation marker."""
        if len(text) <= limit:
            return text
        head = max(1, limit // 2)
        tail = max(1, limit - head)
        return (
            text[:head]
            + f"\n\n[prompt payload truncated: kept first {head} and last {tail} chars]\n\n"
            + text[-tail:]
        )

    def _compact_review_for_transcript(self, review: dict[str, Any]) -> dict[str, Any]:
        """Return review data safe to paste back into the live chat.

        Full review objects can contain reviewer-owned file snapshots, command
        output, and git diffs. Those are kept in `summary.json`, but the next
        implementation turn only needs the decision, concrete requests, and a
        bounded evidence summary. This is the main guard against one verbose
        tool call overflowing the next model context.
        """
        keep_keys = (
            "status",
            "needs_rework",
            "summary",
            "required_changes",
            "cross_check_questions",
            "verification_evidence",
            "compromise_note",
            "planning_confirmation",
        )
        compact = {key: review[key] for key in keep_keys if key in review}
        if "deterministic_evidence_findings" in review:
            compact["deterministic_evidence_findings"] = review["deterministic_evidence_findings"]
        evidence = review.get("feedback_tool_evidence")
        if isinstance(evidence, dict):
            if evidence.get("kind") == "final_feedback_tools":
                compact["feedback_tool_evidence_summary"] = self._compact_final_evidence_for_prompt(evidence)
            else:
                compact["feedback_tool_evidence_summary"] = self._compact_step_evidence_for_prompt(evidence)
        as_json = json.dumps(compact, ensure_ascii=False)
        limit = self.config.context_compaction.transcript_review_max_chars
        if len(as_json) <= limit:
            return compact
        return {
            "status": compact.get("status"),
            "needs_rework": compact.get("needs_rework"),
            "summary": compact.get("summary"),
            "required_changes": self._clip_list_for_transcript(compact.get("required_changes", [])),
            "deterministic_evidence_findings": self._clip_list_for_transcript(
                compact.get("deterministic_evidence_findings", [])
            ),
            "review_truncation_note": clamp_text(as_json, limit, marker="review transcript payload truncated"),
        }

    def _clip_list_for_transcript(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            values = [values]
        return [
            clamp_text(str(value), 1200, marker="list item truncated")
            for value in values[:12]
        ]

    def _final_correction_pass(self, attempt: int, review: dict[str, Any]) -> dict:
        prompt = (
            f"FINAL_PROJECT_CORRECTION_PHASE attempt={attempt}\n"
            "Apply only the final review changes needed to make the whole project consistent with requirements. "
            "Include validation commands and test evidence.\n"
            f"Review: {json.dumps(self._compact_review_for_transcript(review))}\n\n{IMPLEMENTATION_CONTRACT}"
        )
        if any(self._looks_like_browser_step(step) for step in self.plan_steps):
            prompt += "\n" + self._browser_validation_guidance()
        raw = self._implementation_chat(prompt)
        payload = self._extract_json_or_retry(
            raw,
            phase="FINAL_PROJECT_CORRECTION_PHASE",
            contract=IMPLEMENTATION_CONTRACT,
        )
        written = write_files(self.workspace, payload.get("files", []))
        command_results = []
        if self.config.mcp_tools.terminal:
            command_results = run_commands(
                self.workspace,
                payload.get("commands", []),
                self.config.runtime.command_timeout_seconds,
                self.config.runtime.max_command_timeout_seconds,
                output_limit_chars=self.config.context_compaction.tool_output_max_chars,
            )
        self._append_plan_note(f"[final correction attempt {attempt}] {payload.get('plan_note', 'completed')}")
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
        step_limit, limit_is_hard = self._configured_plan_step_limit()
        if step_limit and limit_is_hard and len(self.plan_steps) > step_limit:
            findings.append(
                f"Plan has {len(self.plan_steps)} steps but the project prompt has a hard limit of at most {step_limit}."
            )
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
            findings.extend(self._validation_command_findings(step))
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
            research_present = any(marker in first_text for marker in ("research", "patterns", "knowledge", "investigate"))
            structure_present = any(marker in first_text for marker in ("structure", "architecture", "dependencies", "module"))
            # Existing-project repair work often starts with architecture mapping
            # rather than a greenfield "plan the structure" step. Treat mapping
            # or dependency analysis as satisfying the planning intent: the agent
            # has to inspect the current shape before changing it.
            planning_present = any(
                marker in first_text
                for marker in ("plan", "order", "mapping", "map", "architecture", "dependencies")
            )
            if not (research_present and structure_present and planning_present):
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

    def _configured_plan_step_limit(self) -> tuple[int | None, bool]:
        prompt = self.config.project_design.prompt.lower()
        match = re.search(r"at most\s+(\d+)\s+(?:independently\s+)?verifiable\s+steps", prompt)
        if not match:
            match = re.search(r"at most\s+(\d+)\s+steps", prompt)
        if not match:
            return None, False
        start = max(match.start() - 120, 0)
        end = min(match.end() + 80, len(prompt))
        surrounding = prompt[start:end]
        hard_markers = ("hard", "strict", "exactly", "must", "do not exceed", "never exceed")
        return int(match.group(1)), any(marker in surrounding for marker in hard_markers)

    def _requirements_summary_for_prompt(self) -> str:
        if not isinstance(self.requirements, dict):
            return "No requirements available."
        summary = str(self.requirements.get("project_summary") or self.requirements.get("summary") or "")
        items = [str(item) for item in self.requirements.get("refined_requirements", [])[:8]]
        return json.dumps({"summary": summary, "key_requirements": items})

    def _validation_command_findings(self, step: dict[str, Any]) -> list[str]:
        findings: list[str] = []
        commands = step.get("validation_commands") or []
        command_text = json.dumps(commands).lower()
        step_id = str(step.get("id") or "step")
        for command in commands:
            if isinstance(command, dict):
                if command.get("manual_test"):
                    findings.append(
                        f"{step_id} uses manual_test metadata in validation_commands; replace it with an executable script/report command."
                    )
                parts = [str(part).lower() for part in (command.get("cmd") or command.get("command") or [])]
            else:
                parts = [str(part).lower() for part in command]
            joined = " ".join(parts)
            if "python -m http.server" in joined or (
                len(parts) >= 3 and parts[0].endswith("python") and parts[1] == "-m" and parts[2] == "http.server"
            ):
                findings.append(
                    f"{step_id} validation starts an HTTP server but does not assert behavior; wrap server startup in a validation script that exits."
                )
            if self._looks_like_unwrapped_expected_failure_validation(step, command, parts):
                findings.append(
                    f"{step_id} validation appears to test an expected failure path without declaring expected_returncode "
                    "or wrapping the exception assertion. Replace it with a command object using expected_returncode, "
                    "or a small wrapper command/script that exits 0 only when the expected error occurs."
                )
        return findings

    def _command_expected_returncode(self, command: list[Any] | dict[str, Any]) -> int:
        if isinstance(command, dict):
            return int(command.get("expected_returncode", 0))
        return 0

    def _looks_like_unwrapped_expected_failure_validation(
        self,
        step: dict[str, Any],
        command: list[Any] | dict[str, Any],
        parts: list[str],
    ) -> bool:
        """Catch plan commands that prove errors by accidentally failing.

        Local models often write a correct acceptance criterion such as
        "raises ValueError on empty input" and then add `python -c mean([])` as
        the validation command. That command will fail forever under the
        reviewer-owned validation pass because plain argv-list commands expect
        return code 0. The plan must either declare `expected_returncode` or use
        a wrapper assertion that returns 0 only when the expected exception is
        observed.
        """
        if self._command_expected_returncode(command) != 0:
            return False
        if len(parts) < 3 or not parts[0].endswith("python") or parts[1] != "-c":
            return False
        step_text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", [])),
        ]).lower()
        if not any(marker in step_text for marker in ("raise", "raises", "exception", "error", "empty", "invalid")):
            return False
        code = parts[2].lower()
        if any(marker in code for marker in ("try:", "except", "raises", "assertraises", "pytest.raises", "returncode")):
            return False
        return "[]" in code or "raise " in code or "sys.exit" in code

    def _looks_like_browser_step(self, step: dict[str, Any]) -> bool:
        text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(step.get("acceptance_criteria", [])),
        ]).lower()
        markers = (
            "browser",
            "ui",
            "web",
            "map",
            "click",
            "drag",
            "zoom",
            "pan",
            "render",
            "screenshot",
            "html",
            "css",
            "javascript",
        )
        return any(marker in text for marker in markers)

    def _browser_validation_guidance(self) -> str:
        """Guidance for generated projects that need browser/UI validation."""
        if not self.config.mcp_tools.web_interaction:
            return (
                "BROWSER_VALIDATION_GUIDANCE:\n"
                "Web interaction tools are disabled for this run. Do not claim Playwright/browser evidence. "
                "Use bounded HTTP/content checks or code-level checks and label them as non-browser fallback evidence.\n"
            )
        return (
            "BROWSER_VALIDATION_GUIDANCE:\n"
            "The agent Docker image already includes Python Playwright and Chromium. Generated validation scripts "
            "must not run `playwright install`, `apt install`, `npm install`, or `pip install`; those can hang and "
            "pollute reproducibility. Unless the config explicitly says Node.js is available, assume there is no "
            "`node`, `npm`, `npx`, or `@playwright/test` runner. Prefer a Python validation script that imports "
            "`from playwright.sync_api import sync_playwright` and drives Chromium directly. Set short browser/page "
            "timeouts (for example 10-15 seconds), always close browser contexts and local servers in finally blocks, "
            "and write clear evidence to a log/JSON file. If browser launch fails under those bounded timeouts, fall "
            "back to the most direct non-browser verification available and explicitly label that evidence as a fallback.\n"
        )

    def _default_quality_policy_payload(self) -> dict[str, Any]:
        return {
            "applies": self._default_quality_policy_applies(),
            "assumed_requirement": (
                "Unless the user explicitly says otherwise, code should be well structured, well tested, "
                "well documented, and the first implementation step or first part of the first step should research "
                "required patterns/knowledge, plan project structure, and update the remaining plan if structure "
                "changes the task order. "
                "Cited source URLs are required only when web research fetched sources."
            ),
        }

    def _default_quality_instruction(self) -> str:
        if not self._default_quality_policy_applies():
            return "The user prompt appears to override the default code-quality policy; record that override explicitly."
        return (
            "Default quality policy applies unless the user explicitly says otherwise: add a requirement that the project "
            "is well structured, well tested, and well documented. The first implementation step, or the first part of "
            "the first step when the user requests very few steps, must: "
            "A) research on the web or from available knowledge any required patterns/knowledge, and "
            "B) plan the project structure/architecture and rewrite the remaining plan if that structure changes task order. "
            "Only require cited source URLs when web research actually fetched source URLs; otherwise require available-knowledge notes."
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

        The configured research file is generated by the harness, so it does not count as model
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
            if item.get("path") not in self._harness_doc_names()
        ]
        generated_text = "\n".join(str(item.get("content", "")) for item in generated_files)
        if any(url in generated_text for url in self._research_source_urls()):
            return []
        return [
            "Web research evidence exists but generated project work did not cite/use any researched source URL outside the configured research file."
        ]

    def _git_diff_findings(
        self,
        step: dict[str, Any],
        implementation: dict[str, Any],
        feedback_tool_evidence: dict[str, Any],
    ) -> list[str]:
        """Reject a step when the reviewer has no implementation diff to inspect."""
        if not (self.config.git_policy.enabled and self.config.git_policy.require_step_diff):
            return []
        git = feedback_tool_evidence.get("git") or {}
        changed_paths = git.get("meaningful_changed_paths") or []
        if changed_paths:
            return []
        if self._is_validation_only_step(step) and self._reviewer_validation_passed(feedback_tool_evidence):
            return []
        if self._is_previously_implemented_attempt(implementation) and self._reviewer_validation_passed(feedback_tool_evidence):
            return []
        status_short = str(git.get("status_short") or "")
        step_text = " ".join([
            str(step.get("id", "")),
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", [])),
        ]).lower()
        harness_plan_name = self.config.runtime.plan_file.lower()
        if harness_plan_name in status_short.lower() and harness_plan_name in step_text:
            return []
        step_id = step.get("id", "step")
        requirement = "; ".join(step.get("acceptance_criteria", [])[:2]) or step.get("title", "the planned work")
        return [
            (
                f"Git working tree has no implementation changes for {step_id}. "
                f"Please implement the plan requirement before review can accept it: {requirement}"
            )
        ]

    def _is_previously_implemented_attempt(self, implementation: dict[str, Any]) -> bool:
        """Allow validation of artifacts already created by an earlier step.

        A model can over-complete a file during a skeleton/planning step. If a
        later step explicitly says it is validating already-created work and
        reviewer-owned validation passes, requiring a fresh edit would encourage
        meaningless churn. The feedback model still reviews the file snapshot.
        """
        if implementation.get("written"):
            return False
        raw = implementation.get("raw") if isinstance(implementation.get("raw"), dict) else {}
        text = " ".join([
            str(implementation.get("plan_note", "")),
            str(raw.get("plan_note", "")),
            " ".join(str(item) for item in implementation.get("test_evidence", [])),
            " ".join(str(item) for item in raw.get("test_evidence", [])),
        ]).lower()
        return any(marker in text for marker in ("already implemented", "already created", "already exists"))

    def _is_validation_only_step(self, step: dict[str, Any]) -> bool:
        """Identify QA/checkpoint steps that should not be forced to edit files.

        Most plan steps should leave a meaningful diff. A final integration
        validation step is different: its deliverable is the independently
        rerun command evidence. Without this exception, the feedback loop can
        reject a perfectly good validation step forever because there is
        intentionally no new implementation work to inspect.
        """
        title = str(step.get("title", "")).lower()
        description = str(step.get("description", "")).lower()
        criteria = " ".join(str(item).lower() for item in step.get("acceptance_criteria", []))
        text = " ".join([title, description, criteria])
        validation_markers = (
            "validation",
            "validate",
            "verify",
            "verification",
            "integration",
            "run all tests",
            "all tests pass",
            "end-to-end",
        )
        implementation_markers = (
            "add ",
            "build ",
            "create ",
            "implement ",
            "write ",
            "generate ",
            "exists",
        )
        return any(marker in text for marker in validation_markers) and not any(
            marker in text for marker in implementation_markers
        )

    def _is_failure_investigation_step(self, step: dict[str, Any]) -> bool:
        """Allow failing test commands as evidence for diagnosis-only steps.

        Bug-fix workflows often start with a step whose purpose is to run the
        broken test suite and document the failure. In that narrow phase, a
        non-zero test command is useful evidence instead of an automatic step
        failure, as long as the agent also leaves a meaningful investigation
        artifact for the reviewer.
        """
        text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", [])),
        ]).lower()
        if any(marker in text for marker in ("all tests pass", "final verification", "fix calculation", "fix syntax")):
            return False
        investigation_markers = (
            "investigation",
            "identify failures",
            "failing tests",
            "error messages",
            "document failures",
            "syntax/import blockers",
        )
        return any(marker in text for marker in investigation_markers)

    def _reviewer_validation_passed(self, feedback_tool_evidence: dict[str, Any]) -> bool:
        results = feedback_tool_evidence.get("validation_results") or []
        if not results:
            return False
        for result in results:
            if result.get("timed_out") or not self._command_returncode_matches_expected(result):
                return False
        return True

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
                output_limit_chars=self.config.context_compaction.tool_output_max_chars,
            )
        return {
            "kind": "step_feedback_tools",
            "step_id": step.get("id"),
            "workspace_files": collect_workspace_files(
                self.workspace,
                self.config.context_compaction.workspace_file_max_bytes,
            ),
            "validation_commands": validation_commands,
            "validation_results": validation_results,
            "git": (
                git_evidence(self.workspace, max_diff_chars=self.config.context_compaction.git_diff_max_chars)
                if self.config.git_policy.enabled
                else {"enabled": False}
            ),
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
                    output_limit_chars=self.config.context_compaction.tool_output_max_chars,
                )
            step_validations.append({
                "step_id": step.get("id"),
                "validation_commands": commands,
                "validation_results": results,
            })
        return {
            "kind": "final_feedback_tools",
            "workspace_files": collect_workspace_files(
                self.workspace,
                self.config.context_compaction.workspace_file_max_bytes,
            ),
            "step_validations": step_validations,
            "git": (
                git_evidence(self.workspace, max_diff_chars=self.config.context_compaction.git_diff_max_chars)
                if self.config.git_policy.enabled
                else {"enabled": False}
            ),
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
            if not self._command_returncode_matches_expected(result) and not self._is_failure_investigation_step(step):
                findings.append(
                    f"Feedback validation command returned {result.get('returncode')} but expected "
                    f"{result.get('expected_returncode', 0)}: {result.get('command')}"
                )
                if self._looks_like_malformed_validation_command(result):
                    findings.append(
                        "Plan validation command appears malformed before it can test the project; request a plan change "
                        "with a simpler script or corrected command instead of asking for implementation-only changes."
                    )
                if self._looks_like_unwrapped_expected_failure_result(step, result):
                    findings.append(
                        "Plan validation command appears to run an expected failure path without declaring expected_returncode "
                        "or wrapping the assertion; request a plan change instead of asking for implementation-only changes."
                    )
        for result in implementation_commands:
            if result.get("timed_out"):
                findings.append(f"Implementation command timed out: {result.get('command')}")
            if not self._command_returncode_matches_expected(result) and not self._is_failure_investigation_step(step):
                findings.append(
                    f"Implementation command returned {result.get('returncode')} but expected "
                    f"{result.get('expected_returncode', 0)}: {result.get('command')}"
                )
        findings.extend(self._git_diff_findings(step, implementation, feedback_tool_evidence or {}))
        return findings

    def _command_returncode_matches_expected(self, result: dict[str, Any]) -> bool:
        """Return True when a command produced the intended exit status.

        Negative-path validation is legitimate. For example, argparse should
        exit with code 2 when a required CLI argument is missing. The command
        schema therefore supports expected_returncode so the reviewer can check
        error messages without the deterministic gate treating the test itself
        as failed.
        """
        expected = int(result.get("expected_returncode", 0))
        return int(result.get("returncode", 0)) == expected

    def _looks_like_malformed_validation_command(self, result: dict[str, Any]) -> bool:
        """Detect broken reviewer commands separately from broken project code.

        A failing validation is normally implementation feedback. But a
        `python -c` command that cannot parse at all is often a plan defect: the
        reviewer would keep rerunning the same invalid command forever. In that
        case the next useful action is a plan refinement that replaces the
        command with a small script or simpler assertion.
        """
        command = result.get("command") or []
        command_text = " ".join(str(part) for part in command)
        stderr = str(result.get("stderr") or "")
        return (
            "python -c" in command_text
            and 'File "<string>"' in stderr
            and "SyntaxError" in stderr
        )

    def _looks_like_unwrapped_expected_failure_result(self, step: dict[str, Any], result: dict[str, Any]) -> bool:
        """Identify reviewer failures caused by a bad plan, not bad code."""
        if int(result.get("expected_returncode", 0)) != 0 or int(result.get("returncode", 0)) == 0:
            return False
        command = result.get("command") or []
        command_text = " ".join(str(part) for part in command)
        stderr = str(result.get("stderr") or "")
        if "python -c" not in command_text or "Traceback" not in stderr:
            return False
        step_text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", [])),
        ]).lower()
        if "valueerror" in stderr.lower() and any(marker in step_text for marker in ("valueerror", "raise", "raises", "empty")):
            return True
        return "exception" in step_text or "error" in step_text

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
                step = next((item for item in self.plan_steps if str(item.get("id")) == step_id), {})
                results = validation.get("validation_results", [])
                if validation.get("validation_commands") and not results:
                    findings.append(f"Step {step_id} final feedback validation produced no command evidence.")
                for result in results:
                    if result.get("timed_out"):
                        findings.append(f"Step {step_id} final feedback validation timed out: {result.get('command')}")
                    if not self._command_returncode_matches_expected(result) and not self._is_failure_investigation_step(step):
                        findings.append(
                            f"Step {step_id} final feedback validation returned {result.get('returncode')} "
                            f"but expected {result.get('expected_returncode', 0)}: {result.get('command')}"
                        )
                continue
            if not attempts:
                continue
            implementation = attempts[-1].get("implementation", {})
            commands = implementation.get("commands", [])
            if not commands:
                findings.append(f"Step {step_id} final attempt has no command evidence.")
            for result in commands:
                if result.get("timed_out") or not self._command_returncode_matches_expected(result):
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
        if any("Plan validation command appears" in item for item in evidence_findings):
            existing = [str(item) for item in review.get("required_changes", [])]
            review["status"] = "needs_plan_change"
            review["needs_rework"] = True
            review["summary"] = (
                "Please revise the plan validation command: reviewer-owned evidence shows the command itself is malformed."
            )
            review["required_changes"] = existing + [item for item in evidence_findings if item not in existing]
            return review
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
            return {
                "enabled": True,
                "left_uncommitted": False,
                "git": git_evidence(self.workspace, max_diff_chars=self.config.context_compaction.git_diff_max_chars),
            }
        reset = reset_to_ref(
            self.workspace,
            self.git_baseline_ref,
            mode=self.config.git_policy.final_reset_mode,
        )
        return {
            "enabled": True,
            "left_uncommitted": True,
            "reset": reset,
            "git": git_evidence(self.workspace, max_diff_chars=self.config.context_compaction.git_diff_max_chars),
        }

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
