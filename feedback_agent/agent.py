from __future__ import annotations

import json
from pathlib import Path
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


ANALYSIS_CONTRACT = """
Return strict JSON only:
{
  "problem_restatement": "concise restatement of the user's request",
  "domain_and_constraints": ["important domain fact, constraint, source, environment, or uncertainty"],
  "initial_source_check": {
    "sources_checked": ["workspace file, configured research result, command output, or none"],
    "source_gaps": ["gap or unavailable source"],
    "freshness_risks": ["fact that may need web/tool verification later"]
  },
  "possible_solution_paths": [
    {
      "id": "A",
      "description": "approach summary",
      "advantages": ["why this may work"],
      "risks": ["why this may fail"],
      "verification_strategy": "how this path would be verified"
    }
  ],
  "recommended_path": {
    "path_id": "A",
    "rationale": "why this is the best first approach",
    "fallback_trigger": "what evidence would justify trying another path"
  },
  "analysis_quality": {
    "is_comprehensive": true,
    "is_domain_aware": true,
    "is_actionable_for_planning": true,
    "remaining_unknowns": ["unknown to preserve"]
  }
}
Do not write project files. Do not solve benchmark tasks in the harness. The
purpose is to orient later model-driven planning: restate the request, identify
constraints, name what is possible or impossible, and compare multiple viable
approaches before choosing one. Keep the analysis universal and problem-domain
aware; do not inject instructions that target only one historical failure mode.
Start with `{`, return one JSON object, and stop immediately after the matching
closing `}`.
"""


ANALYSIS_REVIEW_CONTRACT = """
Return strict JSON only:
{
  "status": "resolved|needs_rework|cannot_resolve",
  "needs_rework": false,
  "summary": "review summary",
  "required_changes": ["specific analysis gap to fix"],
  "quality_questions": ["question the next analysis pass must answer"]
}
Reject analysis that jumps straight to a plan, considers only one approach,
ignores available workspace/research/source context, or bakes in a narrow
solution that would make the harness less universal.
"""


APPROACH_REVIEW_CONTRACT = """
Return strict JSON only:
{
  "status": "resolved|try_another_approach|needs_rework|cannot_resolve",
  "needs_rework": false,
  "summary": "whether the executed approach answered the user request",
  "decision": "keep_result|retry_with_new_approach|stop",
  "recommended_next_approach": "only when retrying",
  "evidence_reviewed": ["final review, command evidence, plan state, or transcript fact"],
  "runbook_updates": ["note to preserve for the next approach"]
}
Decide whether the completed workflow was the right response to the user's
request. If another approach is warranted, explain the trigger and provide a
new approach direction. Do not retry merely for variety; retry only when the
evidence shows a meaningful gap, a better angle is needed, or the task itself
requires periodic re-checking.
"""


TOOL_CALL_VERIFICATION_CONTRACT = """
Return strict JSON only:
{
  "status": "approved|blocked|needs_revision",
  "summary": "tool-call verification summary",
  "commands": [
    {
      "index": 0,
      "decision": "approved|blocked",
      "risk_level": "low|medium|high",
      "reason": "why this command is safe or unsafe",
      "safer_alternative": "optional safer command or plan change"
    }
  ]
}
Review each proposed terminal call before the harness executes it. Use the full
chat history, current plan, workspace context, and deterministic findings. Push
back on destructive or misdirected operations, wrong source/destination paths,
malformed quoting, commands that cannot verify the intended behavior, or timeout
requests that are unjustified. Approve only commands that are appropriate for
the current task and bounded by the configured workspace/tool policy.
"""


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
      "title": "task title",
      "description": "what this task changes",
      "depends_on": [],
      "acceptance_criteria": ["verifiable criterion"],
      "validation_commands": [["python", "-m", "unittest", "-v"]]
    }
  ]
}
The plan must be ordered, distinct, and executable one step at a time.
For large projects, group related work into a practical number of high-impact
steps instead of creating many tiny steps. Every step must remain independently
verifiable. Include enough detail for the implementation and feedback agents to
make good decisions later; avoid filler and repetition, but do not omit important
requirements just to make the response shorter.
Do not emit chat-template or fake tool-call markers such as <|channel>,
<tool_call>, call:ls_tool, or similar. This harness cannot execute those
markers; it only runs commands listed inside the parsed JSON `commands` fields
after the response is complete. Start with `{`, return one JSON object, and stop
immediately after the matching closing `}`. Use as much detail as the task
needs, but avoid padding, repetition, or speculative tool-call syntax.
Commands may be argv lists or {"cmd": ["python", "script.py"], "timeout_seconds": 7200}
when one specific tool call legitimately needs longer than the default timeout.
Avoid embedding Python source, f-strings, braces, list comprehensions, or other
quote-heavy snippets inside JSON command strings during planning. Prefer simple
argv checks such as ["test", "-f", "README.md"]. For complex validation, add a
plan step that creates a validation script and then run ["python", "validate.py"].
Do not use `python -m py_compile .` or point `py_compile` at a directory; use
`python -m compileall <dir>` or a small validation script when validating a
whole package.
For expected failure-path tests, use {"cmd": ["python", "-m", "app"], "expected_returncode": 2}
and assert the stderr/stdout message in a small wrapper command when possible.
If a partial-fix step intentionally expects the full test suite to keep failing
(for example "syntax fixed, logic tests still fail"), do not use a plain command
that returns non-zero by accident. Use `expected_returncode` and/or a generated
validation script that asserts the remaining failure is the intended one.
Likewise, if acceptance criteria say failure logs should indicate logic errors,
that is an intentional expected-failure step and the validation command must
declare the expected non-zero return code or wrap the assertion.
"""


PLAN_REFINEMENT_CONTRACT = """
Return strict JSON only:
{
  "planning_confirmation": {
    "is_feasible": true,
    "is_clear": true,
    "is_verifiable": true,
    "verification_strategy": "step-by-step validation strategy",
    "remaining_risks": ["risk"]
  },
  "plan": [
    {
      "id": "S1",
      "title": "task title",
      "description": "description",
      "depends_on": [],
      "acceptance_criteria": ["verifiable criterion"],
      "validation_commands": [["python", "scripts/validate_step.py"]]
    }
  ]
}
Do not repeat the full requirements unless that detail is needed to make the
plan clear. Validation commands must terminate and assert behavior. Do not use
`python -m http.server` by itself; wrap any server startup inside a script that
performs checks and exits.
Do not embed inline Python source in JSON command strings. Prefer simple argv
checks such as ["test", "-f", "index.html"], or run a generated validation
script such as ["python", "validate.py"].
Do not use `python -m py_compile .` or any directory argument with
`py_compile`; use `python -m compileall <dir>` or a generated validation script
for package-wide syntax checks.
If a step intentionally expects a non-zero result, including a partial bug-fix
step where failure logs should now show only logic errors, use a command object
with `expected_returncode` or a wrapper script that returns 0 only when that
specific expected failure is observed.
Do not emit chat-template or fake tool-call markers such as <|channel>,
<tool_call>, call:ls_tool, or similar. This harness cannot execute those
markers; it only runs commands listed inside the parsed JSON `validation_commands`
fields after the response is complete. Start with `{`, return one JSON object,
and stop immediately after the matching closing `}`.
"""


IMPLEMENTATION_CONTRACT = """
Return strict JSON only:
{
  "plan_note": "progress note for the configured plan file",
  "files": [{"path": "relative/path", "content": "complete file content"}],
  "commands": [["python", "-m", "unittest", "-v"]],
  "test_evidence": ["description of command/report/screenshot evidence produced"],
  "resolution_request": "none|needs_requirements_change|needs_plan_change|cannot_resolve"
}
Only write paths inside the project workspace. Prefer validation commands that
terminate quickly, but request a per-command timeout when a legitimate build,
test, or browser check needs longer. Write the files needed to complete the
current plan step or a coherent vertical slice of it. For large steps, it is fine
to split work across feedback attempts, but do not artificially withhold
inseparable files or documentation that is needed for a high-quality result.
The `files` payload creates files, not empty directories. When a step requires
directory scaffolding but no real source file belongs there yet, create a small
placeholder such as `game/js/.gitkeep`, `game/css/.gitkeep`, or
`tests/.gitkeep` so the directory exists and validation commands can prove it.
Avoid unrelated full-project rewrites.
When feedback identifies malformed source, corrupted markup, duplicated tags,
broken imports, or other structural damage, replace the affected file from a
clean minimal template instead of carrying forward suspicious fragments. Quality
and verifiability are more important than preserving a previous bad draft.
When feedback identifies a narrow defect in otherwise valid code, preserve the
known-good file content and change only the defective lines. Do not introduce
new custom tags, invented attributes, placeholder syntax, duplicate imports, or
gratuitous wording/syntax changes unless the requirement explicitly asks for
them. Stable, boring, canonical source is better than a fresh rewrite that adds
new mistakes.
Commands may be argv lists or {"cmd": ["python", "script.py"], "timeout_seconds": 7200}
when one specific tool call legitimately needs longer than the default timeout.
Plain argv lists do not expand shell variables, pipes, redirects, globbing, or
`&&`; use `["bash", "-lc", "export PATH=\"$HOME/.dotnet:$PATH\" && dotnet --version"]`
when a command intentionally needs shell behavior.
Avoid quote-heavy inline Python in JSON command strings, and avoid brittle grep
commands when a check has multiple conditions or nested quotes. If validation
needs logic, write a small validation file and run it as a command.
For expected failure-path tests, use {"cmd": ["python", "-m", "app"], "expected_returncode": 2}
or, better, a small assertion command that checks the non-zero return code and error text.
If the current step intentionally leaves known failures for a later step, make
that explicit with `expected_returncode` or a validation script that proves the
remaining failure is intentional. A plain failing command is ambiguous evidence.
Do not emit chat-template or fake tool-call markers such as <|channel>,
<tool_call>, call:ls_tool, or similar. This harness cannot execute those
markers; it only runs commands listed inside the parsed JSON `commands` field
after the response is complete. If you need information from the workspace,
request a command in the JSON rather than pretending to call a tool. Start with
`{`, return one JSON object, and stop immediately after the matching closing `}`.
Use as much detail as the current plan step needs, but avoid padding,
repetition, or unrelated narration.
"""


REVIEW_STATUSES = {
    "resolved",
    "needs_rework",
    "cannot_resolve",
    "needs_requirements_change",
    "needs_plan_change",
    "skipped_with_note",
    "resolved_with_compromise",
    "try_another_approach",
}


FEEDBACK_SYSTEM_PROMPT = """
You are the feedback/review agent in a two-agent development loop.
Read the full transcript, including implementation attempts, prior feedback,
requirements decisions, plan updates, command results, screenshots/reports when
listed, git status/diffs, and unresolved risks. Always challenge the work.
Do not emit chat-template or fake tool-call markers. The harness gives you
workspace files, command results, and git evidence inside the prompt; respond
with the requested review JSON only and stop after the closing brace.
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
individual model turns parseable and reviewable; the plan itself may group
related deliverables when that is the most practical way to satisfy the user's
requirements.
When user constraints conflict, name the conflict, choose a practical
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
        self.problem_analysis: dict[str, Any] = {}
        self.plan_steps: list[dict[str, Any]] = []
        self.plan_notes: list[str] = []
        self.approach_history: list[dict[str, Any]] = []
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

    def _harness_state_file_guidance(self) -> str:
        docs = ", ".join(sorted(self._harness_doc_names()))
        return (
            "HARNESS_STATE_FILES:\n"
            f"The harness owns these root-level workflow/state files: {docs}. "
            "They are readable context, but they are not project deliverables and implementation agents must not "
            "create, overwrite, or validate them as proof of project work. If the project needs research, architecture, "
            "or design notes, plan project-owned files such as ARCHITECTURE.md, DESIGN_NOTES.md, PROJECT_RESEARCH.md, "
            "or docs/*.md instead."
        )

    def _split_model_writable_files(self, files: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        """Keep implementation turns from overwriting harness-owned state.

        PLAN/REQUIREMENTS/RESEARCH files are the workflow control plane. The
        model may read them and the harness updates them, but implementation
        payloads should not replace them with project-local guesses. Blocking
        here is safer than hoping every local model obeys the prompt forever.
        """
        blocked = {Path(name).as_posix() for name in self._harness_doc_names()}
        allowed: list[dict[str, Any]] = []
        skipped: list[str] = []
        for item in files:
            rel = Path(str(item.get("path", ""))).as_posix()
            if rel in blocked:
                skipped.append(rel)
                continue
            allowed.append(item)
        return allowed, skipped

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

    def _workflow_state_for_prompt(self, current_step: dict[str, Any] | None = None) -> str:
        """Return the small state bundle every later turn should remember.

        The chat transcript is the main memory, but long runs compact older
        turns. These workspace-owned state files are the durable control plane,
        so implementation and feedback turns get a concise current snapshot
        instead of relying on the summarizer to remember every plan detail.
        """
        status_lines = []
        for step in self.plan_steps:
            marker = "current" if current_step and step.get("id") == current_step.get("id") else "step"
            status_lines.append(
                f"- {marker} {step.get('id')}: {step.get('status', 'pending')} | {step.get('title', '')}"
            )
        if not status_lines:
            status_lines.append("- no validated plan steps yet")
        note_tail = self.plan_notes[-8:]
        parts = [
            f"Plan file: {self.config.runtime.plan_file}",
            f"Requirements file: {self.config.runtime.requirements_file}",
            f"Research file: {self.config.runtime.research_file}",
            "Step status:",
            *status_lines,
            "Recent plan notes:",
            *(f"- {note}" for note in note_tail),
            "Requirements summary:",
            self._requirements_summary_for_prompt(),
            "Problem analysis summary:",
            self._analysis_summary_for_prompt(),
            "Approach history:",
            self._approach_history_summary_for_prompt(),
            f"Web research status: {self.web_research_result.get('status', 'not_run')}",
            "Plan file tail:",
            self._safe_file_excerpt(self._plan_path(), 6000, tail=True),
            "Requirements file tail:",
            self._safe_file_excerpt(self._requirements_path(), 3000, tail=True),
            "Research file tail:",
            self._safe_file_excerpt(self._research_path(), 3000, tail=True),
        ]
        return "\n".join(parts)

    def _workflow_memory_snapshot(self) -> str:
        """Pinned memory appended to compaction output.

        This is intentionally deterministic and generic: it carries current
        requirements, ordered plan state, recent notes, and research status
        without embedding any benchmark-specific solution.
        """
        return self._prompt_excerpt(self._workflow_state_for_prompt(), 18000)

    def _safe_file_excerpt(self, path: Path, limit: int, *, tail: bool = False) -> str:
        if not path.exists():
            return "[missing]"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"[unreadable: {exc}]"
        if len(text) <= limit:
            return text
        if tail:
            return f"[file head omitted: showing last {limit} chars]\n{text[-limit:]}"
        return self._prompt_excerpt(text, limit)

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
                    f"{self.config.runtime.requirements_file}, and {self.config.runtime.research_file}. "
                    + self._execution_environment_guidance().replace("\n", " ")
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
            pinned_context=self._workflow_memory_snapshot(),
        )
        self.conversation.append("user", content)
        raw = self.impl_client.chat(self.conversation.messages(), max_tokens=max_tokens)
        self.conversation.append("assistant", "IMPLEMENTATION_AGENT_RESPONSE:\n" + raw)
        maybe_compact(
            self.conversation,
            self.config,
            self.impl_client,
            context_window=self.config.implementation_model.context_window,
            pinned_context=self._workflow_memory_snapshot(),
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
            pinned_context=self._workflow_memory_snapshot(),
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
            pinned_context=self._workflow_memory_snapshot(),
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
            pinned_context=self._workflow_memory_snapshot(),
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
            pinned_context=self._workflow_memory_snapshot(),
        )
        return raw

    def _feedback_response_tokens(self, feedback_cfg) -> int:
        """Keep reviewer JSON bounded even when implementation output can be large.

        The implementation model may need a high ceiling for generated files,
        but feedback turns should normally be structured review JSON, not generated project content. Without a separate
        cap, a local model can spend many minutes filling the full implementation
        ceiling after it has already said enough for review.
        """
        configured = self.config.runtime.feedback_response_max_tokens
        if configured <= 0:
            return feedback_cfg.max_tokens
        return max(512, min(feedback_cfg.max_tokens, configured))

    def _structured_control_tokens(self, ceiling: int = 4096) -> int:
        """Bound non-file-generation JSON phases.

        Analysis, requirements, and plan-refinement turns are orchestration
        control messages. They should be detailed enough to guide later work,
        but they should not inherit the large implementation payload ceiling
        reserved for generated files.
        """
        return max(2048, min(self.config.implementation_model.max_tokens, ceiling))

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
        the transcript honest while asking the same agent to return a
        machine-parseable object that matches the phase contract.
        """
        try:
            return extract_json_object(raw)
        except Exception as exc:
            if feedback:
                inferred = self._feedback_reasoning_intent_fallback(phase, raw, exc)
                if inferred is not None:
                    return inferred
            tail = self._repair_tail_for_prompt(raw)
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
                "Return one valid JSON object only. Do not use markdown fences. "
                "Do not include analysis, <think> text, chat-template markers, or fake tool-call markers. "
                "The harness cannot execute <tool_call> text; commands must be listed in JSON. "
                "Start with { and stop immediately after the matching closing }. "
                "If the previous plan was too large to parse, "
                "merge related tasks into a practical independently verifiable set of steps. Include enough "
                "detail for later implementation and review. If the previous implementation payload was "
                "oversized or malformed, return a coherent parseable slice of the current step; the feedback "
                "loop can request the rest later. Keep validation commands runnable in the project workspace, "
                "terminating, and assertion-based. "
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
                    max_tokens=max(2048, min(self.config.implementation_model.max_tokens, 6144)),
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
                    "Return only one valid JSON object. No markdown, thinking text, chat-template markers, "
                    "or fake tool-call markers. Keep the structure practical and parseable: distinct plan "
                    "steps, clear requirements, explicit assumptions, and simple validation commands. "
                    "Use only simple validation commands such as [\"test\", \"-f\", \"index.html\"] "
                    "or [\"python\", \"validate.py\"]. Do not use inline python -c. "
                    "JSON starts with { and ends with }.\n\n"
                    f"Required contract:\n{contract}"
                )
                repaired_minimal = self._implementation_chat(last_chance_prompt, max_tokens=4096)
                return extract_json_object(repaired_minimal)

    def _feedback_reasoning_intent_fallback(
        self,
        phase: str,
        raw: str,
        parse_error: Exception,
    ) -> dict[str, Any] | None:
        """Recover clear reviewer intent from reasoning-only feedback.

        Some local models decide correctly in `<think>` text but never emit the
        required JSON before hitting the token cap. A second repair call often
        repeats the same loop and wastes minutes. This fallback only fires when
        the unparseable reviewer text itself contains a clear accept/reject
        decision; otherwise the normal JSON repair path still runs.
        """
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            return None
        tail = text[-2400:].lower()
        positive_markers = (
            "i will accept",
            "i'll accept",
            "i accept",
            "will accept",
            "i will approve",
            "i approve",
            "looks complete",
            "looks good",
            "fully comply",
            "fully complies",
            "plan is feasible",
            "feasible, clear, and verifiable",
            "no required changes",
            "no changes required",
        )
        negative_markers = (
            "needs rework",
            "needs_rework",
            "needs plan change",
            "needs_plan_change",
            "needs requirements change",
            "needs_requirements_change",
            "cannot accept",
            "do not accept",
            "reject",
            "missing",
            "must fix",
            "must be fixed",
            "not verifiable",
            "not feasible",
            "not clear",
        )
        positive = any(marker in tail for marker in positive_markers)
        negative = any(marker in tail for marker in negative_markers)
        if not positive and not negative:
            return None
        if positive and not negative:
            review = {
                "status": "resolved",
                "needs_rework": False,
                "summary": f"{phase} reviewer emitted reasoning-only output; harness inferred acceptance from the reviewer text.",
                "required_changes": [],
                "inferred_from_malformed_response": True,
                "parse_error": str(parse_error),
            }
            if "PLAN_VALIDATION" in phase:
                review["planning_confirmation"] = {
                    "feasible": True,
                    "clear": True,
                    "verifiable": True,
                    "verification_matrix": [],
                }
            return review
        status = "needs_rework"
        if "PLAN_VALIDATION" in phase:
            status = "needs_plan_change"
        elif "REQUIREMENTS" in phase and "requirements" in tail:
            status = "needs_requirements_change"
        return {
            "status": status,
            "needs_rework": True,
            "summary": f"{phase} reviewer emitted reasoning-only output; harness inferred requested rework from the reviewer text.",
            "required_changes": [
                "Address the reviewer concern described in the malformed reasoning output.",
                "Return compact JSON evidence on the next pass so review can proceed deterministically.",
            ],
            "inferred_from_malformed_response": True,
            "parse_error": str(parse_error),
        }

    def _repair_tail_for_prompt(self, raw: str, limit: int = 1200) -> str:
        """Return bounded recovery context for malformed model output.

        Repair turns should help the model recover the intended JSON shape, not
        replay a pathological response. Some local models occasionally end a
        malformed turn with thousands of repeated tokens, so this keeps only a
        small tail and collapses obvious word-level loops before adding it to
        the next prompt.
        """
        tail = raw[-limit:]
        tail = re.sub(r"(\b[\w.-]{4,}\b)(?:[\s_]+\1){5,}", r"\1 [repeated]", tail)
        words: list[str] = []
        for word in tail.split():
            if len(word) > 260:
                word = word[:160] + "[long-token-truncated]" + word[-60:]
            words.append(word)
        tail = " ".join(words)
        return tail

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
        side for a focused, directly verifiable next attempt.
        """
        summary = f"{phase} reviewer response was malformed after JSON repair."
        return {
            "status": "needs_rework",
            "needs_rework": True,
            "summary": summary,
            "required_changes": [
                "Retry the current step with a focused directly verifiable change.",
                "Provide concrete test evidence so the next review does not depend on unsupported free-form claims.",
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
        git_baseline: dict[str, Any] = {"committed": False, "reason": "No implementation baseline was created."}
        analysis_result: dict[str, Any] = {}
        req_result: dict[str, Any] = {}
        plan_result: dict[str, Any] = {}
        step_results: list[dict[str, Any]] = []
        final_review: dict[str, Any] = {}
        approach_review: dict[str, Any] = {}
        retry_context = ""
        for approach_attempt in range(1, self.config.loop.max_approach_reattempts + 1):
            self._append_plan_note(f"[approach {approach_attempt}] starting analysis and planning pass.")
            analysis_result = self._analysis_phase(extra_context=retry_context, approach_attempt=approach_attempt)
            req_result = self._requirements_refinement_phase(extra_context=retry_context)
            plan_result = self._plan_validation_phase()
            if approach_attempt == 1:
                git_baseline = self._git_baseline_commit()
            step_results = []
            while True:
                step = self._next_pending_step()
                if step is None:
                    break
                step_results.append(self._implementation_loop_for_step(step))
                self._write_plan_doc()
                if step_results[-1]["status"] == "cannot_resolve" and self.config.resolution_policy.stop_on_cannot_resolve:
                    break
            final_review = self._final_review_phase(step_results)
            approach_review = self._approach_review_phase(approach_attempt, step_results, final_review)
            self.approach_history.append({
                "approach_attempt": approach_attempt,
                "analysis": analysis_result,
                "requirements_refinement": req_result,
                "plan_validation": plan_result,
                "steps": self._compact_step_results_for_prompt(step_results),
                "final_review_status": final_review.get("status"),
                "final_status": self._final_status(step_results, final_review),
                "approach_review": approach_review,
            })
            if not self._approach_review_requests_retry(approach_review):
                break
            if approach_attempt >= self.config.loop.max_approach_reattempts:
                self._append_plan_note(
                    "[approach] retry requested, but max_approach_reattempts was reached; keeping latest result."
                )
                break
            retry_context = json.dumps(self._compact_approach_review_for_retry(approach_review), ensure_ascii=False)
            self._append_plan_note(
                f"[approach {approach_attempt}] reviewer requested another approach: {approach_review.get('summary', '')}"
            )
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
            "problem_analysis": analysis_result,
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
            "approach_review": approach_review,
            "approach_history": self.approach_history,
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

    def _analysis_phase(self, *, extra_context: str | None = None, approach_attempt: int = 1) -> dict[str, Any]:
        """Analyze the problem before requirements and planning.

        The harness asks the model to compare approaches and identify source
        gaps, but it does not decide the solution. A reviewer pass pushes back
        on shallow or single-path analysis before planning can begin.
        """
        iterations: list[dict[str, Any]] = []
        latest: dict[str, Any] = {}
        review: dict[str, Any] = {}
        for index in range(1, self.config.phases.analysis.max_iterations + 1):
            prompt = (
                f"PROBLEM_ANALYSIS_PHASE approach_attempt={approach_attempt} iteration={index}\n"
                "Analyze the user's request before planning. Restate the problem, inspect the available "
                "workspace/research/source context, identify what is possible or uncertain, and compare "
                "multiple solution paths. Do not write project files and do not solve benchmark tasks in "
                "the harness itself. This phase prepares later model-driven requirements and planning.\n"
                "Challenge yourself before returning: Are the analysis and solution paths comprehensive, "
                "domain-aware, and adequate for the user's request? Redo weak analysis inside this response "
                "before emitting final JSON.\n"
                f"{self._execution_environment_guidance()}\n"
                f"{self._harness_state_file_guidance()}\n"
                f"Web research evidence: {compact_research_for_prompt(self.web_research_result)}\n"
                f"Prior approach history: {self._approach_history_summary_for_prompt()}\n"
                f"Extra context from prior approach review: {extra_context or 'none'}\n\n"
                f"{ANALYSIS_CONTRACT}"
            )
            raw = self._implementation_chat(prompt, max_tokens=self._structured_control_tokens(6144))
            try:
                latest = self._extract_json_or_retry(
                    raw,
                    phase="PROBLEM_ANALYSIS_PHASE",
                    contract=ANALYSIS_CONTRACT,
                )
            except Exception as exc:
                latest = {
                    "problem_restatement": "Problem analysis failed to return parseable JSON.",
                    "domain_and_constraints": [f"Analysis parse failure recorded: {exc}"],
                    "initial_source_check": {"sources_checked": [], "source_gaps": ["analysis parse failure"], "freshness_risks": []},
                    "possible_solution_paths": [],
                    "recommended_path": {"path_id": "", "rationale": "", "fallback_trigger": ""},
                    "analysis_quality": {
                        "is_comprehensive": False,
                        "is_domain_aware": False,
                        "is_actionable_for_planning": False,
                        "remaining_unknowns": ["No valid analysis yet."],
                    },
                    "parse_error": str(exc),
                }
            self.problem_analysis = latest
            review = self._analysis_review(index, latest)
            iterations.append({"iteration": index, "analysis": latest, "review": review})
            if self._status(review) == "resolved":
                self._append_plan_note(f"[analysis] resolved after iteration {index}: {review.get('summary', '')}")
                return {"status": "resolved", "iterations": iterations}
            self.conversation.append(
                "user",
                "ANALYSIS_REWORK_DIRECTIVE:\nRevise the problem analysis using this review. "
                "Do not narrow the harness toward one benchmark; preserve general-purpose problem solving:\n"
                + json.dumps(self._compact_review_for_transcript(review), indent=2),
            )
        fallback = self._fallback_resolution("analysis", review)
        return {"status": fallback["status"], "iterations": iterations, "resolution": fallback}

    def _analysis_review(self, index: int, analysis: dict[str, Any]) -> dict[str, Any]:
        prompt = {
            "phase": "PROBLEM_ANALYSIS_REVIEW_PHASE",
            "iteration": index,
            "project_design": self.config.project_design.prompt,
            "analysis": analysis,
            "web_research_evidence": self.web_research_result,
            "prior_approach_history": self.approach_history,
            "checks": [
                "the request is restated before planning",
                "available workspace, research, or source context is acknowledged",
                "uncertainties and impossible constraints are preserved",
                "multiple solution paths are compared",
                "a recommended first path and fallback trigger are named",
                "the analysis does not contain a benchmark-specific solution shortcut",
            ],
            "expected_json": {
                "status": "resolved|needs_rework|cannot_resolve",
                "needs_rework": True,
                "summary": "review summary",
                "required_changes": ["specific analysis gap"],
                "quality_questions": ["question"],
            },
        }
        raw = self._feedback_chat(
            "PROBLEM_ANALYSIS_REVIEW_PHASE\n"
            "Review the pre-plan problem analysis. Push back if it skips source/context checks, "
            "lists only one path, or starts solving the task instead of setting up a universal workflow.\n"
            + json.dumps(prompt),
            temperature=0.1,
        )
        review = self._normalize_review(self._extract_json_or_retry(
            raw,
            phase="PROBLEM_ANALYSIS_REVIEW_PHASE",
            contract=ANALYSIS_REVIEW_CONTRACT,
            feedback=True,
        ))
        deterministic = self._analysis_structural_findings(analysis)
        if deterministic:
            existing = [str(item) for item in review.get("required_changes", [])]
            review["required_changes"] = existing + [item for item in deterministic if item not in existing]
            if self._status(review) == "resolved":
                review["status"] = "needs_rework"
                review["needs_rework"] = True
                review["summary"] = "Deterministic analysis checks found missing pre-plan coverage."
        return review

    def _analysis_structural_findings(self, analysis: dict[str, Any]) -> list[str]:
        findings: list[str] = []
        if not str(analysis.get("problem_restatement") or "").strip():
            findings.append("Analysis is missing a problem_restatement.")
        paths = analysis.get("possible_solution_paths") or []
        if not isinstance(paths, list) or len(paths) < 2:
            findings.append("Analysis must compare at least two possible solution paths before planning.")
        recommended = analysis.get("recommended_path")
        if not isinstance(recommended, dict) or not recommended.get("path_id"):
            findings.append("Analysis is missing a recommended_path.path_id.")
        quality = analysis.get("analysis_quality") or {}
        if isinstance(quality, dict):
            for key in ("is_comprehensive", "is_domain_aware", "is_actionable_for_planning"):
                if quality.get(key) is not True:
                    findings.append(f"analysis_quality.{key} is not true.")
        else:
            findings.append("Analysis is missing analysis_quality.")
        return findings

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
                "Return strict JSON with enough detail to guide later work; do not use markdown or <think> text. Validation commands "
                "must be terminating commands or scripts that assert behavior. Do not use python -m "
                "http.server by itself as a validation command; browser checks should be wrapped in a "
                "script that starts a server, interacts or inspects, writes evidence, and exits.\n"
                "If the user's requested step count conflicts with verifiable implementation, record "
                "that conflict as an assumption and choose a practical feasible verifiable plan. "
                "Do not reinterpret per-attempt file-count guidance as a one-file-per-plan-step rule.\n"
                f"{self._default_quality_instruction()}\n"
                f"{self._execution_environment_guidance()}\n"
                f"{self._harness_state_file_guidance()}\n"
                f"Problem analysis summary: {self._analysis_summary_for_prompt()}\n"
                "Use the recommended path from the analysis as the first planning direction, but preserve fallback "
                "triggers and open unknowns. If planning reveals the recommended path is wrong, record that and "
                "choose a better path instead of forcing the earlier analysis.\n"
                f"Web research evidence: {compact_research_for_prompt(self.web_research_result)}\n"
                "If web research status is completed or partial, use those findings in the requirements and plan; "
                "the first research/structure step must cite the source URLs in generated project notes. "
                "If web research is skipped or disabled, record available-knowledge notes instead and do not invent URLs.\n"
                "Challenge yourself before returning: Are the analysis, requirements, assumptions, and plan comprehensive "
                "and adequate for the problem domain? If not, fix them before emitting JSON.\n"
                f"Extra context: {extra_context or 'none'}\n\n{REQUIREMENTS_CONTRACT}"
            )
            raw = self._implementation_chat(prompt, max_tokens=self._structured_control_tokens(6144))
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
                        "The implementation model must retry with valid JSON before implementation can start."
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
        environment_findings = self._environment_assumption_findings(requirements=requirements)
        prompt = {
            "phase": "REQUIREMENTS_REVIEW_PHASE",
            "iteration": index,
            "project_design": self.config.project_design.prompt,
            "requirements": requirements,
            "web_research_evidence": self.web_research_result,
            "default_quality_policy": self._default_quality_policy_payload(),
            "execution_environment": self._execution_environment_payload(),
            "deterministic_environment_findings": environment_findings,
            "expected_json": {
                "status": "resolved|needs_rework|needs_requirements_change|cannot_resolve|skipped_with_note",
                "needs_rework": True,
                "summary": "review summary",
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
            "or the initial research/structure planning step. If default_quality_policy.applies is false because "
            "the user explicitly constrained deliverables to one output/artifact, do not require extra project files; "
            "require validation evidence that respects the artifact constraint instead.\n"
            "Apply execution_environment strictly. If deterministic_environment_findings is non-empty, request a "
            "requirements or plan correction instead of accepting incompatible assumptions.\n"
            "If WEB_RESEARCH_TOOL_RESULT has completed or partial sources, reject requirements that ignore those sources. "
            "If web research is skipped or disabled, do not require cited source URLs; request available-knowledge notes instead.\n"
            + json.dumps(prompt),
            temperature=0.1,
        )
        review = self._extract_json_or_retry(
            raw,
            phase="REQUIREMENTS_REVIEW_PHASE",
            contract='{"status":"resolved|needs_rework|needs_requirements_change|cannot_resolve|skipped_with_note","needs_rework":true,"summary":"review summary","required_changes":["specific change"]}',
            feedback=True,
        )
        review = self._normalize_review(review)
        if environment_findings:
            existing = [str(item) for item in review.get("required_changes", [])]
            review["required_changes"] = existing + [item for item in environment_findings if item not in existing]
            if self._status(review) == "resolved":
                review["status"] = "needs_requirements_change"
                review["needs_rework"] = True
                review["summary"] = "Deterministic environment checks found incompatible browser/tooling assumptions."
        return review

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
            "execution_environment": self._execution_environment_payload(),
            "plan": self.plan_steps,
            "deterministic_structural_findings": structural_findings,
            "checks": [
                "each step is distinct",
                "dependencies are explicit",
                "each step has acceptance criteria",
                "each step has validation commands or an explicit non-command validation method",
                "validation commands terminate and assert behavior instead of starting a server forever",
                "browser/UI steps have executable browser evidence such as Playwright, screenshots, or a validation report when web interaction tools are enabled",
                "browser/UI plans match the agent container tools; Python Playwright is available, but Node/npm/npx/@playwright/test are not available unless explicitly configured",
                "project deliverables must not be harness-owned state files such as PLAN.md, REQUIREMENTS.md, or RESEARCH.md",
                "the sequence can be executed one step at a time",
                "planning_confirmation says the plan is feasible, clear, and verifiable",
                "the reviewer can name exactly how each step will be verified later",
                "when default quality policy applies, the first step researches needed patterns/knowledge and plans project structure before feature implementation",
                "when web research evidence exists, the plan requires generated notes to cite and apply researched source URLs",
            ],
            "expected_json": {
                "status": "resolved|needs_plan_change|needs_requirements_change|cannot_resolve",
                "needs_rework": True,
                "summary": "review summary",
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
            "hard only when the user explicitly says hard/strict/exactly/must; otherwise prefer a "
            "practical feasible verifiable plan. Per-attempt file-count guidance is not a plan-step limit.\n"
            "Challenge the plan before accepting it: Are the analysis and planning comprehensive and adequate "
            "to the request and domain? Does any step need to be updated because it is impossible, stale, "
            "or no longer useful? Push back with needs_plan_change if so.\n"
            f"{self._execution_environment_guidance()}\n"
            f"{self._harness_state_file_guidance()}\n"
            + json.dumps(prompt),
            temperature=0.1,
        )
        review = self._normalize_review(self._extract_json_or_retry(
            raw,
            phase="PLAN_VALIDATION_PHASE",
            contract='{"status":"resolved|needs_plan_change|needs_requirements_change|cannot_resolve","needs_rework":true,"summary":"review summary","required_changes":["specific change"]}',
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
            "Return the plan/refined planning confirmation contract below; do not repeat "
            "the full requirements list unless those details are needed for clarity. Validation commands must be scripts/commands that exit and "
            "assert behavior. Do not use python -m http.server by itself.\n"
            f"{self._execution_environment_guidance()}\n"
            f"{self._harness_state_file_guidance()}\n"
            f"Requirements summary: {self._requirements_summary_for_prompt()}\n"
            f"Current plan: {json.dumps(self.plan_steps)}\n"
            f"Web research evidence: {compact_research_for_prompt(self.web_research_result)}\n"
            f"Review: {json.dumps(self._compact_review_for_transcript(review))}\n\n{PLAN_REFINEMENT_CONTRACT}"
        )
        raw = self._implementation_chat(prompt, max_tokens=self._structured_control_tokens())
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
        previous_steps = self.plan_steps
        self.plan_steps = self._merge_refined_plan_steps(
            previous_steps,
            normalize_plan_steps(payload.get("plan", self.plan_steps)),
        )
        # Keep the embedded requirements copy consistent with the authoritative
        # top-level plan so later reviews do not see both old and new commands.
        self.requirements["plan"] = self.plan_steps
        self.plan_notes.append(f"Plan refined after review iteration {index}.")
        self._write_plan_doc()
        return payload

    def _merge_refined_plan_steps(
        self,
        previous_steps: list[dict[str, Any]],
        refined_steps: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Replace plan content while preserving dict identity for active loops.

        A feedback review may request a plan change in the middle of a step.
        The implementation loop already holds a reference to that step dict, so
        simply assigning ``self.plan_steps = new_steps`` leaves reviewer-owned
        validation stuck on stale commands. Updating matching step dictionaries
        in place keeps the active loop, plan document, and final review aligned.
        """
        previous_by_id = {str(step.get("id")): step for step in previous_steps if step.get("id") is not None}
        merged: list[dict[str, Any]] = []
        for refined in refined_steps:
            step_id = str(refined.get("id")) if refined.get("id") is not None else ""
            existing = previous_by_id.get(step_id)
            if existing is None:
                merged.append(refined)
                continue
            existing.clear()
            existing.update(refined)
            merged.append(existing)
        return merged

    def _next_pending_step(self) -> dict[str, Any] | None:
        """Return the next unresolved step from the current, possibly refined plan."""
        for step in self.plan_steps:
            if str(step.get("status", "pending")).lower() not in {"resolved", "cannot_resolve", "skipped"}:
                return step
        return None

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
                step = self._current_step_by_id(step["id"]) or step
            elif status == "needs_requirements_change":
                self._requirements_refinement_phase(extra_context=json.dumps(self._compact_review_for_transcript(review)))
                step = self._current_step_by_id(step["id"]) or step
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
                "Keep previous requirements, analysis, plan validation, repair history, and this step context in mind. "
                "Summarize what remains incomplete and complete those gaps if possible. If the plan is now stale, "
                "impossible, or no longer useful, request needs_plan_change instead of burning attempts on it:\n"
                + json.dumps(self._compact_review_for_transcript(review), indent=2),
            )
        resolution = self._fallback_resolution(f"step {step['id']}", attempts[-1]["review"] if attempts else {})
        step["status"] = resolution["status"]
        return {"step_id": step["id"], "status": resolution["status"], "attempts": attempts, "resolution": resolution}

    def _current_step_by_id(self, step_id: Any) -> dict[str, Any] | None:
        """Find the latest plan step by id after requirements or plan refinement."""
        for candidate in self.plan_steps:
            if str(candidate.get("id")) == str(step_id):
                return candidate
        return None

    def _implementation_pass(self, step: dict[str, Any], attempt: int) -> dict:
        """Ask for complete-file edits and run the model-requested validations."""
        prompt = (
            f"IMPLEMENT_PLAN_STEP_PHASE step_id={step['id']} attempt={attempt}\n"
            "Work on this single plan step only. Do not silently jump ahead. If the step is impossible, "
            "use resolution_request and explain why. Cross-check your edits against this step's acceptance "
            "criteria and include validation commands that prove the step whenever terminal tools are enabled.\n"
            "You are responsible for choosing the repair strategy. Use the recorded analysis, review findings, "
            "command evidence, and prior repair history to decide what to change; the harness provides evidence "
            "and boundaries, not a predetermined solution.\n"
            "Do not stage or commit with git. The harness owns git add/commit after feedback accepts a step. "
            "You may run read-only git commands such as git status or git diff for your own evidence.\n"
            f"Do not rewrite {self.config.runtime.plan_file} just to mark the current step complete; put progress in plan_note. "
            f"The harness appends notes and marks resolved after feedback accepts the step. Only edit {self.config.runtime.plan_file} "
            "when the feedback request specifically requires substantive plan content changes.\n"
            f"Do not include harness-owned state files in the files payload: "
            f"{', '.join(sorted(self._harness_doc_names()))}. The harness creates and updates those files.\n"
            "If the current plan step asks for one of those harness-owned files as a project deliverable, request "
            "needs_plan_change instead of trying to satisfy that conflicting instruction.\n"
            "Do not implement future plan steps early. If the current step is setup, structure, or research, create "
            "minimal scaffolding and accurate placeholders only; leave feature mechanics for their own accepted steps.\n"
            "Keep this attempt parseable and focused on the current plan step. Write the files needed to complete "
            "the step or a coherent vertical slice. If feedback requested several unrelated changes, choose a "
            "sensible subset, note what remains, and let the next feedback iteration request the rest. Do not rewrite "
            "unrelated parts of the project.\n"
            "If previous attempts created malformed code, first stabilize the affected files using conservative "
            "canonical source. If a file is already correct, do not rewrite it just for variety; repeated rewrites "
            "should reduce risk, not generate new syntax variants.\n"
            f"Problem analysis summary: {self._analysis_summary_for_prompt()}\n"
            f"Requirements summary: {self._requirements_summary_for_prompt()}\n"
            f"Validated plan step ids: {[step.get('id') for step in self.plan_steps]}\n"
            f"Workflow state context:\n{self._workflow_state_for_prompt(step)}\n"
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
                    "No files were written; next attempt must return a valid parseable JSON payload."
                ),
                "files": [],
                "commands": [],
                "test_evidence": [],
                "resolution_request": "none",
                "parse_error": str(exc),
                "raw_tail": raw[-2000:],
            }
        allowed_files, skipped_harness_files = self._split_model_writable_files(payload.get("files", []))
        written = write_files(self.workspace, allowed_files)
        command_results = []
        if self.config.mcp_tools.terminal:
            command_results = self._run_verified_commands(
                payload.get("commands", []),
                source="implementation",
                context={
                    "step": step,
                    "attempt": attempt,
                    "plan_note": payload.get("plan_note"),
                    "test_evidence": payload.get("test_evidence", []),
                    "purpose": "Implementation-agent requested terminal commands for the current plan step.",
                },
            )
        note = payload.get("plan_note") or f"{step['id']} attempt {attempt} implementation pass completed."
        self._append_plan_note(f"[{step['id']} attempt {attempt}] {note}")
        return {
            "written": written,
            "commands": command_results,
            "raw": payload,
            "skipped_harness_files": skipped_harness_files,
        }

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
                "If evidence remains imperfect in compromise mode, either return needs_rework with a focused bounded fix or resolved_with_compromise/skipped_with_note with an explicit diluted requirement note.",
                "For browser/game work, prefer Playwright-style interaction evidence and screenshot/report artifacts when configured.",
                "Do not request incidental package/browser installation inside generated validation scripts for default browser checks; if a task requires another stack, request an explicit dependency/setup step with bounded commands.",
                "In compromise mode, accept a clearly labelled non-browser fallback only when browser launch cannot be made reliable and the fallback still gives concrete evidence.",
                "Return needs_plan_change if this step cannot be independently verified as written.",
                "Return needs_requirements_change if the requirements are contradictory or impossible.",
                "Return cannot_resolve only when bounded retries are unlikely to help.",
            ],
            "expected_json": {
                "status": "resolved|needs_rework|cannot_resolve|needs_requirements_change|needs_plan_change|skipped_with_note",
                "needs_rework": True,
                "summary": "review summary",
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
            contract='{"status":"resolved|needs_rework|cannot_resolve|needs_requirements_change|needs_plan_change|skipped_with_note","needs_rework":true,"summary":"review summary","required_changes":["specific change"]}',
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

    def _approach_review_phase(
        self,
        approach_attempt: int,
        step_results: list[dict[str, Any]],
        final_review: dict[str, Any],
    ) -> dict[str, Any]:
        """Ask whether the executed approach was the right response.

        This is separate from correctness review. A project can pass its plan
        but still reveal that another approach is needed, for example a
        periodic monitoring task that must check again later or a plan that
        satisfied narrow tests while missing the user's broader intent.
        """
        prompt = {
            "phase": "APPROACH_REVIEW_PHASE",
            "approach_attempt": approach_attempt,
            "max_approach_reattempts": self.config.loop.max_approach_reattempts,
            "project_design": self.config.project_design.prompt,
            "problem_analysis": self._analysis_summary_for_prompt(),
            "requirements": self._requirements_summary_for_prompt(),
            "plan": self._compact_plan_for_prompt(),
            "step_results": self._compact_step_results_for_prompt(step_results),
            "final_review": self._compact_final_review_for_approach(final_review),
            "prior_approach_history": self._approach_history_summary_for_prompt(),
            "expected_json": {
                "status": "resolved|try_another_approach|needs_rework|cannot_resolve",
                "needs_rework": False,
                "summary": "decision summary",
                "decision": "keep_result|retry_with_new_approach|stop",
                "recommended_next_approach": "only when retrying",
                "evidence_reviewed": ["evidence"],
                "runbook_updates": ["note"],
            },
        }
        raw = self._feedback_chat_with_compact_context(
            "APPROACH_REVIEW_PHASE\n"
            "Decide whether the completed approach was the right response to the original user request. "
            "Use the final review and evidence, but evaluate broader fit: whether a different approach, "
            "another timed check, a requirements correction, or a plan rethink is warranted. Do not retry "
            "for style or novelty; retry only when evidence shows a meaningful gap or the task requires "
            "periodic re-checking. Summarize gaps and failures, and if they can be completed within the "
            "existing approach say so without requesting a full retry.\n"
            + json.dumps(prompt),
            context_note=(
                "The full transcript remains in .agent_state/conversation.full.jsonl. "
                "This phase reviews approach adequacy, not implementation details already covered by final review."
            ),
            temperature=0.1,
        )
        review = self._normalize_review(self._extract_json_or_retry(
            raw,
            phase="APPROACH_REVIEW_PHASE",
            contract=APPROACH_REVIEW_CONTRACT,
            feedback=True,
        ))
        decision = str(review.get("decision") or "").strip()
        if decision == "retry_with_new_approach" and self._status(review) == "resolved":
            review["status"] = "try_another_approach"
            review["needs_rework"] = True
        if self._status(review) == "try_another_approach" and not review.get("recommended_next_approach"):
            review["recommended_next_approach"] = "Re-run analysis and planning from the recorded gaps."
        self._append_plan_note(f"[approach review {approach_attempt}] {review.get('summary', 'no summary')}")
        return review

    def _approach_review_requests_retry(self, review: dict[str, Any]) -> bool:
        return self._status(review) == "try_another_approach" or str(review.get("decision")) == "retry_with_new_approach"

    def _compact_approach_review_for_retry(self, review: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "status": review.get("status"),
                "summary": review.get("summary"),
                "decision": review.get("decision"),
                "recommended_next_approach": review.get("recommended_next_approach"),
                "runbook_updates": review.get("runbook_updates", []),
                "required_changes": review.get("required_changes", []),
            }.items()
            if value not in (None, "", [])
        }

    def _compact_final_review_for_approach(self, final_review: dict[str, Any]) -> dict[str, Any]:
        iterations = final_review.get("iterations") or []
        last = iterations[-1] if iterations else {}
        review = last.get("review") or {}
        return {
            "status": final_review.get("status"),
            "resolution": final_review.get("resolution"),
            "last_review_status": review.get("status"),
            "last_review_summary": review.get("summary"),
            "required_changes": self._clip_list_for_transcript(review.get("required_changes", [])),
            "deterministic_evidence_findings": self._clip_list_for_transcript(
                review.get("deterministic_evidence_findings", [])
            ),
        }

    def _final_project_review(self, attempt: int, step_results: list[dict[str, Any]]) -> dict:
        feedback_tool_evidence = self._final_feedback_tool_evidence(step_results)
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
            "skipped_harness_files": implementation.get("skipped_harness_files", []),
            "commands": self._compact_command_results_for_prompt(implementation.get("commands", [])),
            "plan_note": raw.get("plan_note"),
            "test_evidence": raw.get("test_evidence", []),
            "resolution_request": raw.get("resolution_request"),
            "parse_error": raw.get("parse_error"),
        }

    def _compact_step_evidence_for_prompt(self, evidence: dict[str, Any]) -> dict[str, Any]:
        """Summarize reviewer-owned step evidence for local-model context limits."""
        files = []
        for item in self._reviewer_prompt_files(evidence.get("workspace_files", [])):
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
        for item in self._reviewer_prompt_files(evidence.get("workspace_files", [])):
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
        suffix = Path(path).suffix.lower()
        important = path.endswith((
            "validation_evidence.log",
            "results.json",
            "report.json",
            "playwright-report.json",
        ))
        if important:
            limit = 12000
        elif suffix in {".md", ".html", ".css", ".js", ".json", ".py"} and len(content) <= 8000:
            # Small project files are often where the reviewer finds precise
            # inconsistencies. Keep them intact; truncating a 2-3 KB README is
            # more misleading than helpful and can cause false pushbacks.
            limit = 8000
        else:
            limit = default_limit
        prompt_truncated = len(content) > limit
        return {
            "path": path,
            "size": item.get("size", len(content.encode("utf-8"))),
            "source_truncated": item.get("truncated", False),
            "prompt_truncated": prompt_truncated,
            "content": self._prompt_excerpt(content, limit),
        }

    def _reviewer_prompt_files(self, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop harness control-plane documents from reviewer prompt snapshots.

        The reviewer receives the current plan, requirements, and research
        evidence explicitly in structured request fields. Re-sending PLAN.md
        and friends as workspace files wastes context and crowds out the real
        project artifacts the reviewer must inspect.
        """
        harness_names = self._harness_doc_names()
        prompt_files: list[dict[str, Any]] = []
        for item in files:
            path = str(item.get("path", ""))
            if path in harness_names:
                continue
            prompt_files.append(item)
        return prompt_files

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

    def _compact_review_for_correction(self, review: dict[str, Any]) -> dict[str, Any]:
        """Return only the review decision needed by the next correction turn.

        Step/final review objects can contain large file snapshots, rerun output,
        and git diffs. That evidence belongs in `summary.json` and the full
        transcript, but feeding it back into the implementation model can turn a
        simple final correction into a tens-of-thousands-token prompt. The
        implementation side needs the verdict, concrete requested changes, and
        deterministic guardrail findings; it does not need the full reviewer
        evidence bundle again.
        """
        return {
            key: self._clip_list_for_transcript(value) if isinstance(value, list) else value
            for key, value in {
                "status": review.get("status"),
                "needs_rework": review.get("needs_rework"),
                "summary": review.get("summary"),
                "required_changes": review.get("required_changes", []),
                "deterministic_evidence_findings": review.get("deterministic_evidence_findings", []),
                "compromise_note": review.get("compromise_note"),
            }.items()
            if value not in (None, [], "")
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
            f"Do not include harness-owned state files in the files payload: "
            f"{', '.join(sorted(self._harness_doc_names()))}. The harness creates and updates those files.\n"
            f"Review: {json.dumps(self._compact_review_for_correction(review))}\n\n{IMPLEMENTATION_CONTRACT}"
        )
        if any(self._looks_like_browser_step(step) for step in self.plan_steps):
            prompt += "\n" + self._browser_validation_guidance()
        raw = self._implementation_chat(prompt)
        payload = self._extract_json_or_retry(
            raw,
            phase="FINAL_PROJECT_CORRECTION_PHASE",
            contract=IMPLEMENTATION_CONTRACT,
        )
        allowed_files, skipped_harness_files = self._split_model_writable_files(payload.get("files", []))
        written = write_files(self.workspace, allowed_files)
        command_results = []
        if self.config.mcp_tools.terminal:
            command_results = self._run_verified_commands(
                payload.get("commands", []),
                source="final_correction",
                context={
                    "attempt": attempt,
                    "review": self._compact_review_for_correction(review),
                    "purpose": "Implementation-agent requested terminal commands for final correction.",
                },
            )
        self._append_plan_note(f"[final correction attempt {attempt}] {payload.get('plan_note', 'completed')}")
        return {
            "written": written,
            "commands": command_results,
            "raw": payload,
            "skipped_harness_files": skipped_harness_files,
        }

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
            findings.extend(self._harness_state_file_plan_findings(step))
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
                for marker in ("plan", "order", "mapping", "map", "architecture", "dependencies", "assessment", "assess")
            )
            if not planning_present and structure_present:
                # Greenfield plans often express the planning action as
                # "create/document the project structure" rather than using
                # the literal word "plan". Treat those phrases as satisfying
                # the same quality gate so we do not force useless rewording.
                planning_present = any(
                    marker in first_text
                    for marker in (
                        "structure overview",
                        "project structure",
                        "skeleton",
                        "separation of concerns",
                        "documented",
                        "document notes",
                    )
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
        findings.extend(self._environment_assumption_findings(requirements=self.requirements, plan=self.plan_steps))
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

    def _analysis_summary_for_prompt(self) -> str:
        if not isinstance(self.problem_analysis, dict) or not self.problem_analysis:
            return "No problem analysis available yet."
        paths = self.problem_analysis.get("possible_solution_paths") or []
        recommended = self.problem_analysis.get("recommended_path") or {}
        return json.dumps({
            "problem_restatement": self.problem_analysis.get("problem_restatement"),
            "domain_and_constraints": self.problem_analysis.get("domain_and_constraints", [])[:8],
            "source_gaps": (self.problem_analysis.get("initial_source_check") or {}).get("source_gaps", [])[:5],
            "possible_solution_paths": [
                {
                    "id": path.get("id"),
                    "description": path.get("description"),
                    "risks": path.get("risks", [])[:3],
                }
                for path in paths[:5]
                if isinstance(path, dict)
            ],
            "recommended_path": recommended,
        }, ensure_ascii=False)

    def _approach_history_summary_for_prompt(self) -> str:
        if not self.approach_history:
            return "No completed approach attempts yet."
        compact = []
        for item in self.approach_history[-5:]:
            review = item.get("approach_review") or {}
            compact.append({
                "approach_attempt": item.get("approach_attempt"),
                "final_status": item.get("final_status"),
                "approach_decision": review.get("decision") or review.get("status"),
                "summary": review.get("summary"),
                "runbook_updates": review.get("runbook_updates", []),
            })
        return json.dumps(compact, ensure_ascii=False)

    def _harness_state_file_plan_findings(self, step: dict[str, Any]) -> list[str]:
        """Reject plans that turn workflow-control files into project artifacts.

        The harness writes PLAN/REQUIREMENTS/RESEARCH style files before the
        implementation agent runs. If a generated plan asks the model to create
        those same root-level filenames, the step is impossible: the write guard
        will correctly block it later. Catching that mismatch during plan review
        makes the feedback loop repair the plan instead of wasting attempts.
        """
        text = json.dumps(
            {
                "title": step.get("title", ""),
                "description": step.get("description", ""),
                "acceptance_criteria": step.get("acceptance_criteria", []),
                "validation_commands": step.get("validation_commands", []),
            },
            sort_keys=True,
        )
        findings: list[str] = []
        step_id = str(step.get("id") or "step")
        for name in sorted(self._harness_doc_names()):
            normalized = Path(name).as_posix()
            # Flag `RESEARCH.md` and `./RESEARCH.md`, but allow project-owned
            # subpaths such as `docs/RESEARCH.md`.
            pattern = rf"(?<![/\\\w.-])(?:\./)?{re.escape(normalized)}(?![\w.-])"
            if re.search(pattern, text, flags=re.IGNORECASE):
                findings.append(
                    f"{step_id} names harness-owned state file {normalized} as a project deliverable or validation target; "
                    "use ARCHITECTURE.md, DESIGN_NOTES.md, PROJECT_RESEARCH.md, or docs/*.md instead."
                )
        return findings

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
                raw_parts = [str(part) for part in (command.get("cmd") or command.get("command") or [])]
            else:
                raw_parts = [str(part) for part in command]
            parts = [part.lower() for part in raw_parts]
            joined = " ".join(parts)
            if "python -m http.server" in joined or (
                len(parts) >= 3 and parts[0].endswith("python") and parts[1] == "-m" and parts[2] == "http.server"
            ):
                findings.append(
                    f"{step_id} validation starts an HTTP server but does not assert behavior; wrap server startup in a validation script that exits."
                )
            if len(parts) >= 2 and parts[0].endswith("python") and parts[1] == "-mm":
                findings.append(
                    f"{step_id} validation command uses malformed Python flag '-mm'; use '-m' or replace it with a validation script."
                )
            if len(raw_parts) >= 2 and raw_parts[0] == "test" and raw_parts[1] == "-F":
                findings.append(
                    f"{step_id} validation command uses malformed shell test flag '-F'; use '-f' for file existence."
                )
            if self._looks_like_malformed_grep_max_count(raw_parts):
                findings.append(
                    f"{step_id} validation command uses a malformed grep max-count flag; use `grep -q pattern file`, "
                    "a numeric `grep -m <count> ...`, or replace it with a validation script."
                )
            if self._looks_like_py_compile_directory_command(parts):
                findings.append(
                    f"{step_id} validation uses `python -m py_compile` on a directory. "
                    "Use `python -m compileall <dir>` or a generated validation script for package-wide syntax checks."
                )
            if self._looks_like_invalid_inline_python_compound_command(parts):
                findings.append(
                    f"{step_id} validation uses a one-line `python -c` try/except block, which Python cannot parse. "
                    "Replace it with a generated validation script or a multiline command."
                )
            if self._looks_like_unwrapped_expected_failure_validation(step, command, parts):
                findings.append(
                    f"{step_id} validation appears to test an expected failure path without declaring expected_returncode "
                    "or wrapping the exception assertion. Replace it with a command object using expected_returncode, "
                    "or a small wrapper command/script that exits 0 only when the expected error occurs."
                )
            if self.config.mcp_tools.web_interaction:
                step_text = (self.config.project_design.prompt + " " + json.dumps(step, sort_keys=True)).lower()
                if any(part in ("npm", "npx", "node") for part in parts) and not self._explicit_dependency_setup_is_present(step_text):
                    findings.append(
                        f"{step_id} validation uses Node/npm tooling, but the default agent container provides Python "
                        "Playwright and Chromium instead. Use Python Playwright for generic browser validation, or add "
                        "an explicit bounded dependency/setup step for the requested stack."
                    )
        return findings

    def _command_expected_returncode(self, command: list[Any] | dict[str, Any]) -> int:
        if isinstance(command, dict):
            return int(command.get("expected_returncode", 0))
        return 0

    def _looks_like_py_compile_directory_command(self, parts: list[str]) -> bool:
        """Reject a common impossible validation command during plan review.

        `py_compile` compiles files, not directories. Local models often suggest
        `python -m py_compile .` as a package-wide syntax check, which later
        fails before it exercises the project and can trap the feedback loop.
        Catch it at plan-validation time and ask for `compileall` or a script.
        """
        if len(parts) < 4:
            return False
        if not (parts[0].endswith("python") and parts[1] == "-m" and parts[2] == "py_compile"):
            return False
        for arg in parts[3:]:
            normalized = arg.rstrip("/")
            if normalized in {".", "./"}:
                return True
            # Heuristic: common package/workspace directory checks. File paths
            # normally have a suffix; directory names do not.
            if normalized and not Path(normalized).suffix and normalized not in {"-q", "-qq"}:
                return True
        return False

    def _looks_like_invalid_inline_python_compound_command(self, parts: list[str]) -> bool:
        """Catch `python -c "try: ...; except ..."` before a real run wastes time.

        Python accepts some one-line compound statements, but `try/except`
        needs a real block layout. Local models often propose it as a compact
        negative-path assertion; asking for a generated validation script is
        clearer and portable across shells.
        """
        if len(parts) < 3 or not (parts[0].endswith("python") and parts[1] == "-c"):
            return False
        code = parts[2]
        return "try:" in code and "except" in code and "\n" not in code

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
        is_python_inline = len(parts) >= 3 and parts[0].endswith("python") and parts[1] == "-c"
        if is_python_inline:
            code = parts[2].lower()
            wrapper_markers = (
                "try:",
                "except",
                "raises",
                "assertraises",
                "pytest.raises",
                "returncode",
                "subprocess.run",
                "exit(0 if",
                "sys.exit(0",
            )
            if any(marker in code for marker in wrapper_markers):
                return False
        if self._step_expects_nonzero_validation(step):
            return self._looks_like_plain_test_or_app_command(parts)
        if not is_python_inline:
            return False
        step_text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", [])),
        ]).lower()
        if not any(marker in step_text for marker in ("raise", "raises", "exception", "error", "empty", "invalid")):
            return False
        return "[]" in code or "raise " in code or "sys.exit" in code

    def _looks_like_plain_test_or_app_command(self, parts: list[str]) -> bool:
        """Return True for commands likely meant to observe a residual failure.

        Partial bug-fix steps sometimes need one command that must still fail
        for a known reason, usually `unittest` or `pytest`. Syntax/build checks
        in the same step, such as `compileall`, should remain ordinary success
        validations. This distinction keeps the plan checker strict without
        falsely rejecting valid setup/syntax commands.
        """
        if not parts:
            return False
        lowered = [str(part).lower() for part in parts]
        joined = " ".join(lowered)
        if "unittest" in lowered or "pytest" in lowered or "pytest" in joined:
            return True
        if any(part.startswith("test_") or part.startswith("tests/") for part in lowered):
            return True
        if lowered[0].endswith("python") and len(lowered) >= 3 and lowered[1] == "-m":
            return lowered[2] not in {"compileall", "py_compile"}
        return False

    def _step_expects_nonzero_validation(self, step: dict[str, Any]) -> bool:
        """Return True when a step deliberately expects a failing command.

        Existing-project repair workflows often include a partial-fix step such
        as "syntax errors are gone, but logic tests still fail". A plain test
        command returning 1 is ambiguous: it may mean the intended residual
        failure, or it may mean a new syntax/import regression. Plans should
        express that intent with expected_returncode or a validation wrapper.
        """
        text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", [])),
        ]).lower()
        markers = (
            "still fail",
            "still fails",
            "continues to fail",
            "expected failure",
            "expected to fail",
            "fails due to",
            "fail due to",
            "remaining failure",
        "logic error persists",
        "logic failure persists",
        "failure logs clearly indicate logic",
        "failure logs indicate logic",
        "failures clearly indicate logic",
        "logic failures remain",
        "logic tests still fail",
        )
        return any(marker in text for marker in markers)

    def _is_transient_expected_failure_validation(self, step: dict[str, Any], command: Any) -> bool:
        """Identify expected failures that are only valid mid-repair.

        Negative-path behavior, such as "invalid input exits 2", is a final
        product requirement and must still be rerun in final review. This helper
        only skips non-zero commands tied to investigation or partial-fix steps
        whose expected failure should disappear once later plan steps succeed.
        """
        if self._command_expected_returncode(command) == 0:
            return False
        text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", [])),
        ]).lower()
        transient_markers = (
            "syntax/import",
            "syntax or import",
            "test runner successfully discovers",
            "attempts to run tests",
            "still fail",
            "still fails",
            "continues to fail",
            "remaining failure",
            "logic tests still fail",
            "logic failures remain",
            "fix syntax",
            "fix import",
            "partial-fix",
            "partial fix",
        )
        return self._is_failure_investigation_step(step) or any(marker in text for marker in transient_markers)

    def _looks_like_browser_step(self, step: dict[str, Any]) -> bool:
        text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(step.get("acceptance_criteria", [])),
        ]).lower()
        # Use word boundaries so documentation terms such as "guide" do not
        # accidentally match the UI marker and trigger browser-only guidance.
        return bool(
            re.search(
                r"\b(browser|ui|web|map|click|drag|zoom|pan|render|screenshot|html|css|javascript|playwright|chromium|canvas|game)\b",
                text,
            )
        )

    def _execution_environment_payload(self) -> dict[str, Any]:
        """Compact machine-readable environment facts for planning/review prompts."""
        payload = {
            "agent_runs_in_docker": self.config.runtime.docker_isolation,
            "terminal_tools": self.config.mcp_tools.terminal,
            "web_research": self.config.mcp_tools.web_scraping and self.config.web_research.enabled,
            "web_interaction": self.config.mcp_tools.web_interaction,
        }
        if self.config.mcp_tools.web_interaction:
            payload.update({
                "browser_validation_stack": "Python Playwright with preinstalled Chromium",
                "node_js_available_by_default": False,
                "preferred_browser_validation": "Python script importing from playwright.sync_api import sync_playwright",
                "dependency_install_policy": (
                    "Python Playwright is only the default browser-validation path. If the user prompt or plan "
                    "explicitly requires another stack, add a named dependency/setup step with bounded commands "
                    "inside the Docker agent container instead of assuming the tools are already installed."
                ),
            })
        return payload

    def _execution_environment_guidance(self) -> str:
        """Human-readable environment constraints injected before planning starts."""
        if not self.config.mcp_tools.web_interaction:
            return (
                "EXECUTION_ENVIRONMENT:\n"
                "Web/browser interaction is disabled. Do not promise browser or Playwright evidence unless a later "
                "config enables it; use terminating command-line checks instead."
            )
        return (
            "EXECUTION_ENVIRONMENT:\n"
            "The agent runs inside a Docker container with Python, Python Playwright, and Chromium already installed. "
            "The default container does not include Node.js, npm, npx, or @playwright/test. Requirements, assumptions, "
            "plans, and validation commands should therefore prefer Python Playwright scripts using "
            "`from playwright.sync_api import sync_playwright` for generic browser/UI validation. This is a default "
            "preference, not a restriction on the user's technology choice: if the task explicitly requires another "
            "runtime or SDK, add a separate dependency discovery/setup step with bounded commands inside the Docker "
            "agent container, document what was installed, and validate the requested stack directly."
        )

    def _environment_assumption_findings(
        self,
        *,
        requirements: dict[str, Any] | None = None,
        plan: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Detect generated plans that contradict the known container toolchain."""
        if not self.config.mcp_tools.web_interaction:
            return []
        findings: list[str] = []
        text_parts: list[str] = [self.config.project_design.prompt.lower()]
        if requirements:
            text_parts.append(json.dumps(requirements, sort_keys=True).lower())
        if plan:
            text_parts.append(json.dumps(plan, sort_keys=True).lower())
        text = "\n".join(text_parts)
        explicit_setup = self._explicit_dependency_setup_is_present(text)
        unsupported_patterns = [
            (r"\bnpx\b", "npx"),
            (r"\b@playwright/test\b", "@playwright/test"),
            (r"\bnode(?:\.js)?\s+api\b", "Node.js Playwright API"),
            (r"\bplaywright\b[^.\n]{0,80}\binstalled\s+via\s+npm\b", "Playwright installed via npm"),
            (r"\bdependencies?\b[^.\n]{0,80}\bnode(?:\.js)?\b[^.\n]{0,80}\bplaywright\b", "Node.js as a browser-validation dependency"),
        ]
        for pattern, label in unsupported_patterns:
            for match in re.finditer(pattern, text):
                if self._unsupported_tooling_mention_is_negated(text, match.start(), match.end()):
                    continue
                if explicit_setup:
                    continue
                findings.append(
                    f"Generated requirements/plan assume {label}, but the default agent Docker image provides "
                    "Python Playwright with Chromium and no Node/npm/npx/@playwright/test. Either use Python "
                    "Playwright for generic browser validation, or add an explicit bounded dependency/setup step "
                    "inside the Docker agent container for the requested stack."
                )
                break
        return list(dict.fromkeys(findings))

    def _explicit_dependency_setup_is_present(self, text: str) -> bool:
        """Return True when the task or plan explicitly accounts for missing tools."""
        setup_markers = (
            r"\bdependency\s+setup\b",
            r"\bdependencies\s+setup\b",
            r"\bsetup\s+step\b",
            r"\btoolchain\s+setup\b",
            r"\bruntime\s+setup\b",
            r"\bsdk\s+setup\b",
            r"\binstall(?:ing)?\b[^.\n]{0,120}\binside\s+(?:the\s+)?(?:docker|container|agent\s+container)\b",
            r"\b(?:docker|container|agent\s+container)\b[^.\n]{0,120}\binstall(?:ing)?\b",
            r"\bapt(?:-get)?\s+install\b",
            r"\bdotnet-install\b",
            r"\bcheck(?:s|ing)?\b[^.\n]{0,80}\b(?:if\s+)?(?:missing|available|installed)\b[^.\n]{0,120}\binstall\b",
            r"\bbounded\b[^.\n]{0,120}\b(?:install|setup|toolchain|runtime|sdk)\b",
        )
        return any(re.search(pattern, text) for pattern in setup_markers)

    def _unsupported_tooling_mention_is_negated(self, text: str, start: int, end: int) -> bool:
        """Return True for phrases such as "no Node/npm/npx".

        The environment guard should catch accidental Node tooling plans, not
        punish the model for explicitly saying those tools are unavailable.
        """
        context = text[max(0, start - 100):min(len(text), end + 60)]
        negated_patterns = (
            r"\bno\b[^.\n]{0,90}\b(node|npm|npx|@playwright/test)\b",
            r"\bwithout\b[^.\n]{0,90}\b(node|npm|npx|@playwright/test)\b",
            r"\bavoid\b[^.\n]{0,90}\b(node|npm|npx|@playwright/test)\b",
            r"\bdo not\b[^.\n]{0,90}\b(node|npm|npx|@playwright/test)\b",
            r"\bmust not\b[^.\n]{0,90}\b(node|npm|npx|@playwright/test)\b",
            r"\bnot\b[^.\n]{0,60}\bavailable\b[^.\n]{0,60}\b(node|npm|npx|@playwright/test)\b",
            r"\b(node|npm|npx|@playwright/test)\b[^.\n]{0,60}\bnot\b[^.\n]{0,40}\bavailable\b",
        )
        return any(re.search(pattern, context) for pattern in negated_patterns)

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
            "The agent Docker image already includes Python Playwright and Chromium. Unless the config or project "
            "requirements explicitly provide another stack, assume there is no `node`, `npm`, `npx`, or "
            "`@playwright/test` runner. Prefer a Python validation script that imports "
            "`from playwright.sync_api import sync_playwright` and drives Chromium directly. Set bounded browser/page "
            "timeouts appropriate to the expected action (for example 10-15 seconds for simple page loads), always "
            "close browser contexts and local servers in finally blocks, "
            "and write clear evidence to a log/JSON file. If browser launch fails under those bounded timeouts, fall "
            "back to the most direct non-browser verification available and explicitly label that evidence as a fallback. "
            "If the user prompt genuinely requires a different runtime, package manager, SDK, or browser stack, plan "
            "that as a separate dependency/setup step with bounded commands inside the isolated Docker agent container; "
            "do not hide package installation inside an unrelated validation script. "
            "For static HTML/CSS/JS, prefer simple canonical HTML5 files over clever patching; if markup is malformed, "
            "rewrite the complete affected file from one clean template and then stop changing it unless feedback "
            "points to a specific defect. Do not add custom tags, duplicate meta tags, or alternate attribute "
            "spellings while trying to fix unrelated issues. Validate with a browser script or a small Python "
            "structural checker rather than many fragile grep commands.\n"
        )

    def _default_quality_policy_payload(self) -> dict[str, Any]:
        explicit_artifact_only = self._explicit_artifact_only_constraint()
        return {
            "applies": self._default_quality_policy_applies(),
            "explicit_artifact_only_constraint": explicit_artifact_only,
            "assumed_requirement": (
                "Unless the user explicitly says otherwise, code should be well structured, well tested, "
                "well documented, and the first implementation step or first part of the first step should research "
                "required patterns/knowledge, plan project structure, and update the remaining plan if structure "
                "changes the task order. Explicit user constraints such as outputting one named artifact only override "
                "additional documentation or test deliverables, but do not remove the need for validation evidence. "
                "Cited source URLs are required only when web research fetched sources."
            ),
        }

    def _default_quality_instruction(self) -> str:
        if not self._default_quality_policy_applies():
            return (
                "The user prompt appears to override default extra deliverables or code-quality assumptions. "
                "Record that override explicitly, do not add files that violate an output-only constraint, and still "
                "plan validation commands/evidence that prove the requested artifact is correct."
            )
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
        if self._explicit_artifact_only_constraint():
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

    def _explicit_artifact_only_constraint(self) -> bool:
        """Detect explicit user constraints that limit deliverables.

        This is intentionally conservative: it looks for an action verb near
        `only`, so phrases like "A is the only vowel" do not disable the default
        quality policy by themselves.
        """
        prompt = self.config.project_design.prompt.lower()
        patterns = [
            r"\b(?:create|write|produce|output|return|put)\b[^.?!\n]{0,100}\bonly\b",
            r"\bonly\b[^.?!\n]{0,60}\b(?:file|artifact|answer|output)\b",
            r"\b(?:single|one)\b[^.?!\n]{0,40}\b(?:file|artifact|output)\b",
        ]
        return any(re.search(pattern, prompt) for pattern in patterns)

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
            "error identification",
            "identify error",
            "identified error",
            "identifying error",
            "identify failures",
            "failing tests",
            "test suite execution results",
            "error messages",
            "document failures",
            "syntax/import blockers",
            "syntax/import error is identified",
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

    def _run_verified_commands(
        self,
        commands: list[list[str] | dict[str, Any]],
        *,
        source: str,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        commands = [command for command in commands if command]
        if not commands:
            return []
        verification = self._tool_call_verification_phase(commands, source=source, context=context or {})
        decisions = self._tool_verification_decisions(verification, len(commands))
        results: list[dict[str, Any] | None] = [None] * len(commands)
        runnable: list[Any] = []
        runnable_indexes: list[int] = []
        for index, command in enumerate(commands):
            decision = decisions.get(index, {})
            if str(decision.get("decision")) == "blocked":
                results[index] = self._blocked_tool_result(command, decision, verification)
                continue
            runnable.append(command)
            runnable_indexes.append(index)
        if runnable:
            executed = run_commands(
                self.workspace,
                runnable,
                self.config.runtime.command_timeout_seconds,
                self.config.runtime.max_command_timeout_seconds,
                output_limit_chars=self.config.context_compaction.tool_output_max_chars,
            )
            for index, result in zip(runnable_indexes, executed):
                result["tool_verification"] = decisions.get(index, {"decision": "approved"})
                results[index] = result
        return [result for result in results if result is not None]

    def _tool_call_verification_phase(
        self,
        commands: list[Any],
        *,
        source: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        deterministic = self._deterministic_tool_call_findings(commands)
        prompt = {
            "phase": "TOOL_CALL_VERIFICATION_PHASE",
            "source": source,
            "commands": [
                {"index": index, "command": command}
                for index, command in enumerate(commands)
            ],
            "context": context,
            "workflow_state": self._workflow_state_for_prompt(context.get("step") if isinstance(context.get("step"), dict) else None),
            "deterministic_findings": deterministic,
            "runtime_policy": {
                "workspace": str(self.workspace),
                "default_timeout_seconds": self.config.runtime.command_timeout_seconds,
                "max_timeout_seconds": self.config.runtime.max_command_timeout_seconds,
                "tool_output_max_chars": self.config.context_compaction.tool_output_max_chars,
                "terminal_enabled": self.config.mcp_tools.terminal,
            },
            "expected_json": {
                "status": "approved|blocked|needs_revision",
                "summary": "verification summary",
                "commands": [
                    {
                        "index": 0,
                        "decision": "approved|blocked",
                        "risk_level": "low|medium|high",
                        "reason": "reason",
                        "safer_alternative": "optional",
                    }
                ],
            },
        }
        raw = self._feedback_chat(
            "TOOL_CALL_VERIFICATION_PHASE\n"
            "Verify proposed terminal tool calls before execution. Use the whole transcript to understand intent. "
            "Approve commands that are correctly targeted, bounded, and useful for the current plan. Block commands "
            "that may destroy data, target the wrong path/device, depend on malformed quoting, run indefinitely, "
            "or fail to verify the intended behavior. Deterministic findings are authoritative safety signals.\n"
            + json.dumps(prompt),
            temperature=0.0,
        )
        try:
            review = self._extract_json_or_retry(
                raw,
                phase="TOOL_CALL_VERIFICATION_PHASE",
                contract=TOOL_CALL_VERIFICATION_CONTRACT,
                feedback=True,
            )
        except Exception as exc:
            review = {
                "status": "blocked",
                "summary": f"Tool verifier returned malformed JSON: {exc}",
                "commands": [
                    {
                        "index": index,
                        "decision": "blocked",
                        "risk_level": "medium",
                        "reason": "Verifier output was malformed; retry with clearer command intent.",
                    }
                    for index, _command in enumerate(commands)
                ],
                "parse_error": str(exc),
            }
        review = self._normalize_tool_verification(review, commands, deterministic)
        self.conversation.append(
            "user",
            "TOOL_CALL_VERIFICATION_RESULT:\n"
            + json.dumps(self._compact_tool_verification_for_transcript(review), indent=2),
        )
        return review

    def _normalize_tool_verification(
        self,
        review: dict[str, Any],
        commands: list[Any],
        deterministic: list[dict[str, Any]],
    ) -> dict[str, Any]:
        review = dict(review)
        existing = {
            int(item.get("index", -1)): dict(item)
            for item in review.get("commands", [])
            if isinstance(item, dict)
        }
        deterministic_by_index: dict[int, list[str]] = {}
        for finding in deterministic:
            deterministic_by_index.setdefault(int(finding.get("index", -1)), []).append(str(finding.get("reason", "")))
        normalized = []
        any_blocked = False
        for index, command in enumerate(commands):
            item = existing.get(index, {"index": index, "decision": "approved", "risk_level": "low", "reason": "No verifier concern."})
            reasons = [str(item.get("reason") or "")]
            if deterministic_by_index.get(index):
                item["decision"] = "blocked"
                item["risk_level"] = "high"
                reasons.extend(deterministic_by_index[index])
                item["reason"] = "; ".join(reason for reason in reasons if reason)
            if str(item.get("decision")) not in {"approved", "blocked"}:
                item["decision"] = "blocked"
                item["reason"] = str(item.get("reason") or "Verifier did not explicitly approve this command.")
            if item["decision"] == "blocked":
                any_blocked = True
            item["index"] = index
            item["command"] = command
            normalized.append(item)
        review["commands"] = normalized
        if any_blocked:
            review["status"] = "blocked"
        else:
            review["status"] = "approved"
        review.setdefault("summary", "Tool calls verified.")
        return review

    def _tool_verification_decisions(self, verification: dict[str, Any], command_count: int) -> dict[int, dict[str, Any]]:
        decisions: dict[int, dict[str, Any]] = {}
        for item in verification.get("commands", []):
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", -1))
            if 0 <= index < command_count:
                decisions[index] = item
        return decisions

    def _blocked_tool_result(self, command: Any, decision: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
        expected = 0
        if isinstance(command, dict):
            expected = int(command.get("expected_returncode", 0))
            command_parts = command.get("cmd") or command.get("command") or []
        else:
            command_parts = command
        if isinstance(command_parts, str):
            command_parts = [command_parts]
        return {
            "command": [str(part) for part in command_parts],
            "returncode": 126,
            "expected_returncode": expected,
            "returncode_matches_expected": 126 == expected,
            "stdout": "",
            "stderr": (
                "Tool call blocked before execution by verification step: "
                + str(decision.get("reason") or verification.get("summary") or "no reason supplied")
            ),
            "timed_out": False,
            "timeout_seconds": self.config.runtime.command_timeout_seconds,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "blocked_by_tool_verifier": True,
            "tool_verification": decision,
        }

    def _deterministic_tool_call_findings(self, commands: list[Any]) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for index, command in enumerate(commands):
            parts = self._command_parts_for_safety(command)
            if not parts:
                continue
            executable = Path(parts[0]).name
            if executable in {"dd", "mkfs", "fdisk", "parted", "wipefs", "mount", "umount"}:
                findings.append({
                    "index": index,
                    "risk_level": "high",
                    "reason": f"`{executable}` can modify devices or filesystems and requires explicit user approval outside this harness.",
                })
            if executable in {"sudo", "su", "docker", "podman"}:
                findings.append({
                    "index": index,
                    "risk_level": "high",
                    "reason": f"`{executable}` can escape the normal workspace/tool boundary for generated project validation.",
                })
            if executable in {"rm", "cp", "mv", "chmod", "chown"}:
                findings.extend(self._path_sensitive_tool_findings(index, executable, parts))
            if executable == "curl":
                findings.extend(self._curl_payload_findings(index, parts))
            if executable in {"bash", "sh"} and len(parts) >= 3 and parts[1] in {"-c", "-lc"}:
                script = parts[2]
                if any(token in script for token in (" dd ", " mkfs", " fdisk", " parted", " wipefs")):
                    findings.append({
                        "index": index,
                        "risk_level": "high",
                        "reason": "Shell command contains a high-risk disk/filesystem operation.",
                    })
                if "curl" in script and script.count("'") % 2 == 1:
                    findings.append({
                        "index": index,
                        "risk_level": "medium",
                        "reason": "Shell curl command appears to have unbalanced single quotes; use an argv command or write a request body file.",
                    })
        return findings

    def _command_parts_for_safety(self, command: Any) -> list[str]:
        if isinstance(command, dict):
            parts = command.get("cmd") or command.get("command") or []
        else:
            parts = command
        if isinstance(parts, str):
            return [parts]
        return [str(part) for part in parts]

    def _path_sensitive_tool_findings(self, index: int, executable: str, parts: list[str]) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        recursive = any(part in {"-r", "-R", "--recursive"} or (part.startswith("-") and "r" in part.lower()) for part in parts[1:])
        for arg in parts[1:]:
            if arg.startswith("-"):
                continue
            if arg in {"/", "/*", ".", "./"} and executable == "rm" and recursive:
                findings.append({
                    "index": index,
                    "risk_level": "high",
                    "reason": f"`{executable}` recursively targets `{arg}`, which is too broad for a generated tool call.",
                })
            if arg.startswith(("/dev/", "/mnt/", "/media/", "/home/", "/etc/", "/var/")):
                findings.append({
                    "index": index,
                    "risk_level": "high",
                    "reason": f"`{executable}` targets absolute path `{arg}` outside the project workspace.",
                })
        return findings

    def _curl_payload_findings(self, index: int, parts: list[str]) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for pos, part in enumerate(parts):
            if part in {"-d", "--data", "--data-raw", "--data-binary"} and pos + 1 < len(parts):
                payload = parts[pos + 1].strip()
                if payload.startswith("{"):
                    try:
                        json.loads(payload)
                    except json.JSONDecodeError as exc:
                        findings.append({
                            "index": index,
                            "risk_level": "medium",
                            "reason": f"curl JSON payload is malformed before execution: {exc}",
                        })
        return findings

    def _compact_tool_verification_for_transcript(self, review: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": review.get("status"),
            "summary": review.get("summary"),
            "commands": [
                {
                    "index": item.get("index"),
                    "decision": item.get("decision"),
                    "risk_level": item.get("risk_level"),
                    "reason": clamp_text(str(item.get("reason", "")), 800, marker="tool verification reason truncated"),
                }
                for item in review.get("commands", [])[:20]
                if isinstance(item, dict)
            ],
        }

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
            validation_results = self._run_verified_commands(
                validation_commands,
                source="step_feedback_validation",
                context={
                    "step": step,
                    "purpose": "Reviewer-owned validation commands for the current plan step.",
                },
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

    def _final_feedback_tool_evidence(self, step_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Re-run validation commands that still describe the final state.

        A plan may contain intermediate expected-failure checks, for example
        "syntax is fixed, so the test suite now runs but still fails on logic".
        Those commands are useful during that step review, but after later steps
        repair the logic they are historical evidence rather than final-state
        assertions. Final review skips those transient checks so it does not ask
        the implementation agent to make a healthy project fail again.

        The first draft plan can also contain a stale validation path that the
        agents corrected during the step. In that case final review still runs
        the original command, but it also re-runs the accepted validation command
        from the resolved step. Deterministic review can then distinguish stale
        plan evidence from a genuinely broken final project.
        """
        step_result_by_id = {
            str(item.get("step_id")): item
            for item in (step_results or [])
        }
        step_validations: list[dict[str, Any]] = []
        for step in self.plan_steps:
            commands = step.get("validation_commands", [])
            runnable_commands: list[Any] = []
            skipped_commands: list[dict[str, Any]] = []
            for command in commands:
                if self._is_transient_expected_failure_validation(step, command):
                    skipped_commands.append({
                        "command": command,
                        "reason": (
                            "Skipped during final review because this is an intermediate "
                            "expected-failure validation, not a final-state assertion."
                        ),
                    })
                else:
                    runnable_commands.append(command)
            results: list[dict[str, Any]] = []
            if self.config.mcp_tools.terminal and runnable_commands:
                results = self._run_verified_commands(
                    runnable_commands,
                    source="final_feedback_validation",
                    context={
                        "step": step,
                        "purpose": "Final-review validation commands from the accepted plan.",
                    },
                )
            accepted_commands = self._accepted_validation_commands_for_step(
                step_result_by_id.get(str(step.get("id")), {})
            )
            accepted_commands_run: list[Any] = []
            accepted_commands_skipped: list[dict[str, Any]] = []
            runnable_signatures = {
                self._command_signature(existing)
                for existing in runnable_commands
            }
            for command in accepted_commands:
                if self._command_signature(command) in runnable_signatures:
                    continue
                if self._is_transient_expected_failure_validation(step, command):
                    accepted_commands_skipped.append({
                        "command": command,
                        "reason": (
                            "Skipped during final review because this accepted step command "
                            "proved an intermediate expected failure that later steps should resolve."
                        ),
                    })
                    continue
                accepted_commands_run.append(command)
            accepted_results: list[dict[str, Any]] = []
            if self.config.mcp_tools.terminal and accepted_commands_run:
                accepted_results = self._run_verified_commands(
                    accepted_commands_run,
                    source="final_accepted_validation",
                    context={
                        "step": step,
                        "purpose": "Final-review rerun of validation-like commands accepted during implementation.",
                    },
                )
            step_validations.append({
                "step_id": step.get("id"),
                "validation_commands": commands,
                "final_validation_commands_run": runnable_commands,
                "final_validation_commands_skipped": skipped_commands,
                "validation_results": results,
                "accepted_validation_commands_run": accepted_commands_run,
                "accepted_validation_commands_skipped": accepted_commands_skipped,
                "accepted_validation_results": accepted_results,
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

    def _accepted_validation_commands_for_step(self, step_result: dict[str, Any]) -> list[Any]:
        """Return safe validation-like commands from the accepted implementation.

        Implementation turns can contain setup or cleanup commands. Final review
        must not replay those blindly. This helper extracts only commands that
        already passed in the accepted step and look like tests, checks, or
        validation scripts, so they can be rerun as final-state evidence.
        """
        commands: list[Any] = []
        seen: set[tuple[tuple[str, ...], int]] = set()
        for attempt in reversed(step_result.get("attempts") or []):
            review_status = self._status(attempt.get("review") or {})
            if review_status not in {"resolved", "resolved_with_compromise", "skipped_with_note", ""}:
                continue
            implementation = attempt.get("implementation") or {}
            for result in implementation.get("commands") or []:
                if not self._command_returncode_matches_expected(result) or result.get("timed_out"):
                    continue
                if not self._looks_like_validation_evidence_command(result):
                    continue
                command = self._command_spec_from_result(result)
                signature = self._command_signature(command)
                if signature in seen:
                    continue
                seen.add(signature)
                commands.append(command)
            if commands:
                break
        return commands

    def _looks_like_validation_evidence_command(self, result: dict[str, Any]) -> bool:
        command = [str(part) for part in (result.get("command") or [])]
        if not command:
            return False
        if command[0] in {"rm", "mv", "cp", "mkdir", "touch", "git", "sed", "tee", "cat"}:
            return False
        text = " ".join(command).lower()
        if "http.server" in text and not any(marker in text for marker in ("test", "validate", "check")):
            return False
        validation_markers = (
            "unittest",
            "pytest",
            "npm test",
            "test_",
            "/test",
            "tests/",
            "validate",
            "check",
            "assert",
            "playwright",
            "coverage",
        )
        return any(marker in text for marker in validation_markers)

    def _command_spec_from_result(self, result: dict[str, Any]) -> Any:
        command = [str(part) for part in (result.get("command") or [])]
        expected = int(result.get("expected_returncode", 0))
        if expected:
            return {"cmd": command, "expected_returncode": expected}
        return command

    def _command_signature(self, command: Any) -> tuple[tuple[str, ...], int]:
        if isinstance(command, dict):
            parts = command.get("cmd") or command.get("command") or []
            expected = int(command.get("expected_returncode", 0))
        else:
            parts = command
            expected = 0
        return tuple(str(part) for part in parts), expected

    def _evidence_findings(
        self,
        step: dict[str, Any],
        implementation: dict[str, Any],
        feedback_tool_evidence: dict[str, Any] | None = None,
    ) -> list[str]:
        findings: list[str] = []
        implementation_commands = implementation.get("commands", [])
        feedback_results = (feedback_tool_evidence or {}).get("validation_results", [])
        skipped_harness_files = implementation.get("skipped_harness_files", [])
        if skipped_harness_files:
            findings.append(
                "Implementation attempted to overwrite harness-owned state files; these writes were blocked: "
                + ", ".join(str(path) for path in skipped_harness_files)
                + ". Please keep project deliverables in project files and use plan_note for progress."
            )
        expected_validation = bool(step.get("validation_commands"))
        if expected_validation and not feedback_results:
            findings.append(f"{step.get('id', 'step')} has validation criteria but feedback tools produced no validation evidence.")
        for result in feedback_results:
            if result.get("timed_out"):
                findings.append(f"Feedback validation command timed out: {result.get('command')}")
            findings.extend(self._validation_result_integrity_findings(step, result, "Feedback validation"))
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
            findings.extend(self._validation_result_integrity_findings(step, result, "Implementation"))
            if not self._command_returncode_matches_expected(result) and not self._is_failure_investigation_step(step):
                findings.append(
                    f"Implementation command returned {result.get('returncode')} but expected "
                    f"{result.get('expected_returncode', 0)}: {result.get('command')}"
                )
        findings.extend(self._git_diff_findings(step, implementation, feedback_tool_evidence or {}))
        findings.extend(
            self._workspace_reference_findings(
                feedback_tool_evidence or {},
                allow_planned_future_refs=True,
            )
        )
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
        ) or (
            "python -m py_compile" in command_text
            and "is a directory" in stderr.lower()
        )

    def _validation_result_integrity_findings(
        self,
        step: dict[str, Any],
        result: dict[str, Any],
        evidence_label: str,
    ) -> list[str]:
        """Catch commands that pass only because the command itself is broken.

        Expected non-zero return codes are useful for partial-fix and negative
        path evidence, but they can also hide typos such as `python -mm`, where
        the command fails before it touches the project. Keep this deterministic
        so a forgiving reviewer cannot accept a meaningless expected failure.
        """
        command = [str(part) for part in (result.get("command") or [])]
        stderr = str(result.get("stderr") or "")
        findings: list[str] = []
        if len(command) >= 2 and command[0].endswith("python") and command[1] == "-mm":
            findings.append(
                f"{evidence_label} command appears malformed (`python -mm`); request a plan change or corrected command "
                f"before accepting {step.get('id', 'this step')}."
            )
        if len(command) >= 2 and command[0] == "test" and command[1] == "-F":
            findings.append(
                f"{evidence_label} command appears malformed (`test -F`); request a plan change to use `test -f` "
                f"before accepting {step.get('id', 'this step')}."
            )
        if self._looks_like_malformed_grep_max_count(command) or "grep: invalid max count" in stderr.lower():
            findings.append(
                f"{evidence_label} command appears malformed (`grep` max-count flag); request a plan change "
                "to use `grep -q`, a numeric `grep -m <count>`, or a generated validation script."
            )
        if (
            len(command) >= 4
            and command[0].endswith("python")
            and command[1] == "-m"
            and command[2] == "py_compile"
            and "is a directory" in stderr.lower()
        ):
            findings.append(
                f"{evidence_label} command uses `python -m py_compile` on a directory; request a plan change "
                "to use `python -m compileall <dir>` or a generated validation script."
            )
        if "No module named m" in stderr and any(part == "-mm" for part in command):
            findings.append(
                f"{evidence_label} command failed before exercising the project (`No module named m`); "
                "do not treat this as valid expected-failure evidence."
            )
        return findings

    def _looks_like_malformed_grep_max_count(self, command: list[str]) -> bool:
        """Detect grep invocations where `-m` is accidentally glued to text.

        `grep -m` requires a numeric count. Local models sometimes mutate a
        simple `grep -q` check into values such as `grep -md`, which fails
        before validating the workspace and therefore needs plan repair rather
        than another implementation attempt.
        """
        if not command:
            return False
        executable = Path(str(command[0])).name
        if executable != "grep":
            return False
        for index, part in enumerate(str(item) for item in command[1:]):
            if part == "-m":
                next_index = index + 2
                return next_index >= len(command) or not str(command[next_index]).isdigit()
            if part.startswith("--max-count="):
                return not part.removeprefix("--max-count=").isdigit()
            if part.startswith("-m") and part != "-m":
                return not part[2:].isdigit()
        return False

    def _looks_like_unwrapped_expected_failure_result(self, step: dict[str, Any], result: dict[str, Any]) -> bool:
        """Identify reviewer failures caused by a bad plan, not bad code."""
        if int(result.get("expected_returncode", 0)) != 0 or int(result.get("returncode", 0)) == 0:
            return False
        command = result.get("command") or []
        command_text = " ".join(str(part) for part in command)
        stderr = str(result.get("stderr") or "")
        if self._step_expects_nonzero_validation(step):
            return True
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
            if step_result.get("status") != "resolved" and not self._skipped_step_is_superseded_by_final_evidence(
                step_result,
                final_validations.get(step_id),
            ):
                findings.append(f"Step {step_id} ended with status {step_result.get('status')}.")
            attempts = step_result.get("attempts", [])
            if not attempts:
                findings.append(f"Step {step_id} has no attempts.")
            validation = final_validations.get(step_id)
            if validation is not None:
                step = next((item for item in self.plan_steps if str(item.get("id")) == step_id), {})
                results = validation.get("validation_results", [])
                accepted_results = validation.get("accepted_validation_results", [])
                commands_run = validation.get("final_validation_commands_run", validation.get("validation_commands"))
                if commands_run and not results:
                    findings.append(f"Step {step_id} final feedback validation produced no command evidence.")
                for result in results:
                    if result.get("timed_out"):
                        findings.append(f"Step {step_id} final feedback validation timed out: {result.get('command')}")
                    findings.extend(self._validation_result_integrity_findings(step, result, f"Step {step_id} final feedback validation"))
                    if not self._command_returncode_matches_expected(result) and not self._is_failure_investigation_step(step):
                        if self._plan_failure_is_superseded_by_accepted_validation(result, accepted_results):
                            continue
                        findings.append(
                            f"Step {step_id} final feedback validation returned {result.get('returncode')} "
                            f"but expected {result.get('expected_returncode', 0)}: {result.get('command')}"
                        )
                for result in accepted_results:
                    if result.get("timed_out"):
                        findings.append(f"Step {step_id} accepted validation timed out during final review: {result.get('command')}")
                    findings.extend(self._validation_result_integrity_findings(step, result, f"Step {step_id} accepted validation"))
                    if not self._command_returncode_matches_expected(result) and not self._is_failure_investigation_step(step):
                        findings.append(
                            f"Step {step_id} accepted validation returned {result.get('returncode')} "
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
                findings.extend(self._validation_result_integrity_findings({}, result, f"Step {step_id} final attempt"))
        findings.extend(
            self._workspace_reference_findings(
                feedback_tool_evidence or {},
                allow_planned_future_refs=False,
            )
        )
        return findings

    def _skipped_step_is_superseded_by_final_evidence(
        self,
        step_result: dict[str, Any],
        validation: dict[str, Any] | None,
    ) -> bool:
        """Allow final review to rescue a bounded-retry skipped step.

        A step can exhaust its per-step review budget, then be corrected during
        final review. In that case the historical step status remains
        ``skipped_with_note`` even though the final workspace is now healthy.
        This helper keeps deterministic review strict by requiring fresh
        reviewer-owned command evidence before suppressing the historical status
        warning.
        """
        if step_result.get("status") != "skipped_with_note" or validation is None:
            return False
        results = list(validation.get("validation_results") or [])
        accepted_results = list(validation.get("accepted_validation_results") or [])
        all_results = results + accepted_results
        if not all_results:
            return False
        return all(
            self._command_returncode_matches_expected(result) and not result.get("timed_out")
            for result in all_results
        )

    def _plan_failure_is_superseded_by_accepted_validation(
        self,
        plan_result: dict[str, Any],
        accepted_results: list[dict[str, Any]],
    ) -> bool:
        """Treat stale plan-path failures as superseded by rerun accepted tests.

        This does not forgive arbitrary failing tests. It only applies when the
        plan command failed before exercising project behavior, while an
        accepted test/check command from the resolved step was rerun and passed.
        """
        if not accepted_results or not all(
            self._command_returncode_matches_expected(item) and not item.get("timed_out")
            for item in accepted_results
        ):
            return False
        stderr = str(plan_result.get("stderr") or "").lower()
        stdout = str(plan_result.get("stdout") or "").lower()
        text = f"{stdout}\n{stderr}"
        stale_markers = (
            "no module named",
            "can't open file",
            "cannot open file",
            "no such file or directory",
            "module not found",
            "importerror",
            "modulenotfounderror",
        )
        return any(marker in text for marker in stale_markers)

    def _workspace_reference_findings(
        self,
        feedback_tool_evidence: dict[str, Any],
        *,
        allow_planned_future_refs: bool = True,
    ) -> list[str]:
        """Find obvious broken local file references in generated Markdown docs.

        The feedback model sees file snapshots, but local models sometimes miss
        one-character path typos in documentation. This check is intentionally
        narrow: it only scans Markdown-like project files for slash-containing
        local paths and verifies those paths exist in the reviewer snapshot.

        During per-step review, generated README/research notes often mention
        artifacts scheduled for later steps. That is healthy planning, not a
        broken link. The stricter final review disables the planned-reference
        exception once all accepted steps should have produced their artifacts.
        """
        workspace_files = feedback_tool_evidence.get("workspace_files") or []
        existing_files = {str(item.get("path", "")) for item in workspace_files}
        existing_dirs: set[str] = set()
        for path in existing_files:
            parts = path.split("/")
            for index in range(1, len(parts)):
                existing_dirs.add("/".join(parts[:index]))
        ignore_prefixes = (
            ".agent_state/",
            "http://",
            "https://",
        )
        harness_docs = {Path(name).as_posix() for name in self._harness_doc_names()}
        planned_refs = self._planned_local_path_references() if allow_planned_future_refs else set()
        findings: list[str] = []
        for item in workspace_files:
            path = str(item.get("path") or "")
            if path in harness_docs:
                continue
            if not path.endswith((".md", ".markdown", ".txt")):
                continue
            content = str(item.get("content") or "")
            # Paths such as invoice_calc/discounts.py or src/pages/index.html.
            for match in re.finditer(r"(?<![A-Za-z0-9+.-]://)(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", content):
                ref = match.group(0).strip("`'\"),.;:]}")
                if not ref or ref.startswith(ignore_prefixes):
                    continue
                if match.start() > 0 and content[match.start() - 1] in {"$", "~"}:
                    # Environment/home references such as $HOME/.dotnet are
                    # explanatory text, not workspace artifact links.
                    continue
                if "://" in content[max(0, match.start() - 12):match.start() + 3]:
                    # Avoid matching a suffix of an external URL, e.g. the
                    # `ot.net/v1/...` tail inside `https://dot.net/v1/...`.
                    continue
                if ref.split("/", 1)[0].isdigit():
                    # Avoid treating `localhost:8080/game/index.html` as a
                    # local path beginning with the port number.
                    continue
                if "." in ref.split("/", 1)[0] and ref.split("/", 1)[0] not in existing_dirs:
                    # Domain-looking references such as example.com/file are
                    # external, not generated workspace paths.
                    continue
                # Avoid treating prose pairs such as "syntax/import" or
                # "line/logic" as paths. Most real artifact references here
                # name a file with an extension; directory-only references are
                # considered valid only when they already exist.
                basename = ref.rsplit("/", 1)[-1]
                if "." not in basename and ref not in existing_dirs:
                    continue
                if ref in existing_files or ref in existing_dirs:
                    continue
                if ref in planned_refs:
                    continue
                # Skip paths that are clearly external/package-ish rather than
                # generated workspace artifacts.
                if ref.startswith(("usr/", "var/", "tmp/", "workspace/")):
                    continue
                findings.append(
                    f"{path} references missing local path `{ref}`; correct the documentation or create the referenced artifact."
                )
        return sorted(set(findings))

    def _planned_local_path_references(self) -> set[str]:
        """Return local path-looking references already present in the validated plan."""
        text = json.dumps(self.plan_steps, sort_keys=True)
        refs: set[str] = set()
        for match in re.finditer(r"(?<![A-Za-z0-9+.-]://)(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", text):
            ref = match.group(0).strip("`'\"),.;:]}")
            if not ref or ref.split("/", 1)[0].isdigit():
                continue
            if match.start() > 0 and text[match.start() - 1] in {"$", "~"}:
                continue
            if "://" in text[max(0, match.start() - 12):match.start() + 3]:
                continue
            if "." in ref.split("/", 1)[0]:
                continue
            if ref.startswith((".agent_state/", "http://", "https://", "usr/", "var/", "tmp/", "workspace/")):
                continue
            refs.add(ref)
        return refs

    def _enforce_evidence_policy(
        self,
        review: dict[str, Any],
        evidence_findings: list[str],
        review_mode: str,
    ) -> dict[str, Any]:
        if not evidence_findings:
            return review
        review = dict(review)
        if any(
            "Plan validation command appears" in item
            or "command appears malformed" in item
            or "request a plan change" in item
            for item in evidence_findings
        ):
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
        if statuses.issubset({"resolved", "skipped_with_note"}) and final_status in {"resolved", "resolved_with_compromise"}:
            iterations = (final_review or {}).get("iterations") or []
            last_review = iterations[-1].get("review", {}) if iterations else {}
            if not last_review.get("deterministic_evidence_findings"):
                return "resolved"
        if "cannot_resolve" in statuses:
            return "cannot_resolve"
        if "skipped_with_note" in statuses:
            return "resolved_with_skips"
        return "partial"
