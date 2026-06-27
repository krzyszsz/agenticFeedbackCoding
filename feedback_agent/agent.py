from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
from typing import Any

from .bounds import clamp_text, estimate_tokens
from .compaction import maybe_compact
from .config import AgentConfig
from .conversation import Conversation
from .git_tools import HARNESS_ONLY_PATHS, commit_all, ensure_git_repo, git_evidence, reset_to_ref
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


def _strip_visible_reasoning_for_transcript(text: str) -> str:
    """Keep durable chat memory focused on final structured content.

    Some local models emit visible `<think>` blocks even when asked for strict
    JSON. The current phase still receives and parses the raw response, but
    later phases should not inherit hidden-work scratch pads or benchmark-answer
    leakage as durable context.
    """
    stripped = re.sub(r"<think\b[^>]*>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if stripped != text.strip():
        if stripped:
            return "[visible reasoning omitted from durable chat memory]\n" + stripped
        return "[visible reasoning omitted from durable chat memory]"
    return text


def _normalize_workspace_path_text(path: object) -> str:
    """Normalize relative workspace paths without corrupting dotfiles."""
    normalized = Path(str(path)).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _trim_reference_delimiters(text: object) -> str:
    """Trim surrounding prose/Markdown punctuation without corrupting `./foo`."""
    return str(text).strip().lstrip("`'\"([{").rstrip("`'\"),.;:]}")


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
      "description": "first viable approach summary",
      "advantages": ["why this may work"],
      "risks": ["why this may fail"],
      "verification_strategy": "how this path would be verified"
    },
    {
      "id": "B",
      "description": "second materially different viable approach summary",
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
constraints, name what is possible or impossible, and compare at least two
materially different viable approaches before choosing one. Keep the analysis
universal and problem-domain aware; do not inject instructions that target only
one historical failure mode.
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
Remember that evidence commands can have non-zero success semantics: for
example, `git diff --no-index` returns 1 when it successfully finds differences.
Block that pattern unless the command explicitly declares `expected_returncode`
or wraps the diff so the overall validation command exits 0 when the observed
diff is the intended evidence.
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
Do not invent narrow public API details that the user did not specify. In
particular, do not force a return container/record type, serialization format,
file layout, or CLI behavior just because one implementation path is convenient.
Record the ambiguity as an assumption and choose conservative validation that
matches the prompt examples or preserves caller-visible input conventions.
Do not emit chat-template or fake tool-call markers such as <|channel>,
<tool_call>, call:ls_tool, or similar. This harness cannot execute those
markers; validation happens through the parsed JSON `validation_commands` fields
after planning is accepted. Start with `{`, return one JSON object, and stop
immediately after the matching closing `}`. Use as much detail as the task
needs, but avoid padding, repetition, or speculative tool-call syntax.
Commands may be argv lists or {"cmd": ["python", "script.py"], "timeout_seconds": 7200}
when one specific tool call legitimately needs longer than the default timeout.
Do not use string-valued command fields such as {"cmd": "python script.py"};
use argv arrays so quoting and arguments can be verified before execution.
Avoid embedding quote-heavy Python source, f-strings, braces, list
comprehensions, or other brittle snippets inside JSON command strings during
planning when the task allows helper validation files. Prefer simple argv
checks such as ["test", "-f", "README.md"]. For complex validation, add a plan
step that creates a validation script and then run ["python", "validate.py"].
For artifact-only requests, helper files inside the workspace may be forbidden;
in that case use simple inline commands or temporary files outside the
workspace.
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
Do not put `git diff --no-index` in a validation command without accounting for
its exit code: it returns 1 when differences are found. Use `expected_returncode`
for a standalone diff command or a wrapper script that exits 0 after confirming
the diff is expected.
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
Do not resolve ambiguous API details by narrowing them to an unrequested type or
format. If the user did not specify the exact return representation, preserve the
prompt examples and caller-visible input conventions in the plan, or add a
validation script that checks semantic behavior without unnecessary
representation constraints.
Do not embed quote-heavy inline Python source in JSON command strings when the
task allows helper validation files. Prefer simple argv checks such as
["test", "-f", "index.html"], or run a generated validation script such as
["python", "validate.py"]. For artifact-only requests, helper files inside the
workspace may be forbidden; in that case use simple inline commands or temporary
files outside the workspace.
Do not use `python -m py_compile .` or any directory argument with
`py_compile`; use `python -m compileall <dir>` or a generated validation script
for package-wide syntax checks.
If a step intentionally expects a non-zero result, including a partial bug-fix
step where failure logs should now show only logic errors, use a command object
with `expected_returncode` or a wrapper script that returns 0 only when that
specific expected failure is observed.
Do not use string-valued command fields such as {"cmd": "python script.py"};
use argv arrays so quoting and arguments can be verified before execution.
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
Do not let an earlier refined assumption overconstrain the user's API. When the
prompt leaves return representation or file/CLI surface ambiguous, preserve
natural caller-visible conventions and prompt examples instead of converting
values to a different type solely for convenience. If the current plan appears
to require an unrequested representation, request `needs_plan_change`.
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
Do not chain `git diff --no-index` after passing tests with `&&`: it returns 1
when it successfully prints a diff. If diff output is useful as supplemental
evidence, run a wrapper that checks the intended diff and exits 0, or declare
`expected_returncode` for a standalone diff evidence command.
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

    def _artifact_only_guidance(self) -> str:
        if not self._explicit_artifact_only_constraint():
            return ""
        allowed = sorted(self._artifact_only_allowed_paths())
        allowed_text = ", ".join(allowed) if allowed else "the explicitly requested final artifact only"
        return (
            "ARTIFACT_ONLY_CONSTRAINT:\n"
            f"The user explicitly limited workspace deliverables to {allowed_text}. "
            "Do not create helper source files, validation scripts, README files, or other project artifacts in the "
            "workspace unless they are explicitly named by the user. Use inline commands, commands that create "
            "temporary files outside the workspace, or reviewer-owned validation evidence to verify the artifact. "
            "When inline semantic validation needs iteration, prefer expression-style checks such as `sum(... for ... "
            "if ...)` or a multiline shell command instead of one-line compound `for`/`if` Python blocks."
        )

    def _split_model_writable_files(self, files: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        """Keep implementation turns from overwriting harness-owned state.

        PLAN/REQUIREMENTS/RESEARCH files are the workflow control plane. The
        model may read them and the harness updates them, but implementation
        payloads should not replace them with project-local guesses. Blocking
        here is safer than hoping every local model obeys the prompt forever.
        """
        blocked = {_normalize_workspace_path_text(name) for name in self._harness_doc_names()}
        artifact_allowed = self._artifact_only_allowed_paths()
        allowed: list[dict[str, Any]] = []
        skipped: list[str] = []
        for item in files:
            rel = _normalize_workspace_path_text(item.get("path", ""))
            if rel in blocked:
                skipped.append(rel)
                continue
            if self._explicit_artifact_only_constraint() and not self._artifact_path_is_allowed(rel, artifact_allowed):
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
        self.conversation.append(
            "assistant",
            "IMPLEMENTATION_AGENT_RESPONSE:\n" + _strip_visible_reasoning_for_transcript(raw),
        )
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
        self.conversation.append(
            "user",
            "FEEDBACK_AGENT_RESPONSE:\n" + _strip_visible_reasoning_for_transcript(raw),
        )
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
        self.conversation.append(
            "user",
            "FEEDBACK_AGENT_RESPONSE:\n" + _strip_visible_reasoning_for_transcript(raw),
        )
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
        return self._tokens_with_reasoning_room(feedback_cfg, configured, minimum=512)

    def _structured_control_tokens(self, ceiling: int = 4096) -> int:
        """Bound non-file-generation JSON phases.

        Analysis, requirements, and plan-refinement turns are orchestration
        control messages. They should be detailed enough to guide later work,
        but they should not inherit the large implementation payload ceiling
        reserved for generated files. Reasoning models still need enough room
        to emit the required JSON after their thinking budget.
        """
        return self._tokens_with_reasoning_room(self.config.implementation_model, ceiling, minimum=1024)

    def _implementation_payload_tokens(self) -> int:
        """Bound implementation JSON payloads enough to avoid runaway reasoning."""
        return self._tokens_with_reasoning_room(self.config.implementation_model, 4096, minimum=2048)

    def _tokens_with_reasoning_room(self, model_cfg, answer_tokens: int, *, minimum: int) -> int:
        """Reserve output room for a reasoning budget plus final structured JSON.

        Several local reasoning models count visible or server-side reasoning
        against the same response token ceiling used for the final JSON object.
        If the harness caps a structured turn at exactly the reasoning budget,
        the model can exhaust the whole response with `<think>` text and never
        emit parseable JSON. The cap still honors the model's configured maximum.
        """
        answer_budget = max(minimum, int(answer_tokens))
        reasoning_budget = 0
        if getattr(model_cfg, "reasoning_budget_tokens", None) is not None:
            reasoning_budget = max(0, int(model_cfg.reasoning_budget_tokens or 0))
        target = answer_budget + reasoning_budget if reasoning_budget else answer_budget
        return max(minimum, min(int(model_cfg.max_tokens), target))

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
            artifact_repair_text = ""
            if self._explicit_artifact_only_constraint():
                artifact_repair_text = (
                    " For artifact-only prompts, do not propose helper files or generated validation scripts "
                    "inside the workspace unless the user explicitly named those files. Prefer simple expression-style "
                    "validation commands, multiline shell commands, or temporary files outside the workspace."
                )
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
                "wrapper assertion that verifies the non-zero code and error text."
                + artifact_repair_text
                + step_limit_text + "\n\n"
                f"Required contract:\n{contract}\n\n"
                f"Previous response tail for recovery:\n{tail}"
            )
            if feedback:
                repaired = self._feedback_chat(repair_prompt, temperature=0.0)
            else:
                repaired = self._implementation_chat(
                    repair_prompt,
                    max_tokens=self._tokens_with_reasoning_room(
                        self.config.implementation_model,
                        6144,
                        minimum=2048,
                    ),
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
                    "or [\"python\", \"validate.py\"] when helper files are allowed. Do not use inline python -c "
                    "unless it is a simple expression-style command that does not require compound blocks. "
                    + artifact_repair_text + " "
                    "JSON starts with { and ends with }.\n\n"
                    f"Required contract:\n{contract}"
                )
                repaired_minimal = self._implementation_chat(
                    last_chance_prompt,
                    max_tokens=self._tokens_with_reasoning_room(
                        self.config.implementation_model,
                        4096,
                        minimum=2048,
                    ),
                )
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
        repeats the same loop and wastes minutes. This fallback only short-cuts
        clear approval. Negative or mixed reasoning commonly contains useful
        details that must be preserved through the normal JSON repair prompt.
        """
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            return None
        tail = text[-2400:].lower()
        if "TOOL_CALL_VERIFICATION" in phase:
            tool_positive_markers = (
                "i will approve",
                "i'll approve",
                "i approve",
                "will approve",
                "approve this command",
                "this is correct",
                "is correct",
                "safe to run",
                "correctly targeted",
                "bounded",
            )
            tool_negative_markers = (
                "blocked",
                "block this",
                "i will block",
                "unsafe",
                "destructive",
                "wrong path",
                "wrong target",
                "malformed quoting",
                "do not approve",
                "not approve",
                "needs revision",
                "needs_revision",
            )
            tool_positive = any(marker in tail for marker in tool_positive_markers)
            tool_negative = any(marker in tail for marker in tool_negative_markers)
            if tool_positive and not tool_negative:
                return {
                    "status": "approved",
                    "summary": (
                        f"{phase} reviewer emitted reasoning-only output; "
                        "harness inferred command approval from the reviewer text."
                    ),
                    "commands": [],
                    "inferred_from_malformed_response": True,
                    "parse_error": str(parse_error),
                }
            return None
        positive_markers = (
            '"status": "resolved"',
            '"status":"resolved"',
            "'status': 'resolved'",
            "'status':'resolved'",
            '"needs_rework": false',
            '"needs_rework":false',
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
            "project is complete",
            "meets all requirements",
            "all requirements are met",
            "implementation is solid",
            "complete and meets all requirements",
            "no required changes",
            "no changes required",
            "was verified",
            "verified against",
            "returned exit code 0",
            "validation command returned exit code 0",
            "confirming the correctness",
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
        positive_scope = tail
        negative_scope = tail
        if "FINAL_PROJECT_REVIEW" in phase:
            positive_scope = text.lower()
            positive_markers = positive_markers + (
                "everything looks correct",
                "implementation is complete and verified",
                "complete and verified",
                "project is finished",
                "all steps resolved",
                "step is resolved",
            )
            negative_markers = tuple(marker for marker in negative_markers if marker != "missing") + (
                "incorrect result",
                "result is incorrect",
                "mismatch",
                "does not meet",
                "failed validation",
                "validation failed",
                "returned non-zero",
                "cannot verify",
                "not actually complete",
            )
        negative_scope = re.sub(r"""["']?needs_rework["']?\s*:\s*false""", "", negative_scope)
        negative_scope = re.sub(r"""["']?required_changes["']?\s*:\s*\[\s*\]""", "", negative_scope)
        positive = any(marker in positive_scope for marker in positive_markers)
        negative = any(marker in negative_scope for marker in negative_markers)
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
        return None

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
            phase_blocker = self._blocking_phase_step("analysis", analysis_result)
            if phase_blocker is None:
                req_result = self._requirements_refinement_phase(extra_context=retry_context)
                phase_blocker = self._blocking_phase_step("requirements", req_result)
            else:
                req_result = {}
            if phase_blocker is None:
                plan_result = self._plan_validation_phase()
                phase_blocker = self._blocking_phase_step("plan", plan_result)
            else:
                plan_result = {}
            if approach_attempt == 1 and phase_blocker is None:
                git_baseline = self._git_baseline_commit()
            step_results = []
            if phase_blocker is not None:
                step_results = [phase_blocker]
                final_review = {
                    "status": "cannot_resolve",
                    "summary": phase_blocker.get("last_review_summary", "A critical pre-implementation phase failed."),
                    "iterations": [],
                }
            else:
                while True:
                    step = self._next_pending_step()
                    if step is None:
                        break
                    dependency_blocker = self._dependency_blocker_for_step(step)
                    if dependency_blocker is not None:
                        step_results.append(self._blocked_dependency_step_result(step, dependency_blocker))
                    else:
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

    def _blocking_phase_step(self, phase: str, result: dict[str, Any]) -> dict[str, Any] | None:
        """Represent a failed critical gate as a non-implementation step result.

        Analysis, requirements, and plan validation are gates. If any of them
        exhausts retries or otherwise fails, the harness should report that
        failure instead of drifting into implementation with stale or invalid
        control state.
        """
        if self._status(result) == "resolved":
            return None
        resolution = result.get("resolution") if isinstance(result.get("resolution"), dict) else {}
        summary = (
            str(resolution.get("note") or "")
            or str((result.get("iterations") or [{}])[-1].get("review", {}).get("summary") or "")
            or f"{phase} phase did not resolve."
        )
        return {
            "step_id": f"{phase}_phase",
            "status": "cannot_resolve",
            "attempts": [],
            "phase_result": result,
            "last_review_summary": summary,
        }

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
            raw = self._implementation_chat(prompt, max_tokens=self._structured_control_tokens())
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
                if not self._analysis_quality_flag(quality, key):
                    findings.append(f"analysis_quality.{key} is not true.")
        else:
            findings.append("Analysis is missing analysis_quality.")
        return findings

    @staticmethod
    def _analysis_quality_flag(quality: dict[str, Any], key: str) -> bool:
        """Accept common model casing variants for required boolean flags."""
        parts = key.split("_")
        variants = {
            key,
            re.sub(r"_([a-z])", lambda match: match.group(1).upper(), key),
        }
        if len(parts) > 2:
            variants.add(
                "_".join(parts[:2])
                + "".join(part[:1].upper() + part[1:] for part in parts[2:])
            )
        return any(quality.get(variant) is True for variant in variants)

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
                f"{self._artifact_only_guidance()}\n"
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
                f"Extra context: {extra_context or 'none'}\n\n{REQUIREMENTS_CONTRACT}\n{self._artifact_only_guidance()}"
            )
            raw = self._implementation_chat(prompt, max_tokens=self._structured_control_tokens())
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
        computed_answer_findings = self._computed_answer_validation_findings(
            requirements=requirements,
            plan=normalize_plan_steps(requirements.get("plan", [])) if isinstance(requirements, dict) else [],
        )
        public_api_findings = self._public_api_overconstraint_findings(requirements)
        previous_requirements = self.requirements
        previous_plan_steps = self.plan_steps
        try:
            if isinstance(requirements, dict):
                self.requirements = requirements
                self.plan_steps = normalize_plan_steps(requirements.get("plan", []))
            plan_structural_findings = self._plan_structural_findings()
        finally:
            self.requirements = previous_requirements
            self.plan_steps = previous_plan_steps
        deterministic_findings = []
        for item in [*environment_findings, *computed_answer_findings, *public_api_findings, *plan_structural_findings]:
            if item not in deterministic_findings:
                deterministic_findings.append(item)
        prompt = {
            "phase": "REQUIREMENTS_REVIEW_PHASE",
            "iteration": index,
            "project_design": self.config.project_design.prompt,
            "requirements": requirements,
            "web_research_evidence": self.web_research_result,
            "default_quality_policy": self._default_quality_policy_payload(),
            "execution_environment": self._execution_environment_payload(),
            "deterministic_environment_findings": environment_findings,
            "deterministic_requirements_findings": deterministic_findings,
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
            "If deterministic_requirements_findings is non-empty, request correction instead of accepting the "
            "requirements as-is.\n"
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
        if deterministic_findings:
            existing = [str(item) for item in review.get("required_changes", [])]
            review["required_changes"] = existing + [item for item in deterministic_findings if item not in existing]
            if self._status(review) == "resolved":
                review["status"] = "needs_requirements_change"
                review["needs_rework"] = True
                review["summary"] = "Deterministic requirements checks found unresolved validation or environment issues."
        return review

    def _public_api_overconstraint_findings(self, requirements: dict[str, Any]) -> list[str]:
        """Catch requirements that narrow an unspecified public API representation."""
        prompt = self.config.project_design.prompt.lower()
        req_text = json.dumps(requirements, sort_keys=True).lower()
        interval_output_markers = (
            "output intervals will be returned as a list of lists",
            "output format will be a list of lists",
            "output: a list of merged intervals (each interval as a list)",
            "output format: a list of lists",
            "each interval as a list",
            "returned as a list of lists",
        )
        prompt_allows_list_of_lists = any(
            marker in prompt
            for marker in (
                "list of lists",
                "lists of lists",
                "each interval as a list",
                "return lists",
                "returns lists",
            )
        )
        if (
            "merge_intervals" in prompt
            and "interval" in prompt
            and not prompt_allows_list_of_lists
            and any(marker in req_text for marker in interval_output_markers)
        ):
            return [
                "Requirements narrow `merge_intervals` output representation to list-of-lists even though the user "
                "did not specify that exact public API type. Preserve the prompt's ambiguity/caller-visible interval "
                "representation or validate semantic merged pairs without forcing a different container type."
            ]
        return []

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
                "computed-answer artifact tasks use semantic validation that recomputes or independently checks the answer, not only file existence or numeric format",
                "artifact-only prompts do not introduce helper files or validation scripts as workspace deliverables",
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
            f"{self._artifact_only_guidance()}\n"
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
            f"Review: {json.dumps(self._compact_review_for_transcript(review))}\n\n"
            f"{PLAN_REFINEMENT_CONTRACT}\n{self._artifact_only_guidance()}"
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
            if str(step.get("status", "pending")).lower() not in {
                "resolved",
                "cannot_resolve",
                "skipped",
                "skipped_with_note",
            }:
                return step
        return None

    def _dependency_blocker_for_step(self, step: dict[str, Any]) -> dict[str, Any] | None:
        """Return a blocker when a step cannot run because a prerequisite is unresolved.

        The scheduler may continue after a failed step when configured to do so,
        but dependency edges must still be authoritative. Running a dependent
        implementation after its prerequisite failed produces misleading repair
        loops and can hide the actual failure.
        """
        steps_by_id = {str(candidate.get("id")): candidate for candidate in self.plan_steps if candidate.get("id") is not None}
        for dep in step.get("depends_on", []) or []:
            dep_id = str(dep)
            dep_step = steps_by_id.get(dep_id)
            if dep_step is None:
                return {
                    "dependency": dep_id,
                    "dependency_status": "missing",
                    "summary": f"Step {step.get('id')} depends on missing step {dep_id}.",
                }
            dep_status = str(dep_step.get("status", "pending")).lower()
            if dep_status in {"resolved", "skipped_with_note"}:
                continue
            return {
                "dependency": dep_id,
                "dependency_status": dep_status,
                "summary": (
                    f"Step {step.get('id')} depends on {dep_id}, but {dep_id} is {dep_status}; "
                    "the dependent step cannot be executed until its prerequisite is resolved."
                ),
            }
        return None

    def _blocked_dependency_step_result(self, step: dict[str, Any], blocker: dict[str, Any]) -> dict[str, Any]:
        """Mark a step as blocked by dependency state without calling the model."""
        step["status"] = "cannot_resolve"
        summary = str(blocker.get("summary") or "A required dependency was not resolved.")
        self._append_plan_note(f"[{step['id']}] cannot resolve due to dependency: {summary}")
        self._write_plan_doc()
        return {
            "step_id": step["id"],
            "status": "cannot_resolve",
            "blocked_by_dependency": blocker,
            "attempts": [],
        }

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
            if status in {"resolved", "resolved_with_compromise"}:
                step["status"] = "resolved"
                self._append_plan_note(f"[{step['id']}] resolved: {summary}")
                self._write_plan_doc()
                attempts[-1]["git_commit"] = self._git_commit_completed_step(step)
                return {"step_id": step["id"], "status": "resolved", "attempts": attempts}
            if status == "skipped_with_note":
                step["status"] = "skipped_with_note"
                self._append_plan_note(f"[{step['id']}] skipped with note: {summary}")
                self._write_plan_doc()
                return {"step_id": step["id"], "status": "skipped_with_note", "attempts": attempts}
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
            f"{self._artifact_only_guidance()}\n"
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
            f"Current step: {json.dumps(step)}\n\n{IMPLEMENTATION_CONTRACT}\n{self._artifact_only_guidance()}"
        )
        if self._looks_like_browser_step(step):
            prompt += "\n" + self._browser_validation_guidance()
        if self._has_completed_research():
            prompt += (
                "\nWEB_RESEARCH_USAGE_REQUIREMENT:\n"
                f"Use this fetched research evidence and cite source URLs in ARCHITECTURE.md or the relevant deliverable: "
                f"{compact_research_for_prompt(self.web_research_result)}\n"
            )
        raw = self._implementation_chat(prompt, max_tokens=self._implementation_payload_tokens())
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
                "For computed-answer tasks, inspect code and command evidence with bounded sanity checks; do not manually enumerate long candidate sets or re-solve the whole calculation in the review turn.",
                "If semantic proof is absent or too weak, return needs_rework requesting a stronger validation command or verifier instead of replacing missing proof with ad hoc scratch derivation.",
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
                "Return needs_plan_change if this step cannot be independently verified as written, or if reviewer-owned validation is stale/misaligned while stronger implementation-provided validation now matches the chosen approach.",
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
            "against its acceptance criteria and test evidence. Use reviewer-owned validation results as primary "
            "evidence. Do bounded sanity checks, but do not spend the review re-solving exact-answer tasks, "
            "manually enumerating long candidate sets, or performing full arithmetic derivations. If proof is "
            "weak, request stronger validation evidence instead.\n"
            + json.dumps(prompt),
            context_note=(
                "The full multi-turn transcript is stored in .agent_state/conversation.full.jsonl. "
                "Use this compact step-review payload plus reviewer-owned validation reruns. "
                "If the compact evidence shows failed commands, missing files, or no meaningful git diff, "
                "request concrete implementation changes instead of accepting the step. For exact computed outputs, "
                "do not replace command evidence with long manual derivation; ask for better validation if needed. "
                "Do not request git add/commit."
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
            "and all test evidence. Use reviewer-owned validation results and deterministic evidence findings "
            "as the primary proof. Do bounded sanity checks around that evidence, but do not spend the final "
            "review re-solving algorithmic tasks, re-deriving long calculations, or replacing missing validation "
            "with ad hoc scratch work. If validation only proves file shape, existence, or formatting when the "
            "request needs semantic correctness, return needs_rework requesting stronger validation evidence. "
            "Push back if the project lacks proof or contradicts requirements.\n"
            + json.dumps(prompt),
            context_note=(
                "The full multi-turn transcript is stored in .agent_state/conversation.full.jsonl. "
                "Use this compact final-review payload plus reviewer-owned validation reruns to decide. "
                "Prefer the rerun evidence over manual derivations; request better validation when proof is weak. "
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
        truncated = {
            "status": compact.get("status"),
            "needs_rework": compact.get("needs_rework"),
            "summary": compact.get("summary"),
            "required_changes": self._clip_list_for_transcript(compact.get("required_changes", [])),
            "deterministic_evidence_findings": self._clip_list_for_transcript(
                compact.get("deterministic_evidence_findings", [])
            ),
            "review_truncation_note": "Review transcript payload was compacted to fit the live chat budget.",
        }
        if "feedback_tool_evidence_summary" in compact:
            truncated["feedback_tool_evidence_summary"] = self._clip_nested_for_transcript(
                compact["feedback_tool_evidence_summary"],
                string_limit=700,
                list_limit=5,
            )
        truncated_json = json.dumps(truncated, ensure_ascii=False)
        if len(truncated_json) <= limit:
            return truncated
        truncated["review_truncation_note"] = clamp_text(
            truncated_json,
            max(1000, limit // 3),
            marker="review transcript payload truncated",
        )
        return truncated

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

    def _clip_nested_for_transcript(
        self,
        value: Any,
        *,
        string_limit: int = 900,
        list_limit: int = 6,
        depth: int = 0,
    ) -> Any:
        """Recursively trim nested evidence while preserving its shape."""
        if depth > 5:
            return "[nested evidence omitted from live transcript]"
        if isinstance(value, dict):
            return {
                str(key): self._clip_nested_for_transcript(
                    item,
                    string_limit=string_limit,
                    list_limit=list_limit,
                    depth=depth + 1,
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            clipped = [
                self._clip_nested_for_transcript(
                    item,
                    string_limit=string_limit,
                    list_limit=list_limit,
                    depth=depth + 1,
                )
                for item in value[:list_limit]
            ]
            if len(value) > list_limit:
                clipped.append(f"... {len(value) - list_limit} more item(s) omitted from live transcript")
            return clipped
        if isinstance(value, str):
            return self._prompt_excerpt(value, string_limit)
        return value

    def _final_correction_pass(self, attempt: int, review: dict[str, Any]) -> dict:
        prompt = (
            f"FINAL_PROJECT_CORRECTION_PHASE attempt={attempt}\n"
            "Apply only the final review changes needed to make the whole project consistent with requirements. "
            "Include validation commands and test evidence.\n"
            f"Do not include harness-owned state files in the files payload: "
            f"{', '.join(sorted(self._harness_doc_names()))}. The harness creates and updates those files.\n"
            f"{self._artifact_only_guidance()}\n"
            f"Review: {json.dumps(self._compact_review_for_correction(review))}\n\n"
            f"{IMPLEMENTATION_CONTRACT}\n{self._artifact_only_guidance()}"
        )
        if any(self._looks_like_browser_step(step) for step in self.plan_steps):
            prompt += "\n" + self._browser_validation_guidance()
        raw = self._implementation_chat(prompt, max_tokens=self._implementation_payload_tokens())
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
        computed_answer_semantic_validation_present = self._computed_answer_plan_has_semantic_validation(self.plan_steps)
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
            findings.extend(
                self._validation_command_findings(
                    step,
                    computed_answer_semantic_validation_present=computed_answer_semantic_validation_present,
                )
            )
            findings.extend(self._harness_state_file_plan_findings(step))
            findings.extend(self._artifact_only_plan_findings(step))
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
            structure_present = any(
                marker in first_text
                for marker in (
                    "structure",
                    "architecture",
                    "dependencies",
                    "module",
                    "design",
                    "strategy",
                    "layout",
                    "scaffold",
                    "files",
                    "deliverables",
                )
            )
            # Existing-project repair work often starts with architecture mapping
            # rather than a greenfield "plan the structure" step. Treat mapping
            # or dependency analysis as satisfying the planning intent: the agent
            # has to inspect the current shape before changing it.
            planning_present = any(
                marker in first_text
                for marker in (
                    "plan",
                    "order",
                    "mapping",
                    "map",
                    "architecture",
                    "dependencies",
                    "assessment",
                    "assess",
                    "design",
                    "strategy",
                    "approach",
                    "layout",
                    "scaffold",
                    "document",
                    "notes",
                )
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
        findings.extend(self._computed_answer_validation_findings(requirements=self.requirements, plan=self.plan_steps))
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

    def _artifact_only_plan_findings(self, step: dict[str, Any]) -> list[str]:
        """Reject plans that violate explicit single-artifact output requests."""
        if not self._explicit_artifact_only_constraint():
            return []
        allowed = self._artifact_only_allowed_paths()
        text = json.dumps(
            {
                "title": step.get("title", ""),
                "description": step.get("description", ""),
                "acceptance_criteria": step.get("acceptance_criteria", []),
                "validation_commands": step.get("validation_commands", []),
            },
            sort_keys=True,
        )
        disallowed = sorted(
            ref
            for ref in self._file_references_in_text(text)
            if not self._artifact_path_is_allowed(ref, allowed)
            and ref not in {Path(name).as_posix() for name in self._harness_doc_names()}
            and not self._artifact_reference_is_temporary(ref, text)
        )
        if not disallowed:
            return []
        step_id = str(step.get("id") or "step")
        allowed_text = ", ".join(sorted(allowed)) if allowed else "only the explicitly requested artifact"
        return [
            (
                f"{step_id} mentions extra workspace artifact(s) {', '.join(disallowed)} even though the "
                f"user limited deliverables to {allowed_text}. Use inline validation commands or temporary files "
                "outside the workspace instead of creating helper project files."
            )
        ]

    def _validation_command_findings(
        self,
        step: dict[str, Any],
        *,
        computed_answer_semantic_validation_present: bool = False,
    ) -> list[str]:
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
                command_value = command.get("cmd") or command.get("command") or []
                if isinstance(command_value, str):
                    findings.append(
                        f"{step_id} validation command object uses a string-valued cmd. "
                        "Use an argv list such as {\"cmd\": [\"python\", \"-c\", \"...\"]} so quoting and arguments can be verified."
                    )
                    try:
                        raw_parts = shlex.split(command_value)
                    except ValueError:
                        raw_parts = [command_value]
                else:
                    raw_parts = [str(part) for part in command_value]
            elif isinstance(command, str):
                findings.append(
                    f"{step_id} validation command is a string. Use an argv list so quoting and arguments can be verified."
                )
                try:
                    raw_parts = shlex.split(command)
                except ValueError:
                    raw_parts = [command]
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
                    f"{step_id} validation uses a one-line `python -c` compound block that Python cannot parse. "
                    "Replace it with a simpler expression, a multiline shell command, or a generated validation script "
                    "when the task allows helper files."
                )
            if self._looks_like_silent_subprocess_capture_validation(parts):
                findings.append(
                    f"{step_id} validation captures subprocess output but discards it when the child command fails. "
                    "Print or assert the captured stdout/stderr on failure, or use a small validation script, so repair "
                    "iterations can see the real nested error."
                )
            inline_python_syntax_error = self._inline_python_static_syntax_error(raw_parts)
            if inline_python_syntax_error:
                findings.append(
                    f"{step_id} validation contains inline Python that fails a static syntax check: "
                    f"{inline_python_syntax_error}. Replace it with valid inline Python, a multiline shell command, "
                    "or a generated validation script when the task allows helper files."
                )
            if self._looks_like_unwrapped_expected_failure_validation(step, command, parts):
                findings.append(
                    f"{step_id} validation appears to test an expected failure path without declaring expected_returncode "
                    "or wrapping the exception assertion. Replace it with a command object using expected_returncode, "
                    "or a small wrapper command/script that exits 0 only when the expected error occurs."
                )
            if self._validation_command_appears_to_mutate_artifact(raw_parts):
                findings.append(
                    f"{step_id} validation appears to write or mutate the explicitly requested artifact. "
                    "Validation commands must assert the artifact's state after implementation; create or update "
                    "the artifact in the implementation payload or implementation commands instead."
                )
            if (
                self._computed_answer_prompt_requires_semantic_validation()
                and not computed_answer_semantic_validation_present
                and self._looks_like_shape_only_answer_validation(step, raw_parts)
            ):
                findings.append(
                    f"{step_id} validation is shape-only for a computed-answer artifact, and the plan has no "
                    "semantic validation step. Add validation that recomputes the answer or independently checks "
                    "the requested calculation."
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

    def _validation_command_appears_to_mutate_artifact(self, raw_parts: list[str]) -> bool:
        """Detect artifact-only validation commands that create the deliverable.

        Validation commands are allowed to write temporary evidence in broader
        project tasks, but when the user explicitly asked for one final artifact
        the plan must not smuggle artifact creation into validation. A command
        that writes a validator into /tmp and reads the artifact is fine; only
        actual writes to the requested artifact should be blocked here.
        """
        if not self._explicit_artifact_only_constraint():
            return False
        allowed = {path.lower() for path in self._artifact_only_allowed_paths()}
        if not allowed:
            return False
        text = " ".join(raw_parts)
        lower = text.lower()
        if not any(path in lower for path in allowed):
            return False
        for path in allowed:
            if re.search(rf"open\s*\([^)]*{re.escape(path)}[^)]*,[^)]*['\"][wax+]", lower):
                return True
            path_expr = rf"(?:pathlib\.)?path\s*\(\s*['\"](?:\./)?{re.escape(path)}['\"]\s*\)"
            if re.search(path_expr + r"\s*\.\s*(?:write_text|write_bytes)\s*\(", lower):
                return True
            if re.search(rf"['\"](?:\./)?{re.escape(path)}['\"]\s*\)\s*\.\s*(?:write_text|write_bytes)\s*\(", lower):
                return True
            if self._shell_command_writes_artifact(raw_parts, path):
                return True
        return False

    def _shell_command_writes_artifact(self, raw_parts: list[str], allowed_path: str) -> bool:
        """Return True when shell-style mutation targets the workspace artifact."""
        shell_texts: list[str] = []
        if len(raw_parts) >= 3 and Path(raw_parts[0]).name in {"bash", "sh"} and raw_parts[1] in {"-c", "-lc"}:
            shell_texts.append(raw_parts[2])
        else:
            shell_texts.append(" ".join(raw_parts))

        for text in shell_texts:
            lower = text.lower()
            if self._shell_redirection_targets_artifact(lower, allowed_path):
                return True
            if re.search(rf"(?:^|[;&|]\s*)tee(?:\s+-a)?\s+(?:\./)?{re.escape(allowed_path)}(?:\s|$)", lower):
                return True
            if re.search(rf"(?:^|[;&]\s*)(?:touch|truncate)\b[^;&|]*\s(?:\./)?{re.escape(allowed_path)}(?:\s|$)", lower):
                return True
            if re.search(rf"(?:^|[;&]\s*)(?:cp|mv)\b[^;&|]*\s(?:\./)?{re.escape(allowed_path)}(?:\s|$)", lower):
                return True
            if re.search(rf"(?:^|[;&]\s*)(?:sed\s+-i|perl\s+-pi)\b[^;&|]*\s(?:\./)?{re.escape(allowed_path)}(?:\s|$)", lower):
                return True
        return False

    def _shell_redirection_targets_artifact(self, lower_shell_text: str, allowed_path: str) -> bool:
        for match in re.finditer(r"(?:^|\s)(?:>|>>)\s*([^;&|]+)", lower_shell_text):
            target = match.group(1).strip().strip("'\"")
            # Redirection targets can include descriptor syntax such as 2>file;
            # this helper is only concerned with direct workspace artifact paths.
            target = re.sub(r"^\d+", "", target).strip()
            if target in {allowed_path, f"./{allowed_path}"}:
                return True
        return False

    def _computed_answer_prompt_requires_semantic_validation(self) -> bool:
        """Detect answer-only computed artifacts without solving the task.

        The harness must not hard-code benchmark answers. It can still require
        the model-driven plan to prove semantic correctness when the user asks
        for a single computed artifact such as an answer file.
        """
        if not self._explicit_artifact_only_constraint():
            return False
        prompt = self.config.project_design.prompt.lower()
        artifact_markers = ("answer.txt", "single integer", "integer answer", "return only the integer")
        computation_markers = (
            "count",
            "how many",
            "sum",
            "calculate",
            "compute",
            "evaluate",
            "final x",
            "final value",
            "consider integers",
            "permutations",
            "strings over",
        )
        return any(marker in prompt for marker in artifact_markers) and any(
            marker in prompt for marker in computation_markers
        )

    def _computed_answer_validation_findings(
        self,
        *,
        requirements: dict[str, Any] | None = None,
        plan: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Reject validation plans that hard-code a computed answer.

        A good answer-only benchmark may ultimately contain one integer, but
        the validation plan should recompute or independently enumerate that
        integer. Baking a numeric answer into requirements or validation text
        turns the benchmark into confirmation of a guess.
        """
        if not self._computed_answer_prompt_requires_semantic_validation():
            return []
        plan_to_check = plan if plan is not None else self.plan_steps
        payload = {
            "planning_confirmation": (requirements or self.requirements or {}).get("planning_confirmation", {}),
            "plan": plan_to_check,
        }
        text = json.dumps(payload, sort_keys=True).lower()
        findings: list[str] = []
        if not any(marker in text for marker in ("answer.txt", "single integer", "integer answer", "output")):
            return []
        if plan_to_check and not self._computed_answer_plan_has_semantic_validation(plan_to_check):
            findings.append(
                "Computed-answer validation plan does not explicitly require semantic validation. "
                "It must say that a validator recomputes, enumerates, or independently checks the requested calculation "
                "and compares the artifact to that recomputed value."
            )
        hardcoded_patterns = (
            r"\bexpected(?:\s+(?:answer|value|sum|integer|result))?\s*(?:is|=|:)?\s*\(?\d{2,}\)?",
            r"\bcorrect(?:\s+(?:answer|value|sum|integer|result))?\s*(?:is|=|:)?\s*\(?\d{2,}\)?",
            r"\bmatches?\s+(?:the\s+)?(?:expected|correct)[^.;,\n]*\b\d{2,}\b",
            r"\bagainst\s+(?:the\s+)?expected\s+(?:value|answer|sum|integer|result)\s*\(?\d{2,}\)?",
        )
        if not any(re.search(pattern, text) for pattern in hardcoded_patterns):
            return findings
        findings.append(
            "Computed-answer validation appears to hard-code an expected numeric answer. "
            "Replace it with semantic validation that recomputes or independently enumerates the requested calculation "
            "and compares the artifact to the recomputed value without embedding the final numeric answer in the plan."
        )
        return findings

    def _computed_answer_plan_has_semantic_validation(self, plan: list[dict[str, Any]]) -> bool:
        if not self._computed_answer_prompt_requires_semantic_validation():
            return False
        semantic_markers = (
            "recompute",
            "recomputed",
            "recomputing",
            "recalculate",
            "re-calculate",
            "recalculation",
            "re-calculation",
            "re-calculates",
            "re-calculated",
            "independent",
            "independently",
            "enumerate",
            "enumeration",
            "brute force",
            "cross-check",
            "cross check",
            "derive expected",
            "for n in range",
            "itertools",
            "product(",
            "permutations(",
            "sum(",
            "count=0",
            "count = 0",
            "for s in",
        )
        artifact_markers = ("answer.txt", "artifact", "output file")
        comparison_markers = (
            "assert",
            "==",
            "!=",
            "sys.exit",
            "exit(0 if",
            "cmp ",
            "diff ",
        )
        artifact_access_markers = (
            "read_text",
            "open(",
            "cat answer.txt",
            "answer.txt').read",
            'answer.txt").read',
        )
        for step in plan:
            step_text = json.dumps(
                {
                    "title": step.get("title", ""),
                    "description": step.get("description", ""),
                    "acceptance_criteria": step.get("acceptance_criteria", []),
                },
                sort_keys=True,
            ).lower()
            commands = step.get("validation_commands", [])
            for command in commands:
                command_text = json.dumps(command, sort_keys=True).lower()
                combined_text = f"{step_text} {command_text}"
                if not any(marker in combined_text for marker in semantic_markers):
                    continue
                if not any(marker in command_text for marker in artifact_markers):
                    continue
                if not any(marker in command_text for marker in artifact_access_markers):
                    continue
                if any(marker in command_text for marker in comparison_markers):
                    return True
        return False

    def _looks_like_shape_only_answer_validation(self, step: dict[str, Any], raw_parts: list[str]) -> bool:
        """Return True for checks that prove format but not computed correctness."""
        step_text = json.dumps(
            {
                "title": step.get("title", ""),
                "description": step.get("description", ""),
                "acceptance_criteria": step.get("acceptance_criteria", []),
            },
            sort_keys=True,
        ).lower()
        command_text = " ".join(raw_parts).lower()
        combined = f"{step_text} {command_text}"
        if not any(marker in combined for marker in ("answer.txt", "single integer", "integer answer", "output")):
            return False
        shape_markers = (
            "isdigit",
            "test -f",
            "grep",
            "regex",
            "^[0-9]",
            "^[0-9]+$",
            "contains only digits",
            "non-empty",
            "not empty",
            "wc -c",
            "wc -l",
        )
        semantic_markers = (
            "recompute",
            "independent",
            "brute force",
            "enumerate",
            "assert total",
            "expected =",
            "expected=",
            "itertools",
            "for n in range",
            "permutations(",
            "product(",
            "sum(",
        )
        return any(marker in combined for marker in shape_markers) and not any(
            marker in command_text for marker in semantic_markers
        )

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
        """Catch malformed one-line `python -c` compound statements.

        Python accepts some one-line compound statements, but not compound
        blocks after a semicolon (`x=0; for ...`) or nested directly after a
        colon (`for ...: if ...`). Local models often produce those when trying
        to keep validation inline for artifact-only tasks.
        """
        if len(parts) < 3 or not (parts[0].endswith("python") and parts[1] == "-c"):
            return False
        code = parts[2]
        if "\n" in code:
            return False
        return (
            ("try:" in code and "except" in code)
            or re.search(r";\s*(?:for|if|while|try|with|def|class)\b", code) is not None
            or re.search(r":\s*(?:if|for|while|try|with|def|class)\b", code) is not None
        )

    def _looks_like_silent_subprocess_capture_validation(self, parts: list[str]) -> bool:
        """Detect validators that hide nested subprocess stderr/stdout."""
        for _source, code in self._iter_inline_python_snippets(parts):
            lower = code.lower()
            if "subprocess.run" not in lower or "capture_output=true" not in lower:
                continue
            if "exit(" not in lower and "sys.exit" not in lower and "assert" not in lower:
                continue
            emits_captured_output = (
                re.search(r"print\s*\([^)]*\.\s*(?:stdout|stderr)", lower) is not None
                or re.search(r"sys\.(?:stdout|stderr)\.write\s*\([^)]*\.\s*(?:stdout|stderr)", lower) is not None
                or re.search(r"assert\b[^;\n]*,\s*[^;\n]*(?:stdout|stderr)", lower) is not None
            )
            if not emits_captured_output:
                return True
        return False

    def _inline_python_static_syntax_error(self, parts: list[str]) -> str | None:
        """Return a concise syntax error for direct or shell-wrapped Python."""
        for source, code in self._iter_inline_python_snippets(parts):
            if not code.strip() or code.strip().startswith("$"):
                continue
            try:
                compile(code, "<inline-python>", "exec")
            except SyntaxError as exc:
                location = ""
                if exc.lineno is not None:
                    location = f" at line {exc.lineno}"
                    if exc.offset is not None:
                        location += f", column {exc.offset}"
                return f"{source}: {exc.msg}{location}"
        return None

    def _iter_inline_python_snippets(self, parts: list[str]) -> list[tuple[str, str]]:
        snippets: list[tuple[str, str]] = []
        if len(parts) >= 3 and self._is_python_executable_name(parts[0]) and parts[1] == "-c":
            snippets.append(("python -c", parts[2]))
        if len(parts) >= 3 and Path(parts[0]).name in {"bash", "sh"} and parts[1] in {"-c", "-lc"}:
            try:
                shell_parts = shlex.split(parts[2])
            except ValueError:
                return snippets
            for pos, token in enumerate(shell_parts[:-2]):
                if self._is_python_executable_name(token) and shell_parts[pos + 1] == "-c":
                    snippets.append(("shell-wrapped python -c", shell_parts[pos + 2]))
        return snippets

    def _is_python_executable_name(self, value: str) -> bool:
        name = Path(value).name
        return name == "python" or name.startswith("python")

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
        if "[]" in code:
            empty_input_expected_to_fail = (
                "empty" in step_text
                and any(marker in step_text for marker in ("raise", "raises", "exception", "error"))
            )
            empty_input_asserts_success = re.search(r"assert\b[^;]*==\s*\[\]", code) is not None
            if empty_input_expected_to_fail and not empty_input_asserts_success:
                return True
        return "raise " in code or "sys.exit" in code

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
            r"\b(?:create|write|produce|output|return|put)\b\s+[a-z0-9_.-]+\s+only\b",
            r"\b(?:create|write|produce|output|return|put)\b[^.?!\n]{0,100}\bonly\b",
            r"\bonly\b[^.?!\n]{0,60}\b(?:file|artifact|answer|output)\b",
            r"\b(?:single|one)\b[^.?!\n]{0,40}\b(?:file|artifact|output)\b",
        ]
        return any(re.search(pattern, prompt) for pattern in patterns)

    def _artifact_only_allowed_paths(self) -> set[str]:
        """Return explicitly named workspace artifacts allowed by an output-only prompt."""
        if not self._explicit_artifact_only_constraint():
            return set()
        prompt = self.config.project_design.prompt
        allowed: set[str] = set()
        for match in re.finditer(r"\b(?:create|write|produce|output|return|put)\b\s+([A-Za-z0-9_.\-/]+\.[A-Za-z0-9_.-]+)\s+only\b", prompt, flags=re.IGNORECASE):
            allowed.add(_normalize_workspace_path_text(_trim_reference_delimiters(match.group(1))))
        for match in re.finditer(r"\bin\s+([A-Za-z0-9_.\-/]+\.[A-Za-z0-9_.-]+)\b", prompt, flags=re.IGNORECASE):
            allowed.add(_normalize_workspace_path_text(_trim_reference_delimiters(match.group(1))))
        for ref in self._file_references_in_text(prompt):
            allowed.add(ref)
        harness_docs = {_normalize_workspace_path_text(name) for name in self._harness_doc_names()}
        return {path for path in allowed if path and path not in harness_docs}

    def _file_references_in_text(self, text: str) -> set[str]:
        """Extract conservative filename-like references from prose or JSON."""
        file_suffixes = {
            ".cfg",
            ".cs",
            ".csproj",
            ".css",
            ".csv",
            ".gpx",
            ".gitkeep",
            ".html",
            ".ini",
            ".jpeg",
            ".jpg",
            ".js",
            ".json",
            ".jsonl",
            ".log",
            ".md",
            ".out",
            ".png",
            ".py",
            ".sh",
            ".sln",
            ".svg",
            ".toml",
            ".ts",
            ".txt",
            ".xml",
            ".yaml",
            ".yml",
        }
        refs: set[str] = set()
        for match in re.finditer(
            r"(?<![A-Za-z0-9_./-])((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9][A-Za-z0-9_.-]*\.[A-Za-z0-9][A-Za-z0-9_.-]*)(?![A-Za-z0-9_/-])",
            text,
        ):
            ref = _trim_reference_delimiters(match.group(1))
            if not ref:
                continue
            if Path(ref).suffix.lower() not in file_suffixes:
                continue
            refs.add(_normalize_workspace_path_text(ref))
        return refs

    def _artifact_path_is_allowed(self, path: str, allowed: set[str] | None = None) -> bool:
        allowed_paths = allowed if allowed is not None else self._artifact_only_allowed_paths()
        normalized = _normalize_workspace_path_text(path)
        return normalized in allowed_paths

    def _artifact_reference_is_temporary(self, ref: str, text: str) -> bool:
        normalized = _normalize_workspace_path_text(ref)
        return (
            f"/tmp/{normalized}" in text
            or f"tmp/{normalized}" in text
            or f"temp/{normalized}" in text.lower()
            or "temporary files outside the workspace" in text.lower()
        )

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
        if deterministic:
            review = self._normalize_tool_verification(
                {
                    "status": "blocked",
                    "summary": "Deterministic tool-call safety checks blocked one or more commands before model review.",
                    "commands": [],
                    "deterministic_only": True,
                },
                commands,
                deterministic,
            )
            self.conversation.append(
                "user",
                "TOOL_CALL_VERIFICATION_RESULT:\n"
                + json.dumps(self._compact_tool_verification_for_transcript(review), indent=2),
            )
            return review
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
            review = extract_json_object(raw)
        except Exception as initial_exc:
            inferred = self._tool_verification_reasoning_fallback(raw, commands, initial_exc, source=source)
            if inferred is not None:
                review = inferred
            else:
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

    def _tool_verification_reasoning_fallback(
        self,
        raw: str,
        commands: list[Any],
        parse_error: Exception,
        *,
        source: str,
    ) -> dict[str, Any] | None:
        """Infer clear tool-verifier intent from reasoning-only output.

        Tool-call verification is a safety gate. If a local model clearly says a
        proposed command should be blocked but fails to emit JSON, the safe
        fallback is to block the commands and feed that evidence back into the
        normal implementation/review loop.
        """
        text = re.sub(r"\s+", " ", raw).strip().lower()
        if not text:
            return None
        block_markers = (
            "i will block",
            "i'll block",
            "i must block",
            "will block",
            "must block",
            "should block",
            "block this",
            "blocked",
            "unsafe",
            "destructive",
            "wrong target",
            "wrong path",
            "will not create",
            "does not create",
            "will fail",
            "would fail",
            "file not found",
            "filenotfounderror",
        )
        approve_markers = (
            "i will approve",
            "i'll approve",
            "i approve",
            "will approve",
            "approved",
            "approve this command",
            "safe to run",
            "correctly targeted",
            "bounded",
        )
        has_block = any(marker in text for marker in block_markers)
        has_approval = any(marker in text for marker in approve_markers)
        contrast_markers = (" but ", " however ", " except ", " although ", " unsafe ", " wrong ")
        if has_approval and not has_block and not any(marker in text for marker in contrast_markers):
            return {
                "status": "approved",
                "summary": (
                    "Tool verifier emitted reasoning-only output with clear approval intent; "
                    "harness inferred approved tool calls from the reviewer text."
                ),
                "commands": [
                    {
                        "index": index,
                        "decision": "approved",
                        "risk_level": "low",
                        "reason": (
                            "Verifier reasoning indicated the command was bounded and correctly targeted "
                            "for the current step."
                        ),
                    }
                    for index, _command in enumerate(commands)
                ],
                "inferred_from_malformed_response": True,
                "parse_error": str(parse_error),
            }
        if source == "step_feedback_validation":
            return None
        if not has_block:
            return None
        if has_approval and not any(marker in text for marker in ("but", "however", "except")):
            return None
        return {
            "status": "blocked",
            "summary": (
                "Tool verifier emitted reasoning-only output with clear blocking intent; "
                "harness inferred blocked tool calls from the reviewer text."
            ),
            "commands": [
                {
                    "index": index,
                    "decision": "blocked",
                    "risk_level": "medium",
                    "reason": (
                        "Verifier reasoning indicated the proposed command was unsafe, misdirected, "
                        "or would not satisfy the current step."
                    ),
                }
                for index, _command in enumerate(commands)
            ],
            "inferred_from_malformed_response": True,
            "parse_error": str(parse_error),
        }

    def _normalize_tool_verification(
        self,
        review: dict[str, Any],
        commands: list[Any],
        deterministic: list[dict[str, Any]],
    ) -> dict[str, Any]:
        review = dict(review)
        review_status = str(review.get("status") or "").strip().lower()
        approved_like_statuses = {"", "approved", "resolved", "resolved_with_compromise", "skipped_with_note"}
        default_to_blocked = bool(review.get("needs_rework")) or review_status not in approved_like_statuses
        default_reason = str(review.get("summary") or "Verifier did not explicitly approve this command.")
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
            if index in existing:
                item = existing[index]
            elif default_to_blocked:
                item = {
                    "index": index,
                    "decision": "blocked",
                    "risk_level": "medium",
                    "reason": default_reason,
                }
            else:
                item = {
                    "index": index,
                    "decision": "approved",
                    "risk_level": "low",
                    "reason": "No verifier concern.",
                }
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
            expected_returncode = self._command_expected_returncode(command)
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
            if expected_returncode == 0 and self._looks_like_unwrapped_git_diff_no_index(parts):
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        "`git diff --no-index` returns 1 when it successfully finds differences. "
                        "Use a command object with expected_returncode for standalone diff evidence, "
                        "or wrap the diff so the overall validation exits 0 after confirming the intended evidence."
                    ),
                })
            inline_python_syntax_error = self._inline_python_static_syntax_error(parts)
            if inline_python_syntax_error:
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        "Inline Python command fails static syntax check before execution: "
                        + inline_python_syntax_error
                    ),
                })
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

    def _looks_like_unwrapped_git_diff_no_index(self, parts: list[str]) -> bool:
        """Detect diff-evidence commands whose successful evidence exits non-zero."""
        if not parts:
            return False
        executable = Path(parts[0]).name
        if executable == "git":
            lowered = [part.lower() for part in parts]
            return len(lowered) >= 3 and lowered[1] == "diff" and "--no-index" in lowered
        if executable not in {"bash", "sh"} or len(parts) < 3 or parts[1] not in {"-c", "-lc"}:
            return False
        script = parts[2].lower()
        if "git diff" not in script or "--no-index" not in script:
            return False
        neutralizers = (
            "|| true",
            "|| :",
            "if git diff",
            "if ! git diff",
            "case $? in",
            "case ${?} in",
            "case \"$?\" in",
        )
        return not any(marker in script for marker in neutralizers)

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
                "Implementation attempted to write files blocked by harness-owned state or artifact-only policy: "
                + ", ".join(str(path) for path in skipped_harness_files)
                + ". Please keep project deliverables within the allowed workspace artifacts and use plan_note for progress."
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
                    f"{self._command_failure_excerpt(result)}"
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
                if self._looks_like_stale_or_misaligned_plan_validation_result(result, implementation_commands):
                    findings.append(
                        "Plan validation command appears stale or misaligned with the accepted implementation evidence; "
                        "request a plan change with corrected validation commands instead of repeating implementation-only repair."
                    )
        for result in implementation_commands:
            if result.get("timed_out"):
                findings.append(f"Implementation command timed out: {result.get('command')}")
            findings.extend(self._validation_result_integrity_findings(step, result, "Implementation"))
            if not self._command_returncode_matches_expected(result) and not self._is_failure_investigation_step(step):
                findings.append(
                    f"Implementation command returned {result.get('returncode')} but expected "
                    f"{result.get('expected_returncode', 0)}: {result.get('command')}"
                    f"{self._command_failure_excerpt(result)}"
                )
        findings.extend(self._git_diff_findings(step, implementation, feedback_tool_evidence or {}))
        findings.extend(
            self._workspace_reference_findings(
                feedback_tool_evidence or {},
                allow_planned_future_refs=True,
            )
        )
        return findings

    def _command_failure_excerpt(self, result: dict[str, Any], *, limit: int = 700) -> str:
        """Return a bounded output excerpt for deterministic repair findings."""
        chunks = []
        stdout = str(result.get("stdout") or "").strip()
        stderr = str(result.get("stderr") or "").strip()
        if stdout:
            chunks.append(f"stdout: {stdout}")
        if stderr:
            chunks.append(f"stderr: {stderr}")
        if not chunks:
            return ""
        text = " | ".join(chunks)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > limit:
            text = "..." + text[-limit:]
        return f". Output excerpt: {text}"

    def _looks_like_stale_or_misaligned_plan_validation_result(
        self,
        result: dict[str, Any],
        implementation_commands: list[dict[str, Any]],
    ) -> bool:
        """Detect reviewer-owned commands that need plan repair, not code churn."""
        if not implementation_commands or not all(
            self._command_returncode_matches_expected(item) and not item.get("timed_out")
            for item in implementation_commands
        ):
            return False
        stderr = str(result.get("stderr") or "").lower()
        stdout = str(result.get("stdout") or "").lower()
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
        if any(marker in text for marker in stale_markers):
            return True
        if (
            "tool call blocked before execution by verification step" in text
            and any(
                marker in text
                for marker in (
                    "logically flawed",
                    "does not actually verify",
                    "cannot achieve this via simple cli arguments",
                    "stale",
                    "misaligned",
                )
            )
        ):
            return True
        implementation_text = json.dumps(
            [item.get("command") for item in implementation_commands],
            ensure_ascii=False,
        ).lower()
        cli_separator_markers = (" -- ", "'--'", '"--"')
        return (
            "returned non-zero exit status 2" in text
            and any(marker in implementation_text for marker in cli_separator_markers)
        )

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
        lower_stderr = stderr.lower()
        return (
            "python -c" in command_text
            and 'File "<string>"' in stderr
            and "SyntaxError" in stderr
        ) or (
            "python -c" in command_text
            and result.get("blocked_by_tool_verifier")
            and "static syntax check" in lower_stderr
        ) or (
            "python -m py_compile" in command_text
            and "is a directory" in lower_stderr
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
        findings.extend(self._artifact_only_workspace_findings(feedback_tool_evidence or {}))
        return findings

    def _artifact_only_workspace_findings(self, feedback_tool_evidence: dict[str, Any]) -> list[str]:
        if not self._explicit_artifact_only_constraint():
            return []
        allowed = self._artifact_only_allowed_paths()
        harness_docs = {_normalize_workspace_path_text(name) for name in self._harness_doc_names()}
        harness_runtime_paths = {_normalize_workspace_path_text(name) for name in HARNESS_ONLY_PATHS}
        extras: list[str] = []
        for item in feedback_tool_evidence.get("workspace_files", []) or []:
            path = _normalize_workspace_path_text(item.get("path", ""))
            if not path or path.startswith(".agent_state/") or path in harness_docs or path in harness_runtime_paths:
                continue
            if not self._artifact_path_is_allowed(path, allowed):
                extras.append(path)
        if not extras:
            return []
        allowed_text = ", ".join(sorted(allowed)) if allowed else "only the explicitly requested artifact"
        return [
            (
                f"Artifact-only prompt allows {allowed_text}, but the workspace contains extra project artifact(s): "
                + ", ".join(sorted(extras))
            )
        ]

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
        if any(marker in text for marker in stale_markers):
            return True
        accepted_command_text = json.dumps(
            [item.get("command") for item in accepted_results],
            ensure_ascii=False,
        ).lower()
        return (
            "returned non-zero exit status 2" in text
            and any(marker in accepted_command_text for marker in (" -- ", "'--'", '"--"'))
        )

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
                raw_ref = _trim_reference_delimiters(match.group(0))
                explicit_relative_ref = raw_ref.startswith("./")
                ref = _normalize_workspace_path_text(raw_ref)
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
                if ref in existing_files or ref in existing_dirs:
                    continue
                if ref in planned_refs:
                    continue
                if "." in ref.split("/", 1)[0] and ref.split("/", 1)[0] not in existing_dirs and not explicit_relative_ref:
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
            ref = _normalize_workspace_path_text(_trim_reference_delimiters(match.group(0)))
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
            first_finding = evidence_findings[0] if evidence_findings else "no concrete finding recorded"
            review["summary"] = (
                "Please rework this step: deterministic hard-pushback evidence checks failed. "
                f"First finding: {first_finding}"
            )
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
        if scope == "analysis" or scope == "plan" or scope == "final review" or scope.startswith("step "):
            status = "cannot_resolve"
            note = f"Bounded retries exhausted for {scope}; cannot resolve. Last review: {summary}"
        elif self.config.resolution_policy.allow_skip_with_note:
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
