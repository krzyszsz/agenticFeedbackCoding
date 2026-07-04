from __future__ import annotations

import ast
from collections import Counter
import difflib
import itertools
import json
from pathlib import Path
import re
import shlex
import subprocess
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


RUNTIME_STATE_BASENAMES = {
    ".cache",
    ".checkpoint",
    ".lock",
    ".pid",
    ".progress",
    ".state",
    ".watch_state",
}

RUNTIME_STATE_SUFFIXES = (
    ".checkpoint",
    ".lock",
    ".pid",
    ".state",
)

RUNTIME_STATE_EXCLUDED_BASENAMES = {
    ".agent_state",
    ".dockerignore",
    ".editorconfig",
    ".env.example",
    ".eslintignore",
    ".eslintrc",
    ".git",
    ".gitattributes",
    ".github",
    ".gitignore",
    ".prettierignore",
    ".prettierrc",
}

EXECUTABLE_DELIVERABLE_SUFFIXES = (
    ".bash",
    ".cjs",
    ".js",
    ".mjs",
    ".pl",
    ".py",
    ".rb",
    ".sh",
)


JSON_OUTPUT_RULES = """
Output rules:
Return one valid JSON object matching the requested schema. Do not use markdown
fences, `<think>` text, chat-template markers, fake tool-call markers, or
speculative tool syntax. Never wrap the object in ```json or any other code
block. The first character of your response must be `{`; stop after the matching
`}` and avoid padding or unrelated narration. Do not repeat the same JSON key
twice in one object. Inside JSON strings, prefer plain ASCII prose and code-safe notation;
do not use LaTeX/backslash commands such as `\\le`, `\\sum`, or `\\{...\\}` because
they are easy for local models to emit as invalid JSON escapes or corrupted
requirements text.
"""


PROTOCOL_DISCIPLINE_GUIDANCE = """
Protocol discipline:
Workflow decisions are made only through the required JSON fields for the
current phase. Prose, reasoning text, or a near miss from an earlier turn is
context, not an accepted decision. If a previous response missed the schema,
repair it by answering the exact current question in the requested JSON shape;
do not rely on special keywords or implied agreement.
"""


REVIEW_DECISION_OUTPUT_GUIDANCE = """
Review decision output:
This is a feedback/review decision phase. Return only the decision object named
by the current phase schema. Do not answer with the artifact being reviewed:
do not return a replacement plan, requirements payload, implementation files, or
commands unless that exact key is listed in the phase schema. When reviewing a
plan, judge whether it is acceptable and name required changes; do not rewrite
the plan. When reviewing implementation evidence, judge the evidence; do not
emit a new implementation payload.
"""


EXECUTABLE_DELIVERABLE_GUIDANCE = """
Executable deliverables:
When a generated file must be directly executable, include an appropriate
shebang in the file content. The harness marks shebang files executable when it
applies the JSON `files` payload. Validation commands should prove that state
with `test -x ./script`, direct invocation, or both. Do not use `chmod`, `chown`,
or other workspace metadata/source mutation commands on project files as
validation; if executability is missing, repair the file content or request a
plan/implementation correction.
Do not use "executable" merely to mean that a Python script can be run with
`python script.py`. For Python CLI files, require direct `./script.py`
executability only when the user request, existing project convention, or plan
explicitly requires direct invocation or a shebang.
"""


SCOPE_BOUNDARY_GUIDANCE = """
Scope boundary:
Use the original user request, supplied examples, workspace evidence, and
accepted requirements as the scope boundary. Do not turn an unspecified detail
into a requirement, public API, persistence rule, traversal rule, normalization
rule, retry policy, output representation, or validation-only interface. When a
detail is unclear, preserve it as an assumption/open question or validate the
observable behavior without making a hidden caller-visible promise.
"""


REQUIREMENTS_SCOPE_PRESERVATION_GUIDANCE = """
Requirements scope preservation:
During requirements refinement, preserve explicit caller-visible names, data
shapes, invocation surfaces, and output formats from the user's prompt. Do not
turn a requested JSON list into wrapper objects, machine-readable stdout into
pretty/human-readable output, a named setting or environment-style control into
a different interface, or a named artifact into a different filename. If a
documentation artifact is requested without a filename, prefer the conventional
`README.md` unless the request points to a more specific document. Assumptions
may fill gaps, but they must not replace explicit prompt words or create hidden
caller-visible requirements.
For generated CLIs, machine-readable stdout JSON should stay compact unless the
user asks for pretty output. For scripts, uppercase controls named in the prompt
should remain supported as environment variables or equivalent named controls,
not be replaced by flags-only or positional-only interfaces. For validation
scripts requested without an argument contract, provide a useful zero-argument
bounded default.
"""


ORIGINAL_REQUEST_FIT_CHECK_GUIDANCE = """
Original-request fit check:
Before accepting requirements, a plan, final output, or an approach decision,
compare it to the original user prompt instead of only to the refined
requirements or generated tests. Ask what a reasonable user would run, open,
read, or parse without knowing the harness's internal assumptions. If the
workflow invented a filename, required argument, output shape, artifact
location, or validation-only interface that the prompt did not ask for, request
a smaller compatibility repair, requirements correction, or plan correction
instead of accepting the invented surface. If repeated repair cycles keep
validating the invented surface rather than the user-facing one, consider
whether to stop with an explicit compromise or retry from a corrected approach.
"""


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
Do not write project files or compute final deliverables in this phase. The
purpose is to orient later model-driven planning: restate the request, identify
constraints, name what is possible or impossible, compare at least two materially
different viable approaches, and choose the best first approach. Keep the
analysis domain-neutral except for facts supplied by the active request.
If a workspace source snapshot is provided, use it as initial source evidence:
cite relevant paths in `sources_checked`, preserve any gaps, and do not pretend
to have run commands that were not actually run.
""" + SCOPE_BOUNDARY_GUIDANCE + """
When naming tools or libraries, keep each approach internally consistent. Do not
put optional external dependencies inside a dependency-free, standard-library,
or "tiny" path unless the user or workspace already requires them. If tests are
requested but no runner convention is visible, describe the path as using the
standard-library test runner or the existing project runner; do not speculate
about external runners as part of the default path.
""" + JSON_OUTPUT_RULES + """
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
Reject analysis that jumps straight to implementation, considers only one
approach, ignores available workspace/research/source context, or bakes in a
narrow solution that would make the harness less universal.
""" + REVIEW_DECISION_OUTPUT_GUIDANCE + JSON_OUTPUT_RULES + """
"""


APPROACH_REVIEW_CONTRACT = """
Return strict JSON only:
{
  "status": "resolved|try_another_approach|needs_rework|cannot_resolve",
  "needs_rework": false,
  "summary": "whether the executed approach answered the user request",
  "decision": "keep_result|retry_with_new_approach|stop",
  "recommended_next_approach": "only when retrying",
  "evidence_reviewed": ["IDs copied from available_evidence, such as final_review:summary"],
  "runbook_updates": ["note to preserve for the next approach"]
}
Decide whether the completed workflow was the right response to the user's
request. If another approach is warranted, explain the trigger and provide a
new approach direction. Do not retry merely for variety; retry only when the
evidence shows a meaningful gap, a better angle is needed, or the task itself
requires periodic re-checking.
The evidence_reviewed field is a citation protocol, not prose: copy only IDs
from the supplied available_evidence list. Put interpretation in summary or
runbook_updates, not in evidence_reviewed.
""" + REVIEW_DECISION_OUTPUT_GUIDANCE + JSON_OUTPUT_RULES + """
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
Commands are supplied as argv arrays, not through an outer harness shell. For
`["bash", "-lc", "script"]`, the `script` value is one argument passed to bash;
shell expansion inside that script is evaluated by that bash process.
For validation commands that compare literal stdout or file content, check that
regex tools are used safely: a literal structured-output pattern containing
characters such as brackets or braces should use fixed-string matching, an exact
captured-output comparison, or a small validator script instead of accidental
regular-expression matching.
Long-running commands are acceptable when their purpose, bounds, and observable
progress are justified by the task. Prefer commands that emit useful progress
or write bounded evidence, because the harness may ask a progress reviewer
whether a still-running command should continue. `timeout_seconds: 0` means no
hard wall-clock deadline for that command; approve that only when the task
plausibly needs open-ended monitoring or the command has clear progress/evidence
signals for later progress reviews. Use a positive timeout for ordinary tests,
builds, and probes.
Only judge commands present in the supplied `commands` array. Do not mention,
approve, or rely on command indexes that are not in that array. If the supplied
commands are safe but do not prove the stated acceptance criteria, return
`needs_revision` with a safer or stronger validation command.
Remember that evidence commands can have non-zero success semantics: for
example, `git diff --no-index` returns 1 when it successfully finds differences.
Block that pattern unless the command explicitly declares `expected_returncode`
or wraps the diff so the overall validation command exits 0 when the observed
diff is the intended evidence.
""" + REVIEW_DECISION_OUTPUT_GUIDANCE + JSON_OUTPUT_RULES + """
"""


ANTI_TUNNEL_VISION_GUIDANCE = """
Anti-tunnel-vision rule:
Do not agree with a prior model message merely because it sounds confident, and
do not reject it merely because this prompt asks you to challenge it. Compare the
current evidence to the original request, named constraints, and plan state. If
the current direction is still best, say why and continue. If evidence shows it
is stale, wrong, unsafe, or under-verified, request the smallest useful change or
a plan/approach update. If multiple repair cycles keep changing the same
validation/protocol detail without improving the user-visible deliverable, ask
whether the loop is still serving the original request. Prefer a simpler
validation method, explicit compromise, or plan update over another near-identical
retry.
"""


SELF_CHECK_GUIDANCE = """
Evidence-bound self-check:
Silently compare your answer to the original request, available evidence,
current plan state, and named constraints before you return JSON. Keep the
answer if those facts support it. Revise only concrete gaps, contradictions,
missing verification, stale assumptions, or unsafe operations; do not add work
merely to answer hypothetical doubt. Do not include the self-check, uncertainty
notes, or repeated confirmations in the response; return only the requested
schema fields.
"""


REVIEW_CHALLENGE_GUIDANCE = """
Evidence-bound review check:
Look for ways the proposal could be wrong, stale, unsafe, under-verified, or
tunnelled into one approach. Push back only when you can name a concrete issue
or a better verified path. Accept when the evidence satisfies the original
request and current plan. When requesting rework, name the evidence gap,
boundary, and verification standard. Do not prescribe a complete replacement
command, patch, or implementation recipe unless the defect is only command
syntax/quoting and the exact shape is necessary to explain the issue; the next
planning or implementation pass should choose the repair. Report evidence
precisely: do not say tests or commands verified a behavior unless the result
or test content actually exercises it. If you rely on source or file inspection,
label it as source evidence and request stronger validation when inspection is
not enough. Generated tests and validators are evidence, not authority over the
user's requested behavior. If a generated test expectation conflicts with the
original request or accepted requirements, request validator/plan repair or
requirements clarification; do not ask implementation to adopt test-only
expanded semantics. A zero exit code proves only what the command actually
checked; require evidence that would fail for a plausible wrong implementation
of the requested user-visible behavior. Positive presence evidence is not enough
when the request constrains count, absence, uniqueness, ordering,
idempotence, or a one-time side effect; require validation that checks the
relevant boundary. When exact stdout, compact machine-readable output, or a
fixed file format is caller-visible, parsed-only checks can miss regressions;
request at least one captured string or byte-level comparison for that surface.
"""


def _review_prompt_guidance(*extras: str, executable_deliverables: bool = False) -> str:
    """Shared suffix for all feedback/review prompts.

    Contracts alone are not enough for small local models: the live prompt that
    asks for a decision must repeat the role boundary and JSON-only output rule.
    """
    parts = [PROTOCOL_DISCIPLINE_GUIDANCE, REVIEW_DECISION_OUTPUT_GUIDANCE]
    if executable_deliverables:
        parts.append(EXECUTABLE_DELIVERABLE_GUIDANCE)
    parts.extend([REVIEW_CHALLENGE_GUIDANCE, ANTI_TUNNEL_VISION_GUIDANCE])
    parts.extend(extra for extra in extras if extra)
    parts.append(JSON_OUTPUT_RULES)
    return "\n".join(part.strip() for part in parts if part.strip()) + "\n"


STRUCTURAL_REPAIR_GUIDANCE = """
Structural repair rule:
If feedback, command output, or file evidence reports SyntaxError, parser errors,
malformed markup, broken imports, duplicated tags, delimiter mismatches, or
repeated structural damage, treat the whole affected file as suspect. Rebuild the
file from a clean minimal template or scan the complete file for the same class
of defect before returning. Do not claim a structural repair based only on
changing the named line; include a validation command that proves the file parses
or the relevant suite loads.
"""


TOOL_PROGRESS_REVIEW_CONTRACT = """
Return strict JSON only:
{
  "status": "continue|terminate",
  "decision": "continue|terminate",
  "summary": "why the running command should continue or stop",
  "evidence": ["specific current-output or context fact"],
  "risks": ["risk if continued or stopped"],
  "next_check_seconds": 300
}
Review a command that is already running. Use the chat history, current plan,
original request, tool-call verification result, and the bounded live stdout/stderr
snapshot. Do not terminate merely because elapsed time is large or because the
command is quiet; different tasks have different legitimate timelines, and some
commands may intentionally have no hard wall-clock deadline. Terminate only when
the current evidence shows the command is waiting for unavailable input, running
the wrong target, stuck in a hopeless loop, producing irrelevant output,
violating safety/workspace policy, or no longer useful for the current plan. If
output is repetitive, compare it to the last review and ask whether it shows
new useful progress, a stable wait state, or a repeated failure. If the command
is still plausibly advancing toward the current plan, continue it and state the
specific progress signal you relied on. Heartbeats, health checks, elapsed
time, and repeated generic log lines are observability, not task progress;
treat them as weak evidence unless they are tied to a requested monitoring task.
If the right action is uncertain, continue and choose a context-appropriate
next_check_seconds.
""" + REVIEW_DECISION_OUTPUT_GUIDANCE + JSON_OUTPUT_RULES + """
"""


PLAN_SCOPE_RULES = """
Plan scope rules, in priority order:
1. User-requested deliverables and explicit constraints control the scope.
1.5. Preserve the user's behavior boundaries. Do not infer extra behavior,
   caller-visible representation, or validation-only public knobs from nearby
   words, common examples, or a previous failed attempt. If a boundary is
   unclear, keep it explicit as an assumption/open question.
2. For bounded utilities, scripts, functions, exact artifacts, and small bug fixes,
   prefer one compact vertical-slice implementation step with its validation
   commands attached to that same step. For explicit artifact-only outputs, the
   step should write the requested artifact and validate that artifact; do not
   add a separate calculation, probing, or stdout-only step that produces no
   requested deliverable.
3. Use separate plan steps only for real dependencies, broad project phases, or
   explicitly requested deliverables. Do not add a standalone final-verification
   or QA-only step that merely reruns checks the harness already runs during step
   and final review.
4. Preserve named public entrypoints. If the prompt says a script runs, validates,
   or checks a bounded default and does not require arguments, keep a useful direct
   invocation such as `python script.py`; make extra knobs optional. If the prompt
   says the script takes provided input, preserve that required input.
5. Do not invent public API details that the user did not specify. Preserve any
   explicit caller-visible shape, and otherwise validate behavior without
   unnecessary representation constraints.
6. Validation for scripts, CLIs, validators, pages, documents, and generated
   artifacts should include at least one check of the default or most likely
   user-facing invocation, artifact path, or output surface implied by the
   original prompt. Do not validate only a newly invented optional interface.
"""


VALIDATION_COMMAND_RULES = """
Command and validation rules, in priority order:
1. Validation must observe the requested behavior or artifact, not merely create
   files, start services, or restate success. Prefer checks that exit non-zero on
   mismatch and print concise diagnostic evidence on failure. If the requested
   behavior constrains count, absence, uniqueness, ordering, idempotence, or
   one-time side effects, assert that boundary explicitly instead of only
   checking that some positive output exists.
1.5. When stdout or file formatting is part of the caller-visible behavior, such
   as exact stdout, compact machine-readable output, or a fixed file format,
   include at least one captured string or byte-level comparison. Parsing JSON,
   HTML, CSV, or text into an equivalent structure is useful but not sufficient
   to prove the caller-visible formatting surface.
2. Commands are data, not prose. Use a plain argv list for ordinary commands, or
   a command object with `cmd`, `expected_returncode`, and/or
   `timeout_seconds` when metadata is needed. The `cmd` value is always an argv
   list. Put shell syntax inside one `bash -lc` script string when shell
   expansion, pipes, redirects, or multi-command coordination are required; that
   script string is passed as a single argv item, not pre-expanded by the
   harness.
2.5. Avoid fragile shell quoting in validation commands. If inline Python needs
   JSON strings, dict/list literals, or nested quotes inside `bash -lc`, prefer a
   quoted here-doc, temporary validator script, or direct argv-list `python -c`
   command over repeatedly repairing escaped quotes.
3. Expected failure checks must make the intended non-zero outcome explicit
   through `expected_returncode` or a wrapper that exits 0 only after confirming
   the intended failure. Do not hide assertion or child-command failures with
   cleanup, fallback branches, or status-masking shell tails.
4. Long-running or open-ended validation must expose bounded progress evidence
   so the live progress reviewer can decide whether it still serves the plan.
   Use command-specific timeout metadata for known-long finite work, and reserve
   open-ended monitoring for cases where progress review should decide when to
   stop.
5. Validation should not mutate project source or public interfaces merely to
   create a test scenario. Use temporary fixtures/state when needed, clean them
   up without hiding failures, and wire them into the command being tested.
6. Documentation, UI, API, and semantic behavior requirements need content or
   behavior evidence. File existence alone is enough only when existence is the
   actual requirement.
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
""" + SCOPE_BOUNDARY_GUIDANCE + REQUIREMENTS_SCOPE_PRESERVATION_GUIDANCE + PLAN_SCOPE_RULES + EXECUTABLE_DELIVERABLE_GUIDANCE + VALIDATION_COMMAND_RULES + JSON_OUTPUT_RULES + """
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
plan clear.
""" + PLAN_SCOPE_RULES + EXECUTABLE_DELIVERABLE_GUIDANCE + VALIDATION_COMMAND_RULES + JSON_OUTPUT_RULES + """
"""


IMPLEMENTATION_CONTRACT = """
Return strict JSON only:
{
  "plan_note": "progress note for the configured plan file",
  "files": [{"path": "relative/path", "content": "complete file content"}],
  "commands": [
    ["python", "-m", "unittest", "-v"],
    ["bash", "-lc", "test \\\"$(cat out.txt)\\\" = expected"],
    {"cmd": ["python", "cli.py", "--bad-input"], "expected_returncode": 2},
    {"cmd": ["npm", "test"], "timeout_seconds": 600}
  ],
  "test_evidence": ["description of command/report/screenshot evidence produced"],
  "resolution_request": "none|needs_requirements_change|needs_plan_change|cannot_resolve"
}
Only write paths inside the project workspace. Prefer validation commands that
terminate quickly, but request a per-command timeout when a legitimate build,
test, or browser check needs longer. Write the files needed to complete the
current plan step or a coherent vertical slice of it. For large steps, it is fine
to split work across feedback attempts, but do not artificially withhold
inseparable files or documentation that is needed for a high-quality result.
Return a JSON object only. Put every validation command needed to prove the
current step in the machine-readable `commands` field. Do not claim in
`plan_note` or `test_evidence` that a rejected gap was validated unless the
corresponding command is present in `commands` or the requested change is
captured in `resolution_request`. In `test_evidence`, describe validation that
is already present in prior command results or validation that this response is
requesting through `commands`; do not say a current-attempt command passed,
failed, or was verified until the harness has executed it and returned output.
The `files[].content` values are JSON strings. Escape every literal double quote
inside generated file text, including test fixtures and expected JSON strings,
or prefer parsed-value assertions such as `json.loads(...)` over exact embedded
JSON text. Raw content like `{"a":1}` inside a `content` string breaks the
response JSON unless those quotes are escaped.
Do not let an earlier refined assumption overconstrain the user's API. When the
prompt leaves return representation or file/CLI surface ambiguous, keep the API
representation-neutral unless the prompt or supplied examples explicitly
constrain caller-visible shape. If the current plan requires an unrequested
public interface detail, request `needs_plan_change` instead of baking that
detail into the implementation.
For machine-readable stdout JSON, emit compact JSON without indentation or
unsolicited spaces unless the user asks for pretty/human-readable output. For
scripts, preserve uppercase controls named in the prompt as environment-variable
inputs or support them in addition to flags and positional arguments. For
validation scripts, if the user asks for a script that validates/runs a bounded
case and does not specify required arguments, make the script runnable with no
arguments using a small safe default.
The `files` payload creates files, not empty directories. When a step requires
directory scaffolding but no real source file belongs there yet, create a small
placeholder such as `game/js/.gitkeep`, `game/css/.gitkeep`, or
`tests/.gitkeep` so the directory exists and validation commands can prove it.
Avoid unrelated full-project rewrites.
""" + EXECUTABLE_DELIVERABLE_GUIDANCE + STRUCTURAL_REPAIR_GUIDANCE + """
When feedback identifies a narrow defect in otherwise valid code, preserve the
known-good file content and change only the defective lines. Do not introduce
new custom tags, invented attributes, placeholder syntax, duplicate imports, or
gratuitous wording/syntax changes unless the requirement explicitly asks for
them. Stable, boring, canonical source is better than a fresh rewrite that adds
new mistakes.
The `commands` field uses the same command format as validation commands. If you
need information from the workspace, request a command in JSON instead of
pretending to call a tool. If a command needs metadata such as
`expected_returncode` or `timeout_seconds`, use a command object with `cmd`;
otherwise use a plain argv list. The `cmd` value must itself be an argv list,
not a shell string: use {"cmd": ["bash", "-lc", "..."], "timeout_seconds": 600},
not {"cmd": "bash -lc ..."}; never place metadata keys inside an argv list.
""" + VALIDATION_COMMAND_RULES + JSON_OUTPUT_RULES + """
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

PHASE_STATUS_VALUES: dict[str, set[str]] = {
    "PROBLEM_ANALYSIS_REVIEW_PHASE": {"resolved", "needs_rework", "cannot_resolve"},
    "REQUIREMENTS_REVIEW_PHASE": {
        "resolved",
        "needs_rework",
        "needs_requirements_change",
        "cannot_resolve",
        "skipped_with_note",
    },
    "PLAN_VALIDATION_PHASE": {
        "resolved",
        "needs_plan_change",
        "needs_requirements_change",
        "cannot_resolve",
    },
    "STEP_REVIEW_PHASE": {
        "resolved",
        "needs_rework",
        "cannot_resolve",
        "needs_requirements_change",
        "needs_plan_change",
        "skipped_with_note",
        "resolved_with_compromise",
    },
    "FINAL_PROJECT_REVIEW_PHASE": {
        "resolved",
        "needs_rework",
        "cannot_resolve",
        "needs_requirements_change",
        "needs_plan_change",
        "skipped_with_note",
        "resolved_with_compromise",
    },
    "APPROACH_REVIEW_PHASE": {
        "resolved",
        "try_another_approach",
        "needs_rework",
        "cannot_resolve",
    },
    "TOOL_CALL_VERIFICATION_PHASE": {"approved", "blocked", "needs_revision"},
    "TOOL_PROGRESS_REVIEW_PHASE": {"continue", "terminate"},
}

PHASE_DECISION_VALUES: dict[str, set[str]] = {
    "APPROACH_REVIEW_PHASE": {"keep_result", "retry_with_new_approach", "stop"},
    "TOOL_PROGRESS_REVIEW_PHASE": {"continue", "terminate"},
}

SCHEMA_PLACEHOLDER_VALUES = {
    "review summary",
    "whole project review",
    "decision summary",
    "verification summary",
    "concrete final review summary",
    "specific change",
    "specific final change",
    "concrete final change, or empty when resolved",
    "specific analysis gap",
    "specific analysis gap to fix",
    "question",
    "evidence",
    "evidence reviewed",
    "specific command result, file evidence, or reviewer fact",
    "why continue or terminate",
}


FEEDBACK_SYSTEM_PROMPT = """
You are the feedback/review agent in a two-agent development loop.
Read the full transcript, including implementation attempts, prior feedback,
requirements decisions, plan updates, command results, screenshots/reports when
listed, git status/diffs, and unresolved risks. Actively challenge the work
before accepting it, then accept when the evidence is sufficient.
The harness gives you workspace files, command results, and git evidence inside
the prompt. Inspect reviewer-owned validation first. Inspect git diff when git
evidence is present and the step should change workspace files. If an
implementation step made no meaningful workspace changes, request the missing
work instead of accepting statements. Phrase feedback as clear requests:
"Please change X", "Please provide evidence Y", "Please rerun Z".
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
Do not be sycophantic. Treat prior model claims, earlier assumptions, and
exploratory user wording as hypotheses to verify against evidence and the
original request. Challenge tunnel vision, but do not create doubt for its own
sake when the evidence already supports the current path.
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
""" + EXECUTABLE_DELIVERABLE_GUIDANCE + ANTI_TUNNEL_VISION_GUIDANCE + """
""" + PROTOCOL_DISCIPLINE_GUIDANCE + REVIEW_DECISION_OUTPUT_GUIDANCE + JSON_OUTPUT_RULES + """
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
        self.active_repair_findings: list[str] = []
        self.last_requirements_review: dict[str, Any] = {}
        self.approach_history: list[dict[str, Any]] = []
        self.web_research_result: dict[str, Any] = {
            "status": "not_run",
            "requested": False,
            "targets": [],
        }
        self.git_baseline_ref = ""

    def _legacy_semantic_phrase_checks_enabled(self) -> bool:
        """Return whether legacy deterministic semantic phrase tables are enabled.

        The default harness should treat natural-language semantics as model
        judgment, not as a growing table of English phrases. Structural checks
        and command-safety checks still run regardless of this switch.
        """
        return bool(getattr(self.config.quality_policy, "deterministic_semantic_scope_checks", False))

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
                "workspace unless they are explicitly named by the user. Put the requested final artifact itself in "
                "the `files` payload; do not create or overwrite it from the `commands` array. Validation commands "
                "must read or compare the actual requested artifact after the `files` payload is applied. Temporary "
                "files outside the workspace may hold validator code, fixtures, or logs, but must not be used as a "
                "substitute artifact when the real requested artifact can be read directly. "
                "When inline semantic validation needs iteration, prefer JSON-safe single-line expression checks such as "
                "`sum(... for ... if ...)` in `python -c`; do not use compound `def`, `class`, `for`, `while`, `with`, `try`, "
                "shell here-docs, or multiline validators inside plan or command JSON."
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
        active_findings = self.active_repair_findings[-8:] if self.active_repair_findings else ["none"]
        requirements_rejected = self._requirements_draft_is_unaccepted()
        requirements_review_lines = self._latest_requirements_review_lines()
        requirements_label = (
            "Unaccepted requirements draft summary (revise per latest requirements review; do not copy unchanged):"
            if requirements_rejected
            else "Requirements summary:"
        )
        plan_tail = self._safe_file_excerpt(self._plan_path(), 6000, tail=True)
        requirements_tail = self._safe_file_excerpt(self._requirements_path(), 3000, tail=True)
        if requirements_rejected:
            plan_tail = (
                "[unaccepted draft omitted from pinned context because the latest requirements review rejected it; "
                "use the required changes above, the original request, and the current JSON draft only as repair evidence]"
            )
            requirements_tail = plan_tail
        parts = [
            f"Plan file: {self.config.runtime.plan_file}",
            f"Requirements file: {self.config.runtime.requirements_file}",
            f"Research file: {self.config.runtime.research_file}",
            "Step status:",
            *status_lines,
            "Recent plan notes:",
            *(f"- {note}" for note in note_tail),
            "Active repair findings:",
            *(f"- {finding}" for finding in active_findings),
            "Latest requirements review:",
            *requirements_review_lines,
            requirements_label,
            self._requirements_summary_for_prompt(),
            "Problem analysis summary:",
            self._analysis_summary_for_prompt(),
            "Approach history:",
            self._approach_history_summary_for_prompt(),
            f"Web research status: {self.web_research_result.get('status', 'not_run')}",
            "Plan file tail:",
            plan_tail,
            "Requirements file tail:",
            requirements_tail,
            "Research file tail:",
            self._safe_file_excerpt(self._research_path(), 3000, tail=True),
        ]
        return "\n".join(parts)

    def _requirements_draft_is_unaccepted(self) -> bool:
        status = str(self.last_requirements_review.get("status", "")).lower()
        if not status:
            return False
        return status not in {"resolved", "accepted", "ok"}

    def _latest_requirements_review_lines(self) -> list[str]:
        if not self.last_requirements_review:
            return ["- none"]
        status = str(self.last_requirements_review.get("status", "unknown"))
        summary = str(self.last_requirements_review.get("summary", "") or "no summary")
        lines = [
            f"- status: {status}",
            f"- summary: {clamp_text(summary, 500, marker='requirements review summary truncated')}",
        ]
        changes = self.last_requirements_review.get("required_changes", [])
        if changes:
            lines.append("- required changes:")
            lines.extend(f"  - {item}" for item in self._clip_list_for_transcript(changes))
        if self._requirements_draft_is_unaccepted():
            lines.append("- current PLAN/REQUIREMENTS files are unaccepted draft evidence, not approved scope.")
        return lines

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
                    "problem analysis, requirements refinement, plan validation, implementation/review loops, "
                    "final review, and approach review. "
                    f"The harness maintains {self.config.runtime.plan_file} and "
                    f"{self.config.runtime.requirements_file}; read them as workflow memory and return plan_note "
                    "updates instead of editing them as project deliverables. "
                    "Keep all work inside the project workspace. "
                    "The workspace is a git repository when git_policy is enabled; accepted plan steps are "
                    "committed only by the harness after feedback review agrees they are complete. "
                    "Implementation turns may inspect git status and diffs, but must not run git add, "
                    "git commit, git reset, git checkout, or other repository-mutating git commands. "
                    "This transcript is durable chat memory: IMPLEMENTATION_AGENT_REQUEST/RESPONSE and "
                    "FEEDBACK_AGENT_REQUEST/RESPONSE blocks are cumulative context, not isolated prompts. "
                    f"Harness-owned state files are {self.config.runtime.plan_file}, "
                    f"{self.config.runtime.requirements_file}, and {self.config.runtime.research_file}; "
                    "they are control-plane files, not proof of user deliverables. "
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

    def _record_effective_review_if_needed(
        self,
        phase: str,
        review: dict[str, Any],
        *,
        reason: str | None = None,
    ) -> None:
        """Persist deterministic reviewer overrides for later compacted context.

        Feedback model responses are appended before deterministic guardrails
        normalize them. When a guardrail suppresses an unsupported reviewer
        objection, later compaction must see the effective review, not only the
        raw stale rejection.
        """
        if not review.get("suppressed_reviewer_findings") and not reason:
            return
        payload = {
            "phase": phase,
            "harness_effective_review": True,
            "reason": reason or "suppressed_reviewer_finding",
            "status": review.get("status"),
            "needs_rework": bool(review.get("needs_rework")),
            "summary": review.get("summary"),
            "required_changes": self._clip_list_for_transcript(review.get("required_changes", [])),
        }
        if review.get("suppressed_reviewer_findings"):
            payload["suppressed_reviewer_findings"] = self._clip_nested_for_transcript(
                review.get("suppressed_reviewer_findings", []),
                string_limit=800,
                list_limit=3,
            )
        if review.get("deterministic_evidence_findings"):
            payload["deterministic_evidence_findings"] = self._clip_nested_for_transcript(
                review.get("deterministic_evidence_findings", []),
                string_limit=800,
                list_limit=5,
            )
        if review.get("resolution"):
            payload["resolution"] = self._clip_nested_for_transcript(
                review.get("resolution"),
                string_limit=800,
                list_limit=3,
            )
        self.conversation.append("user", "FEEDBACK_AGENT_RESPONSE:\n" + json.dumps(payload, indent=2))

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
        current_question_context: str | None = None,
    ) -> dict[str, Any]:
        """Parse a model JSON response, with one repair turn for malformed output.

        Local models often produce useful content but wrap it in markdown,
        include thinking text, or run out of tokens midway through a long JSON
        object. Crashing loses the whole long run. A bounded repair turn keeps
        the transcript honest while asking the same agent to return a
        machine-parseable object that matches the phase contract.
        """
        try:
            return self._extract_phase_json(raw, phase=phase)
        except Exception as exc:
            recovery_context, omitted_unsafe_tail, omission_reason = self._json_repair_recovery_context(
                raw,
                phase=phase,
                feedback=feedback,
            )
            if omitted_unsafe_tail and not feedback:
                self._replace_last_malformed_response_for_repair(
                    phase,
                    raw,
                    exc,
                    omission_reason=omission_reason,
                )
            parse_error_text = self._json_repair_parse_error_for_prompt(
                exc,
                omitted_unsafe_tail=omitted_unsafe_tail,
                omission_reason=omission_reason,
            )
            step_limit, limit_is_hard = self._configured_plan_step_limit()
            if step_limit and limit_is_hard:
                step_limit_text = f" Keep plans to the hard limit of at most {step_limit} steps."
            elif step_limit:
                step_limit_text = f" Prefer at most {step_limit} steps if that remains verifiable."
            else:
                step_limit_text = ""
            phase_upper = phase.upper()
            include_command_guidance = (
                not feedback
                and (
                    "REQUIREMENTS" in phase_upper
                    or "IMPLEMENT" in phase_upper
                    or '"commands"' in contract
                    or "validation_commands" in contract
                )
            )
            include_artifact_guidance = not feedback and (
                "REQUIREMENTS" in phase_upper or "IMPLEMENT" in phase_upper
            )
            artifact_repair_text = ""
            artifact_only_scope = self._explicit_artifact_only_constraint()
            if artifact_only_scope and include_artifact_guidance:
                artifact_repair_text = (
                    "\nArtifact-only boundary: do not add workspace helper files or generated validation scripts "
                    "unless the user explicitly named them. Use temporary evidence outside the workspace when "
                    "validation needs extra scaffolding."
                )
            command_repair_text = ""
            if include_command_guidance:
                command_repair_text = "\nCommand protocol:\n" + VALIDATION_COMMAND_RULES.strip()
            authoritative_repair_context = ""
            if feedback:
                authoritative_repair_context = (
                    "\n\nAuthoritative current workflow state:\n"
                    + self._prompt_excerpt(self._workflow_state_for_prompt(), 6000)
                )
                if current_question_context:
                    authoritative_repair_context += (
                        "\n\nCurrent review question and supplied evidence:\n"
                        + self._prompt_excerpt(current_question_context, 8000)
                    )
                authoritative_repair_context += (
                    "\n\nThe previous malformed or off-contract response was rejected. Do not treat it as an "
                    "accepted plan, accepted requirements, or completed work. Use it only as a clue about the "
                    "intended review decision, then decide against the authoritative current state above and the "
                    "required contract. Do not name, approve, or preserve commands, files, public options, "
                    "or step references from the rejected response unless the same item also appears in the "
                    "authoritative workflow state or deterministic findings."
                )
            if feedback:
                phase_role_text = (
                    "This is a feedback/review phase. Return a review decision object that matches the "
                    "required contract; do not return requirements, plan, files, or implementation payloads. "
                    "Do not tell the implementation agent to add review-only fields such as `status` to a "
                    "requirements-refinement or implementation response; `status` is your reviewer decision "
                    "field for this repair response. If deterministic findings are present, summarize the "
                    "underlying correction needed in `required_changes` without inventing a task-specific "
                    "solution that the user did not request. "
                )
            else:
                phase_role_text = ""
            command_protocol_sentence = (
                "The harness cannot execute <tool_call> text; commands must be listed in JSON. "
                if include_command_guidance
                else ""
            )
            plan_repair_sentence = (
                "If the previous plan was too large to parse, merge related tasks into a practical "
                "independently verifiable set of steps. Include enough detail for later implementation "
                "and review. "
                if "REQUIREMENTS" in phase_upper
                else ""
            )
            implementation_repair_sentence = (
                "If the previous implementation payload was oversized or malformed, return a coherent "
                "parseable slice of the current step; the feedback loop can request the rest later. "
                "For implementation payloads, remember that `files[].content` values are JSON strings: "
                "escape literal double quotes inside generated source or prefer parsed-value assertions over "
                "raw embedded JSON text such as {\"a\":1}. "
                if "IMPLEMENT" in phase_upper
                else ""
            )
            file_limit_sentence = (
                "Per-attempt file limits are not plan-step limits. "
                if include_artifact_guidance
                else ""
            )
            repair_prompt = (
                f"{phase}_JSON_REPAIR\n"
                f"The previous response could not be parsed as JSON: {parse_error_text}\n"
                f"{PROTOCOL_DISCIPLINE_GUIDANCE.strip()}\n"
                "Return one valid JSON object only. Do not use markdown fences. "
                "Do not include analysis, <think> text, chat-template markers, or fake tool-call markers. "
                + command_protocol_sentence
                + phase_role_text +
                "Start with { and stop immediately after the matching closing }. "
                + plan_repair_sentence
                + implementation_repair_sentence
                + "Use the required contract as the protocol; do not add "
                "task-specific behavior only to satisfy this formatting repair. "
                "Schema example strings are placeholders, not answers: do not copy values such as "
                "`review summary`, `whole project review`, `decision summary`, `specific change`, or "
                "`evidence`; replace them with concrete current content or an empty list when nothing is required. "
                + file_limit_sentence
                + command_repair_text
                + artifact_repair_text
                + step_limit_text
                + authoritative_repair_context
                + "\n\n"
                f"Required contract:\n{contract}\n\n"
                f"{recovery_context}"
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
                return self._extract_phase_json(repaired, phase=phase)
            except Exception as repair_exc:
                if feedback:
                    try:
                        return self._feedback_minimal_json_repair(
                            phase=phase,
                            contract=contract,
                            parse_error=exc,
                            repair_error=repair_exc,
                            current_question_context=current_question_context,
                        )
                    except Exception as final_repair_exc:
                        return self._malformed_feedback_fallback(phase, exc, repair_exc, final_repair_exc)
                if "REQUIREMENTS" not in phase:
                    return self._malformed_implementation_fallback(phase, exc, repair_exc)
                last_chance_prompt = (
                    f"{phase}_MINIMAL_JSON_REPAIR\n"
                    f"The previous repair also failed: {repair_exc}\n"
                    "Return only one valid JSON object. No markdown, thinking text, chat-template markers, "
                    "or fake tool-call markers. Keep the structure practical and parseable: distinct plan "
                    "steps, clear requirements, explicit assumptions, and validation commands that follow "
                    "the command protocol. "
                    + command_repair_text + artifact_repair_text + " "
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
                return self._extract_phase_json(repaired_minimal, phase=phase)

    def _extract_phase_json(self, raw: str, *, phase: str) -> dict[str, Any]:
        payload = extract_json_object(raw)
        payload = self._normalize_schema_placeholders(payload, phase=phase)
        issue = self._phase_contract_issue(payload, phase)
        if issue:
            raise ValueError(f"JSON object did not match {phase} contract: {issue}")
        return payload

    @staticmethod
    def _normalize_schema_placeholders(payload: dict[str, Any], *, phase: str) -> dict[str, Any]:
        """Repair harmless schema-example text without accepting missing decisions.

        Some local reviewers copy a schema example into a low-value prose field
        while still returning concrete status and evidence. Treat that as a
        protocol blemish only when the rest of the structured decision is usable;
        keep missing evidence, placeholder evidence, and non-resolved reviews
        with no required changes as contract failures.
        """
        if not isinstance(payload, dict) or phase != "FINAL_PROJECT_REVIEW_PHASE":
            return payload
        normalized = dict(payload)

        def is_placeholder(value: object) -> bool:
            return isinstance(value, str) and value.strip().lower() in SCHEMA_PLACEHOLDER_VALUES

        for key in ("required_changes", "verification_evidence"):
            value = normalized.get(key)
            if isinstance(value, list):
                normalized[key] = [item for item in value if not is_placeholder(item)]

        if is_placeholder(normalized.get("summary")) and normalized.get("verification_evidence"):
            normalized["summary"] = "Final review resolved based on supplied verification evidence."
        return normalized

    @staticmethod
    def _phase_contract_issue(payload: dict[str, Any], phase: str) -> str:
        if not isinstance(payload, dict):
            return "top-level JSON value is not an object"
        if phase == "PROBLEM_ANALYSIS_PHASE":
            required = {
                "problem_restatement": str,
                "possible_solution_paths": list,
                "recommended_path": dict,
                "analysis_quality": dict,
            }
            return FeedbackLoopAgent._missing_or_mistyped_contract_field(payload, required)
        if phase == "REQUIREMENTS_REFINEMENT_PHASE":
            required = {
                "project_summary": str,
                "refined_requirements": list,
                "assumptions": list,
                "planning_confirmation": dict,
                "plan": list,
            }
            return FeedbackLoopAgent._missing_or_mistyped_contract_field(payload, required)
        if phase == "PLAN_REFINEMENT_PHASE":
            required = {"plan": list, "planning_confirmation": dict}
            return FeedbackLoopAgent._missing_or_mistyped_contract_field(payload, required)
        if phase in {"IMPLEMENT_PLAN_STEP_PHASE", "FINAL_PROJECT_CORRECTION_PHASE"}:
            if not any(key in payload for key in ("files", "commands", "test_evidence", "plan_note", "resolution_request")):
                return "implementation payload is missing files, commands, test_evidence, plan_note, and resolution_request"
        if phase == "TOOL_CALL_VERIFICATION_PHASE":
            issue = FeedbackLoopAgent._missing_or_mistyped_contract_field(
                payload,
                {"status": str, "commands": list},
            )
            if issue:
                return issue
            issue = FeedbackLoopAgent._enum_contract_issue(payload, "status", PHASE_STATUS_VALUES[phase])
            if issue:
                return issue
            issue = FeedbackLoopAgent._schema_placeholder_contract_issue(payload)
            if issue:
                return issue
            return FeedbackLoopAgent._tool_command_decision_contract_issue(payload)
        if phase == "TOOL_PROGRESS_REVIEW_PHASE":
            issue = FeedbackLoopAgent._missing_or_mistyped_contract_field(
                payload,
                {"status": str, "decision": str, "summary": str, "evidence": list, "risks": list},
            )
            if issue:
                return issue
            issue = FeedbackLoopAgent._enum_contract_issue(payload, "status", PHASE_STATUS_VALUES[phase])
            if issue:
                return issue
            issue = FeedbackLoopAgent._enum_contract_issue(payload, "decision", PHASE_DECISION_VALUES[phase])
            if issue:
                return issue
            return FeedbackLoopAgent._schema_placeholder_contract_issue(payload)
        if phase == "APPROACH_REVIEW_PHASE":
            issue = FeedbackLoopAgent._missing_or_mistyped_contract_field(
                payload,
                {"status": str, "summary": str, "decision": str},
            )
            if issue:
                return issue
            issue = FeedbackLoopAgent._enum_contract_issue(payload, "status", PHASE_STATUS_VALUES[phase])
            if issue:
                return issue
            issue = FeedbackLoopAgent._enum_contract_issue(payload, "decision", PHASE_DECISION_VALUES[phase])
            if issue:
                return issue
            issue = FeedbackLoopAgent._schema_placeholder_contract_issue(payload)
            if issue:
                return issue
            return FeedbackLoopAgent._approach_review_evidence_contract_issue(payload)
        if phase == "PLAN_VALIDATION_PHASE":
            issue = FeedbackLoopAgent._missing_or_mistyped_contract_field(
                payload,
                {"status": str, "summary": str},
            )
            if issue:
                return issue
            issue = FeedbackLoopAgent._enum_contract_issue(payload, "status", PHASE_STATUS_VALUES[phase])
            if issue:
                return issue
            if "required_changes" in payload and not isinstance(payload["required_changes"], list):
                return "required_changes is not list"
            issue = FeedbackLoopAgent._schema_placeholder_contract_issue(payload)
            if issue:
                return issue
            status = str(payload.get("status") or "").strip()
            if status == "resolved":
                return FeedbackLoopAgent._plan_validation_resolution_contract_issue(payload)
            if "required_changes" not in payload:
                return "missing required_changes"
            return ""
        if phase == "FINAL_PROJECT_REVIEW_PHASE":
            issue = FeedbackLoopAgent._missing_or_mistyped_contract_field(
                payload,
                {
                    "status": str,
                    "summary": str,
                    "required_changes": list,
                    "verification_evidence": list,
                },
            )
            if issue:
                return issue
            if "needs_rework" in payload and not isinstance(payload["needs_rework"], bool):
                return "needs_rework is not bool"
            issue = FeedbackLoopAgent._enum_contract_issue(payload, "status", PHASE_STATUS_VALUES[phase])
            if issue:
                return issue
            issue = FeedbackLoopAgent._schema_placeholder_contract_issue(payload)
            if issue:
                return issue
            status = str(payload.get("status") or "").strip()
            if status not in {"resolved", "skipped_with_note", "resolved_with_compromise"} and not payload["required_changes"]:
                return "required_changes is empty for non-resolved final review"
            if status in {"resolved", "resolved_with_compromise"} and not payload["verification_evidence"]:
                return "verification_evidence is empty for resolved final review"
            return ""
        if phase.endswith("_REVIEW_PHASE"):
            issue = FeedbackLoopAgent._missing_or_mistyped_contract_field(
                payload,
                {"status": str, "summary": str},
            )
            if issue:
                return issue
            allowed_statuses = PHASE_STATUS_VALUES.get(phase, REVIEW_STATUSES)
            issue = FeedbackLoopAgent._enum_contract_issue(payload, "status", allowed_statuses)
            if issue:
                return issue
            if "required_changes" in payload and not isinstance(payload["required_changes"], list):
                return "required_changes is not list"
            issue = FeedbackLoopAgent._schema_placeholder_contract_issue(payload)
            if issue:
                return issue
            status = str(payload.get("status") or "").strip()
            if status not in {"resolved", "skipped_with_note", "resolved_with_compromise"}:
                if "required_changes" not in payload:
                    return "missing required_changes"
            return ""
        return ""

    @staticmethod
    def _plan_validation_resolution_contract_issue(payload: dict[str, Any]) -> str:
        """Require concrete verification coverage before accepting a plan."""
        confirmation = payload.get("planning_confirmation")
        if not isinstance(confirmation, dict):
            return "missing planning_confirmation"

        def _true_flag(primary: str, alias: str) -> bool:
            return confirmation.get(primary) is True or confirmation.get(alias) is True

        for primary, alias in (
            ("feasible", "is_feasible"),
            ("clear", "is_clear"),
            ("verifiable", "is_verifiable"),
        ):
            if not _true_flag(primary, alias):
                return f"planning_confirmation.{primary} is not true"
        matrix = confirmation.get("verification_matrix")
        if not isinstance(matrix, list) or not matrix:
            return "planning_confirmation.verification_matrix is missing or empty"
        for index, item in enumerate(matrix):
            if not isinstance(item, dict):
                return f"planning_confirmation.verification_matrix[{index}] is not object"
            step_id = item.get("step_id")
            how_verified = item.get("how_verified")
            if not isinstance(step_id, str) or not step_id.strip():
                return f"planning_confirmation.verification_matrix[{index}].step_id is missing"
            if not isinstance(how_verified, str) or not how_verified.strip():
                return f"planning_confirmation.verification_matrix[{index}].how_verified is missing"
        return ""

    @staticmethod
    def _missing_or_mistyped_contract_field(payload: dict[str, Any], required: dict[str, type]) -> str:
        for key, expected_type in required.items():
            if key not in payload:
                return f"missing {key}"
            if not isinstance(payload[key], expected_type):
                return f"{key} is not {expected_type.__name__}"
        return ""

    @staticmethod
    def _enum_contract_issue(payload: dict[str, Any], key: str, allowed: set[str]) -> str:
        value = str(payload.get(key) or "").strip()
        if value not in allowed:
            return f"{key} must be one of {sorted(allowed)}, got {value!r}"
        return ""

    @staticmethod
    def _schema_placeholder_contract_issue(payload: dict[str, Any]) -> str:
        for key in ("summary", "compromise_note"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip().lower() in SCHEMA_PLACEHOLDER_VALUES:
                return f"{key} still contains a schema placeholder instead of a concrete current value"
        for key in (
            "required_changes",
            "quality_questions",
            "cross_check_questions",
            "verification_evidence",
            "evidence_reviewed",
            "runbook_updates",
        ):
            value = payload.get(key)
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, str) and item.strip().lower() in SCHEMA_PLACEHOLDER_VALUES:
                    return f"{key} contains a schema placeholder instead of concrete current content"
        return ""

    @staticmethod
    def _tool_command_decision_contract_issue(payload: dict[str, Any]) -> str:
        for item in payload.get("commands", []):
            if not isinstance(item, dict):
                return "commands contains a non-object item"
            if "index" not in item:
                return "command decision is missing index"
            decision = str(item.get("decision") or "").strip()
            if decision not in {"approved", "blocked"}:
                return f"command decision must be approved or blocked, got {decision!r}"
        return ""

    @staticmethod
    def _approach_review_evidence_contract_issue(payload: dict[str, Any]) -> str:
        evidence = payload.get("evidence_reviewed")
        if not isinstance(evidence, list) or not evidence:
            return "evidence_reviewed must list available_evidence IDs"
        allowed_prefixes = (
            "project_design:",
            "analysis:",
            "requirements:",
            "plan:",
            "step_result:",
            "final_review:",
            "approach_history:",
        )
        for item in evidence:
            if not isinstance(item, str) or not item.strip().startswith(allowed_prefixes):
                return "evidence_reviewed entries must be IDs copied from available_evidence, not prose evidence"
        return ""

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

    def _json_repair_recovery_context(
        self,
        raw: str,
        *,
        phase: str,
        feedback: bool = False,
    ) -> tuple[str, bool, str | None]:
        """Return safe recovery context for a malformed structured response.

        Short malformed JSON often benefits from a tail excerpt. Repetitive
        implementation output is different: showing the tail can cause the next
        local model turn to continue the repeated source text instead of
        regenerating a clean JSON payload.
        """
        if feedback:
            return (
                "Previous response recovery note:\n"
                "The previous feedback/review response did not match the requested JSON protocol. "
                "Its free-form text is omitted here because rejected reviewer scratch work is not "
                "accepted workflow evidence and can anchor the repair to the wrong decision. Answer the "
                "same current review question from the authoritative workflow state and required contract. "
                "Do not ask for implementation changes merely because the previous reviewer response was "
                "malformed; request changes only when the current evidence has a concrete gap.",
                True,
                "feedback review repairs should answer the protocol again instead of replaying rejected scratch text",
            )
        omission_reason = self._repair_tail_omission_reason(raw, phase=phase)
        if omission_reason:
            return (
                "Previous response recovery note:\n"
                "The previous response was long, line-repetitive, or malformed in a way that makes its "
                "tail unsafe as recovery context. Discard that text instead of continuing it. Regenerate "
                "a fresh minimal JSON payload for this phase from the current plan, current requirements, "
                "authoritative chat history, and the required contract. If the discarded response repeated "
                "source comments, shell commands, generated file content, or partial JSON fragments, do not "
                "copy those fragments into the repair response."
            ), True, omission_reason
        return "Previous response tail for recovery:\n" + self._repair_tail_for_prompt(raw), False, None

    @staticmethod
    def _json_repair_parse_error_for_prompt(
        parse_error: Exception,
        *,
        omitted_unsafe_tail: bool,
        omission_reason: str | None = None,
    ) -> str:
        if omission_reason:
            return (
                f"{type(parse_error).__name__}: malformed or off-contract response. Raw response text is "
                f"omitted because {omission_reason}."
            )
        if omitted_unsafe_tail:
            return (
                f"{type(parse_error).__name__}: malformed or off-contract response. Raw response text is "
                "omitted because it was repetitive enough to be unsafe recovery context."
            )
        return clamp_text(str(parse_error), 1200, marker="parse error truncated")

    def _repair_tail_omission_reason(self, raw: str, *, phase: str) -> str | None:
        if self._text_has_line_or_block_repetition(raw):
            return "it was repetitive enough to be unsafe recovery context"
        if self._text_has_reasoning_or_template_markup(raw):
            return (
                "it contained visible reasoning, chat-template markers, or fake tool-call syntax that "
                "must not be preserved as recovery evidence"
            )
        return None

    @staticmethod
    def _text_has_reasoning_or_template_markup(text: str) -> bool:
        lowered = text.lower()
        return (
            "<think" in lowered
            or "</think" in lowered
            or "<|tool_call" in lowered
            or "tool_call|>" in lowered
            or "<|channel" in lowered
            or "channel|>" in lowered
            or "<|start" in lowered
            or "<|end" in lowered
            or "<|assistant" in lowered
            or "<|user" in lowered
        )

    @staticmethod
    def _text_has_line_or_block_repetition(text: str) -> bool:
        """Detect repeated generated lines/blocks in a bounded text sample."""
        sample = text[-30000:]
        raw_lines = sample.splitlines()
        lines = [line.strip() for line in raw_lines if len(line.strip()) >= 16]
        if len(lines) < 10:
            return False
        counts = Counter(lines)
        most_common = counts.most_common(1)[0][1]
        if most_common >= 5 and most_common / len(lines) >= 0.18:
            return True
        unique_ratio = len(counts) / len(lines)
        if len(lines) >= 40 and unique_ratio <= 0.35 and most_common >= 3:
            return True
        for block_size in (2, 3, 4, 5, 6):
            if len(lines) < block_size * 4:
                continue
            blocks = ["\n".join(lines[index:index + block_size]) for index in range(0, len(lines) - block_size + 1)]
            block_counts = Counter(blocks)
            repeated = block_counts.most_common(1)[0][1]
            if repeated >= 4 and repeated / len(blocks) >= 0.12:
                return True
        return False

    def _replace_last_malformed_response_for_repair(
        self,
        phase: str,
        raw: str,
        parse_error: Exception,
        *,
        omission_reason: str | None = None,
    ) -> None:
        """Keep pathological implementation output out of the next active model call."""
        reason = omission_reason or "the rejected response was unsafe recovery context"
        note = (
            "IMPLEMENTATION_AGENT_RESPONSE:\n"
            "[malformed implementation response omitted from active context before JSON repair]\n"
            f"Phase: {phase}\n"
            "Parse error: "
            + self._json_repair_parse_error_for_prompt(
                parse_error,
                omitted_unsafe_tail=True,
                omission_reason=reason,
            )
            + "\n"
            f"Reason: {reason}.\n"
            f"Original response length: {len(raw)} chars.\n"
            "Repair instruction: regenerate a fresh minimal JSON payload from the current requirements, "
            "plan step, chat history, and required contract. Do not continue or quote the omitted response."
        )
        self.conversation.replace_last_turn(
            role="assistant",
            content_prefix="IMPLEMENTATION_AGENT_RESPONSE:\n",
            new_content=note,
        )

    def _feedback_minimal_json_repair(
        self,
        *,
        phase: str,
        contract: str,
        parse_error: Exception,
        repair_error: Exception,
        current_question_context: str | None = None,
    ) -> dict[str, Any]:
        """Give review phases one more protocol-only chance before fallback.

        A malformed reviewer turn is a reviewer protocol problem, not evidence
        that the implementation is wrong. This retry deliberately excludes the
        bad response tail and asks the same natural-language question again in
        the required JSON shape, using only current workflow state.
        """
        current_context_text = ""
        if current_question_context:
            current_context_text = (
                "\n\nCurrent review question and supplied evidence:\n"
                + self._prompt_excerpt(current_question_context, 8000)
            )
        prompt = (
            f"{phase}_MINIMAL_JSON_REPAIR\n"
            "Your last response still did not match the review protocol. This is only a protocol repair; "
            "do not restart implementation and do not request task changes just because the reviewer response "
            "was malformed.\n"
            "Answer the current review question now. Use the authoritative workflow state below, the original "
            "phase name, and the required contract as the protocol. If the evidence is sufficient, return the "
            "resolved form allowed by the contract. If the evidence has a concrete gap, return the appropriate "
            "non-resolved status allowed by the contract and name that gap. If you cannot decide from the "
            "available evidence, say so through the closest allowed non-resolved status rather than inventing "
            "evidence.\n"
            "First parse/protocol error: "
            + self._json_repair_parse_error_for_prompt(
                parse_error,
                omitted_unsafe_tail=True,
                omission_reason="the rejected response is not accepted evidence",
            )
            + "\nSecond parse/protocol error: "
            + self._json_repair_parse_error_for_prompt(
                repair_error,
                omitted_unsafe_tail=True,
                omission_reason="the rejected repair response is not accepted evidence",
            )
            + "\n"
            f"{PROTOCOL_DISCIPLINE_GUIDANCE.strip()}\n"
            "Return one valid JSON object only. No markdown, no analysis text, no chat-template markers, "
            "and no fake tool-call markers. Start with { and stop after the matching }. "
            "Schema example strings are placeholders, not answers: do not copy values such as "
            "`review summary`, `whole project review`, `decision summary`, `specific change`, or `evidence`; "
            "replace them with concrete current content or an empty list when nothing is required.\n\n"
            f"Authoritative current workflow state:\n{self._prompt_excerpt(self._workflow_state_for_prompt(), 6000)}\n\n"
            f"{current_context_text}\n\n"
            f"Required contract:\n{contract}"
        )
        repaired = self._feedback_chat(prompt, temperature=0.0)
        return self._extract_phase_json(repaired, phase=phase)

    def _malformed_feedback_fallback(
        self,
        phase: str,
        parse_error: Exception,
        repair_error: Exception,
        final_repair_error: Exception | None = None,
    ) -> dict[str, Any]:
        """Convert an unparseable reviewer turn into an explicit protocol failure.

        Some local models occasionally drift into half-JSON or repeated analysis
        even during repair turns. Crashing at that point loses a long run, but
        converting the protocol failure into implementation rework is worse: it
        misattributes the defect and can make a correct implementation churn.
        """
        summary = f"{phase} reviewer response was malformed after JSON repair."
        if phase == "TOOL_PROGRESS_REVIEW_PHASE":
            return {
                "status": "continue",
                "decision": "continue",
                "summary": summary + " The running command remains under progress review.",
                "evidence": ["Harness parser could not extract valid progress-review JSON."],
                "risks": ["No parseable reviewer decision was available to justify terminating the command."],
                "next_check_seconds": self.config.runtime.command_progress_review_interval_seconds,
                "review_protocol_error": True,
                "parse_error": str(parse_error),
                "repair_error": str(repair_error),
                "final_repair_error": str(final_repair_error or ""),
            }
        if phase == "TOOL_CALL_VERIFICATION_PHASE":
            return {
                "status": "blocked",
                "summary": summary + " Commands are blocked because approval requires parseable verifier decisions.",
                "commands": [],
                "review_protocol_error": True,
                "parse_error": str(parse_error),
                "repair_error": str(repair_error),
                "final_repair_error": str(final_repair_error or ""),
            }
        return {
            "status": "cannot_resolve",
            "needs_rework": True,
            "summary": summary,
            "required_changes": [
                "Reviewer protocol repair failed; repeat the review decision in the requested JSON contract before using it as workflow guidance.",
            ],
            "verification_evidence": [
                "Harness parser could not extract valid reviewer JSON from the original or repair response."
            ],
            "review_protocol_error": True,
            "parse_error": str(parse_error),
            "repair_error": str(repair_error),
            "final_repair_error": str(final_repair_error or ""),
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
        status = self._status(result)
        if status in {"resolved", "resolved_with_compromise"}:
            return None
        if phase == "requirements" and self._requirements_validation_only_skip_can_continue(result):
            self._append_plan_note(
                "[requirements] continuing after validation-command-only retry exhaustion; "
                "plan validation must repair or replace the recorded validation commands before implementation."
            )
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

    def _requirements_validation_only_skip_can_continue(self, result: dict[str, Any]) -> bool:
        """Let plan validation handle exhausted requirements validation-command repairs.

        Requirements refinement can otherwise spend all attempts repairing only
        reviewer-owned command syntax and then restart the whole approach, even
        though the user-facing requirements are clear. Continue only when every
        recorded unresolved change is plainly about validation command mechanics.
        """
        if self._status(result) != "skipped_with_note":
            return False
        iterations = result.get("iterations")
        if not isinstance(iterations, list) or not iterations:
            return False
        last_review = iterations[-1].get("review") if isinstance(iterations[-1], dict) else None
        if not isinstance(last_review, dict):
            return False
        return self._review_required_changes_are_validation_command_mechanics(last_review)

    def _review_required_changes_are_validation_command_mechanics(self, review: dict[str, Any]) -> bool:
        changes = [str(item).strip() for item in review.get("required_changes", []) if str(item).strip()]
        return self._findings_are_validation_command_mechanics(changes)

    @classmethod
    def _findings_are_validation_command_mechanics(cls, findings: list[str]) -> bool:
        """Return true only for findings about reviewer-owned validation mechanics.

        This deliberately does not include scope, safety, output-shape, or
        missing-deliverable findings. It is only used to stop repeated repairs
        of brittle validation commands from blocking otherwise clear work.
        """
        normalized = [str(finding).strip().lower() for finding in findings if str(finding).strip()]
        return bool(normalized) and all(cls._finding_is_validation_command_mechanic(item) for item in normalized)

    @staticmethod
    def _finding_is_validation_command_mechanic(finding: str) -> bool:
        normalized = re.sub(r"\s+", " ", finding.lower()).strip()
        text = f" {normalized} "
        validation_markers = (
            " validation ",
            "validation command",
            "validation_commands",
            "shell syntax",
            "shell parse",
            "inline python",
            "python -c",
            "grep",
            "pytest",
            "test runner",
            "workspace path",
            "workspace source path",
            "temporary command output",
            "static parse check",
            "static syntax check",
            "unsupported assertion metadata",
            "manual_test metadata",
        )
        return any(marker in text for marker in validation_markers)

    def _validation_command_compromise_review(
        self,
        scope: str,
        findings: list[str],
        *,
        status: str,
    ) -> dict[str, Any]:
        unique_findings: list[str] = []
        for finding in findings:
            if finding not in unique_findings:
                unique_findings.append(finding)
        return {
            "status": status,
            "needs_rework": False,
            "summary": (
                f"Repeated {scope} repairs are only changing validation-command mechanics. "
                "Continue with the user-visible requirements and require fresh executable evidence later."
            ),
            "required_changes": unique_findings,
            "cross_check_questions": [
                "Is another validation-command repair actually improving the deliverable, or only repeating shell/quoting work?"
            ],
            "verification_evidence": [],
        }

    def _apply_validation_command_compromise_to_plan(self, review: dict[str, Any]) -> None:
        """Remove only validation commands that deterministic checks can show are brittle.

        Later implementation and review phases can still use fresh commands the
        model actually runs, and those commands are verified before execution.
        """
        removed = 0
        for step in self.plan_steps:
            commands = list(step.get("validation_commands") or [])
            kept_commands: list[Any] = []
            removed_commands: list[Any] = []
            for command in commands:
                probe_step = dict(step)
                probe_step["validation_commands"] = [command]
                command_findings = self._validation_command_findings(probe_step)
                if command_findings and self._findings_are_validation_command_mechanics(command_findings):
                    removed_commands.append(command)
                else:
                    kept_commands.append(command)
            if not removed_commands:
                continue
            removed += len(removed_commands)
            step["validation_commands"] = kept_commands
            notes = list(step.get("validation_notes") or [])
            notes.append(
                "Removed brittle planned validation command(s) after repeated validation-command-only repairs; "
                "implementation and review must provide fresh executable evidence for this step."
            )
            step["validation_notes"] = notes
        if removed:
            self.requirements["plan"] = self.plan_steps
            self._append_plan_note(
                f"[plan] removed {removed} brittle planned validation command(s) after compromise: "
                f"{review.get('summary', '')}"
            )

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
            workspace_context = self._initial_workspace_context_for_prompt()
            prompt = (
                f"PROBLEM_ANALYSIS_PHASE approach_attempt={approach_attempt} iteration={index}\n"
                "Analyze the user's request before planning. Restate the problem, inspect the available "
                "workspace/research/source context, identify what is possible or uncertain, and compare "
                "multiple solution paths. Do not write project files or compute final deliverables in this "
                "analysis phase. This phase prepares later model-driven requirements and planning.\n"
                    f"{SELF_CHECK_GUIDANCE}\n"
                    f"{ANTI_TUNNEL_VISION_GUIDANCE}\n"
                    f"{self._execution_environment_guidance()}\n"
                f"{self._harness_state_file_guidance()}\n"
                f"Workspace source snapshot: {json.dumps(workspace_context, ensure_ascii=False)}\n"
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
                "Do not narrow future workflow toward one past failure or one fixed solution; "
                "preserve general-purpose problem solving:\n"
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
                "the analysis does not contain a narrow precomputed solution shortcut",
            ],
            "expected_json": {
                "status": "resolved|needs_rework|cannot_resolve",
                "needs_rework": False,
                "summary": "review summary",
                "required_changes": ["specific analysis gap"],
                "quality_questions": ["question"],
            },
        }
        raw = self._feedback_chat(
            "PROBLEM_ANALYSIS_REVIEW_PHASE\n"
                "Review the pre-plan problem analysis. Push back if it skips source/context checks, "
                "lists only one path, or starts solving the task instead of preparing a reusable workflow "
                "for the active request.\n"
                f"{_review_prompt_guidance()}\n"
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
        findings.extend(self._analysis_internal_contradiction_findings(analysis))
        if self._prompt_requests_scalar_json_list_stdout() and self._requirements_wrap_scalar_json_list(analysis):
            findings.append(
                "Analysis turns a requested JSON list of scalar values into wrapper objects or records. "
                "Restate the caller-visible stdout shape from the prompt as list values directly, and leave "
                "object fields such as timestamps out unless the original request explicitly asks for them."
            )
        if self._legacy_semantic_phrase_checks_enabled():
            findings.extend(self._analysis_public_api_shape_findings(analysis))
            findings.extend(self._unrequested_scope_expansion_findings(analysis, source_label="Analysis"))
        return findings

    def _unrequested_scope_expansion_findings(self, payload: Any, *, source_label: str) -> list[str]:
        """Legacy phrase-table scope checks were removed from the default harness.

        Scope drift is now reviewed through the generic prompt contract and
        reviewer judgment. The harness should not maintain a growing vocabulary
        of English phrases that only works for past benchmark failures.
        """
        return []

    def _analysis_public_api_shape_findings(self, analysis: dict[str, Any]) -> list[str]:
        """Prevent pre-plan analysis from anchoring unrequested API representation."""
        if not self._prompt_public_entrypoints(self.config.project_design.prompt):
            return []
        if self._prompt_explicitly_names_public_io_representation(self.config.project_design.prompt):
            return []
        text = json.dumps(analysis, sort_keys=True).lower()
        scan_text = re.sub(r"\be\.g\.", "eg", text)
        findings: list[str] = []
        public_io_patterns = (
            r"\b(?:input|inputs|output|outputs|return|returns|returned)\b[^.\n]{0,180}",
            r"\b(?:representation|data structure|datastructure|container|format|shape|structure)\b[^.\n]{0,220}",
        )
        for match in itertools.chain.from_iterable(re.finditer(pattern, scan_text) for pattern in public_io_patterns):
            window = match.group(0)
            context = scan_text[max(0, match.start() - 80):match.end() + 220]
            if self._output_shape_window_is_neutral_question_or_negation(window, context):
                continue
            if (
                re.search(r"\b(?:list|tuple|dict|dictionary|json|string|csv|xml)\b", window)
                or "[[" in window
                or "((" in window
            ):
                findings.append(
                    "Analysis invents a concrete caller-visible input/output representation for a public API. "
                    "Restate the API data semantically, for example as pair-like values or records, and preserve "
                    "representation as unspecified unless the original user request named it."
                )
                break
        return findings

    def _analysis_internal_contradiction_findings(self, analysis: dict[str, Any]) -> list[str]:
        findings: list[str] = []
        paths = analysis.get("possible_solution_paths") or []
        external_default_markers = (
            "pytest",
            "py.test",
        )
        dependency_free_markers = (
            "standard library",
            "stdlib",
            "dependency-free",
            "dependency free",
            "no external",
            "without external",
            "built-in",
            "builtin",
            "tiny",
        )
        iterable_paths = paths if isinstance(paths, list) else []
        for path in iterable_paths:
            if not isinstance(path, dict):
                continue
            approach_text = " ".join(
                str(path.get(key) or "")
                for key in ("description", "rationale")
            ).lower()
            approach_text += " " + " ".join(str(item) for item in path.get("advantages", []) if item is not None).lower()
            verification_text = str(path.get("verification_strategy") or "").lower()
            full_text = f"{approach_text} {verification_text}"
            if (
                any(marker in approach_text for marker in dependency_free_markers)
                and any(marker in full_text for marker in external_default_markers)
            ):
                path_id = str(path.get("id") or "?")
                findings.append(
                    f"Analysis path {path_id} mixes a dependency-free/standard-library approach with an external test runner."
                )
        return findings

    def _requirements_test_runner_consistency_findings(self, requirements: dict[str, Any]) -> list[str]:
        """Reject unsupported default test-runner assumptions.

        This is intentionally about evidence and consistency, not about one
        preferred runner. If the user asked for pytest, the workspace already
        shows a pytest convention, or the plan has an explicit dependency/setup
        decision, pytest is a valid choice. Otherwise, a blank Python workspace
        should not drift from "standard library" into an external runner merely
        because the model treats that as a default.
        """
        if not isinstance(requirements, dict):
            return []
        req_text = json.dumps(requirements, sort_keys=True).lower()
        if not self._requirements_choose_pytest(requirements):
            return []
        prompt_text = self.config.project_design.prompt.lower()
        if re.search(r"\b(?:pytest|py\.test)\b", prompt_text):
            return []
        if self._workspace_has_pytest_convention():
            return []
        if self._explicit_dependency_setup_is_present(req_text):
            return []
        return [
            (
                "Requirements or plan choose pytest as a default test runner even though the user did not "
                "request it and no workspace pytest convention is visible. Use the existing project runner, "
                "the standard-library test runner for a blank Python workspace, or add an explicit bounded "
                "dependency/setup decision before relying on an external runner."
            )
        ]

    @classmethod
    def _requirements_choose_pytest(cls, requirements: dict[str, Any]) -> bool:
        """Return True only for affirmative pytest usage, not negative mentions."""
        if cls._plan_commands_invoke_pytest(requirements.get("plan", [])):
            return True
        text_chunks: list[str] = []
        for key in ("project_summary", "refined_requirements", "assumptions"):
            value = requirements.get(key)
            if isinstance(value, list):
                text_chunks.extend(str(item) for item in value)
            elif value is not None:
                text_chunks.append(str(value))
        for item in requirements.get("open_questions", []) or []:
            if isinstance(item, dict):
                text_chunks.extend(str(item.get(key) or "") for key in ("resolution_strategy", "decision"))
        confirmation = requirements.get("planning_confirmation")
        if isinstance(confirmation, dict):
            text_chunks.append(str(confirmation.get("verification_strategy") or ""))
        return any(cls._text_affirmatively_chooses_pytest(chunk) for chunk in text_chunks)

    @staticmethod
    def _plan_commands_invoke_pytest(plan: object) -> bool:
        if not isinstance(plan, list):
            return False
        for step in plan:
            if not isinstance(step, dict):
                continue
            for command in step.get("validation_commands", []) or []:
                text = json.dumps(command, sort_keys=True).lower()
                if re.search(r"\b(?:pytest|py\.test)\b", text):
                    return True
        return False

    @staticmethod
    def _text_affirmatively_chooses_pytest(text: object) -> bool:
        lower = re.sub(r"\s+", " ", str(text).strip().lower())
        if not re.search(r"\b(?:pytest|py\.test)\b", lower):
            return False
        for match in re.finditer(r"\b(?:pytest|py\.test)\b", lower):
            window = lower[max(0, match.start() - 80): match.end() + 80]
            if re.search(
                r"\b(?:avoid|avoids|avoiding|not|no|without|instead of|rather than|replace|replacing|remove|removing)\b",
                window,
            ):
                continue
            if re.search(
                r"\b(?:use|uses|using|choose|chooses|chosen|select|selected|run|runs|invoke|provide|write|test runner|testing framework|suite|validation)\b",
                window,
            ):
                return True
        return False

    def _workspace_has_pytest_convention(self) -> bool:
        try:
            files = collect_workspace_files(self.workspace, max_file_bytes=4096)
        except Exception:
            return False
        config_names = {
            "pytest.ini",
            ".pytest.ini",
            "tox.ini",
            "setup.cfg",
            "pyproject.toml",
            "conftest.py",
        }
        for item in files:
            path = str(item.get("path", ""))
            name = Path(path).name
            content = str(item.get("content", "")).lower()
            if name in {"pytest.ini", ".pytest.ini", "conftest.py"}:
                return True
            if name == "pyproject.toml" and "[tool.pytest" in content:
                return True
            if name in {"setup.cfg", "tox.ini"} and "[pytest" in content:
                return True
            if path.startswith("tests/") and re.search(r"\bimport\s+pytest\b|\bfrom\s+pytest\s+import\b", content):
                return True
        return False

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
                "Return strict JSON with enough detail to guide later work. Follow the shared plan-scope and "
                "validation-command rules in the contract below.\n"
                "If the user's requested step count conflicts with verifiable implementation, record "
                "that conflict as an assumption and choose a practical feasible verifiable plan. "
                "Do not reinterpret per-attempt file-count guidance as a one-file-per-plan-step rule.\n"
                "If a prior review or extra context names a blocker, treat older PLAN/REQUIREMENTS snippets "
                "as failed evidence rather than a template. Remove or replace commands and public options "
                "that were called out as invalid; do not copy them into the next plan unchanged.\n"
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
                "generated source-using deliverables or project notes must cite/apply fetched source URLs. "
                "If web research is skipped or disabled, record available-knowledge notes instead and do not invent URLs.\n"
                    f"{SELF_CHECK_GUIDANCE}\n"
                    f"{ANTI_TUNNEL_VISION_GUIDANCE}\n"
                    f"Extra context: {extra_context or 'none'}\n\n{REQUIREMENTS_CONTRACT}"
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
            review_status = self._status(review)
            if review_status == "resolved" or (
                review_status == "skipped_with_note"
                and self._review_required_changes_are_validation_command_mechanics(review)
            ):
                self.last_requirements_review = {}
                doc_review = self._requirements_review_for_doc(review)
                self._write_requirements_doc(doc_review)
                if review_status == "skipped_with_note":
                    self._append_plan_note(
                        "[requirements] accepted with validation-command compromise after iteration "
                        f"{index}: {doc_review.get('summary', '')}"
                    )
                else:
                    self._append_plan_note(f"[requirements] resolved after iteration {index}: {doc_review.get('summary', '')}")
                return {"status": review_status, "iterations": iterations}
            self.last_requirements_review = self._compact_review_for_transcript(review)
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
        semantic_phrase_checks = self._legacy_semantic_phrase_checks_enabled()
        environment_findings = (
            self._environment_assumption_findings(requirements=requirements)
            if semantic_phrase_checks
            else []
        )
        consistency_findings = self._requirements_internal_consistency_findings(requirements)
        test_runner_findings = self._requirements_test_runner_consistency_findings(requirements)
        scope_findings = (
            self._unrequested_scope_expansion_findings(
                requirements,
                source_label="Requirements",
            )
            if semantic_phrase_checks
            else []
        )
        computed_answer_findings = (
            self._computed_answer_validation_findings(
                requirements=requirements,
                plan=normalize_plan_steps(requirements.get("plan", [])) if isinstance(requirements, dict) else [],
            )
            if semantic_phrase_checks
            else []
        )
        public_api_findings = self._public_api_overconstraint_findings(requirements) if semantic_phrase_checks else []
        stdout_json_findings = self._stdout_json_format_requirements_findings(requirements)
        documentation_filename_findings = self._documentation_filename_requirements_findings(requirements)
        script_invocation_findings = (
            self._script_direct_invocation_findings(
                requirements=requirements,
                plan=normalize_plan_steps(requirements.get("plan", [])) if isinstance(requirements, dict) else [],
            )
            if semantic_phrase_checks
            else []
        )
        previous_requirements = self.requirements
        previous_plan_steps = self.plan_steps
        try:
            if isinstance(requirements, dict):
                self.requirements = requirements
                self.plan_steps = normalize_plan_steps(requirements.get("plan", []))
            plan_structural_findings = self._plan_structural_findings(
                include_diagnostic_quality=False,
            )
        finally:
            self.requirements = previous_requirements
            self.plan_steps = previous_plan_steps
        deterministic_findings = []
        for item in [
            *environment_findings,
            *consistency_findings,
            *test_runner_findings,
            *scope_findings,
            *computed_answer_findings,
            *public_api_findings,
            *stdout_json_findings,
            *documentation_filename_findings,
            *script_invocation_findings,
            *plan_structural_findings,
        ]:
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
                "needs_rework": False,
                "summary": "review summary",
                "required_changes": ["specific change"],
                "cross_check_questions": ["requirement question the next pass must answer"],
            },
        }
        if deterministic_findings:
            if index > 1 and self._findings_are_validation_command_mechanics(deterministic_findings):
                return self._validation_command_compromise_review(
                    "requirements",
                    deterministic_findings,
                    status="skipped_with_note",
                )
            return self._deterministic_requirements_review(deterministic_findings)

        raw = self._feedback_chat(
            "REQUIREMENTS_REVIEW_PHASE\n"
            "Check whether the requirements are complete enough to support a distinct, verifiable plan. "
            "Reject vague requirements, missing gap decisions, and missing verification strategy.\n"
            "Treat project_design as the highest-priority scope source. Reject requirements that add behavior "
            "modifiers, relations, public surface, or validation duties not present in the original request unless "
            "they are explicitly recorded as assumptions or open questions needing clarification.\n"
            "If constraints conflict, request a clear compromise instead of repeatedly enforcing both sides "
            "of an impossible constraint. Per-attempt output-size guidance is not a plan-step limit.\n"
            "Use default_quality_policy exactly as provided: when applies=true, require the requested quality "
            "deliverables; require a separate research/structure planning step only when "
            "requires_research_structure_step=true. When applies=false, do not invent extra project files, "
            "documentation, tests, or research steps that the user did not ask for; require direct validation "
            "evidence that respects the user's requested scope instead.\n"
            "Apply execution_environment strictly. If deterministic_environment_findings is non-empty, request a "
            "requirements or plan correction instead of accepting incompatible assumptions.\n"
            "If deterministic_requirements_findings is non-empty, request correction instead of accepting the "
            "requirements as-is. Do not rewrite a deterministic public-API representation finding into a new "
            "concrete output type or same-input-type policy unless the original user request explicitly named "
            "that behavior; request removal of the invented representation or clarification instead. "
            "If deterministic_requirements_findings is empty, do not invent command-syntax objections; "
            "focus on semantic coverage, scope, and requirement clarity.\n"
                "If WEB_RESEARCH_TOOL_RESULT has completed or partial sources, reject requirements that ignore those sources. "
                "If web research is skipped or disabled, do not require cited source URLs; request available-knowledge notes instead.\n"
                f"{_review_prompt_guidance(ORIGINAL_REQUEST_FIT_CHECK_GUIDANCE)}\n"
                + json.dumps(prompt),
            temperature=0.1,
        )
        review = self._extract_json_or_retry(
            raw,
            phase="REQUIREMENTS_REVIEW_PHASE",
            contract='{"status":"resolved|needs_rework|needs_requirements_change|cannot_resolve|skipped_with_note","needs_rework":false,"summary":"review summary","required_changes":["specific change"]}',
            feedback=True,
        )
        review = self._normalize_review(review)
        if self._legacy_semantic_phrase_checks_enabled():
            review = self._suppress_unsupported_validation_syntax_objection(review, scope="requirements review")
        self._record_effective_review_if_needed("REQUIREMENTS_REVIEW_PHASE", review)
        return review

    def _deterministic_requirements_review(self, findings: list[str]) -> dict[str, Any]:
        """Return an authoritative requirements review for deterministic blockers.

        Deterministic checks are local harness invariants, not reviewer opinions.
        Asking the feedback model anyway can produce contradictory durable
        history such as "resolved" immediately followed by a rework directive.
        """
        unique_findings: list[str] = []
        for finding in findings:
            if finding not in unique_findings:
                unique_findings.append(finding)
        return {
            "status": "needs_requirements_change",
            "needs_rework": True,
            "summary": "Deterministic requirements checks found unresolved validation or environment issues.",
            "required_changes": unique_findings,
            "cross_check_questions": [],
            "verification_evidence": [],
        }

    def _requirements_internal_consistency_findings(self, requirements: dict[str, Any]) -> list[str]:
        """Detect unresolved alternatives that were recorded as assumptions.

        Small local models often keep both sides of an earlier uncertainty in a
        single assumption, then append a tentative "let's assume" clause. That
        leaves later phases with two incompatible targets and one vague decision.
        """
        if not isinstance(requirements, dict):
            return []
        findings: list[str] = []
        for index, assumption in enumerate(requirements.get("assumptions", []) or [], start=1):
            text = str(assumption)
            if self._text_has_unresolved_assumption_alternatives(text):
                excerpt = clamp_text(text, 260, marker="assumption truncated")
                findings.append(
                    f"Assumption {index} still contains unresolved alternatives instead of one clear decision: "
                    f"{excerpt!r}. Choose one explicit assumption, move the uncertainty into open_questions, "
                    "or state the remaining risk without preserving incompatible options in the same assumption."
                )
        return findings

    @staticmethod
    def _text_has_unresolved_assumption_alternatives(text: str) -> bool:
        lower = re.sub(r"\s+", " ", str(text).strip().lower())
        if not lower:
            return False
        if "or better" in lower or "let's assume" in lower or "lets assume" in lower:
            return True
        if "assume" not in lower:
            return False
        alternative_marker = bool(re.search(r"\b(?:either|or)\b", lower))
        decision_marker = bool(
            re.search(
                r"\b(?:target|default|primary|format|representation|filesystem|runtime|path|surface|mode)\b",
                lower,
            )
        )
        return alternative_marker and decision_marker

    def _requirements_review_for_doc(self, review: dict[str, Any]) -> dict[str, Any]:
        """Avoid preserving stale reviewer prose as authoritative requirements memory."""
        sanitized = dict(review)
        summary = str(sanitized.get("summary") or "")
        if self._legacy_semantic_phrase_checks_enabled() and self._review_summary_conflicts_with_current_requirements(summary):
            sanitized["summary"] = (
                "Requirements review resolved against the current requirements payload. "
                "The refined requirements above are authoritative; older rejected representation wording was omitted."
            )
        return sanitized

    def _review_summary_conflicts_with_current_requirements(self, summary: str) -> bool:
        """Detect summaries that repeat superseded shape assumptions."""
        if not summary:
            return False
        req_text = json.dumps(self.requirements, sort_keys=True).lower()
        lower = summary.lower()
        stale_markers = (
            "match the input",
            "matching the input",
            "same input",
            "same container",
            "same iterable",
            "preserves the natural shape",
            "preserve the natural shape",
            "list of lists",
            "list-of-lists",
            "list of tuples",
            "list-of-tuples",
        )
        return any(marker in lower and marker not in req_text for marker in stale_markers)

    def _public_api_overconstraint_findings(self, requirements: dict[str, Any]) -> list[str]:
        """Catch public API output-shape assumptions that are not validated.

        This stays deliberately domain-neutral. If the prompt names a callable
        entrypoint and the generated requirements introduce caller-visible
        representation details, the harness should make the model prove that
        representation with direct examples instead of accepting a vague test
        suite name or silently changing the public contract.
        """
        entrypoints = self._prompt_public_entrypoints(self.config.project_design.prompt)
        if not entrypoints:
            return []
        req_text = json.dumps(requirements, sort_keys=True)
        if not self._requirements_discuss_public_output_shape(req_text):
            return []
        shape_preservation = self._requirements_describe_shape_preservation(req_text)
        findings: list[str] = []
        for entrypoint in entrypoints:
            if self._requirements_force_unrequested_canonical_shape(req_text):
                findings.append(
                    f"Requirements force a canonical caller-visible output representation or same-input-type "
                    f"preservation policy for public API `{entrypoint}` even though the user did not request "
                    "that representation. Remove the fixed representation or same-input-type preservation "
                    "policy, or ask for clarification. For pair-like APIs, do not fix this by choosing a "
                    "different fixed container type such as list-of-tuples, list-of-lists, or input-type "
                    "preservation unless the original user request named that behavior. Do not hide a "
                    "concrete default in assumptions or open_questions.decision; keep those fields semantic "
                    "too."
                )
                continue
            if shape_preservation:
                continue
            if not self._requirements_have_concrete_public_output_representation(req_text):
                continue
            if self._public_api_output_shape_validation_present(requirements, entrypoint):
                continue
            findings.append(
                f"Requirements introduce caller-visible output representation for public API `{entrypoint}` "
                "without representative validation. Do not invent a concrete return container to fix this. "
                "Either remove the unrequested representation or ask for clarification. Only if the original "
                "request names a fixed return representation or same-input-type policy should the plan require "
                "that behavior for every caller input. Otherwise validate semantic values across representative "
                "input shapes without requiring a specific pair container shape."
            )
        if findings:
            return findings
        return []

    def _stdout_json_format_requirements_findings(self, requirements: dict[str, Any]) -> list[str]:
        """Reject unrequested shape or presentation changes for JSON stdout."""
        findings: list[str] = []
        if self._prompt_requests_scalar_json_list_stdout() and self._requirements_wrap_scalar_json_list(requirements):
            findings.append(
                "Requirements turn a requested JSON list of scalar values into wrapper objects or records. "
                "Preserve the caller-visible stdout shape from the prompt: output the list values directly, "
                "and only add object fields such as timestamps when the original request explicitly asks for them."
            )
        if not self._prompt_requests_machine_json_stdout():
            return findings
        if any(
            self._requirement_string_adds_pretty_json_stdout(text)
            for text in self._requirements_stdout_format_scan_strings(requirements)
        ):
            findings.append(
                "Requirements add pretty-printing or indentation to machine-readable JSON stdout even though "
                "the user did not request presentation formatting. Preserve the prompt-implied stdout data "
                "contract with compact deterministic JSON, and validate stdout as the value the caller receives."
            )
        return findings

    def _prompt_requests_scalar_json_list_stdout(self) -> bool:
        prompt = re.sub(r"\s+", " ", self.config.project_design.prompt.lower())
        if "json" not in prompt or "list" not in prompt:
            return False
        if not any(marker in prompt for marker in ("stdout", "standard output", "prints", "print ", "writes")):
            return False
        for match in re.finditer(r"\bjson\s+list\s+of\b", prompt):
            window = prompt[match.end(): match.end() + 140]
            if any(marker in window for marker in ("object", "objects", "record", "records", "dict", "dictionary")):
                continue
            return True
        return False

    def _requirements_wrap_scalar_json_list(self, requirements: dict[str, Any]) -> bool:
        for text in self._requirements_stdout_format_scan_strings(requirements):
            lower = re.sub(r"\s+", " ", text.lower())
            for match in re.finditer(r"\bjson\s+list\s+of\s+(?:objects|records|dicts|dictionaries)\b", lower):
                if not self._object_list_shape_phrase_is_negated(lower, match.start(), match.end()):
                    return True
            for match in re.finditer(r"\blist\s+of\s+(?:objects|records|dicts|dictionaries)\b", lower):
                if self._object_list_shape_phrase_is_negated(lower, match.start(), match.end()):
                    continue
                if any(marker in lower for marker in ("stdout", "output", "prints", "json")):
                    return True
        return False

    @staticmethod
    def _object_list_shape_phrase_is_negated(text: str, start: int, end: int) -> bool:
        prefix = text[max(0, start - 80):start]
        suffix = text[end:end + 80]
        if re.search(r"\b(?:not|never|no|without)\b[^.;:\n]{0,50}$", prefix):
            return True
        if re.search(r"\b(?:rather than|instead of)\b[^.;:\n]{0,50}$", prefix):
            return True
        if re.search(r"^[^.;:\n]{0,50}\b(?:is not|are not|must not|should not)\b", suffix):
            return True
        return False

    def _documentation_filename_requirements_findings(self, requirements: dict[str, Any]) -> list[str]:
        """Reject invented documentation filenames when the prompt leaves it unnamed."""
        if not self._prompt_requests_unnamed_documentation():
            return []
        paths = self._markdown_paths_in_value(requirements)
        non_readme_docs = sorted(
            path
            for path in paths
            if Path(path).name.lower() not in {"readme", "readme.md"}
        )
        if not non_readme_docs:
            return []
        return [
            (
                "Requirements or plan place unnamed documentation in "
                f"{', '.join(non_readme_docs)}. The original request asked for documentation without naming a "
                "file, so prefer the conventional `README.md` unless the prompt points to a more specific document."
            )
        ]

    def _prompt_requests_unnamed_documentation(self) -> bool:
        prompt = re.sub(r"\s+", " ", self.config.project_design.prompt.lower())
        if not any(marker in prompt for marker in ("document ", "documentation", "docs", "readme")):
            return False
        if re.search(r"\b[\w./-]+\.md\b", prompt) or "readme" in prompt:
            return False
        return True

    @classmethod
    def _markdown_paths_in_value(cls, value: Any) -> set[str]:
        paths: set[str] = set()
        if isinstance(value, dict):
            for item in value.values():
                paths.update(cls._markdown_paths_in_value(item))
            return paths
        if isinstance(value, list):
            for item in value:
                paths.update(cls._markdown_paths_in_value(item))
            return paths
        if not isinstance(value, str):
            return paths
        for match in re.finditer(r"\b[\w./-]+\.md\b", value, flags=re.IGNORECASE):
            paths.add(match.group(0))
        return paths

    def _prompt_requests_machine_json_stdout(self) -> bool:
        prompt = self.config.project_design.prompt.lower()
        if "json" not in prompt:
            return False
        if not any(marker in prompt for marker in ("stdout", "standard output", "prints", "print ", "writes")):
            return False
        user_format_markers = (
            "pretty-print",
            "pretty print",
            "pretty-printed",
            "pretty printed",
            "indent",
            "human-readable",
            "human readable",
            "formatted",
        )
        return not any(marker in prompt for marker in user_format_markers)

    @classmethod
    def _requirements_stdout_format_scan_strings(cls, value: Any, *, parent_key: str | None = None) -> list[str]:
        skipped_keys = {
            "evidence",
            "freshness_risks",
            "question",
            "remaining_risks",
            "remaining_unknowns",
            "source_gaps",
        }
        if parent_key in skipped_keys:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            strings: list[str] = []
            for key, child in value.items():
                strings.extend(cls._requirements_stdout_format_scan_strings(child, parent_key=str(key)))
            return strings
        if isinstance(value, list):
            strings = []
            for child in value:
                strings.extend(cls._requirements_stdout_format_scan_strings(child, parent_key=parent_key))
            return strings
        return []

    @classmethod
    def _requirement_string_adds_pretty_json_stdout(cls, text: str) -> bool:
        lowered = text.lower()
        pretty_markers = (
            "pretty-print",
            "pretty print",
            "pretty-printed",
            "pretty printed",
            "formatted with indentation",
            "with indentation",
            "4-space indentation",
            "two-space indentation",
            "human readability",
            "human-readable",
            "for readability",
            "indent=",
        )
        for marker in pretty_markers:
            start = 0
            while True:
                index = lowered.find(marker, start)
                if index == -1:
                    break
                if not cls._pretty_json_marker_is_negated(lowered, index, index + len(marker)):
                    return True
                start = index + len(marker)
        return False

    @staticmethod
    def _pretty_json_marker_is_negated(text: str, start: int, end: int) -> bool:
        stripped = text.strip()
        if "?" in stripped and stripped.startswith(("should ", "whether ", "must ", "do ", "does ", "is ")):
            return True
        prefix = text[max(0, start - 90):start]
        suffix = text[end:end + 90]
        local = f"{prefix} {suffix}"
        negation_markers = (
            "compact json",
            "compact deterministic",
            "compact, deterministic",
            "no indentation",
            "without indentation",
            "not pretty",
            "not be pretty",
            "do not pretty",
            "don't pretty",
            "avoid pretty",
            "rather than pretty",
            "instead of pretty",
            "no,",
        )
        if any(marker in local for marker in negation_markers):
            return True
        return bool(
            re.search(
                r"\b(?:no|not|without|avoid|reject|forbid|forbids|forbidden|disable|disallow)\b.{0,35}$",
                prefix,
            )
        )

    def _list_null_element_validation_scope_finding(self, result: dict[str, Any], source_label: str) -> str:
        """Detect validators that broaden null-key removal into list-element removal."""
        if not self._prompt_requests_null_key_removal_with_list_order():
            return ""
        if self._prompt_explicitly_requests_list_null_element_removal():
            return ""
        output_text = f"{result.get('stdout') or ''}\n{result.get('stderr') or ''}"
        if not self._validation_failure_suggests_list_null_element_removal(output_text):
            return ""
        return (
            f"{source_label} appears to fail because generated validation expects a null/None element inside a "
            "list to be removed. The original request only established null-valued object-key removal while "
            "preserving list order. Treat this as a possible validator or plan defect: request validator repair, "
            "plan repair, or requirements clarification instead of changing implementation behavior solely to "
            "remove list elements."
        )

    def _prompt_requests_null_key_removal_with_list_order(self) -> bool:
        prompt = self.config.project_design.prompt.lower()
        if not any(marker in prompt for marker in ("null", "none")):
            return False
        if not any(marker in prompt for marker in ("key", "keys", "object", "objects", "property", "properties")):
            return False
        if not any(marker in prompt for marker in ("remove", "removes", "drop", "omit", "skip", "filter")):
            return False
        list_order_markers = (
            "keep list order",
            "keeps list order",
            "preserve list order",
            "preserves list order",
            "maintain list order",
            "maintains list order",
            "list order",
            "array order",
            "preserve arrays",
            "preserve lists",
            "leave arrays",
            "leave lists",
        )
        return any(marker in prompt for marker in list_order_markers)

    def _prompt_explicitly_requests_list_null_element_removal(self) -> bool:
        return self._text_explicitly_requests_list_null_element_removal(self.config.project_design.prompt)

    @staticmethod
    def _text_explicitly_requests_list_null_element_removal(text: str) -> bool:
        lowered = text.lower()
        verbs = r"(?:remove|removes|drop|drops|omit|omits|exclude|excludes|discard|discards|filter|filters)(?:\s+out)?"
        patterns = (
            rf"\b{verbs}\b.{{0,40}}"
            r"\b(?:null|none)\b.{0,30}\b(?:from|in|inside|within)\b.{0,30}\b(?:lists?|arrays?)\b",
            rf"\b{verbs}\b.{{0,40}}"
            r"\b(?:null|none)\b.{0,30}\b(?:lists?|arrays?)\b.{0,30}\b(?:elements?|items?|values?)\b",
            r"\b(?:lists?|arrays?)\b.{0,40}"
            rf"\b{verbs}\b.{{0,40}}\b(?:null|none)\b",
        )
        return any(re.search(pattern, lowered, flags=re.DOTALL) for pattern in patterns)

    @classmethod
    def _validation_failure_suggests_list_null_element_removal(cls, text: str) -> bool:
        lowered = text.lower()
        if "none" not in lowered and "null" not in lowered:
            return False
        if not any(marker in lowered for marker in ("assertionerror", "!=", "expected", "actual", "failed")):
            return False
        parsed_lists: list[tuple[str, ...]] = []
        lists_with_null: list[tuple[str, ...]] = []
        for match in re.finditer(r"\[[^\[\]]{0,240}\]", text, flags=re.DOTALL):
            tokens = cls._flat_literal_list_tokens(match.group(0))
            if not tokens:
                continue
            if any(token in {"none", "null"} for token in tokens):
                without_null = tuple(token for token in tokens if token not in {"none", "null"})
                if without_null:
                    lists_with_null.append(without_null)
            else:
                parsed_lists.append(tuple(tokens))
        return any(without_null in parsed_lists for without_null in lists_with_null)

    @staticmethod
    def _flat_literal_list_tokens(list_text: str) -> list[str]:
        inner = list_text.strip()[1:-1]
        if "{" in inner or "}" in inner:
            return []
        tokens: list[str] = []
        for raw_token in inner.split(","):
            token = raw_token.strip().strip("\"'")
            if not token:
                continue
            token = re.sub(r"\s+", "", token).lower()
            tokens.append(token)
        return tokens

    def _requirements_force_unrequested_canonical_shape(self, requirements_text: str) -> bool:
        """Detect invented canonical output types for flexible public APIs."""
        if self._prompt_explicitly_names_output_representation(self.config.project_design.prompt):
            return False
        lower = requirements_text.lower()
        if self._requirements_assume_concrete_output_representation(requirements_text):
            return True
        if self._requirements_describe_shape_preservation(lower):
            return not self._prompt_explicitly_requests_shape_preservation(self.config.project_design.prompt)
        accepts_multiple_input_shapes = any(
            marker in lower
            for marker in (
                "list of lists or list of tuples",
                "list of lists or a list of tuples",
                "list of tuples or list of lists",
                "list of tuples or a list of lists",
                "lists/tuples",
                "tuples/lists",
                "list/tuple",
                "tuple/list",
                "iterables (lists or tuples)",
                "iterables such as lists or tuples",
                "lists or tuples",
                "tuples or lists",
                "lists and tuples",
                "tuples and lists",
                "list-of-list input",
                "list-of-tuple input",
                "multiple input containers",
                "any iterable of iterables",
                "input contains lists",
                "whether input contains tuples or lists",
                "whether input contains lists or tuples",
                "regardless of whether input contains tuples or lists",
                "regardless of whether input contains lists or tuples",
            )
        )
        strong_forces_one_output_shape = any(
            marker in lower
            for marker in (
                "canonical output shape",
                "canonical output type",
                "canonical representation",
                "output must be a list of lists",
                "output must be list of lists",
                "output must be a `list` of `tuples`",
                "output must be a list of tuples",
                "output must be list of tuples",
                "output is a list of lists",
                "output is a list of tuples",
                "output format: a list of lists",
                "output format: a list of tuples",
                "output format: a list of tuples",
                "output will be a list of tuples",
                "output will be a list of lists",
                "output representation (tuples)",
                "as tuples",
                "(as tuples)",
                "list[list",
                "list[tuple",
                "-> list[tuple",
                "-> list[list",
                "returns `list[tuple",
                "returns `list[list",
                "return list[tuple",
                "return list[list",
                "list-of-lists structure",
                "list-of-tuples structure",
                "returned value from",
                "returned object follows",
                "return value follows",
                "output should maintain the same list-of-lists",
                "output should maintain the same list-of-tuples",
                "list of tuples representing",
                "list of lists representing",
                "return a list of lists",
                "returns a list of lists",
                "return a list of tuples",
                "returns a list of tuples",
                "always return a list",
                "must be `list` of `list`",
                "must be `list` of `tuple`",
                "must be list of list",
                "must be list of tuple",
                "must be a list of list",
                "must be a list of tuple",
                "even if the input contains",
                "even for list-of-list input",
                "regardless of whether input",
            )
        )
        if strong_forces_one_output_shape:
            return True
        numbered_pair_container_patterns = (
            r"\boutput\s+(?:is|must be|will be|should be)\s+a\s+list\s+of\s+\d+\s*-?\s*element\s+(?:lists|tuples)\b",
            r"\breturns?\s+a\s+list\s+of\s+\d+\s*-?\s*element\s+(?:lists|tuples)\b",
            r"\breturn\s+a\s+list\s+of\s+\d+\s*-?\s*element\s+(?:lists|tuples)\b",
            r"\boutput\s+format\s*:\s*a\s+list\s+of\s+\d+\s*-?\s*element\s+(?:lists|tuples)\b",
        )
        if any(re.search(pattern, lower) for pattern in numbered_pair_container_patterns):
            return True
        if not accepts_multiple_input_shapes:
            return False
        if re.search(r"\boutput\s*:\s*a list\b", lower):
            return True
        if re.search(r"\boutput\b[^.\n]{0,120}\[\[", lower):
            return True
        return False

    @classmethod
    def _requirements_assume_concrete_output_representation(cls, requirements_text: str) -> bool:
        """Detect open-question or assumption text that resolves output shape by fiat."""
        try:
            payload = json.loads(requirements_text)
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False

        for assumption in payload.get("assumptions", []):
            assumption_text = str(assumption).lower()
            if (
                cls._text_mentions_output_representation_gap(assumption_text)
                and cls._text_names_concrete_representation(assumption_text)
                and not cls._text_rejects_concrete_representation(assumption_text)
            ):
                return True

        output_shape_question_seen = False
        for item in payload.get("open_questions", []):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).lower()
            strategy = str(item.get("resolution_strategy", "")).lower()
            decision = str(item.get("decision", "")).lower()
            if not cls._text_mentions_output_representation_gap(question):
                continue
            output_shape_question_seen = True
            if (
                (strategy == "assume" or cls._text_chooses_concrete_representation(decision))
                and cls._text_names_concrete_representation(decision)
                and not cls._text_rejects_concrete_representation(decision)
            ):
                return True

        if output_shape_question_seen:
            for assumption in payload.get("assumptions", []):
                assumption_text = str(assumption).lower()
                if (
                    cls._text_names_concrete_representation(assumption_text)
                    and not cls._text_rejects_concrete_representation(assumption_text)
                ):
                    return True
        return False

    @staticmethod
    def _text_mentions_output_representation_gap(text: str) -> bool:
        return bool(
            re.search(r"\b(?:output|outputs|return|returns|returned|result|results|input/output)\b", text)
            and re.search(r"\b(?:type|format|container|record|representation|shape|serialization)\b", text)
        )

    @staticmethod
    def _text_chooses_concrete_representation(text: str) -> bool:
        return bool(
            re.search(r"\b(?:assume|choose|use|return|returns|will be|should be|must be|standardize|normalize)\b", text)
        )

    @staticmethod
    def _text_names_concrete_representation(text: str) -> bool:
        markers = (
            "list of lists",
            "list of tuples",
            "list-of-lists",
            "list-of-tuples",
            "list[list",
            "list[tuple",
            "tuple",
            "dictionary",
            "dict",
            "json",
            "object",
            "record",
            "string",
            "csv",
            "xml",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _text_rejects_concrete_representation(text: str) -> bool:
        neutral_markers = (
            "not a requirement",
            "not required",
            "not specified",
            "not specify",
            "did not specify",
            "does not specify",
            "does not mandate",
            "does not require",
            "not mandated",
            "not required",
            "no concrete",
            "no specific",
            "representation-neutral",
            "without requiring",
            "without forcing",
            "leave to the implementation",
            "left to the implementation",
            "implementation's discretion",
            "implementation discretion",
            "implementation detail",
            "implementation details",
            "not part",
            "not part of the functional requirement",
            "not part of the functional requirements",
        )
        return any(marker in text for marker in neutral_markers)

    @classmethod
    def _requirements_describe_shape_preservation(cls, requirements_text: str) -> bool:
        """Detect requirements that preserve caller shape instead of choosing one shape."""
        lower = requirements_text.lower()
        anti_preservation_markers = (
            "canonical output",
            "canonical representation",
            "always return",
            "-> list[",
            "return list[",
            "returns `list[",
            "list[list",
            "list[tuple",
            "output must be",
            "output format:",
            "regardless of whether input",
            "regardless of whether the input",
            "even if input",
            "even if the input",
            "even for list-of-list input",
        )
        if any(marker in lower for marker in anti_preservation_markers):
            return False
        preservation_patterns = (
            r"\b(?:preserve|preserves|preserving)\b[^.\n]{0,140}\b(?:natural\s+)?(?:input|original|caller|shape|type|format|container|representation|iterable)\b",
            r"\b(?:same|natural)\b[^.\n]{0,100}\b(?:shape|type|format|container|representation|iterable)\b[^.\n]{0,100}\b(?:input|original|caller)\b",
            r"\boutput\b[^.\n]{0,140}\b(?:match|matches|matching|preserve|preserves|preserving|same)\b[^.\n]{0,140}\binput\b",
            r"\binput\s+and\s+output\b[^.\n]{0,140}\b(?:preserve|preserves|preserving|match|matches|matching)\b",
        )
        return any(re.search(pattern, lower) for pattern in preservation_patterns)

    @classmethod
    def _requirements_have_concrete_public_output_representation(cls, requirements_text: str) -> bool:
        lower = requirements_text.lower()
        concrete_markers = (
            "list of lists",
            "list of tuples",
            "list-of-lists",
            "list-of-tuples",
            "list[list",
            "list[tuple",
            "tuple",
            "dict",
            "dictionary",
            "json",
            "set",
            "object",
            "string",
            "output format",
            "container type",
            "record type",
            "record",
            "representation",
        )
        for match in re.finditer(r"\b(?:output|outputs|return|returns|returned|result|results|input/output)\b[^.\n]{0,180}", lower):
            window = match.group(0)
            context = lower[max(0, match.start() - 80): match.end() + 220]
            if cls._output_shape_window_is_neutral_question_or_negation(window, context):
                continue
            if cls._output_shape_window_is_generic_pair_values(window):
                continue
            if "sequence" in window and any(
                marker in window
                for marker in (
                    "e.g., lists or tuples",
                    "e.g. lists or tuples",
                    "such as lists or tuples",
                    "lists or tuples",
                    "tuples or lists",
                )
            ):
                continue
            if "[[" in window or "((" in window:
                return True
            if re.search(r"\b(?:output|outputs|result|results)\b[^.\n]{0,100}\blist\b", window):
                return True
            if re.search(r"\breturns?\b[^.\n]{0,100}\blist of\b", window):
                return True
            if any(marker in window for marker in concrete_markers):
                return True
        return False

    @staticmethod
    def _output_shape_window_is_neutral_question_or_negation(window: str, context: str) -> bool:
        """Ignore discussion that explicitly avoids choosing an API representation."""
        lower_window = window.lower()
        lower_context = context.lower()
        question_markers = (
            '"question"',
            "'question'",
            "question:",
            "should the",
            "whether the",
        )
        if any(marker in lower_context for marker in question_markers) and re.search(
            r"\b(?:output|input/output|return|result)\b[^?]{0,160}\b(?:type|format|container|representation)\b",
            lower_context,
        ):
            return True
        neutral_markers = (
            "not a requirement",
            "not specified",
            "not specify",
            "did not specify",
            "does not specify",
            "does not mandate",
            "does not require",
            "representation-neutral",
            "no concrete",
            "no specific",
            "not mandated",
            "not required",
            "not part",
            "unknown",
            "unspecified",
            "not defined",
            "not explicitly defined",
            "will be decided",
            "to be decided",
            "defer",
            "deferred",
            "requirements refinement",
            "exact data structure",
            "specific data structure",
            "exact representation",
            "specific representation",
            "left to the implementation",
            "implementation's discretion",
            "implementation discretion",
            "implementation detail",
            "implementation details",
            "without requiring",
            "without forcing",
        )
        if not any(marker in lower_context for marker in neutral_markers):
            return False
        forcing_markers = (
            "output must be",
            "output will be",
            "always return",
            "canonical output",
            "canonical representation",
            "required output",
            "assume a",
            "assume an",
            "will assume",
            "assumed to be",
            "will use",
            "standardize",
            "normalize to",
        )
        if any(marker in lower_window for marker in forcing_markers):
            return False
        return True

    @staticmethod
    def _output_shape_window_is_generic_pair_values(window: str) -> bool:
        """Allow value-level pair wording without treating it as a container contract."""
        if re.search(r"(?:^|[,\s])e$", window.strip()):
            return False
        if not re.search(r"\b(?:pair|pairs|interval|intervals|two-element|2-element)\b", window):
            return False
        concrete_markers = (
            "list of lists",
            "list of tuples",
            "list-of-lists",
            "list-of-tuples",
            "list[list",
            "list[tuple",
            "tuple",
            "dict",
            "dictionary",
            "json",
            "set",
            "object",
            "record",
            "container type",
            "same input format",
            "same iterable type",
            "same container type",
            "type preservation",
            "shape preservation",
            "representation",
            "format",
            "[[",
            "((",
        )
        if any(marker in window for marker in concrete_markers):
            return False
        return bool(re.search(r"\b(?:list|sequence|collection|iterable)\s+of\b", window))

    @classmethod
    def _prompt_explicitly_names_output_representation(cls, prompt: str) -> bool:
        lower = prompt.lower()
        explicit_markers = (
            "json",
            "list of lists",
            "list of tuples",
            "tuple",
            "dict",
            "dictionary",
            "object",
            "string",
            "csv",
        )
        for match in re.finditer(r"\b(?:output|outputs|return|returns|returned|result|print|prints)\b[^.\n]{0,180}", lower):
            window = match.group(0)
            if any(marker in window for marker in explicit_markers):
                return True
            if "[[" in window or "((" in window or "{" in window:
                return True
        return False

    @classmethod
    def _prompt_explicitly_names_public_io_representation(cls, prompt: str) -> bool:
        lower = prompt.lower()
        explicit_markers = (
            "json",
            "list",
            "tuple",
            "dict",
            "dictionary",
            "object",
            "string",
            "csv",
            "xml",
            "[[",
            "((",
        )
        for match in re.finditer(
            r"\b(?:input|inputs|output|outputs|return|returns|returned|result|results|accept|accepts|argument|parameter)\b[^.\n]{0,180}",
            lower,
        ):
            if any(marker in match.group(0) for marker in explicit_markers):
                return True
        return False

    @classmethod
    def _prompt_explicitly_requests_shape_preservation(cls, prompt: str) -> bool:
        lower = prompt.lower()
        preservation_patterns = (
            r"\b(?:preserve|preserves|preserving|keep|keeps|keeping|maintain|maintains|retains?|match|matches)\b"
            r"[^.\n]{0,140}\b(?:input|original|same)\b[^.\n]{0,140}"
            r"\b(?:shape|type|format|container|representation|list|tuple)\b",
            r"\b(?:same|original|input)\b[^.\n]{0,120}"
            r"\b(?:shape|type|format|container|representation|list|tuple)\b",
        )
        return any(re.search(pattern, lower) for pattern in preservation_patterns)

    @staticmethod
    def _prompt_public_entrypoints(prompt: str) -> list[str]:
        entrypoints: list[str] = []
        ignored = {
            "assert",
            "dict",
            "int",
            "len",
            "list",
            "max",
            "min",
            "open",
            "print",
            "range",
            "set",
            "sorted",
            "str",
            "sum",
            "tuple",
        }
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", prompt):
            name = match.group(1)
            if name.lower() in ignored:
                continue
            window = prompt[max(0, match.start() - 120): match.end() + 80].lower()
            if not any(
                marker in window
                for marker in (
                    "api",
                    "callable",
                    "called",
                    "define",
                    "entrypoint",
                    "expose",
                    "function",
                    "implement",
                    "method",
                    "module",
                    "named",
                    "provide",
                    "with ",
                )
            ):
                continue
            if name not in entrypoints:
                entrypoints.append(name)
        return entrypoints

    @classmethod
    def _requirements_discuss_public_output_shape(cls, text: str) -> bool:
        lower = text.lower()
        if any(
            marker in lower
            for marker in (
                "caller-visible",
                "output format",
                "return format",
                "result format",
                "same input format",
                "same iterable type",
                "same container type",
                "shape preservation",
                "type preservation",
                "representation",
            )
        ):
            return True
        shape_markers = (
            "container",
            "dict",
            "iterable",
            "json",
            "list",
            "object",
            "pair",
            "record",
            "set",
            "tuple",
            "type",
        )
        for match in re.finditer(r"\b(?:output|outputs|return|returns|returned|result)\b[^.\n]{0,180}", lower):
            window = match.group(0)
            if any(marker in window for marker in shape_markers):
                return True
        return False

    def _public_api_output_shape_validation_present(self, requirements: dict[str, Any], entrypoint: str) -> bool:
        payload = {
            "plan": self.plan_steps or normalize_plan_steps(requirements.get("plan", [])),
        }
        text = json.dumps(payload, sort_keys=True).lower()
        entry_lower = entrypoint.lower()
        for match in re.finditer(rf"\b{re.escape(entry_lower)}\s*\(", text):
            window = text[max(0, match.start() - 160): match.end() + 300]
            if not any(marker in window for marker in ("==", "assert", "equals", "expected", "produces", "return", "returns")):
                continue
            if any(
                marker in window
                for marker in (
                    "[",
                    "dict",
                    "isinstance",
                    "json",
                    "list",
                    "same",
                    "shape",
                    "tuple",
                    "type(",
                )
            ):
                return True
        return False

    def _script_direct_invocation_findings(
        self,
        *,
        requirements: dict[str, Any],
        plan: list[dict[str, Any]],
    ) -> list[str]:
        """Preserve prompt-implied direct script entrypoints.

        This is intentionally a narrow heuristic. It does not try to infer every
        CLI contract. It catches the common drift where a prompt says "create
        validate_x.py that runs/checks a bounded scenario" and the plan changes
        that into a generic script requiring positional arguments, so the named
        script is no longer useful when invoked directly.
        """
        prompt = self.config.project_design.prompt
        direct_scripts = [
            script for script in self._prompt_named_scripts(prompt)
            if self._prompt_implies_direct_script_run(prompt, script)
        ]
        if not direct_scripts:
            return []
        requirements_text = json.dumps(requirements, sort_keys=True).lower()
        findings: list[str] = []
        for script in direct_scripts:
            lowered_script = script.lower()
            if self._requirements_make_script_args_mandatory(requirements_text, lowered_script):
                findings.append(
                    f"{script} is described in the prompt as a bounded script action, but the requirements appear "
                    "to make positional arguments mandatory. Preserve a useful direct invocation with sensible "
                    "defaults and make extra controls optional unless the user asked for required arguments."
                )
            if plan and not self._plan_validates_direct_script_invocation(plan, script):
                findings.append(
                    f"The plan does not validate the prompt-implied direct invocation of {script}. Add a bounded "
                    f"validation command such as {self._direct_script_command_example(script)} and keep any extra "
                    "arguments optional unless the user asked for mandatory arguments."
                )
        return findings

    def _script_primary_input_surface_findings(
        self,
        *,
        requirements: dict[str, Any],
        plan: list[dict[str, Any]],
    ) -> list[str]:
        """Reject invented required-looking flags for prompt-described primary CLI input.

        If the user describes a primary input such as "a file path" but only
        names flags for secondary settings, the harness should not let the plan
        quietly turn that primary input into a new required public option. The
        model may still choose an optional convenience flag, but the runbook must
        preserve and validate the prompt-implied positional input surface unless
        the prompt itself named the flag.
        """
        prompt = self.config.project_design.prompt
        prompt_lower = prompt.lower()
        prompt_flags = set(re.findall(r"--[a-z0-9][a-z0-9_-]*", prompt_lower))
        payload_text = json.dumps({"requirements": requirements, "plan": plan}, sort_keys=True).lower()
        findings: list[str] = []
        for script in self._prompt_named_scripts(prompt):
            input_kind = self._prompt_implied_primary_input_kind(prompt_lower, script)
            if not input_kind:
                continue
            invented_flags = [
                flag for flag in self._primary_input_candidate_flags(input_kind)
                if flag in payload_text and flag not in prompt_flags
            ]
            if not invented_flags:
                continue
            if self._plan_validates_script_positional_input(plan, script):
                continue
            flags = ", ".join(f"`{flag}`" for flag in invented_flags)
            findings.append(
                f"{script} is prompted as taking a primary {input_kind} input, but requirements or plan introduce "
                f"{flags} even though the prompt did not name that flag. Do not convert prompt-implied primary "
                "input into a new required public option. Preserve and validate a positional input form, or "
                "explicitly mark the flag as optional while still validating the prompt-implied input surface."
            )
        return findings

    @staticmethod
    def _prompt_implied_primary_input_kind(prompt_lower: str, script: str) -> str:
        script_lower = script.lower()
        idx = prompt_lower.find(script_lower)
        if idx < 0:
            return ""
        window = prompt_lower[max(0, idx - 80): idx + len(script_lower) + 260]
        if any(marker in window for marker in ("file path", "path argument", "path as", "watched file", "log file")):
            return "file path"
        if any(marker in window for marker in ("provided string", "input string", "text argument", "provided text")):
            return "text"
        if any(marker in window for marker in ("provided input", "input argument")):
            return "input"
        return ""

    @staticmethod
    def _primary_input_candidate_flags(input_kind: str) -> tuple[str, ...]:
        if input_kind == "file path":
            return (
                "--file",
                "--path",
                "--input",
                "--input-file",
                "--log",
                "--log-file",
                "--source",
                "--source-file",
                "--target",
                "--target-file",
            )
        if input_kind == "text":
            return ("--input", "--text", "--string")
        if input_kind == "input":
            return ("--input",)
        return ()

    def _plan_validates_script_positional_input(self, plan: list[dict[str, Any]], script: str) -> bool:
        for step in plan:
            for command in step.get("validation_commands", []) or []:
                if self._command_validates_script_positional_input(command, script):
                    return True
        return False

    def _command_validates_script_positional_input(self, command: Any, script: str) -> bool:
        argv = self._command_argv_for_static_check(command)
        if self._argv_invokes_script_with_positional_input(argv, script):
            return True
        for shell_text in self._shell_texts_for_static_check(argv):
            for segment in self._shell_command_segments(shell_text):
                if self._argv_invokes_script_with_positional_input(self._safe_shell_split(segment), script):
                    return True
        return False

    @classmethod
    def _argv_invokes_script_with_positional_input(cls, argv: list[str], script: str) -> bool:
        if not argv:
            return False
        script_positions = [idx for idx, item in enumerate(argv) if Path(item).name == script]
        if not script_positions:
            return False
        args = argv[script_positions[0] + 1 :]
        index = 0
        while index < len(args):
            token = args[index]
            if token == "--":
                return any(not str(item).startswith("-") for item in args[index + 1 :])
            if token.startswith("-"):
                if "=" not in token and index + 1 < len(args) and not args[index + 1].startswith("-"):
                    index += 2
                else:
                    index += 1
                continue
            return True
        return False

    @staticmethod
    def _prompt_named_scripts(prompt: str) -> list[str]:
        scripts: list[str] = []
        for match in re.finditer(r"\b[\w.-]+\.(?:py|sh)\b", prompt):
            script = match.group(0)
            if script not in scripts:
                scripts.append(script)
        return scripts

    @staticmethod
    def _prompt_implies_direct_script_run(prompt: str, script: str) -> bool:
        lowered = prompt.lower()
        script_lower = script.lower()
        idx = lowered.find(script_lower)
        if idx < 0:
            return False
        before_window = lowered[max(0, idx - 80): idx]
        after_window = lowered[idx + len(script_lower): idx + len(script_lower) + 180]
        window = before_window + script_lower + after_window
        action_markers = (
            " that runs",
            " that validates",
            " that checks",
            " should run",
            " should validate",
            " should check",
            " runs it",
            " validates it",
            " checks it",
        )
        if not any(marker in window for marker in action_markers):
            return False
        mandatory_markers = (
            "argument",
            "arguments",
            "positional",
            "required",
            "required arg",
            "accept",
            "takes",
            "take a",
            "option",
            "flag",
            "parameter",
            "configurable",
            "environment override",
            "env override",
            "provided ",
            "input ",
            "input string",
            "file path",
            "path ",
        )
        return not any(marker in after_window for marker in mandatory_markers)

    @staticmethod
    def _requirements_make_script_args_mandatory(requirements_text: str, script: str) -> bool:
        escaped = re.escape(script)
        patterns = (
            rf"{escaped}[^.:\n]{{0,120}}\bmust accept\b",
            rf"{escaped}[^.:\n]{{0,120}}\bmust take\b",
            rf"{escaped}[^.:\n]{{0,120}}\brequires?\b[^.:\n]{{0,60}}\b(argument|arguments|positional|command|count)\b",
            rf"{escaped}[^.:\n]{{0,120}}\baccepts?\b[^.:\n]{{0,60}}\b(argument|arguments|positional|command|count)\b",
        )
        optional_markers = (
            "optional",
            "default",
            "without any positional",
            "without positional",
            "without arguments",
            "without any arguments",
            "without args",
            "no positional",
            "no arguments",
            "direct invocation",
            "can be run directly",
            "support direct invocation",
            "supports direct invocation",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, requirements_text):
                window = requirements_text[max(0, match.start() - 80): match.end() + 120]
                if any(marker in window for marker in optional_markers):
                    continue
                return True
        return False

    def _plan_validates_direct_script_invocation(self, plan: list[dict[str, Any]], script: str) -> bool:
        for step in plan:
            for command in step.get("validation_commands", []) or []:
                if self._command_validates_direct_script_invocation(command, script):
                    return True
        return False

    def _command_validates_direct_script_invocation(self, command: Any, script: str) -> bool:
        argv = self._command_argv_for_static_check(command)
        if self._argv_is_direct_script_invocation(argv, script):
            return True
        for shell_text in self._shell_texts_for_static_check(argv):
            if self._shell_text_has_direct_script_invocation(shell_text, script):
                return True
        return False

    @staticmethod
    def _command_argv_for_static_check(command: Any) -> list[str]:
        if isinstance(command, dict):
            command = command.get("cmd", command.get("command", []))
        if isinstance(command, list):
            return [str(item) for item in command]
        if isinstance(command, str):
            try:
                return shlex.split(command)
            except ValueError:
                return []
        return []

    @staticmethod
    def _argv_is_direct_script_invocation(argv: list[str], script: str) -> bool:
        if not argv:
            return False
        script_positions = [idx for idx, item in enumerate(argv) if Path(item).name == script]
        if not script_positions:
            return False
        script_index = script_positions[0]
        if script.endswith(".py"):
            if script_index == 0:
                return len(argv) == 1
            if Path(argv[0]).name.startswith("python") and script_index == 1:
                return len(argv) == 2
            return False
        if script.endswith(".sh"):
            if script_index == 0:
                return len(argv) == 1
            if Path(argv[0]).name in {"bash", "sh"} and script_index == 1:
                return len(argv) == 2
        return False

    @classmethod
    def _shell_text_has_direct_script_invocation(cls, shell_text: str, script: str) -> bool:
        for segment in cls._shell_command_segments(shell_text):
            argv = cls._safe_shell_split(segment)
            if cls._argv_is_direct_script_invocation(argv, script):
                return True
        return False

    @staticmethod
    def _shell_command_segments(shell_text: str) -> list[str]:
        return [
            segment.strip()
            for segment in re.split(r"(?:&&|\|\||[;\n])", shell_text)
            if segment.strip()
        ]

    @staticmethod
    def _direct_script_command_example(script: str) -> list[str]:
        if script.endswith(".py"):
            return ["python", script]
        if script.endswith(".sh"):
            return ["bash", script]
        return [script]

    def _plan_validation_phase(self) -> dict:
        """Block implementation until the ordered plan is executable and checkable."""
        iterations: list[dict[str, Any]] = []
        review: dict[str, Any] = {}
        for index in range(1, self.config.phases.plan_validation.max_iterations + 1):
            review = self._plan_validation_review(index)
            iterations.append({"iteration": index, "review": review, "plan": self.plan_steps})
            status = self._status(review)
            if status in {"resolved", "resolved_with_compromise"}:
                if status == "resolved_with_compromise":
                    self._apply_validation_command_compromise_to_plan(review)
                self._append_plan_note(f"[plan] validated after iteration {index}: {review.get('summary', '')}")
                self._write_plan_doc()
                return {"status": status, "iterations": iterations}
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
            "default_quality_policy": self._default_quality_policy_payload(),
            "web_research_evidence": self.web_research_result,
            "execution_environment": self._execution_environment_payload(),
            "plan": self.plan_steps,
            "deterministic_structural_findings": structural_findings,
            "checks": self._plan_validation_prompt_checks(),
            "expected_json": {
                "status": "resolved|needs_plan_change|needs_requirements_change|cannot_resolve",
                "needs_rework": False,
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
        if structural_findings:
            if index > 1 and self._findings_are_validation_command_mechanics(structural_findings):
                return self._validation_command_compromise_review(
                    "plan",
                    structural_findings,
                    status="resolved_with_compromise",
                )
            return self._deterministic_plan_validation_review(structural_findings)

        raw = self._feedback_chat(
            "PLAN_VALIDATION_PHASE\n"
            "Before implementation starts, explicitly confirm whether the plan is feasible, clear, "
            "and verifiable. If any step cannot be independently verified, return needs_plan_change. "
            "Apply the shared plan-scope and validation-command rules. Treat step-count limits as "
            "hard only when the user explicitly says hard/strict/exactly/must; otherwise prefer a "
            "practical feasible verifiable plan. Per-attempt file-count guidance is not a plan-step limit. "
            "If deterministic_structural_findings is empty, do not invent command-syntax objections; "
            "focus on semantic coverage, scope, dependency order, and verifiability.\n"
                f"{_review_prompt_guidance(PLAN_SCOPE_RULES, VALIDATION_COMMAND_RULES, ORIGINAL_REQUEST_FIT_CHECK_GUIDANCE, self._execution_environment_guidance(), self._harness_state_file_guidance(), self._artifact_only_guidance())}\n"
            + json.dumps(prompt),
            temperature=0.1,
        )
        review = self._normalize_review(self._extract_json_or_retry(
            raw,
            phase="PLAN_VALIDATION_PHASE",
            contract=(
                '{"status":"resolved|needs_plan_change|needs_requirements_change|cannot_resolve",'
                '"needs_rework":false,"summary":"review summary","required_changes":["specific change"],'
                '"planning_confirmation":{"feasible":true,"clear":true,"verifiable":true,'
                '"verification_matrix":[{"step_id":"S1","how_verified":"command or explicit review method"}]}}'
            ),
            feedback=True,
        ))
        if self._legacy_semantic_phrase_checks_enabled():
            review = self._suppress_unsupported_validation_syntax_objection(review, scope="plan validation")
        self._record_effective_review_if_needed("PLAN_VALIDATION_PHASE", review)
        return review

    def _plan_validation_prompt_checks(self) -> list[str]:
        """Return review checks that match active deterministic policy.

        The model reviewer may still make semantic judgments from the request,
        requirements, and plan. This checklist avoids implying that default
        runs also use legacy phrase tables for scope-specific deterministic
        findings.
        """
        checks = [
            "each step is distinct",
            "dependencies are explicit",
            "each step has acceptance criteria",
            "each step has validation commands or an explicit non-command validation method",
            "validation commands terminate and assert behavior instead of starting a server forever",
            "validation evidence proves the requested behavior, not only superficial file existence or command success",
            "validation includes the default or most likely user-facing invocation, artifact path, or output surface implied by the original prompt",
            "browser/UI steps have executable browser evidence such as Playwright, screenshots, or a validation report when web interaction tools are enabled",
            "browser/UI plans match the agent container tools; Python Playwright is available, but Node/npm/npx/@playwright/test are not available unless explicitly configured",
            "project deliverables must not be harness-owned state files such as PLAN.md, REQUIREMENTS.md, or RESEARCH.md",
            "the sequence can be executed one step at a time",
            "planning_confirmation says the plan is feasible, clear, and verifiable",
            "the reviewer can name exactly how each step will be verified later",
            "requested documentation/design-note deliverables have acceptance criteria or bounded evidence for relevant content, not only file existence",
            "when default_quality_policy.requires_research_structure_step is true, the first step researches needed patterns/knowledge and plans project structure before feature implementation",
            "when default quality policy does not apply, the plan avoids unrequested documentation, tests, and research steps while still validating the requested deliverables",
            "when web research evidence exists, generated source-using notes or deliverables cite and apply researched source URLs",
        ]
        if self._legacy_semantic_phrase_checks_enabled():
            checks.extend([
                "computed-answer artifact tasks use semantic validation that recomputes or independently checks the answer, not only file existence or numeric format",
                "artifact-only prompts do not introduce helper files or validation scripts as workspace deliverables",
                "bounded tasks do not add standalone final-verification or QA-only steps that duplicate step validation and final review",
                "named scripts keep the prompt-implied direct invocation surface unless mandatory arguments were requested",
                "public function/API plans avoid unrequested caller-visible representation constraints and validate representation only when the user requested it",
            ])
        return checks

    def _deterministic_plan_validation_review(self, findings: list[str]) -> dict[str, Any]:
        """Return an authoritative plan review for local structural blockers."""
        unique_findings: list[str] = []
        for finding in findings:
            if finding not in unique_findings:
                unique_findings.append(finding)
        return {
            "status": "needs_plan_change",
            "needs_rework": True,
            "summary": "Deterministic plan checks found unresolved structural validation issues.",
            "required_changes": unique_findings,
            "cross_check_questions": [],
            "verification_evidence": [],
        }

    def _plan_refinement_pass(self, index: int, review: dict[str, Any]) -> dict:
        """Let the implementation model repair the plan while preserving context."""
        prompt = (
            f"PLAN_REFINEMENT_PHASE iteration={index}\n"
            "Revise only the ordered plan so every step is distinct, sequential, and verifiable. "
            "Keep requirements unless the review explicitly says they must change.\n"
            "Return the plan/refined planning confirmation contract below; do not repeat "
            "the full requirements list unless those details are needed for clarity. Apply the shared plan-scope "
                "and validation-command rules in the contract below.\n"
                f"{SELF_CHECK_GUIDANCE}\n"
                f"{ANTI_TUNNEL_VISION_GUIDANCE}\n"
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
                self._adopt_accepted_validation_commands_for_step(step, attempts[-1])
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
            self.conversation.append("user", self._next_implementation_directive(review))
        resolution = self._fallback_resolution(f"step {step['id']}", attempts[-1]["review"] if attempts else {})
        step["status"] = resolution["status"]
        return {"step_id": step["id"], "status": resolution["status"], "attempts": attempts, "resolution": resolution}

    def _next_implementation_directive(self, review: dict[str, Any]) -> str:
        compact_review = self._compact_review_for_transcript(review)
        deterministic_note = ""
        if compact_review.get("deterministic_evidence_findings"):
            deterministic_note = (
                "Deterministic evidence findings are authoritative repair blockers. "
                "Address them even when the model-written review summary is narrower or omits one of them. "
                "Do not claim a deterministic finding is fixed unless the next files and commands remove or prove it.\n"
            )
        return (
            "NEXT_IMPLEMENTATION_DIRECTIVE:\nApply this step review in the next attempt. "
            "Keep previous requirements, analysis, plan validation, repair history, and this step context in mind. "
            "Summarize what remains incomplete and complete those gaps if possible. Put proof for each rejected "
            "validation gap in the next response's machine-readable `commands` field; do not rely on plan_note "
            "claims for evidence. If the plan is now stale, impossible, or no longer useful, request "
            "needs_plan_change instead of burning attempts on it.\n"
            + deterministic_note
            + f"{STRUCTURAL_REPAIR_GUIDANCE}\n"
            "Review to apply:\n"
            + json.dumps(compact_review, indent=2)
        )

    def _update_active_repair_findings(
        self,
        step: dict[str, Any],
        attempt: int,
        review: dict[str, Any],
        deterministic_findings: list[str],
    ) -> None:
        """Persist unresolved repair blockers into deterministic workflow memory."""
        status = self._status(review)
        if status in {"resolved", "resolved_with_compromise", "skipped_with_note"} and not deterministic_findings:
            self.active_repair_findings = []
            return
        findings: list[str] = []
        for item in deterministic_findings:
            findings.append(f"{step.get('id', 'step')} attempt {attempt} deterministic: {item}")
        for item in review.get("required_changes", []) or []:
            findings.append(f"{step.get('id', 'step')} attempt {attempt} reviewer: {item}")
        self.active_repair_findings = self._clip_list_for_transcript(findings)

    def _deterministic_findings_plan_note(self, step: dict[str, Any], attempt: int, findings: list[str]) -> str:
        clipped = [clamp_text(str(item), 320, marker="finding truncated") for item in findings[:5]]
        return f"[{step.get('id', 'step')} attempt {attempt}] deterministic findings: " + " | ".join(clipped)

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
                f"{SELF_CHECK_GUIDANCE}\n"
                f"{ANTI_TUNNEL_VISION_GUIDANCE}\n"
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
            "Do not implement future plan steps early. If the current step is explicitly only setup, structure, or "
            "research, create minimal scaffolding and accurate placeholders only; leave feature mechanics for "
            "their own accepted steps.\n"
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
        payload = self._normalize_implementation_payload(payload)
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
                    "test_evidence_note": (
                        "Implementation test_evidence in this proposal is model-provided prose before "
                        "the harness executes these commands. Treat it as intended validation, not proof "
                        "that the command already passed."
                    ),
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

    @staticmethod
    def _as_list_field(value: Any) -> list[Any]:
        """Canonicalize model fields that are schema lists but often arrive as strings."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @classmethod
    def _normalize_command_field(cls, value: Any) -> list[Any]:
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            return [value]
        return cls._as_list_field(value)

    def _normalize_implementation_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["files"] = self._as_list_field(normalized.get("files"))
        normalized["commands"] = self._normalize_command_field(normalized.get("commands"))
        normalized["test_evidence"] = self._as_list_field(normalized.get("test_evidence"))
        return normalized

    def _step_review_pass(
        self,
        step: dict[str, Any],
        attempt: int,
        implementation: dict[str, Any],
        review_mode: str,
    ) -> dict:
        """Critique one step using reviewer-owned file and command evidence."""
        plan_text = self._plan_path().read_text(encoding="utf-8")
        feedback_tool_evidence = self._step_feedback_tool_evidence(step, implementation=implementation)
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
                "If original plan validation is stale but accepted implementation validation was rerun and passed, judge the current step against the fresh accepted validation evidence and request a plan update only when coverage is still weak.",
                "For computed-answer tasks, inspect code and command evidence with bounded sanity checks; do not manually enumerate long candidate sets or re-solve the whole calculation in the review turn.",
                "If semantic proof is absent or too weak, return needs_rework requesting a stronger validation command or verifier instead of replacing missing proof with ad hoc scratch derivation.",
                "If a previous review requested a specific evidence gap, require that the latest implementation includes corresponding validation in its machine-readable commands or a clear resolution_request; do not accept a prose claim alone.",
                "Use feedback_tool_evidence.git.status_short/diff_stat/diff to review changes since the last accepted step commit.",
                "Untracked meaningful paths are valid pre-acceptance implementation evidence; the harness will stage and commit after acceptance.",
                "If git meaningful_changed_paths is empty for an implementation step, request the missing change and name the current plan requirement.",
                "Validation-only steps may have no git diff when reviewer-owned validation commands pass; do not reject those solely for an empty working tree.",
                "Do not ask the implementation agent to run git add or git commit; repository mutation is harness-owned.",
                f"Do not require the implementation agent to pre-mark the current step completed in {self.config.runtime.plan_file}; the harness marks resolved after acceptance.",
                "Do not accept a step just because the implementation agent claims tests passed.",
                "For files that must be directly executable, require a shebang in the file content and evidence such as `test -x ./script` or direct invocation. Do not treat `python script.py` support as direct executability unless the user request, project convention, or accepted plan explicitly requires it. Do not request `chmod` or `chown` on workspace source files as validation; the harness marks shebang files executable when applying the JSON files payload.",
                "Treat implementation-requested commands as evidence, not as an automatic veto. If a model-side self-check fails or is blocked, discount it only when reviewer-owned validation and file evidence independently prove the same acceptance criterion; otherwise request stronger current evidence or a plan update.",
                "Reject validation that is too shallow for the requirement; require evidence that exercises the feature from the user's perspective.",
                "For negative-path behavior, prefer non-destructive wrapper commands, test doubles, fixture scripts, environment hooks, or expected_returncode; avoid asking the implementation to mutate source files temporarily just to simulate failure.",
                "If web_research_evidence has completed sources, confirm the generated work actually cites and applies those source URLs.",
                "If test evidence is absent in hard_pushback mode, return needs_rework.",
                "Do not use resolved_with_compromise merely because retries are taking time. If an unmet acceptance criterion can still be proven with a bounded command or plan change, return needs_rework or needs_plan_change. Use compromise only for impossible or explicitly diluted requirements and name the dilution.",
                "For browser/game work, prefer Playwright-style interaction evidence and screenshot/report artifacts when configured.",
                "Do not request incidental package/browser installation inside generated validation scripts for default browser checks; if a task requires another stack, request an explicit dependency/setup step with bounded commands.",
                "In compromise mode, accept a clearly labelled non-browser fallback only when browser launch cannot be made reliable and the fallback still gives concrete evidence.",
                    "Return needs_plan_change if this step cannot be independently verified as written, or if reviewer-owned validation is stale/misaligned while stronger implementation-provided validation now matches the chosen approach.",
                    "Return needs_requirements_change if the requirements are contradictory or impossible.",
                    "Return cannot_resolve only when bounded retries are unlikely to help.",
                    "Use the evidence-bound review check: accept only when evidence supports the current path; request a change only when a concrete gap, stale plan, or safer alternative is visible.",
                ],
            "expected_json": {
                "status": "resolved|needs_rework|cannot_resolve|needs_requirements_change|needs_plan_change|skipped_with_note|resolved_with_compromise",
                "needs_rework": False,
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
                "weak, shallow, or only proves that a command exited successfully, request stronger validation evidence "
                "instead.\n"
                f"{_review_prompt_guidance()}\n"
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
            contract='{"status":"resolved|needs_rework|cannot_resolve|needs_requirements_change|needs_plan_change|skipped_with_note|resolved_with_compromise","needs_rework":false,"summary":"review summary","required_changes":["specific change"]}',
            feedback=True,
            current_question_context=json.dumps(prompt, ensure_ascii=False),
        ))
        if self._legacy_semantic_phrase_checks_enabled():
            review = self._suppress_unsupported_negative_path_shell_objection(
                review,
                feedback_tool_evidence=feedback_tool_evidence,
            )
        review = self._enforce_evidence_policy(review, evidence_findings, review_mode)
        review["deterministic_evidence_findings"] = evidence_findings
        self._update_active_repair_findings(step, attempt, review, evidence_findings)
        self._record_effective_review_if_needed(
            "STEP_REVIEW_PHASE",
            review,
            reason="deterministic_evidence_findings" if evidence_findings else None,
        )
        review["feedback_tool_evidence"] = feedback_tool_evidence
        self._append_plan_note(f"[{step['id']} attempt {attempt}] review: {review.get('summary', 'no summary')}")
        if evidence_findings:
            self._append_plan_note(self._deterministic_findings_plan_note(step, attempt, evidence_findings))
        return review

    def _final_review_phase(self, step_results: list[dict[str, Any]]) -> dict:
        """Run whole-project review after individual plan steps complete."""
        iterations: list[dict[str, Any]] = []
        max_corrections = max(0, self.config.review_policy.final_review_iterations)
        corrections_used = 0
        attempt = 1
        while True:
            review = self._final_project_review(attempt, step_results)
            item: dict[str, Any] = {"attempt": attempt, "review": review}
            if self._status(review) in {"resolved", "resolved_with_compromise", "skipped_with_note"}:
                self._apply_final_review_rescues(step_results, review)
                self._append_plan_note(f"[final review] resolved: {review.get('summary', '')}")
                self._write_plan_doc()
                item["git_commit"] = self._git_commit_final_review()
                iterations.append(item)
                return {"status": self._status(review), "iterations": iterations}
            if corrections_used >= max_corrections:
                iterations.append(item)
                break
            correction = self._final_correction_pass(attempt, review)
            item["correction"] = correction
            iterations.append(item)
            corrections_used += 1
            attempt += 1
        last_review = iterations[-1]["review"] if iterations else {}
        protocol_compromise = self._final_review_protocol_failure_compromise(step_results, last_review)
        if protocol_compromise:
            iterations[-1]["protocol_compromise"] = protocol_compromise
            self._append_plan_note(f"[final review] resolved_with_compromise: {protocol_compromise['summary']}")
            self._write_plan_doc()
            iterations[-1]["git_commit"] = self._git_commit_final_review()
            self._record_effective_review_if_needed(
                "FINAL_PROJECT_REVIEW_PHASE",
                protocol_compromise,
                reason="final_review_protocol_failure_compromise",
            )
            return {
                "status": "resolved_with_compromise",
                "iterations": iterations,
                "resolution": protocol_compromise["resolution"],
            }
        fallback = self._fallback_resolution("final review", last_review)
        self._append_plan_note(f"[final review] {fallback['status']}: {fallback['note']}")
        final_review = {
            "status": fallback["status"],
            "needs_rework": fallback["status"] not in {"resolved", "skipped_with_note", "resolved_with_compromise"},
            "summary": fallback["note"],
            "required_changes": (iterations[-1]["review"].get("required_changes", []) if iterations else []),
            "resolution": fallback,
        }
        self._record_effective_review_if_needed(
            "FINAL_PROJECT_REVIEW_PHASE",
            final_review,
            reason="bounded_final_review_resolution",
        )
        return {"status": fallback["status"], "iterations": iterations, "resolution": fallback}

    def _final_review_protocol_failure_compromise(
        self,
        step_results: list[dict[str, Any]],
        review: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self._is_reviewer_protocol_failure(review):
            return None
        if not step_results or any(str(item.get("status")) != "resolved" for item in step_results):
            return None
        if review.get("deterministic_evidence_findings"):
            return None
        if not self._final_feedback_evidence_all_passed(review.get("feedback_tool_evidence")):
            return None
        summary = (
            "All plan steps and final validation commands passed, but the final-review model could not "
            "return a parseable protocol response after repair. Keeping the verified result with an explicit "
            "review-protocol compromise instead of restarting the whole approach."
        )
        resolution = {
            "status": "resolved_with_compromise",
            "note": summary,
        }
        return {
            "status": "resolved_with_compromise",
            "needs_rework": False,
            "summary": summary,
            "required_changes": [],
            "verification_evidence": [
                "Step reviews are resolved.",
                "Final reviewer-owned validation commands passed.",
                "Only the reviewer JSON protocol repair failed.",
            ],
            "resolution": resolution,
            "review_protocol_error": True,
        }

    @staticmethod
    def _is_reviewer_protocol_failure(review: dict[str, Any]) -> bool:
        if review.get("review_protocol_error"):
            return True
        text = "\n".join([
            str(review.get("summary") or ""),
            "\n".join(str(item) for item in review.get("required_changes", []) or []),
        ])
        return "Reviewer protocol repair failed" in text or "reviewer response was malformed" in text

    def _final_feedback_evidence_all_passed(self, evidence: Any) -> bool:
        if not isinstance(evidence, dict):
            return False
        validation_groups = evidence.get("step_validations") or []
        seen_result = False
        for group in validation_groups:
            for result in group.get("validation_results", []) or []:
                seen_result = True
                if result.get("timed_out") or result.get("stopped_by_progress_review"):
                    return False
                if not self._command_returncode_matches_expected(result):
                    return False
        return seen_result

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
        available_evidence = self._approach_review_evidence_catalog(step_results, final_review)
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
            "available_evidence": available_evidence,
            "expected_json": {
                "status": "resolved|try_another_approach|needs_rework|cannot_resolve",
                "needs_rework": False,
                "summary": "decision summary",
                "decision": "keep_result|retry_with_new_approach|stop",
                "recommended_next_approach": "only when retrying",
                "evidence_reviewed": ["available_evidence id"],
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
                "existing approach say so without requesting a full retry. Use evidence_reviewed only as a "
                "citation list: copy IDs from available_evidence exactly, and put interpretation in summary or "
                "runbook_updates. Do not add new facts, manual derivations, calculations, or proof claims that "
                "are not present in available_evidence.\n"
                f"{_review_prompt_guidance(ORIGINAL_REQUEST_FIT_CHECK_GUIDANCE)}\n"
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
            current_question_context=json.dumps(prompt, ensure_ascii=False),
        ))
        decision = str(review.get("decision") or "").strip()
        if decision == "retry_with_new_approach" and self._status(review) == "resolved":
            review["status"] = "try_another_approach"
            review["needs_rework"] = True
        final_status = self._final_status(step_results, final_review)
        if final_status != "resolved" and self._status(review) == "resolved" and decision in {"", "keep_result"}:
            failure_details = self._approach_retry_failure_details(step_results, final_review)
            existing_changes = [str(item) for item in review.get("required_changes", [])]
            existing_notes = [str(item) for item in review.get("runbook_updates", [])]
            review["status"] = "try_another_approach"
            review["needs_rework"] = True
            review["decision"] = "retry_with_new_approach"
            review["summary"] = (
                "The approach cannot be kept as resolved because the workflow final status is "
                f"{final_status}. Re-run analysis and planning using the recorded failure evidence."
            )
            evidence_reviewed = review.setdefault("evidence_reviewed", [])
            if "final_review:status" not in evidence_reviewed:
                evidence_reviewed.append("final_review:status")
            review["required_changes"] = existing_changes + [
                item for item in failure_details["required_changes"] if item not in existing_changes
            ]
            review["runbook_updates"] = existing_notes + [
                item for item in failure_details["runbook_updates"] if item not in existing_notes
            ]
            if failure_details["recommended_next_approach"]:
                review["recommended_next_approach"] = failure_details["recommended_next_approach"]
        if self._status(review) == "try_another_approach" and not review.get("recommended_next_approach"):
            review["recommended_next_approach"] = "Re-run analysis and planning from the recorded gaps."
        review = self._canonicalize_approach_review_summary(review)
        self.conversation.append(
            "user",
            "APPROACH_REVIEW_RESULT:\n"
            + json.dumps(self._compact_approach_review_for_transcript(review), indent=2),
        )
        self._append_plan_note(f"[approach review {approach_attempt}] {review.get('summary', 'no summary')}")
        return review

    @staticmethod
    def _compact_approach_review_for_transcript(review: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "status",
            "needs_rework",
            "summary",
            "decision",
            "recommended_next_approach",
            "evidence_reviewed",
            "runbook_updates",
            "required_changes",
            "reviewer_rationale",
        )
        return {
            key: review.get(key)
            for key in keys
            if review.get(key) not in (None, "", [])
        }

    @staticmethod
    def _canonicalize_approach_review_summary(review: dict[str, Any]) -> dict[str, Any]:
        review = dict(review)
        evidence_ids = [
            str(item)
            for item in review.get("evidence_reviewed", []) or []
            if isinstance(item, str) and item.strip()
        ]
        evidence_text = ", ".join(evidence_ids[:6]) if evidence_ids else "no cited evidence IDs"
        status = str(review.get("status") or "").strip() or "unknown"
        decision = str(review.get("decision") or "").strip() or "unknown"
        if decision == "keep_result" and status == "resolved":
            summary = f"Approach review kept the result based on cited evidence IDs: {evidence_text}."
        elif decision == "retry_with_new_approach" or status == "try_another_approach":
            summary = f"Approach review requested another approach based on cited evidence IDs: {evidence_text}."
        elif decision == "stop":
            summary = f"Approach review stopped without another attempt based on cited evidence IDs: {evidence_text}."
        else:
            summary = f"Approach review returned status {status} and decision {decision} based on cited evidence IDs: {evidence_text}."
        original_summary = str(review.get("summary") or "").strip()
        if original_summary and original_summary != summary:
            review["reviewer_rationale"] = original_summary
        review["summary"] = summary
        return review

    def _approach_review_evidence_catalog(
        self,
        step_results: list[dict[str, Any]],
        final_review: dict[str, Any],
    ) -> list[dict[str, str]]:
        evidence: list[dict[str, str]] = [
            {
                "id": "project_design:prompt",
                "text": self._prompt_excerpt(self.config.project_design.prompt, 1200),
            },
            {
                "id": "analysis:summary",
                "text": self._prompt_excerpt(self._analysis_summary_for_prompt(), 1200),
            },
            {
                "id": "requirements:summary",
                "text": self._prompt_excerpt(self._requirements_summary_for_prompt(), 1200),
            },
            {
                "id": "plan:summary",
                "text": self._prompt_excerpt(json.dumps(self._compact_plan_for_prompt(), ensure_ascii=False), 1200),
            },
        ]
        compact_final = self._compact_final_review_for_approach(final_review)
        for key in ("status", "summary"):
            value = compact_final.get(key)
            if value:
                evidence.append({
                    "id": f"final_review:{key}",
                    "text": self._prompt_excerpt(str(value), 1200),
                })
        for key in ("required_changes", "verification_evidence"):
            values = compact_final.get(key)
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values[:8]):
                evidence.append({
                    "id": f"final_review:{key}:{index}",
                    "text": self._prompt_excerpt(str(value), 1200),
                })
        for index, result in enumerate(step_results[:8]):
            summary = {
                "status": result.get("status"),
                "step_id": result.get("step_id"),
                "attempts": len(result.get("attempts", []) or []),
            }
            evidence.append({
                "id": f"step_result:{index}",
                "text": self._prompt_excerpt(json.dumps(summary, ensure_ascii=False), 1200),
            })
        if self.approach_history:
            evidence.append({
                "id": "approach_history:summary",
                "text": self._prompt_excerpt(self._approach_history_summary_for_prompt(), 1200),
            })
        return evidence

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

    def _approach_retry_failure_details(
        self,
        step_results: list[dict[str, Any]],
        final_review: dict[str, Any],
    ) -> dict[str, list[str] | str]:
        """Collect concrete failure evidence for the next approach attempt."""
        required_changes: list[str] = []
        runbook_updates: list[str] = []

        def add_change(value: object) -> None:
            text = str(value).strip()
            if text and text not in required_changes:
                required_changes.append(text)

        def add_note(value: object) -> None:
            text = str(value).strip()
            if text and text not in runbook_updates:
                runbook_updates.append(text)

        for result in step_results:
            phase_result = result.get("phase_result")
            if isinstance(phase_result, dict):
                phase = str(result.get("step_id") or "phase")
                compact = self._compact_phase_result_for_prompt(phase_result)
                summary = compact.get("last_review_summary") or compact.get("resolution_note")
                if summary:
                    add_note(f"{phase}: {summary}")
                for item in compact.get("required_changes", []) or []:
                    add_change(item)
            attempts = result.get("attempts") or []
            if attempts:
                review = attempts[-1].get("review") or {}
                if review.get("summary"):
                    add_note(f"{result.get('step_id', 'step')}: {review.get('summary')}")
                for item in review.get("required_changes", []) or []:
                    add_change(item)

        compact_final = self._compact_final_review_for_approach(final_review)
        if compact_final.get("last_review_summary"):
            add_note(f"final review: {compact_final.get('last_review_summary')}")
        for item in compact_final.get("required_changes", []) or []:
            add_change(item)
        for item in compact_final.get("deterministic_evidence_findings", []) or []:
            add_change(item)

        clipped_changes = self._clip_list_for_transcript(required_changes)
        clipped_notes = self._clip_list_for_transcript(runbook_updates)
        if clipped_changes:
            next_approach = "Before retrying, address these recorded blockers: " + "; ".join(clipped_changes[:3])
        elif clipped_notes:
            next_approach = "Re-run analysis and planning using these recorded blockers: " + "; ".join(clipped_notes[:3])
        else:
            next_approach = "Re-run analysis and planning from the recorded gaps."
        return {
            "required_changes": clipped_changes,
            "runbook_updates": clipped_notes,
            "recommended_next_approach": next_approach,
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
            "project_design": self.config.project_design.prompt,
            "requirements": self._requirements_summary_for_prompt(),
            "plan": self._compact_plan_for_prompt(),
            "step_results": self._compact_step_results_for_prompt(step_results),
            "feedback_tool_evidence": self._compact_final_evidence_for_prompt(feedback_tool_evidence),
            "deterministic_evidence_findings": evidence_findings,
            "expected_json": {
                "status": "resolved|needs_rework|cannot_resolve|needs_requirements_change|needs_plan_change|skipped_with_note|resolved_with_compromise",
                "needs_rework": False,
                "summary": "concrete final review summary",
                "required_changes": ["concrete final change, or empty when resolved"],
                "verification_evidence": ["specific command result, file evidence, or reviewer fact"],
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
                "Do the same when validation passes but would not fail for a plausible wrong implementation of the "
                "requested user-visible behavior. "
                "Treat project_design as the highest-priority scope source: if refined requirements, plan text, "
                "documentation, tests, or code add behavior not present in the original request, request a "
                "requirements or plan correction instead of accepting the broadened scope. "
                "If a step accepted a compromise for a verifiable acceptance criterion, do not silently upgrade that "
                "to resolved at final review; request the missing bounded evidence or a plan/requirements change. "
                "Push back if the project lacks proof or contradicts requirements.\n"
                f"{_review_prompt_guidance(ORIGINAL_REQUEST_FIT_CHECK_GUIDANCE)}\n"
                + json.dumps(prompt),
            context_note=(
                "The full multi-turn transcript is stored in .agent_state/conversation.full.jsonl. "
                "Use this compact final-review payload plus reviewer-owned validation reruns to decide. "
                "Prefer the rerun evidence over manual derivations; request better validation when proof is weak. "
                "Do not convert unverified acceptance criteria into source-inspection-only acceptance when a bounded "
                "validation command could prove them. "
                "All individual plan steps were reviewed before this final pass."
            ),
            temperature=0.1,
        )
        review = self._normalize_review(self._extract_json_or_retry(
            raw,
            phase="FINAL_PROJECT_REVIEW_PHASE",
            contract='{"status":"resolved|needs_rework|cannot_resolve|needs_requirements_change|needs_plan_change|skipped_with_note|resolved_with_compromise","needs_rework":false,"summary":"concrete final review summary","required_changes":["concrete final change, or empty when resolved"],"verification_evidence":["specific command result, file evidence, or reviewer fact"]}',
            feedback=True,
            current_question_context=json.dumps(prompt, ensure_ascii=False),
        ))
        if evidence_findings and self._status(review) in {"resolved", "resolved_with_compromise", "skipped_with_note"}:
            review["status"] = "needs_rework"
            review["needs_rework"] = True
            review["summary"] = "Final review cannot resolve because deterministic evidence checks found gaps."
            review["required_changes"] = evidence_findings
        review["feedback_tool_evidence"] = feedback_tool_evidence
        review["deterministic_evidence_findings"] = evidence_findings
        if evidence_findings:
            self._record_effective_review_if_needed(
                "FINAL_PROJECT_REVIEW_PHASE",
                review,
                reason="deterministic_evidence_findings",
            )
        return review

    def _apply_final_review_rescues(self, step_results: list[dict[str, Any]], review: dict[str, Any]) -> None:
        """Update current plan state when final evidence supersedes a skipped step.

        The attempt history still records that a step exhausted its local
        review budget, but later prompts and the runbook should show the current
        verified state. Otherwise approach review sees a contradictory workflow:
        fresh final validation passed, while the current plan still says the
        step is skipped.
        """
        evidence = review.get("feedback_tool_evidence")
        if not isinstance(evidence, dict):
            return
        final_validations = {
            str(item.get("step_id")): item
            for item in evidence.get("step_validations", [])
        }
        rescued_ids: list[str] = []
        for step_result in step_results:
            step_id = str(step_result.get("step_id"))
            if not self._skipped_step_is_superseded_by_final_evidence(step_result, final_validations.get(step_id)):
                continue
            step_result["historical_status"] = step_result.get("status")
            step_result["status"] = "resolved"
            step_result["resolved_by_final_review"] = True
            rescued_ids.append(step_id)
            for step in self.plan_steps:
                if str(step.get("id")) == step_id:
                    step["historical_status"] = step.get("status")
                    step["status"] = "resolved"
                    step["resolved_by_final_review"] = True
                    break
        if rescued_ids:
            self._append_plan_note(
                "[final review] fresh reviewer-owned evidence resolved previously skipped step(s): "
                + ", ".join(rescued_ids)
            )

    def _compact_plan_for_prompt(self) -> list[dict[str, Any]]:
        """Summarize the plan without embedding large command/file payloads."""
        compact: list[dict[str, Any]] = []
        for step in self.plan_steps:
            acceptance_criteria, omitted_count = self._compact_acceptance_criteria_for_prompt(
                step.get("acceptance_criteria", [])
            )
            compact.append({
                "id": step.get("id"),
                "title": step.get("title"),
                "status": step.get("status"),
                "acceptance_criteria": acceptance_criteria,
                "acceptance_criteria_total": len(step.get("acceptance_criteria", []) or []),
                "acceptance_criteria_omitted_count": omitted_count,
                "validation_command_count": len(step.get("validation_commands") or []),
            })
        return compact

    @staticmethod
    def _compact_acceptance_criteria_for_prompt(criteria: Any) -> tuple[list[str], int]:
        """Preserve criteria for review while keeping each step bounded.

        Final and approach reviews need the whole acceptance surface, not only
        the first few items. Keep all normal-sized criteria, clip individual
        long items, and only omit tail items when a pathological list would
        dominate the prompt. The omitted count is explicit so a reviewer does
        not mistake a partial view for complete evidence.
        """
        if not isinstance(criteria, list):
            criteria = []
        max_total_chars = 6000
        compact: list[str] = []
        used_chars = 0
        for index, item in enumerate(criteria):
            clipped = clamp_text(str(item), 700, marker="acceptance criterion truncated")
            projected = used_chars + len(clipped)
            if compact and projected > max_total_chars:
                return compact, len(criteria) - index
            compact.append(clipped)
            used_chars = projected
        return compact, 0

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
            raw = implementation.get("raw") or {}
            implementation_command_summary = self._command_result_counts(
                implementation.get("commands", [])
            )
            reviewer_evidence = (review.get("feedback_tool_evidence") or {})
            reviewer_validation_summary = self._command_result_counts(
                reviewer_evidence.get("validation_results", [])
            )
            item = {
                "step_id": result.get("step_id"),
                "status": result.get("status"),
                "attempt_count": len(attempts),
                "written_paths": implementation.get("written", []),
                "last_review_status": review.get("status"),
                "last_review_summary": review.get("summary"),
                "implementation_command_summary": implementation_command_summary,
                "reviewer_validation_summary": reviewer_validation_summary,
            }
            phase_result = result.get("phase_result")
            if isinstance(phase_result, dict):
                item["phase_failure"] = self._compact_phase_result_for_prompt(phase_result)
            claimed_evidence = raw.get("test_evidence", [])
            if claimed_evidence:
                item["implementation_test_evidence_claims"] = self._clip_list_for_transcript(claimed_evidence)
                if (
                    implementation_command_summary["blocked"]
                    or implementation_command_summary["failed"]
                    or implementation_command_summary["timed_out"]
                    or implementation_command_summary["stopped_by_progress_review"]
                ):
                    item["evidence_note"] = (
                        "Implementation test_evidence is model-provided prose. Prefer reviewer-owned "
                        "validation and executed command summaries because at least one implementation "
                        "command was blocked, failed, timed out, or stopped."
                    )
            compact.append(item)
        return compact

    def _compact_phase_result_for_prompt(self, phase_result: dict[str, Any]) -> dict[str, Any]:
        iterations = phase_result.get("iterations") or []
        last = iterations[-1] if iterations else {}
        review = last.get("review") if isinstance(last, dict) else {}
        if not isinstance(review, dict):
            review = {}
        resolution = phase_result.get("resolution") if isinstance(phase_result.get("resolution"), dict) else {}
        return {
            "status": phase_result.get("status"),
            "resolution_status": resolution.get("status"),
            "resolution_note": resolution.get("note"),
            "last_review_status": review.get("status"),
            "last_review_summary": review.get("summary"),
            "required_changes": self._clip_list_for_transcript(review.get("required_changes", [])),
            "cross_check_questions": self._clip_list_for_transcript(review.get("cross_check_questions", [])),
        }

    def _command_result_counts(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize command outcomes without preserving misleading prose claims."""
        summary: dict[str, Any] = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "blocked": 0,
            "timed_out": 0,
            "stopped_by_progress_review": 0,
            "not_run_or_unknown": 0,
            "problem_examples": [],
        }
        examples: list[dict[str, Any]] = []
        for result in results or []:
            if not isinstance(result, dict):
                continue
            summary["total"] += 1
            command = result.get("command")
            status = ""
            if result.get("blocked_by_tool_verifier") or "tool call blocked before execution by verification step" in str(
                result.get("stderr") or ""
            ).lower():
                summary["blocked"] += 1
                status = "blocked"
            elif result.get("timed_out"):
                summary["timed_out"] += 1
                status = "timed_out"
            elif result.get("stopped_by_progress_review"):
                summary["stopped_by_progress_review"] += 1
                status = "stopped_by_progress_review"
            elif "returncode" not in result:
                summary["not_run_or_unknown"] += 1
                status = "unknown"
            elif self._command_returncode_matches_expected(result):
                summary["passed"] += 1
            else:
                summary["failed"] += 1
                status = "failed"
            if status and len(examples) < 3:
                examples.append({
                    "status": status,
                    "command": command,
                    "returncode": result.get("returncode"),
                    "expected_returncode": result.get("expected_returncode", 0),
                    "output_excerpt": self._command_failure_excerpt(result, limit=350).removeprefix(
                        ". Output excerpt: "
                    ),
                })
        if examples:
            summary["problem_examples"] = examples
        return summary

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
                "elapsed_seconds": result.get("elapsed_seconds"),
                "returncode": result.get("returncode"),
                "expected_returncode": result.get("expected_returncode"),
                "returncode_matches_expected": result.get("returncode_matches_expected"),
                "timed_out": result.get("timed_out"),
                "stopped_by_progress_review": result.get("stopped_by_progress_review"),
                "progress_reviews": self._compact_progress_reviews(result.get("progress_reviews", [])),
                "stdout": self._prompt_excerpt(stdout, limit),
                "stderr": self._prompt_excerpt(stderr, limit),
                "stdout_prompt_truncated": len(stdout) > limit,
                "stderr_prompt_truncated": len(stderr) > limit,
            })
        return compact

    def _compact_progress_reviews(self, reviews: Any) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        if not isinstance(reviews, list):
            return compact
        for review in reviews[-3:]:
            if not isinstance(review, dict):
                continue
            compact.append({
                "status": review.get("status"),
                "decision": review.get("decision"),
                "summary": self._prompt_excerpt(str(review.get("summary", "")), 500),
                "evidence": self._clip_list_for_transcript(review.get("evidence", [])),
                "next_check_seconds": review.get("next_check_seconds"),
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
            "accepted_validation_commands": evidence.get("accepted_validation_commands", []),
            "accepted_validation_results": self._compact_command_results_for_prompt(
                evidence.get("accepted_validation_results", [])
            ),
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
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "returncode": result.get("returncode"),
                    "expected_returncode": result.get("expected_returncode"),
                    "returncode_matches_expected": result.get("returncode_matches_expected"),
                    "timed_out": result.get("timed_out"),
                    "stopped_by_progress_review": result.get("stopped_by_progress_review"),
                    "progress_reviews": self._compact_progress_reviews(result.get("progress_reviews", [])),
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

    def _initial_workspace_context_for_prompt(self) -> dict[str, Any]:
        """Provide bounded source evidence before the model starts planning.

        The harness should not diagnose or solve the task here. It only supplies
        a compact file snapshot so the analysis phase can cite real available
        sources instead of inventing or deferring basic discovery.
        """
        try:
            files = collect_workspace_files(
                self.workspace,
                max_file_bytes=min(self.config.context_compaction.workspace_file_max_bytes, 6000),
            )
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": str(exc),
                "file_count": 0,
                "files": [],
                "omitted_count": 0,
            }
        prompt_files = self._reviewer_prompt_files(files)
        selected: list[dict[str, Any]] = []
        total_chars = 0
        max_total_chars = 12000
        max_files = 12
        for item in prompt_files:
            compact = self._compact_file_for_prompt(item, default_limit=1200)
            encoded = json.dumps(compact, ensure_ascii=False, sort_keys=True)
            if len(selected) >= max_files or total_chars + len(encoded) > max_total_chars:
                continue
            selected.append(compact)
            total_chars += len(encoded)
        return {
            "status": "available",
            "file_count": len(prompt_files),
            "files": selected,
            "omitted_count": max(0, len(prompt_files) - len(selected)),
            "note": "Bounded pre-analysis snapshot only; run explicit commands later when more evidence is needed.",
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
            f"{ANTI_TUNNEL_VISION_GUIDANCE}\n"
            f"Do not include harness-owned state files in the files payload: "
            f"{', '.join(sorted(self._harness_doc_names()))}. The harness creates and updates those files.\n"
            f"{self._artifact_only_guidance()}\n"
            f"Review: {json.dumps(self._compact_review_for_correction(review))}\n\n"
            f"{IMPLEMENTATION_CONTRACT}"
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

    def _plan_structural_findings(self, *, include_diagnostic_quality: bool = True) -> list[str]:
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
        semantic_phrase_checks = self._legacy_semantic_phrase_checks_enabled()
        computed_answer_semantic_validation_present = (
            self._computed_answer_plan_has_semantic_validation(self.plan_steps)
            if semantic_phrase_checks
            else False
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
            findings.extend(
                self._validation_command_findings(
                    step,
                    computed_answer_semantic_validation_present=computed_answer_semantic_validation_present,
                    include_diagnostic_quality=include_diagnostic_quality,
                )
            )
            findings.extend(self._harness_state_file_plan_findings(step))
            if semantic_phrase_checks:
                findings.extend(self._artifact_only_plan_findings(step))
                findings.extend(self._proportional_scope_plan_findings(step))
            for dep in step.get("depends_on", []):
                if dep not in seen_ids:
                    findings.append(f"{step_id} depends on {dep}, which has not appeared earlier in the ordered plan.")
        if semantic_phrase_checks:
            findings.extend(self._artifact_only_plan_deliverable_findings())
            findings.extend(self._named_failure_mode_validation_findings())
            findings.extend(self._requirements_negative_path_validation_findings())
        if semantic_phrase_checks:
            findings.extend(self._computed_answer_validation_findings(requirements=self.requirements, plan=self.plan_steps))
        if semantic_phrase_checks:
            findings.extend(self._script_primary_input_surface_findings(requirements=self.requirements, plan=self.plan_steps))
        if semantic_phrase_checks and self._default_quality_policy_requires_research_structure_step():
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
                    "default_quality_policy.requires_research_structure_step is true, so the first plan step must "
                    "research needed patterns/knowledge, plan project structure/architecture, and rewrite the "
                    "remaining plan if structure changes task order."
                )
            if self._has_completed_research() and not any(marker in first_text for marker in ("source", "url", "cite", "citation")):
                findings.append(
                    "Web research evidence exists, so generated source-using notes or deliverables must require citing and applying source URLs."
                )
        if semantic_phrase_checks:
            findings.extend(self._redundant_final_verification_step_findings())
            findings.extend(self._unrequested_test_deliverable_findings())
            findings.extend(self._script_direct_invocation_findings(requirements=self.requirements, plan=self.plan_steps))
            findings.extend(self._public_api_overconstraint_findings(self.requirements))
            findings.extend(self._environment_assumption_findings(requirements=self.requirements, plan=self.plan_steps))
            if isinstance(self.requirements, dict):
                scoped_requirements = dict(self.requirements)
                scoped_requirements["plan"] = self.plan_steps
                findings.extend(self._requirements_test_runner_consistency_findings(scoped_requirements))
            findings.extend(self._validation_only_public_flag_findings())
            findings.extend(self._unexpected_named_script_reference_findings())
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

    def _validation_only_public_flag_findings(self) -> list[str]:
        """Reject public failure-injection switches invented only to make validation easy."""
        prompt = self.config.project_design.prompt.lower()
        scope_text_parts: list[str] = []
        if isinstance(self.requirements, dict):
            scope_text_parts.extend(str(item) for item in self.requirements.get("refined_requirements", []) or [])
            scope_text_parts.extend(str(item) for item in self.requirements.get("assumptions", []) or [])
        for step in self.plan_steps:
            scope_text_parts.extend([
                str(step.get("title", "")),
                str(step.get("description", "")),
                " ".join(str(item) for item in step.get("acceptance_criteria", []) or []),
            ])
        text = "\n".join(scope_text_parts).lower()
        if not text:
            return []
        suspicious_flags = sorted(set(re.findall(
            r"--(?:[\w-]*(?:test|fail|failure|bad|wrong|invalid)[\w-]*)",
            text,
        )))
        invented = [
            flag for flag in suspicious_flags
            if flag not in prompt and any(marker in flag for marker in ("fail", "bad", "wrong", "invalid"))
        ]
        flags = sorted(set(re.findall(r"--[a-z0-9][a-z0-9_-]*", text)))
        validation_only_flags: list[str] = []
        segments = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?])\s+|\n+", text)
            if segment.strip()
        ]
        for flag in flags:
            if flag in prompt or flag in invented:
                continue
            for segment in segments:
                if flag not in segment:
                    continue
                if (
                    (
                        re.search(r"\b(?:facilitat(?:e|es|ing)|allow(?:s|ing)?|support(?:s|ing)?)\b.{0,80}\b(?:testing|validation)\b", segment)
                        and re.search(r"\b(?:failure|negative|error|bad|wrong|invalid)\b", segment)
                    )
                    or re.search(r"\b(?:test|testing|validation)\b.{0,80}\b(?:failure|negative|error|bad|wrong|invalid)\b", segment)
                    or "failure mode" in segment
                ):
                    validation_only_flags.append(flag)
                    break
        invented.extend(flag for flag in validation_only_flags if flag not in invented)
        if not invented:
            return []
        return [
            (
                "Requirements or plan introduce public failure-injection/test switches "
                f"{invented} that the user did not request. Do not change the user-visible artifact API just "
                "to make validation easier; use temporary fixtures, test doubles, wrapper commands, or "
                "expected_returncode-based checks instead."
            )
        ]

    def _unexpected_named_script_reference_findings(self) -> list[str]:
        prompt_scripts = set(re.findall(
            r"\b[\w.-]+\.py\b",
            self.config.project_design.prompt,
            flags=re.IGNORECASE,
        ))
        if not prompt_scripts:
            return []
        payload = {
            "requirements": self.requirements if isinstance(self.requirements, dict) else {},
            "plan": [
                {
                    "title": step.get("title", ""),
                    "description": step.get("description", ""),
                    "acceptance_criteria": step.get("acceptance_criteria", []),
                }
                for step in self.plan_steps
            ],
        }
        text = json.dumps(payload, ensure_ascii=False)
        mentioned = set(re.findall(r"\b[\w.-]+\.py\b", text, flags=re.IGNORECASE))
        unexpected = sorted(mentioned - prompt_scripts)
        findings: list[str] = []
        for script in unexpected:
            if self._is_expected_test_script_variant(script, prompt_scripts):
                continue
            closest = difflib.get_close_matches(script, sorted(prompt_scripts), n=1, cutoff=0.78)
            if not closest:
                continue
            findings.append(
                f"Generated requirements or plan mention `{script}`, which looks like a typo or unintended variant "
                f"of user-named script `{closest[0]}`. Preserve named entrypoints exactly unless the user asks "
                "for additional scripts."
            )
        return findings

    def _is_expected_test_script_variant(self, script: str, prompt_scripts: set[str]) -> bool:
        prompt = self.config.project_design.prompt.lower()
        if not re.search(r"\b(test|tests|testing|unittest|unit tests?|test suite)\b", prompt):
            return False
        script_lower = script.lower()
        for prompt_script in prompt_scripts:
            stem = Path(prompt_script).stem.lower()
            expected = {
                f"test_{stem}.py",
                f"{stem}_test.py",
                f"tests_{stem}.py",
            }
            if script_lower in expected:
                return True
        return False

    def _named_failure_mode_validation_findings(self) -> list[str]:
        """Ensure distinct named failure categories are not collapsed into one generic negative test."""
        required_modes = self._required_named_failure_modes()
        if len(required_modes) <= 1:
            return []
        command_text = self._all_plan_validation_command_text()
        missing = sorted(
            mode
            for mode in required_modes
            if not self._failure_mode_covered_by_commands(mode, command_text)
            and not self._failure_mode_covered_by_test_plan(mode)
        )
        if not missing:
            return []
        readable = ", ".join(self._failure_mode_label(mode) for mode in missing)
        return [
            (
                "Requirements or acceptance criteria name multiple distinct failure modes, but validation_commands "
                f"do not clearly target: {readable}. Add bounded validation for each named failure category, "
                "or revise the requirements/acceptance criteria if those categories are not actually required. "
                "Good generic proof can be a generated test/validation script, a copied temporary workspace, "
                "or a wired temporary fixture that the validator actually consumes. For simple expected "
                "non-zero CLI cases, prefer one command object with expected_returncode for each bad invocation "
                "instead of one broad shell chain. Do not mutate implemented workspace source files or add "
                "public failure-injection flags solely for validation."
            )
        ]

    def _requirements_negative_path_validation_findings(self) -> list[str]:
        required_modes = self._required_named_failure_modes()
        if not required_modes:
            return []
        command_text = self._all_plan_validation_command_text()
        missing = sorted(
            mode
            for mode in required_modes
            if not self._failure_mode_covered_by_commands(mode, command_text)
            and not self._failure_mode_covered_by_test_plan(mode)
        )
        if not missing:
            return []
        readable = ", ".join(self._failure_mode_label(mode) for mode in missing)
        return [
            (
                "Requirements, assumptions, or acceptance criteria include negative-path behavior, but "
                f"validation_commands do not clearly prove: {readable}. Add a bounded command with "
                "expected_returncode, a generated test/validation script, a copied temporary workspace, "
                "or a wired temporary fixture that the validator actually consumes. Require error-text "
                "assertions only when the requirements name the text or a wrapper hides the child status. "
                "For simple CLI argument-count or bad-input cases, prefer a separate command object for "
                "each failing invocation. "
                "Do not treat a normal success invocation with a different valid argument value as a "
                "negative-path check; the command must create, pass, or run bad input/test data that the "
                "program under validation actually consumes. Do not mutate implemented workspace source "
                "files or add public failure-injection flags solely for validation."
            )
        ]

    def _required_named_failure_modes(self) -> set[str]:
        blocks: list[str] = []
        if isinstance(self.requirements, dict):
            for item in self.requirements.get("refined_requirements", []) or []:
                if self._text_requires_negative_path(str(item)):
                    blocks.append(str(item))
            for item in self.requirements.get("assumptions", []) or []:
                if self._text_requires_negative_path(str(item)):
                    blocks.append(str(item))
            confirmation = self.requirements.get("planning_confirmation") or {}
            if isinstance(confirmation, dict):
                strategy = str(confirmation.get("verification_strategy") or "")
                if self._text_requires_negative_path(strategy):
                    blocks.append(strategy)
        for step in self.plan_steps:
            for item in step.get("acceptance_criteria", []) or []:
                if self._text_requires_negative_path(str(item)):
                    blocks.append(str(item))
        modes: set[str] = set()
        for block in blocks:
            lower = block.lower()
            for mode, patterns in self._failure_mode_requirement_patterns().items():
                if not any(pattern in lower for pattern in patterns):
                    continue
                if mode == "empty_input" and not self._empty_input_is_failure_requirement(lower):
                    continue
                modes.add(mode)
        return modes

    def _all_plan_validation_command_text(self) -> str:
        chunks: list[str] = []
        for step in self.plan_steps:
            commands = step.get("validation_commands") or []
            chunks.append(json.dumps(commands, ensure_ascii=False, sort_keys=True).lower())
            for command in commands:
                argv = self._command_argv_for_static_check(command)
                chunks.append(" ".join(argv).lower())
                chunks.extend(shell_text.lower() for shell_text in self._shell_texts_for_static_check(argv))
        return "\n".join(chunks)

    @staticmethod
    def _failure_mode_requirement_patterns() -> dict[str, tuple[str, ...]]:
        return {
            "invalid_argument": (
                "invalid argument",
                "invalid arguments",
                "invalid input",
                "invalid value",
                "bad argument",
                "bad input",
                "non-integer",
                "non integer",
            ),
            "incorrect_count": (
                "wrong count",
                "incorrect count",
                "count is incorrect",
                "count mismatch",
                "mismatched count",
                "wrong number of lines",
                "incorrect number of lines",
                "wrong line count",
                "incorrect line count",
            ),
            "incorrect_format": (
                "wrong format",
                "incorrect format",
                "format is incorrect",
                "format mismatch",
                "invalid format",
                "malformed format",
            ),
            "missing_argument": (
                "missing arg",
                "missing args",
                "exactly one argument",
                "exactly one positional argument",
                "takes exactly one argument",
                "takes exactly one positional argument",
                "take exactly one argument",
                "take exactly one positional argument",
                "missing argument",
                "missing arguments",
                "missing input",
                "required argument",
                "required arguments",
                "0 argument",
                "0 arguments",
                "no argument",
                "no arguments",
                "no args",
            ),
            "too_many_arguments": (
                "exactly one argument",
                "exactly one positional argument",
                "takes exactly one argument",
                "takes exactly one positional argument",
                "take exactly one argument",
                "take exactly one positional argument",
                "more than one argument",
                "more than one positional argument",
                ">1 argument",
                ">1 arguments",
                ">1 arg",
                ">1 args",
                "too many arguments",
                "too many args",
                "extra argument",
                "extra arguments",
                "unexpected argument",
                "unexpected arguments",
                "wrong number of arguments",
                "invalid argument count",
            ),
            "empty_input": (
                "empty input",
                "empty iterable",
                "empty list",
            ),
        }

    @staticmethod
    def _failure_mode_command_patterns() -> dict[str, tuple[str, ...]]:
        return {
            "invalid_argument": (
                "invalid",
                "bad-input",
                "bad_input",
                "--bad",
                " abc",
                " not-a-number",
                " non-integer",
                "non integer",
            ),
            "incorrect_count": (
                "bad_cnt",
                "bad-cnt",
                "bad_count",
                "bad-count",
                "fail_count",
                "fail-count",
                "wrong_count",
                "wrong-count",
                "incorrect_count",
                "incorrect-count",
                "wrong count",
                "incorrect count",
                "count mismatch",
                "mismatched count",
                "wrong number of lines",
                "incorrect number of lines",
                "wrong line count",
                "incorrect line count",
            ),
            "incorrect_format": (
                "fail_fmt",
                "fail-fmt",
                "fail_format",
                "fail-format",
                "wrong_format",
                "wrong-format",
                "incorrect_format",
                "incorrect-format",
                "wrong format",
                "incorrect format",
                "format mismatch",
                "invalid format",
                "malformed format",
            ),
            "missing_argument": (
                "missing argument",
                "missing input",
                "required argument",
                "usage:",
            ),
            "too_many_arguments": (
                "too many arguments",
                "too many args",
                "extra argument",
                "extra arguments",
                "unexpected argument",
                "unexpected arguments",
                "more than one argument",
                "wrong number of arguments",
                "invalid argument count",
                "usage:",
            ),
            "empty_input": (
                "empty input",
                "empty iterable",
                "empty list",
                "[]",
            ),
        }

    @staticmethod
    def _empty_input_is_failure_requirement(text: str) -> bool:
        """Distinguish valid empty-input edge cases from empty-input failures."""
        lower = text.lower()
        empty_pattern = r"(?:empty input|empty iterable|empty list)"
        if not re.search(empty_pattern, lower):
            return False
        positive_patterns = (
            r"\breturns?\s+\[\]\s+(?:for|on|with)\s+" + empty_pattern,
            r"\bhandles?\s+" + empty_pattern,
            empty_pattern + r"[^.\n]{0,80}\b(?:returns?\s+\[\]|is valid|success|succeeds?)\b",
        )
        if any(re.search(pattern, lower) for pattern in positive_patterns):
            return False
        failure_pattern = (
            r"(?:raises?|throws?|errors?|fails?|non[- ]?zero|valueerror|invalid)"
            r"[^.\n]{0,100}"
            + empty_pattern
            + r"|"
            + empty_pattern
            + r"[^.\n]{0,100}"
            r"(?:raises?|throws?|errors?|fails?|non[- ]?zero|valueerror|invalid)"
        )
        return bool(re.search(failure_pattern, lower))

    @classmethod
    def _failure_mode_covered_by_commands(cls, mode: str, command_text: str) -> bool:
        if mode == "missing_argument":
            return cls._command_text_proves_missing_argument(command_text)
        if mode == "too_many_arguments":
            return cls._command_text_proves_too_many_arguments(command_text)
        if mode == "incorrect_count" and cls._command_text_proves_incorrect_count(command_text):
            return True
        if mode == "incorrect_format" and cls._command_text_proves_incorrect_format(command_text):
            return True
        return any(pattern in command_text for pattern in cls._failure_mode_command_patterns().get(mode, ()))

    @classmethod
    def _command_text_proves_incorrect_count(cls, command_text: str) -> bool:
        text = command_text.lower()
        if any(pattern in text for pattern in cls._failure_mode_command_patterns().get("incorrect_count", ())):
            return True
        has_expected_failure_status = any(
            marker in text
            for marker in (
                "expected_returncode",
                "$?",
                "-ne 0",
                "!= 0",
                "returncode",
                "exit 1",
                "exit 2",
            )
        )
        if not has_expected_failure_status:
            return False
        if not re.search(r"\bvalidate[a-z0-9_-]*\.py\b|\bvalidator\b", text):
            return False
        one_line_fixture = (
            re.search(r"\b(?:echo|printf|cat)\b[^;&|]*(?:line\s+1|print\s*\(\s*['\"]line\s+1)", text)
            or re.search(r"\bprint\s*\(\s*['\"]line\s+1", text)
        )
        return bool(one_line_fixture)

    @classmethod
    def _command_text_proves_incorrect_format(cls, command_text: str) -> bool:
        text = command_text.lower()
        if any(pattern in text for pattern in cls._failure_mode_command_patterns().get("incorrect_format", ())):
            return True
        has_expected_failure_status = any(
            marker in text
            for marker in (
                "expected_returncode",
                "$?",
                "-ne 0",
                "!= 0",
                "returncode",
                "exit 1",
                "exit 2",
            )
        )
        if not has_expected_failure_status:
            return False
        if not re.search(r"\bvalidate[a-z0-9_-]*\.py\b|\bvalidator\b", text):
            return False
        malformed_fixture = (
            re.search(
                r"\b(?:echo|printf|cat)\b[^;&|]*(?:bad\s+format|bad[-_ ]?output|not\s+a\s+line|malformed)",
                text,
            )
            or re.search(
                r"\bprint\s*\(\s*['\"](?:bad\s+format|bad[-_ ]?output|not\s+a\s+line|malformed)",
                text,
            )
        )
        return bool(malformed_fixture)

    @staticmethod
    def _command_text_proves_missing_argument(command_text: str) -> bool:
        """Recognize bounded no-argument CLI checks without requiring magic words.

        Local models often validate argparse-style behavior with shell wrappers
        such as `out=$(python cli.py 2>&1); test $? -ne 0 && grep usage`.  That
        proves the missing-argument path even when the command text does not
        contain the exact phrase "missing argument".
        """
        text = command_text.lower()
        has_status_check = any(
            marker in text
            for marker in (
                "expected_returncode",
                "$?",
                "-ne 0",
                "!= 0",
                "returncode",
                "exit 2",
                "exit 1",
                "if !",
            )
        )
        if not has_status_check:
            return False
        has_error_observation = any(
            marker in text
            for marker in (
                "2>&1",
                "usage",
                "required",
                "error",
                "stderr",
                "no argument",
                "missing",
            )
        )
        if not has_error_observation:
            return False
        return bool(
            re.search(
                r"\bpython(?:\d+(?:\.\d+)?)?\s+[\w./-]+\.py\s*(?:2?>|[;&|)]|$)",
                text,
            )
            or re.search(
                r"['\"]python(?:\d+(?:\.\d+)?)?['\"]\s*,\s*['\"][^'\"]+\.py['\"]\s*[\]\)]",
                text,
            )
        )

    @staticmethod
    def _command_text_proves_too_many_arguments(command_text: str) -> bool:
        """Recognize bounded checks for a CLI receiving extra positional args."""
        text = command_text.lower()
        has_status_check = any(
            marker in text
            for marker in (
                "expected_returncode",
                "$?",
                "-ne 0",
                "!= 0",
                "returncode",
                "exit 2",
                "exit 1",
                "if !",
            )
        )
        if not has_status_check:
            return False
        if any(
            marker in text
            for marker in (
                "too many arguments",
                "too many args",
                "extra argument",
                "extra arguments",
                "unexpected argument",
                "unexpected arguments",
                "more than one argument",
                "wrong number of arguments",
                "invalid argument count",
            )
        ):
            return True
        return bool(
            re.search(
                r"['\"]python(?:\d+(?:\.\d+)?)?['\"]\s*,\s*['\"][^'\"]+\.py['\"]\s*,\s*['\"][^'\"]+['\"]\s*,\s*['\"][^'\"]+['\"]",
                text,
            )
            or re.search(
                r"\bpython(?:\d+(?:\.\d+)?)?\s+[\w./-]+\.py\s+(?:--\s+)?"
                r"(?:'[^']*'|\"[^\"]*\"|[^\s;&|)'\"]+)\s+"
                r"(?:'[^']*'|\"[^\"]*\"|[^\s;&|)'\"]+)(?=$|[\s;&|)])",
                text,
            )
        )

    def _failure_mode_covered_by_test_plan(self, mode: str) -> bool:
        patterns = self._failure_mode_requirement_patterns().get(mode, ())
        if not patterns:
            return False
        for step in self.plan_steps:
            if not any(self._command_is_test_runner(command) for command in step.get("validation_commands") or []):
                continue
            step_text = " ".join([
                str(step.get("title", "")),
                str(step.get("description", "")),
                " ".join(str(item) for item in step.get("acceptance_criteria", []) or []),
                self._requirements_test_coverage_text(),
            ]).lower()
            if not any(marker in step_text for marker in ("test", "tests", "unittest", "pytest")):
                continue
            if any(pattern in step_text for pattern in patterns):
                return True
        return False

    def _requirements_test_coverage_text(self) -> str:
        if not isinstance(self.requirements, dict):
            return ""
        chunks: list[str] = []
        for key in ("refined_requirements", "assumptions"):
            for item in self.requirements.get(key, []) or []:
                text = str(item)
                if re.search(r"\b(?:test|tests|testing|unittest|pytest|suite|cases?)\b", text, flags=re.IGNORECASE):
                    chunks.append(text)
        confirmation = self.requirements.get("planning_confirmation") or {}
        if isinstance(confirmation, dict):
            strategy = str(confirmation.get("verification_strategy") or "")
            if re.search(r"\b(?:test|tests|testing|unittest|pytest|suite|cases?)\b", strategy, flags=re.IGNORECASE):
                chunks.append(strategy)
        return " ".join(chunks)

    @staticmethod
    def _failure_mode_label(mode: str) -> str:
        return mode.replace("_", " ")

    def _redundant_final_verification_step_findings(self) -> list[str]:
        """Reject pure duplicate final-check steps for bounded tasks.

        Step and final review already rerun validation evidence. For small,
        bounded prompts a separate "final verification" step often costs a full
        implementation/review loop without adding a deliverable. Keep this
        generic and conservative: only flag non-quality-scope prompts and only
        when the step text is clearly a final QA/review/check step rather than a
        requested verifier artifact.
        """
        if self._default_quality_policy_applies() or len(self.plan_steps) < 2:
            return []
        findings: list[str] = []
        for step in self.plan_steps[1:]:
            text = " ".join([
                str(step.get("id", "")),
                str(step.get("title", "")),
                str(step.get("description", "")),
                " ".join(str(item) for item in step.get("acceptance_criteria", []) or []),
            ]).lower()
            if not self._looks_like_redundant_final_verification_text(text):
                continue
            if self._step_mentions_requested_deliverable(step):
                continue
            findings.append(
                f"{step.get('id', '<missing>')} appears to be a standalone final verification/QA step for a bounded task. "
                "Merge those terminating validation commands into the relevant implementation step unless the user "
                "explicitly requested a separate testing deliverable or there is a real dependency."
            )
        return findings

    @staticmethod
    def _looks_like_redundant_final_verification_text(text: str) -> bool:
        final_markers = (
            "final verification",
            "final validation",
            "final check",
            "final review",
            "comprehensive check",
            "qa",
            "quality assurance",
            "run all tests",
            "rerun tests",
            "integration verification",
            "verify everything",
        )
        if any(marker in text for marker in final_markers):
            return True
        return (
            any(marker in text for marker in ("verify", "validate", "test", "check"))
            and not any(marker in text for marker in ("implement", "create", "build", "write", "fix", "add"))
        )

    def _step_mentions_requested_deliverable(self, step: dict[str, Any]) -> bool:
        """Allow separate steps that create user-requested verifier/test artifacts."""
        prompt = self.config.project_design.prompt.lower()
        step_text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", []) or []),
        ]).lower()
        for filename in re.findall(r"\b[\w.-]+\.(?:py|js|ts|html|css|json|md|txt|sh|yml|yaml)\b", step_text):
            if filename in prompt and self._step_creates_or_updates_named_deliverable(step_text, filename):
                return True
        requested_test_markers = (
            "include tests",
            "add tests",
            "write tests",
            "test suite",
            "unit tests",
            "pytest",
            "unittest",
        )
        return any(marker in prompt for marker in requested_test_markers) and any(
            marker in step_text for marker in ("test", "tests", "pytest", "unittest")
        )

    @staticmethod
    def _step_creates_or_updates_named_deliverable(step_text: str, filename: str) -> bool:
        escaped = re.escape(filename.lower())
        action = r"(?:create|creates|creating|write|writes|writing|implement|implements|build|builds|add|adds|generate|generates|update|updates)"
        return bool(
            re.search(rf"\b{action}\b[^.\n]{{0,120}}\b{escaped}\b", step_text)
            or re.search(rf"\b{escaped}\b[^.\n]{{0,120}}\b{action}\b", step_text)
        )

    def _unrequested_test_deliverable_findings(self) -> list[str]:
        """Keep bounded prompts from growing unrequested project test suites."""
        if self._default_quality_policy_applies():
            return []
        prompt = self.config.project_design.prompt.lower()
        requested_test_markers = (
            "include tests",
            "add tests",
            "write tests",
            "test suite",
            "unit tests",
            "pytest",
            "unittest",
            "spec file",
        )
        if any(marker in prompt for marker in requested_test_markers):
            return []
        findings: list[str] = []
        for step in self.plan_steps:
            text = " ".join([
                str(step.get("title", "")),
                str(step.get("description", "")),
                " ".join(str(item) for item in step.get("acceptance_criteria", []) or []),
                json.dumps(step.get("validation_commands", []), ensure_ascii=False),
            ]).lower()
            unrequested = []
            for filename in re.findall(r"\b[\w.-]*(?:test|spec)[\w.-]*\.(?:py|js|ts|sh)\b", text):
                if filename not in prompt:
                    unrequested.append(filename)
            if unrequested:
                findings.append(
                    f"{step.get('id', '<missing>')} introduces unrequested test deliverable(s) "
                    f"{sorted(set(unrequested))} while proportional quality policy is off. "
                    "Use bounded validation commands or reviewer-owned validation instead, unless the user asks "
                    "for tests or names that file."
                )
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
                "required_changes": self._clip_list_for_transcript(review.get("required_changes", [])),
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
                f"user limited deliverables to {allowed_text}. Use inline validation commands, or temporary "
                "validator code/fixtures outside the workspace, instead of creating helper project files."
            )
        ]

    def _proportional_scope_plan_findings(self, step: dict[str, Any]) -> list[str]:
        """Reject unrequested helper artifacts for bounded named-artifact tasks."""
        if self._default_quality_policy_applies() or self._explicit_artifact_only_constraint():
            return []
        requested = self._requested_artifact_paths_from_prompt()
        if not requested:
            return []
        text = json.dumps(
            {
                "title": step.get("title", ""),
                "description": step.get("description", ""),
                "acceptance_criteria": step.get("acceptance_criteria", []),
                "validation_commands": step.get("validation_commands", []),
            },
            sort_keys=True,
        )
        harness_docs = {Path(name).as_posix() for name in self._harness_doc_names()}
        extras = sorted(
            ref
            for ref in self._file_references_in_text(text)
            if ref not in requested
            and ref not in harness_docs
            and not self._artifact_reference_is_temporary(ref, text)
        )
        findings: list[str] = []
        step_id = str(step.get("id") or "step")
        if extras:
            requested_text = ", ".join(sorted(requested))
            findings.append(
                f"{step_id} introduces unrequested workspace helper artifact(s) {', '.join(extras)} "
                f"for a bounded prompt whose named deliverable(s) are {requested_text}. Keep validation inline "
                "or in temporary files outside the workspace unless the user explicitly requested those helper files."
            )
        install_commands = self._validation_dependency_install_commands(step.get("validation_commands") or [])
        if install_commands and not self._prompt_requests_dependency_setup():
            findings.append(
                f"{step_id} hides package installation inside validation command(s): {', '.join(install_commands)}. "
                "For bounded prompts without requested dependency setup, validate with available tools or make "
                "dependency discovery/setup an explicit plan step only when the user or existing project requires it."
            )
        return findings

    def _requested_artifact_paths_from_prompt(self) -> set[str]:
        prompt = self.config.project_design.prompt
        requested = {
            ref
            for ref in self._file_references_in_text(prompt)
            if ref not in {Path(name).as_posix() for name in self._harness_doc_names()}
        }
        return requested

    def _validation_dependency_install_commands(self, commands: list[Any]) -> list[str]:
        install_commands: list[str] = []
        for command in commands:
            parts = self._command_argv_for_static_check(command)
            texts = [" ".join(parts), *self._shell_texts_for_static_check(parts)]
            if any(self._text_contains_dependency_install(text) for text in texts):
                install_commands.append(self._validation_command_text_for_similarity(command))
        return install_commands

    @staticmethod
    def _text_contains_dependency_install(text: str) -> bool:
        lower = text.lower()
        patterns = (
            r"\bpython[0-9.]*\s+-m\s+pip\s+install\b",
            r"\bpip[0-9.]*\s+install\b",
            r"\bnpm\s+install\b",
            r"\bnpm\s+i\b",
            r"\byarn\s+add\b",
            r"\bpnpm\s+add\b",
            r"\bapt-get\s+install\b",
            r"\bapt\s+install\b",
        )
        return any(re.search(pattern, lower) for pattern in patterns)

    def _prompt_requests_dependency_setup(self) -> bool:
        prompt = self.config.project_design.prompt.lower()
        markers = (
            "install",
            "dependency",
            "dependencies",
            "package manager",
            "pip ",
            "npm ",
            "apt ",
            "requires requests",
            "use requests",
        )
        return any(marker in prompt for marker in markers)

    def _artifact_only_plan_deliverable_findings(self) -> list[str]:
        if not self._explicit_artifact_only_constraint():
            return []
        allowed = sorted(self._artifact_only_allowed_paths())
        if not allowed:
            return []
        deliverable_text = json.dumps(
            [
                {
                    "title": step.get("title", ""),
                    "description": step.get("description", ""),
                    "acceptance_criteria": step.get("acceptance_criteria", []),
                }
                for step in self.plan_steps
            ],
            sort_keys=True,
        ).lower()
        validation_text = json.dumps(
            [step.get("validation_commands", []) for step in self.plan_steps],
            sort_keys=True,
        ).lower()
        mentioned_deliverables = [
            path for path in allowed if path.lower() in deliverable_text
        ]
        inspected_deliverables = [
            path for path in allowed if path.lower() in validation_text
        ]
        allowed_text = ", ".join(allowed)
        findings: list[str] = []
        if len(self.plan_steps) > 1:
            non_deliverable_steps = []
            for step in self.plan_steps:
                step_text = json.dumps(
                    {
                        "title": step.get("title", ""),
                        "description": step.get("description", ""),
                        "acceptance_criteria": step.get("acceptance_criteria", []),
                    },
                    sort_keys=True,
                ).lower()
                if not any(path.lower() in step_text for path in allowed):
                    non_deliverable_steps.append(str(step.get("id") or "<missing>"))
            if non_deliverable_steps:
                findings.append(
                    "Artifact-only plan splits non-deliverable work into separate step(s) "
                    f"{', '.join(non_deliverable_steps)} even though the user limited deliverables to "
                    f"{allowed_text}. Merge calculation, probing, or stdout-only work into the step that "
                    "writes and validates the requested artifact unless the user explicitly requested a "
                    "separate deliverable or a real dependency requires the split."
                )
        if not mentioned_deliverables:
            findings.append(
                "Artifact-only plan does not preserve the explicitly requested artifact "
                f"({allowed_text}) in step title, description, or acceptance criteria. "
                "Revise the plan so implementation writes the requested artifact rather than only printing or computing a value."
            )
        if not inspected_deliverables:
            findings.append(
                "Artifact-only plan validation does not inspect the explicitly requested artifact "
                f"({allowed_text}). Validation must read or compare the artifact itself, not only print or compute the underlying value."
            )
        return findings

    def _validation_command_findings(
        self,
        step: dict[str, Any],
        *,
        computed_answer_semantic_validation_present: bool = False,
        include_diagnostic_quality: bool = True,
    ) -> list[str]:
        findings: list[str] = []
        commands = step.get("validation_commands") or []
        command_text = json.dumps(commands).lower()
        step_id = str(step.get("id") or "step")
        semantic_phrase_checks = self._legacy_semantic_phrase_checks_enabled()
        for command in commands:
            if isinstance(command, dict):
                if command.get("manual_test"):
                    findings.append(
                        f"{step_id} uses manual_test metadata in validation_commands; replace it with an executable script/report command."
                    )
                unsupported_assertion_keys = sorted(
                    key
                    for key in command
                    if key
                    in {
                        "expected_output",
                        "expected_stdout",
                        "stdout_equals",
                        "stdout_contains",
                        "stderr_contains",
                    }
                )
                if unsupported_assertion_keys:
                    findings.append(
                        f"{step_id} validation command uses unsupported assertion metadata "
                        f"{unsupported_assertion_keys}. The harness only enforces return codes; wrap stdout/stderr "
                        "checks in an executable command or validation script that exits non-zero on mismatch."
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
            argv_metadata = self._looks_like_metadata_inside_argv(raw_parts)
            if argv_metadata:
                findings.append(
                    f"{step_id} validation command puts `{argv_metadata}` inside the argv list. "
                    f"Use a command object such as {{\"cmd\": {json.dumps(raw_parts[:3])}, "
                    f"\"{argv_metadata}\": 0}} instead."
                )
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
            if self._timeout_wraps_shell_builtin_wait(raw_parts):
                findings.append(
                    f"{step_id} validation tries to run shell builtin `wait` through external `timeout`. "
                    "`wait` is not a standalone program, so `timeout ... wait $PID` fails before observing "
                    "the child process. Use a shell script that starts the child and waits inside the same "
                    "shell, or use a polling/communicate timeout pattern."
                )
            grep_option_pattern_finding = self._grep_option_like_pattern_finding(raw_parts)
            if grep_option_pattern_finding:
                findings.append(f"{step_id} validation {grep_option_pattern_finding}")
            grep_literal_pattern_finding = self._grep_literal_regex_pattern_finding(raw_parts)
            if grep_literal_pattern_finding:
                findings.append(f"{step_id} validation {grep_literal_pattern_finding}")
            if self._looks_like_py_compile_directory_command(parts):
                findings.append(
                    f"{step_id} validation uses `python -m py_compile` on a directory. "
                    "Use `python -m compileall <dir>` or a generated validation script for package-wide syntax checks."
                )
            if self._looks_like_invalid_inline_python_compound_command(parts):
                findings.append(
                    f"{step_id} validation uses an invalid one-line `python -c` compound statement that Python cannot parse. "
                    "Replace it with a simple argv check, a JSON-safe single-line expression validator, "
                    "or a generated validation script when the requested scope permits helper/test files. "
                    "Do not put raw multiline Python inside a JSON command string. Simple semicolon-separated "
                    "imports, assignments, file reads, assertions, and print calls are allowed when the static "
                    "syntax check passes."
                )
            if include_diagnostic_quality and self._looks_like_silent_subprocess_capture_validation(parts):
                findings.append(
                    f"{step_id} validation captures subprocess output but discards it when the child command fails. "
                    "Print or assert the captured stdout/stderr on failure, or use a small validation script, so repair "
                    "iterations can see the real nested error."
                )
            if (
                include_diagnostic_quality
                and semantic_phrase_checks
                and self._looks_like_silent_semantic_validation_command(step, raw_parts)
            ):
                findings.append(
                    f"{step_id} semantic validation exits non-zero on mismatch without diagnostic output. "
                    "Print concise expected/actual values, a representative failing case, or relevant stderr/stdout "
                    "before exiting non-zero so repair iterations have useful evidence."
                )
            if self._looks_like_placeholder_validation_command(raw_parts):
                findings.append(
                    f"{step_id} validation contains placeholder or stub test logic that can pass without "
                    "exercising the requested artifact. Replace it with assertions that run the implemented "
                    "program or inspect the requested deliverable behavior."
                )
            workspace_output_targets = self._validation_workspace_output_targets(raw_parts)
            if workspace_output_targets:
                targets = ", ".join(f"`{target}`" for target in workspace_output_targets)
                findings.append(
                    f"{step_id} validation writes temporary command output to workspace path {targets}. "
                    "Use `/tmp`, `mktemp`, or another trap-cleaned temporary path so validation evidence "
                    "does not become an unrequested project artifact."
                )
            stateful_help_finding = self._stateful_validation_help_only_finding(step, raw_parts)
            if stateful_help_finding:
                findings.append(f"{step_id} validation {stateful_help_finding}")
            else:
                runtime_state_finding = self._stateful_validation_runtime_state_finding(step, raw_parts)
                if runtime_state_finding:
                    findings.append(f"{step_id} validation {runtime_state_finding}")
            printf_format_findings = self._printf_literal_percent_findings(raw_parts)
            for finding in printf_format_findings:
                findings.append(f"{step_id} validation {finding}")
            misplaced_env_assignments = self._misplaced_environment_assignments_after_program(raw_parts)
            if misplaced_env_assignments:
                names = ", ".join(f"`{name}=...`" for name in misplaced_env_assignments)
                findings.append(
                    f"{step_id} validation passes {names} after the command-under-test. Shell treats that as "
                    "an argument, not an environment override. Put environment assignments before the command "
                    "or use `env VAR=value command`."
                )
            if self._looks_like_precedence_prone_arithmetic_validation(raw_parts):
                findings.append(
                    f"{step_id} validation appears to rely on a dense integer-division, multiplication, and modulo "
                    "expression whose operator precedence can change the intended arithmetic. Use explicit "
                    "parentheses or named intermediate variables for extracted values before multiplying them."
                )
            raw_numeric_comparison = self._raw_text_numeric_comparison_finding(raw_parts)
            if raw_numeric_comparison:
                findings.append(f"{step_id} validation {raw_numeric_comparison}")
            inline_python_syntax_error = self._inline_python_static_syntax_error(raw_parts)
            if inline_python_syntax_error:
                shell_wrapped_hint = (
                    " For shell-wrapped Python with JSON/dict literals or nested quotes, switch to a quoted "
                    "here-doc, a temporary validator script, or a direct argv-list `python -c` command instead "
                    "of trying another escaped-quote variant."
                    if "shell-wrapped python -c" in inline_python_syntax_error
                    else ""
                )
                findings.append(
                    f"{step_id} validation contains inline Python that fails a static syntax check: "
                    f"{inline_python_syntax_error}. Replace it with a simple argv check, a JSON-safe single-line "
                    "expression validator, or a generated validation script when the requested scope permits helper/test files. "
                    "Do not put raw multiline Python inside a JSON command string. Simple semicolon-separated "
                    "imports, assignments, file reads, assertions, and print calls are allowed when they compile. "
                    "Never use `bash -c python -c ...` or split the shell script across argv elements."
                    f"{shell_wrapped_hint}"
                )
            inline_python_unreachable = self._inline_python_unreachable_after_return(raw_parts)
            if inline_python_unreachable:
                findings.append(
                    f"{step_id} validation contains inline Python with unreachable statements after return: "
                    f"{inline_python_unreachable}. Move assertions, artifact reads/writes, and proof output outside "
                    "the function body, or use a generated validation script when the requested scope permits helper/test files."
                )
            heredoc_error = self._shell_heredoc_static_error(raw_parts)
            if heredoc_error:
                findings.append(
                    f"{step_id} validation contains malformed shell here-doc syntax: {heredoc_error}. "
                    "For a quoted here-doc opener such as `<<'PY'`, the closing delimiter line must be exactly `PY`."
                )
            shell_syntax_error = self._shell_static_syntax_error(raw_parts)
            if shell_syntax_error:
                findings.append(
                    f"{step_id} validation contains shell syntax that fails a static parse check: "
                    f"{shell_syntax_error}. Replace it with a parseable `bash -lc`/`sh -c` script or a "
                    "small validation script before accepting the plan."
                )
            artifact_heredoc_error = self._artifact_only_heredoc_finding(raw_parts) if semantic_phrase_checks else ""
            if artifact_heredoc_error:
                findings.append(f"{step_id} validation {artifact_heredoc_error}")
            if self._looks_like_unwrapped_expected_failure_validation(step, command, parts):
                findings.append(
                    f"{step_id} validation appears to test an expected failure path without declaring expected_returncode "
                    "or wrapping the exception assertion. Replace it with a command object using expected_returncode, "
                    "or a small wrapper command/script that exits 0 only when the expected error occurs."
                )
            if self._looks_like_swallowed_expected_failure_validation(step, command, raw_parts):
                findings.append(
                    f"{step_id} validation appears to mask an expected failure with `|| exit 0`, `|| true`, "
                    "or an equivalent always-success fallback. Use expected_returncode, or a wrapper that exits 1 "
                    "when the command unexpectedly succeeds and exits 0 only after confirming the intended failure."
                )
            elif self._looks_like_validation_failure_masking_shell_fallback(command, raw_parts):
                findings.append(
                    f"{step_id} validation appears to mask an assertion failure with `|| exit 0`, `|| true`, "
                    "`|| echo ...`, or an equivalent always-success fallback. Let the assertion command's "
                    "non-zero status fail the validation, or capture `$?` immediately before cleanup and "
                    "exit with that status."
                )
            if self._looks_like_expected_failure_status_masked_by_shell_tail(command, raw_parts):
                findings.append(
                    f"{step_id} validation declares an expected non-zero return code, but the shell script appears "
                    "to run cleanup or another trailing command after the command-under-test without preserving its "
                    "status. Capture `$?` immediately, run cleanup, then `exit $status`, or use `trap` for cleanup."
                )
            if self._looks_like_validation_status_masked_by_shell_tail(command, raw_parts):
                findings.append(
                    f"{step_id} validation appears to run cleanup or another trailing command after a validation "
                    "assertion without preserving the assertion status. Use `trap` for cleanup, capture `$?` and "
                    "exit with it after cleanup, or chain cleanup so a failed assertion cannot be hidden."
                )
            if semantic_phrase_checks and self._looks_like_negative_path_pipeline_without_status_check(step, command, raw_parts):
                findings.append(
                    f"{step_id} validation pipes an expected failure-path command into grep without checking the "
                    "command-under-test exit status. Capture `$?`/a status variable and assert both the non-zero "
                    "code and error text, or use a command object with expected_returncode."
                )
            filtered_absence_finding = (
                self._filtered_absence_check_finding(step, raw_parts)
                if semantic_phrase_checks
                else ""
            )
            if filtered_absence_finding:
                findings.append(f"{step_id} validation {filtered_absence_finding}")
            if self._validation_command_appears_to_mutate_artifact(raw_parts):
                findings.append(
                    f"{step_id} validation appears to write or mutate the explicitly requested artifact. "
                    "Validation commands must assert the artifact's state after implementation; create or update "
                    "the artifact through the implementation `files` payload instead."
                )
            source_mutation_target = self._workspace_source_mutation_target(raw_parts)
            if source_mutation_target:
                findings.append(
                    f"{step_id} validation appears to write or mutate workspace source path "
                    f"`{source_mutation_target}`. Validation commands must observe implemented files and assert "
                    "behavior; use /tmp fixtures, wrapper commands, test doubles, or expected_returncode for "
                    "negative-path checks instead of temporarily overwriting project source. If this is an "
                    "executable generated file, include a shebang in the files payload and validate executability "
                    "with `test -x` or direct invocation instead of `chmod`."
                )
            if self._looks_like_unwired_temp_fixture_validation(step, command, raw_parts):
                findings.append(
                    f"{step_id} validation creates a temporary fixture or test double, but the command appears "
                    "not to wire that fixture into the program being validated. Pass the fixture path through a "
                    "real option or dependency-injection hook, import it by name through a test runner/mock, or "
                    "run a copied workspace from the temporary directory; `PYTHONPATH` alone is not evidence that "
                    "the validated program consumes the fixture."
                )
            if (
                semantic_phrase_checks
                and
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
        if semantic_phrase_checks:
            findings.extend(self._executable_deliverable_validation_findings(step, commands))
        findings.extend(self._documentation_content_validation_findings(step, commands, "Plan validation"))
        if (
            semantic_phrase_checks
            and self._step_requires_negative_path_evidence(step)
            and not self._validation_commands_include_negative_path(commands, step=step)
        ):
            findings.append(
                f"{step_id} has acceptance criteria for an error, invalid-input, non-zero, or failure path, "
                "but validation_commands only show success-path evidence. Add a bounded command object with "
                "expected_returncode, a generated test/validation script, a copied temporary workspace, or a "
                "wired temporary fixture that the program under validation actually consumes. Require error-text "
                "assertions only when the requirements name the text or a wrapper hides the child status."
            )
        return findings

    def _executable_deliverable_validation_findings(self, step: dict[str, Any], commands: list[Any]) -> list[str]:
        findings: list[str] = []
        for path in self._executable_deliverable_paths_for_step(step):
            if self._commands_prove_executable_path(commands, path):
                continue
            if self._executable_requirement_may_be_unrequested_python_scope(path):
                findings.append(
                    f"{step.get('id', 'step')} appears to require `{path}` to be directly executable, but the "
                    "original request does not clearly require `./` invocation or a shebang. If direct "
                    "executability is not intended, revise the acceptance criteria to validate interpreter "
                    f"execution such as `python {path}` instead of adding scope. If direct executability is "
                    f"intended, add bounded evidence such as `test -x ./{path}` or direct `./{path}` invocation."
                )
                continue
            findings.append(
                f"{step.get('id', 'step')} requires `{path}` to be executable, but validation_commands do not prove "
                f"that with `test -x ./{path}` or a direct `./{path}` invocation. Add an actual "
                f"validation_commands entry or shell segment such as `test -x ./{path}`; saying that a "
                "validator script will check executability is not command evidence unless the listed command itself "
                "shows that probe. Keep executability in the file content via a shebang; do not use chmod/chown as "
                "validation."
            )
        return findings

    def _executable_deliverable_evidence_findings(
        self,
        step: dict[str, Any],
        results: list[dict[str, Any]],
        feedback_tool_evidence: dict[str, Any],
        source: str,
    ) -> list[str]:
        findings: list[str] = []
        workspace_files = feedback_tool_evidence.get("workspace_files", []) if isinstance(feedback_tool_evidence, dict) else []
        for path in self._executable_deliverable_paths_for_step(step):
            if not self._workspace_snapshot_has_shebang(workspace_files, path):
                findings.append(
                    f"{source}: `{path}` is required to be executable, but reviewer-owned file evidence does not show "
                    "a shebang at the start of the file. Put executability in the generated file content rather than "
                    "repairing it with chmod."
                )
                continue
            if self._passing_results_prove_executable_path(results, path):
                continue
            findings.append(
                f"{source}: `{path}` is required to be executable, but passing command evidence does not include "
                f"`test -x ./{path}` or direct `./{path}` execution. Add an actual command entry or shell segment "
                "for that probe; prose saying a validator checks executability is not independent command evidence."
            )
        return findings

    def _executable_deliverable_paths_for_step(self, step: dict[str, Any]) -> list[str]:
        fields = [
            str(step.get("title", "")),
            str(step.get("description", "")),
            *[str(item) for item in step.get("acceptance_criteria", []) or []],
        ]
        executable_fields = [field for field in fields if self._text_requires_executable_evidence(field)]
        if not executable_fields:
            return []
        paths: set[str] = set()
        for field in executable_fields:
            paths.update(self._script_paths_in_text(field))
        if not paths:
            all_paths = self._script_paths_in_text(" ".join(fields))
            paths.update(path for path in all_paths if not self._path_looks_like_test_helper(path))
            if not paths:
                paths.update(all_paths)
        return sorted(paths)

    @staticmethod
    def _text_requires_executable_evidence(text: str) -> bool:
        lower = text.lower()
        if "executable" in lower:
            return True
        return bool(
            re.search(r"\b(?:direct invocation|directly run|run directly|runs directly)\b", lower)
            or re.search(r"\b(?:run|runs|execute|executes|invoke|invokes)\s+(?:as\s+)?\./", lower)
        )

    def _executable_requirement_may_be_unrequested_python_scope(self, path: str) -> bool:
        if Path(path).suffix.lower() != ".py":
            return False
        prompt = self.config.project_design.prompt.lower()
        basename = Path(path).name.lower()
        direct_markers = (
            "executable",
            "shebang",
            "chmod",
            "direct invocation",
            "directly executable",
            "run directly",
            "runs directly",
            f"./{basename}",
            f"./{path.lower()}",
        )
        return not any(marker in prompt for marker in direct_markers)

    @staticmethod
    def _script_paths_in_text(text: str) -> set[str]:
        suffixes = "|".join(re.escape(suffix.lstrip(".")) for suffix in EXECUTABLE_DELIVERABLE_SUFFIXES)
        pattern = rf"(?<![\w/.-])(?:\./)?[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:{suffixes})(?![\w.-])"
        paths: set[str] = set()
        for match in re.finditer(pattern, text):
            path = _normalize_workspace_path_text(_trim_reference_delimiters(match.group(0)))
            if not path or path.startswith(("/", "$")) or ".." in Path(path).parts:
                continue
            paths.add(path)
        return paths

    @staticmethod
    def _path_looks_like_test_helper(path: str) -> bool:
        basename = Path(path).name.lower()
        return basename.startswith(("test_", "tests_")) or basename.endswith(("_test.py", ".test.js", ".spec.js"))

    def _commands_prove_executable_path(self, commands: list[Any], path: str) -> bool:
        return any(self._command_proves_executable_path(command, path) for command in commands)

    def _passing_results_prove_executable_path(self, results: list[dict[str, Any]], path: str) -> bool:
        for result in results:
            if result.get("timed_out") or not self._command_returncode_matches_expected(result):
                continue
            if self._command_proves_executable_path(result.get("command") or [], path):
                return True
        return False

    def _command_proves_executable_path(self, command: Any, path: str) -> bool:
        argv = self._command_argv_for_static_check(command)
        if self._argv_proves_executable_path(argv, path):
            return True
        for shell_text in self._shell_texts_for_static_check(argv):
            if self._shell_text_proves_executable_path(shell_text, path):
                return True
        return False

    @classmethod
    def _shell_text_proves_executable_path(cls, shell_text: str, path: str) -> bool:
        for segment in cls._shell_command_segments(shell_text):
            if cls._argv_proves_executable_path(cls._safe_shell_split(segment), path):
                return True
        return False

    @classmethod
    def _argv_proves_executable_path(cls, argv: list[str], path: str) -> bool:
        if len(argv) >= 3 and Path(argv[0]).name == "test" and argv[1] == "-x":
            return cls._workspace_path_token_matches(argv[2], path)
        if len(argv) >= 4 and Path(argv[0]).name == "[" and argv[1] == "-x":
            return cls._workspace_path_token_matches(argv[2], path)
        subject_index = cls._command_subject_index_after_leading_env(argv)
        if subject_index is None:
            return False
        return cls._token_is_direct_workspace_invocation(argv[subject_index], path)

    @staticmethod
    def _workspace_path_token_matches(token: object, path: str) -> bool:
        return _normalize_workspace_path_text(_trim_reference_delimiters(token)) == path

    @classmethod
    def _token_is_direct_workspace_invocation(cls, token: object, path: str) -> bool:
        raw = _trim_reference_delimiters(token)
        normalized = _normalize_workspace_path_text(raw)
        if normalized != path:
            return False
        return raw.startswith("./") or "/" in raw

    @staticmethod
    def _workspace_snapshot_has_shebang(workspace_files: list[dict[str, Any]], path: str) -> bool:
        for item in workspace_files:
            if _normalize_workspace_path_text(item.get("path", "")) != path:
                continue
            return str(item.get("content") or "").startswith("#!")
        return False

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

    def _workspace_source_mutation_target(self, raw_parts: list[str]) -> str | None:
        """Return a workspace source target if a command appears to mutate it.

        The implementation model should edit project files through the JSON
        `files` payload. Terminal commands are for validation, evidence, and
        bounded setup. Temporarily corrupting source files to prove a negative
        path is especially risky because a quoting error can turn the check into
        a false pass and leave the reviewer with misleading evidence.
        """
        if not raw_parts:
            return None
        python_target = self._python_source_write_target(" ".join(raw_parts))
        if python_target:
            return python_target
        direct_target = self._direct_source_mutation_target(raw_parts)
        if direct_target:
            return direct_target
        for shell_text in self._shell_texts_for_static_check(raw_parts):
            shell_target = self._shell_source_mutation_target(shell_text)
            if shell_target:
                return shell_target
        return None

    def _looks_like_unwired_temp_fixture_validation(
        self,
        step: dict[str, Any],
        command: list[Any] | dict[str, Any],
        raw_parts: list[str],
    ) -> bool:
        """Detect negative-path fixtures that are created but probably unused.

        Temporary fixtures are a good non-destructive way to validate failure
        paths, but only if the command under test actually consumes them. A
        common local-model mistake is to write `/tmp/mock.py`, set PYTHONPATH,
        and then run a workspace script that never imports that module.
        """
        texts = [" ".join(raw_parts), json.dumps(command, sort_keys=True)]
        texts.extend(self._shell_texts_for_static_check(raw_parts))
        combined = "\n".join(texts).lower()
        if "pythonpath" not in combined:
            return False
        temp_paths = self._temp_fixture_paths_in_text(combined)
        if not temp_paths:
            return False
        if not self._text_creates_temp_fixture(combined):
            return False
        step_text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", []) or []),
        ]).lower()
        negative_intent = (
            self._text_requires_negative_path(step_text)
            or self._command_expected_returncode(command) != 0
            or any(marker in combined for marker in ("returncode != 0", "returncode!=0", "-ne 0", "expected_returncode"))
        )
        if not negative_intent:
            return False
        return not self._temp_fixture_is_wired_into_command(combined, temp_paths)

    @staticmethod
    def _temp_fixture_paths_in_text(text: str) -> list[str]:
        paths = []
        for match in re.finditer(r"/(?:tmp|var/tmp|dev/shm)/[A-Za-z0-9_./+-]+", text):
            path = match.group(0).rstrip(".,;:)'\"")
            if path not in paths:
                paths.append(path)
        return paths

    @staticmethod
    def _text_creates_temp_fixture(text: str) -> bool:
        return bool(
            re.search(r"\bopen\s*\(\s*['\"]/(?:tmp|var/tmp|dev/shm)/", text)
            or re.search(r"\.write_(?:text|bytes)\s*\([^)]*/(?:tmp|var/tmp|dev/shm)/", text)
            or re.search(r"(?:^|[;&]\s*)(?:echo|printf|cat|tee|cp)\b[^;&|]*>?\s*/(?:tmp|var/tmp|dev/shm)/", text)
            or re.search(r"(?:^|[;&]\s*)mkdir\s+-p\s+/(?:tmp|var/tmp|dev/shm)/", text)
        )

    @staticmethod
    def _temp_fixture_is_wired_into_command(text: str, temp_paths: list[str]) -> bool:
        if re.search(r"\b(?:cd|pushd)\s+/(?:tmp|var/tmp|dev/shm)/", text):
            return True
        if re.search(r"\b(?:cwd|chdir)\s*[=(]\s*['\"]/(?:tmp|var/tmp|dev/shm)/", text):
            return True
        for path in temp_paths:
            escaped = re.escape(path)
            stem = re.sub(r"\W+", "_", Path(path).stem).strip("_")
            if re.search(rf"\b(?:python(?:\d+(?:\.\d+)?)?|node|ruby|php|bash|sh)\s+['\"]?{escaped}\b", text):
                return True
            if re.search(rf"--[a-z0-9_-]+(?:=|\s+)['\"]?{escaped}\b", text):
                return True
            if stem and re.search(rf"\b(?:import\s+{re.escape(stem)}|from\s+{re.escape(stem)}\s+import)\b", text):
                return True
        return False

    @staticmethod
    def _source_mutation_suffixes() -> set[str]:
        return {
            ".cfg",
            ".css",
            ".html",
            ".ini",
            ".js",
            ".json",
            ".jsx",
            ".md",
            ".py",
            ".sh",
            ".toml",
            ".ts",
            ".tsx",
            ".xml",
            ".yaml",
            ".yml",
        }

    @classmethod
    def _workspace_source_target(cls, target: object) -> str | None:
        normalized = _normalize_workspace_path_text(_trim_reference_delimiters(target))
        if not normalized or normalized in {"-", "/dev/null"}:
            return None
        if normalized.startswith(("/tmp/", "/var/tmp/", "/dev/shm/")):
            return None
        temp_variable_match = re.match(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?/", normalized)
        if temp_variable_match and re.search(r"(?:^|_)(?:tmp|temp)(?:_|$)", temp_variable_match.group(1), flags=re.IGNORECASE):
            return None
        if normalized.startswith(("http://", "https://")):
            return None
        if Path(normalized).suffix.lower() not in cls._source_mutation_suffixes():
            return None
        return normalized

    def _python_source_write_target(self, text: str) -> str | None:
        patterns = (
            r"open\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][^'\"]*[wax+]",
            r"(?:pathlib\.)?path\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\.\s*(?:write_text|write_bytes)\s*\(",
            r"['\"]([^'\"]+)['\"]\s*\)\s*\.\s*(?:write_text|write_bytes)\s*\(",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                target = self._workspace_source_target(match.group(1))
                if target:
                    return target
        return None

    def _direct_source_mutation_target(self, raw_parts: list[str]) -> str | None:
        executable = Path(raw_parts[0]).name if raw_parts else ""
        if executable in {"cp", "mv"}:
            operands = [part for part in raw_parts[1:] if not part.startswith("-")]
            if operands:
                return self._workspace_source_target(operands[-1])
        if executable in {"rm", "touch", "truncate"}:
            for operand in raw_parts[1:]:
                if operand.startswith("-"):
                    continue
                target = self._workspace_source_target(operand)
                if target:
                    return target
        if executable in {"chmod", "chown"}:
            operands = [part for part in raw_parts[1:] if not part.startswith("-")]
            for operand in operands[1:]:
                target = self._workspace_source_target(operand)
                if target:
                    return target
        if executable in {"sed", "perl"} and any(part.startswith(("-i", "-pi")) for part in raw_parts[1:]):
            for operand in raw_parts[1:]:
                if operand.startswith("-"):
                    continue
                target = self._workspace_source_target(operand)
                if target:
                    return target
        if executable == "tee":
            for operand in raw_parts[1:]:
                if operand.startswith("-"):
                    continue
                target = self._workspace_source_target(operand)
                if target:
                    return target
        return None

    @staticmethod
    def _shell_texts_for_static_check(raw_parts: list[str]) -> list[str]:
        if len(raw_parts) >= 3 and Path(raw_parts[0]).name in {"bash", "sh"} and raw_parts[1] in {"-c", "-lc"}:
            return [raw_parts[2]]
        return []

    def _shell_source_mutation_target(self, shell_text: str) -> str | None:
        python_target = self._python_source_write_target(shell_text)
        if python_target:
            return python_target
        for match in re.finditer(r"(?:^|\s)(?:\d*)>{1,2}\s*([^;&|]+)", shell_text):
            raw_target = re.sub(r"^\d+", "", match.group(1)).strip()
            if self._shell_relative_write_is_in_temp_cwd(shell_text, match.start(), raw_target):
                continue
            target = self._workspace_source_target(raw_target)
            if target:
                return target
        for match in re.finditer(r"(?:^|[;&|]\s*)tee(?:\s+-a)?\s+([^;&|]+)", shell_text):
            for operand in self._safe_shell_split(match.group(1)):
                if self._shell_relative_write_is_in_temp_cwd(shell_text, match.start(), operand):
                    continue
                target = self._workspace_source_target(operand)
                if target:
                    return target
        for command in ("cp", "mv"):
            for match in re.finditer(rf"(?:^|[;&]\s*){command}\b([^;&|]*)", shell_text):
                operands = [part for part in self._safe_shell_split(match.group(1)) if not part.startswith("-")]
                if operands:
                    if self._shell_relative_write_is_in_temp_cwd(shell_text, match.start(), operands[-1]):
                        continue
                    target = self._workspace_source_target(operands[-1])
                    if target:
                        return target
        for command in ("rm", "touch", "truncate"):
            for match in re.finditer(rf"(?:^|[;&]\s*){command}\b([^;&|]*)", shell_text):
                for operand in self._safe_shell_split(match.group(1)):
                    if operand.startswith("-"):
                        continue
                    if self._shell_relative_write_is_in_temp_cwd(shell_text, match.start(), operand):
                        continue
                    target = self._workspace_source_target(operand)
                    if target:
                        return target
        for command in ("chmod", "chown"):
            for match in re.finditer(rf"(?:^|[;&]\s*){command}\b([^;&|]*)", shell_text):
                operands = [part for part in self._safe_shell_split(match.group(1)) if not part.startswith("-")]
                for operand in operands[1:]:
                    if self._shell_relative_write_is_in_temp_cwd(shell_text, match.start(), operand):
                        continue
                    target = self._workspace_source_target(operand)
                    if target:
                        return target
        for match in re.finditer(r"(?:^|[;&]\s*)(?:sed\s+-i|perl\s+-pi)\b([^;&|]*)", shell_text):
            for operand in self._safe_shell_split(match.group(1)):
                if operand.startswith("-"):
                    continue
                if self._shell_relative_write_is_in_temp_cwd(shell_text, match.start(), operand):
                    continue
                target = self._workspace_source_target(operand)
                if target:
                    return target
        return None

    @classmethod
    def _shell_relative_write_is_in_temp_cwd(cls, shell_text: str, position: int, target: object) -> bool:
        normalized_target = _normalize_workspace_path_text(_trim_reference_delimiters(target))
        if not normalized_target or normalized_target.startswith(("/", "$", "<", ">", "&")):
            return False
        last_cd = ""
        prefix = shell_text[:position]
        temp_dir_vars = set(
            match.group(1)
            for match in re.finditer(
                r"\b([A-Za-z_][A-Za-z0-9_]*)=\$\(\s*mktemp\s+-d\b",
                prefix,
            )
        )
        for match in re.finditer(r"(?:^|[;&(]\s*)cd\s+([^;&|)]+)", prefix):
            parts = cls._safe_shell_split(match.group(1))
            if parts:
                last_cd = _normalize_workspace_path_text(_trim_reference_delimiters(parts[0]))
        if last_cd.startswith(("tmp/", "/tmp/", "var/tmp/", "/var/tmp/", "dev/shm/", "/dev/shm/")):
            return True
        variable_match = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", last_cd)
        return bool(variable_match and variable_match.group(1) in temp_dir_vars)

    @staticmethod
    def _safe_shell_split(text: str) -> list[str]:
        try:
            return shlex.split(text)
        except ValueError:
            return text.split()

    @staticmethod
    def _shell_static_syntax_error(raw_parts: list[str]) -> str | None:
        """Return a shell parser error for `bash/sh -c` scripts without executing them."""
        if len(raw_parts) < 3:
            return None
        executable = Path(str(raw_parts[0])).name
        if executable not in {"bash", "sh"} or raw_parts[1] not in {"-c", "-lc"}:
            return None
        script = str(raw_parts[2])
        shell = "bash" if executable == "bash" else "sh"
        try:
            result = subprocess.run(
                [shell, "-n"],
                input=script,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode == 0:
            return None
        stderr = " ".join(line.strip() for line in result.stderr.splitlines() if line.strip())
        return clamp_text(stderr or f"{shell} -n returned {result.returncode}", 300)

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

    def _command_requested_timeout(self, command: list[Any] | dict[str, Any]) -> int | None:
        if not isinstance(command, dict) or "timeout_seconds" not in command:
            return None
        try:
            return int(command.get("timeout_seconds"))
        except (TypeError, ValueError):
            return None

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

    def _looks_like_metadata_inside_argv(self, parts: list[str]) -> str | None:
        """Detect command-object metadata accidentally placed in argv lists."""
        if not parts:
            return None
        metadata_keys = {"expected_returncode", "timeout_seconds"}
        executable = Path(parts[0]).name.lower()
        for pos, part in enumerate(parts):
            normalized = str(part).strip().strip("'\"{}, ").lower()
            normalized = normalized.rstrip(":")
            if normalized not in metadata_keys:
                continue
            next_value_is_metadata = pos + 1 < len(parts) and re.fullmatch(r"\d+", str(parts[pos + 1]).strip()) is not None
            command_shape_is_likely_misplaced_metadata = executable in {"bash", "sh"} and pos >= 3
            python_shape_is_likely_misplaced_metadata = self._is_python_executable_name(executable) and pos >= 2
            if next_value_is_metadata or command_shape_is_likely_misplaced_metadata or python_shape_is_likely_misplaced_metadata:
                return normalized
        return None

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

    def _looks_like_silent_semantic_validation_command(self, step: dict[str, Any], raw_parts: list[str]) -> bool:
        """Detect computed-answer validators that intentionally hide mismatch facts."""
        if not self._computed_answer_prompt_requires_semantic_validation():
            return False
        text = " ".join(raw_parts)
        lower = text.lower()
        if not any(marker in lower for marker in ("answer.txt", "expected", "actual", "target", "total", "count")):
            return False
        diagnostic_markers = (
            "print(",
            "sys.stderr.write",
            "sys.stdout.write",
            ".stderr.write",
            ".stdout.write",
            "echo ",
            "printf ",
        )
        has_diagnostics = any(marker in lower for marker in diagnostic_markers)
        if re.search(r"\bassert\b", lower):
            assert_has_message = re.search(r"\bassert\b[^;\n]*,\s*[^;\n]+", lower) is not None
            if not assert_has_message and not has_diagnostics:
                return True
        if not (
            "exit(0 if" in lower
            or "sys.exit(0 if" in lower
            or re.search(r"\bexit\s*\(\s*0\s+if\b", lower)
            or re.search(r"\bsys\.exit\s*\(\s*0\s+if\b", lower)
        ):
            return False
        if not re.search(r"\belse\s+1\b", lower):
            return False
        return not has_diagnostics

    def _looks_like_precedence_prone_arithmetic_validation(self, raw_parts: list[str]) -> bool:
        """Detect compact arithmetic validators that are easy to misread.

        Expressions such as `n // 10 * n % 10` are legal Python, but they do not
        mean `(n // 10) * (n % 10)`. Local models commonly write that shape when
        validating digit-product tasks. The harness should request a clearer
        validator instead of letting a bad reviewer-owned command steer repairs.
        """
        for _source, code in self._iter_inline_python_snippets(raw_parts):
            compact = re.sub(r"\s+", " ", code.lower())
            if re.search(r"\b([a-z_]\w*)\s*//\s*[^;,\]\)\n]+?\s*\*\s*\1\s*%", compact):
                return True
            if re.search(r"\b([a-z_]\w*)\s*%\s*[^;,\]\)\n]+?\s*\*\s*\1\s*//", compact):
                return True
        return False

    def _raw_text_numeric_comparison_finding(self, raw_parts: list[str]) -> str:
        """Detect validators that compare numeric computations to unparsed file text."""
        for source, code in self._iter_inline_python_snippets(raw_parts):
            if not code.strip() or code.strip().startswith("$"):
                continue
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            raw_text_vars = self._raw_file_text_variables(tree)
            if not raw_text_vars:
                continue
            numeric_vars = self._numeric_expression_variables(tree, raw_text_vars)
            for compare in (node for node in ast.walk(tree) if isinstance(node, ast.Compare)):
                if not any(isinstance(operator, (ast.Eq, ast.NotEq)) for operator in compare.ops):
                    continue
                operands = [compare.left, *compare.comparators]
                for left, right in zip(operands, operands[1:]):
                    if self._comparison_mixes_raw_text_and_numeric(left, right, raw_text_vars, numeric_vars):
                        return (
                            f"compares raw file text to a numeric expression in {source}. Convert the file value "
                            "with `int(...)`/`float(...)`, or compare against `str(expected)`, so validation "
                            "can pass only when the representations actually match."
                        )
        return ""

    def _raw_file_text_variables(self, tree: ast.AST) -> set[str]:
        raw_text_vars: set[str] = set()
        for node in ast.walk(tree):
            for target, value in self._assignment_pairs(node):
                if self._expr_is_raw_file_text(value):
                    raw_text_vars.update(self._assigned_names(target))
        return raw_text_vars

    def _numeric_expression_variables(self, tree: ast.AST, raw_text_vars: set[str]) -> set[str]:
        numeric_vars: set[str] = set()
        for node in ast.walk(tree):
            for target, value in self._assignment_pairs(node):
                names = self._assigned_names(target)
                if not names or any(name in raw_text_vars for name in names):
                    continue
                if self._expr_is_numeric_value(value, numeric_vars):
                    numeric_vars.update(names)
            if isinstance(node, ast.AugAssign):
                names = self._assigned_names(node.target)
                if names and self._expr_is_numeric_value(node.value, numeric_vars):
                    numeric_vars.update(name for name in names if name not in raw_text_vars)
        return numeric_vars

    @staticmethod
    def _assignment_pairs(node: ast.AST) -> list[tuple[ast.AST, ast.AST]]:
        if isinstance(node, ast.Assign):
            return [(target, node.value) for target in node.targets]
        if isinstance(node, ast.AnnAssign):
            return [(node.target, node.value)] if node.value is not None else []
        if isinstance(node, ast.NamedExpr):
            return [(node.target, node.value)]
        return []

    @staticmethod
    def _assigned_names(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            names: set[str] = set()
            for element in target.elts:
                names.update(FeedbackAgent._assigned_names(element))
            return names
        return set()

    def _comparison_mixes_raw_text_and_numeric(
        self,
        left: ast.AST,
        right: ast.AST,
        raw_text_vars: set[str],
        numeric_vars: set[str],
    ) -> bool:
        left_raw = self._expr_is_raw_file_text_reference(left, raw_text_vars)
        right_raw = self._expr_is_raw_file_text_reference(right, raw_text_vars)
        if left_raw == right_raw:
            return False
        raw_expr = left if left_raw else right
        numeric_expr = right if left_raw else left
        if self._expr_is_explicit_numeric_conversion(raw_expr):
            return False
        if self._expr_is_string_conversion(numeric_expr):
            return False
        return self._expr_is_numeric_value(numeric_expr, numeric_vars)

    def _expr_is_raw_file_text_reference(self, expr: ast.AST, raw_text_vars: set[str]) -> bool:
        return (
            (isinstance(expr, ast.Name) and expr.id in raw_text_vars)
            or self._expr_is_raw_file_text(expr)
        )

    def _expr_is_raw_file_text(self, expr: ast.AST) -> bool:
        return self._expr_is_file_read_call(self._strip_string_methods(expr))

    def _strip_string_methods(self, expr: ast.AST) -> ast.AST:
        current = expr
        while (
            isinstance(current, ast.Call)
            and isinstance(current.func, ast.Attribute)
            and current.func.attr in {"strip", "rstrip", "lstrip"}
            and not current.args
        ):
            current = current.func.value
        return current

    @staticmethod
    def _expr_is_file_read_call(expr: ast.AST) -> bool:
        if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Attribute):
            return False
        if expr.func.attr == "read_text":
            return True
        if expr.func.attr != "read":
            return False
        owner = expr.func.value
        if isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id == "open":
            return True
        return isinstance(owner, ast.Name) and owner.id in {"file", "f", "fh", "handle"}

    @staticmethod
    def _expr_is_explicit_numeric_conversion(expr: ast.AST) -> bool:
        return (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id in {"int", "float"}
        )

    @staticmethod
    def _expr_is_string_conversion(expr: ast.AST) -> bool:
        return isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "str"

    def _expr_is_numeric_value(self, expr: ast.AST, numeric_vars: set[str]) -> bool:
        if self._expr_is_string_conversion(expr):
            return False
        if isinstance(expr, ast.Name):
            return expr.id in numeric_vars
        if isinstance(expr, ast.Constant):
            return isinstance(expr.value, (int, float)) and not isinstance(expr.value, bool)
        if isinstance(expr, (ast.BinOp, ast.UnaryOp)):
            return True
        if isinstance(expr, ast.Compare):
            return False
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name) and expr.func.id in {
                "sum",
                "len",
                "int",
                "float",
                "abs",
                "round",
                "min",
                "max",
            }:
                return True
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "count":
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

    def _shell_heredoc_static_error(self, parts: list[str]) -> str | None:
        if len(parts) < 3 or Path(parts[0]).name not in {"bash", "sh"} or parts[1] not in {"-c", "-lc"}:
            return None
        script = parts[2]
        for match in re.finditer(r"<<(?P<dash>-?)\s*(?P<quote>['\"]?)(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)\2", script):
            delimiter = match.group("delimiter")
            line_start = script.rfind("\n", 0, match.start()) + 1
            line_end = script.find("\n", match.end())
            if line_end == -1:
                return f"here-doc opener for `{delimiter}` has no following body or closing delimiter"
            body_lines = script[line_end + 1:].splitlines()
            allows_tabs = bool(match.group("dash"))
            for line in body_lines:
                candidate = line.lstrip("\t") if allows_tabs else line
                if candidate == delimiter:
                    break
            else:
                for line in body_lines:
                    stripped = line.strip()
                    if stripped in {f"'{delimiter}'", f'"{delimiter}"'} or stripped.strip("'\"") == delimiter:
                        return (
                            f"here-doc opener on line `{script[line_start:line_end].strip()}` "
                            f"is closed with quoted delimiter `{stripped}` instead of bare `{delimiter}`"
                        )
                return f"here-doc opener on line `{script[line_start:line_end].strip()}` is missing closing delimiter `{delimiter}`"
        return None

    def _shell_command_uses_heredoc(self, parts: list[str]) -> bool:
        if len(parts) < 3 or Path(parts[0]).name not in {"bash", "sh"} or parts[1] not in {"-c", "-lc"}:
            return False
        return bool(re.search(r"<<-?\s*['\"]?[A-Za-z_][A-Za-z0-9_]*['\"]?", parts[2]))

    def _artifact_only_heredoc_finding(self, parts: list[str]) -> str:
        if not self._explicit_artifact_only_constraint() or not self._shell_command_uses_heredoc(parts):
            return ""
        return (
            "uses a shell here-doc in an artifact-only task. Artifact-only validation must stay JSON-safe "
            "and compact; use a one-line `python -c` validator, a simple argv command, or temporary files "
            "outside the workspace instead of multiline plan/command validators."
        )

    def _inline_python_unreachable_after_return(self, parts: list[str]) -> str | None:
        """Return a finding when inline Python hides proof work after return.

        Python accepts `def f(): return value; open(...).write(...)`, but every
        statement after the return belongs to the function body and is
        unreachable. Small local models often use this shape when compressing a
        validator into one line, producing a successful no-op command.
        """
        for source, code in self._iter_inline_python_snippets(parts):
            if not code.strip() or code.strip().startswith("$"):
                continue
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for pos, statement in enumerate(node.body[:-1]):
                    if not isinstance(statement, ast.Return):
                        continue
                    later = node.body[pos + 1 :]
                    if any(not isinstance(later_statement, ast.Pass) for later_statement in later):
                        return (
                            f"{source}: function `{node.name}` has statements after `return`; "
                            "they will not execute."
                        )
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

    def _looks_like_swallowed_expected_failure_validation(
        self,
        step: dict[str, Any],
        command: list[Any] | dict[str, Any],
        raw_parts: list[str],
    ) -> bool:
        if self._command_expected_returncode(command) != 0:
            return False
        shell_texts = self._shell_texts_for_static_check(raw_parts)
        if not shell_texts:
            return False
        step_text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", [])),
        ]).lower()
        expected_failure_markers = (
            "expected failure",
            "negative path",
            "non-zero",
            "nonzero",
            "must fail",
            "should fail",
            "exits non-zero",
            "exit non-zero",
            "invalid",
            "error",
            "exception",
            "mismatch",
            "incorrect",
        )
        combined = step_text + "\n" + "\n".join(shell_texts).lower()
        if not any(marker in combined for marker in expected_failure_markers):
            return False
        return any(self._shell_swallows_failure_as_success(shell_text) for shell_text in shell_texts)

    def _looks_like_validation_failure_masking_shell_fallback(
        self,
        command: list[Any] | dict[str, Any],
        raw_parts: list[str],
    ) -> bool:
        if self._command_expected_returncode(command) != 0:
            return False
        return any(
            self._shell_swallows_failure_as_success(shell_text)
            for shell_text in self._shell_texts_for_static_check(raw_parts)
        )

    @classmethod
    def _shell_swallows_failure_as_success(cls, shell_text: str) -> bool:
        fallback = r"(?:exit\s+0|true|:|echo\b[^;&|]*)"
        for match in re.finditer(rf"\|\|\s*{fallback}(?=$|[\s);&|])", shell_text, flags=re.IGNORECASE):
            prefix = shell_text[:match.start()]
            probe = re.split(r"[;\n]", prefix)[-1].strip(" \t(").lower()
            if not probe:
                continue
            suffix = shell_text[match.end(): match.end() + 80].lower()
            if re.search(r"(?:^|[;&|]\s*)exit\s+1\s*$", probe):
                continue
            if re.match(r"\s*;\s*exit\s+1\b", suffix):
                continue
            if cls._looks_like_validation_probe_text(probe):
                return True
            if probe.startswith(("rm ", "rmdir ", "mkdir ", "touch ", "cleanup ")):
                continue
        return False

    def _looks_like_expected_failure_status_masked_by_shell_tail(
        self,
        command: list[Any] | dict[str, Any],
        raw_parts: list[str],
    ) -> bool:
        """Detect expected-failure shell commands whose cleanup hides the failure.

        A command object such as
        `{"cmd": ["bash", "-lc", "python validator.py; rm /tmp/x"],
        "expected_returncode": 1}` looks like it validates a negative path, but
        the shell returns the status of `rm`, not the validator.  The command
        must either preserve the status explicitly or use a trap for cleanup.
        """
        if self._command_expected_returncode(command) == 0:
            return False
        for shell_text in self._shell_texts_for_static_check(raw_parts):
            lower = shell_text.lower()
            if re.search(r"\btrap\b", lower) or re.search(r"\bset\s+-e\b", lower):
                continue
            if self._shell_text_preserves_last_status_after_cleanup(lower):
                continue
            if self._shell_text_has_probe_then_trailing_status_mask(lower):
                return True
        return False

    def _looks_like_validation_status_masked_by_shell_tail(
        self,
        command: list[Any] | dict[str, Any],
        raw_parts: list[str],
    ) -> bool:
        """Detect ordinary validators whose cleanup can hide assertion failure."""
        if self._command_expected_returncode(command) != 0:
            return False
        for shell_text in self._shell_texts_for_static_check(raw_parts):
            lower = shell_text.lower()
            if re.search(r"\btrap\b", lower) or re.search(r"\bset\s+-e\b", lower):
                continue
            if self._shell_text_preserves_last_status_after_cleanup(lower):
                continue
            if self._shell_text_has_assertion_then_trailing_status_mask(lower):
                return True
        return False

    def _misplaced_environment_assignments_after_program(self, raw_parts: list[str]) -> list[str]:
        env_names = self._declared_environment_override_names()
        if not env_names:
            return []
        command_argvs: list[list[str]] = []
        shell_texts = self._shell_texts_for_static_check(raw_parts)
        if shell_texts:
            for shell_text in shell_texts:
                for segment in self._shell_command_segments_with_pipes(shell_text):
                    argv = self._safe_shell_split(segment)
                    if argv:
                        command_argvs.append(argv)
        elif raw_parts:
            command_argvs.append(raw_parts)

        misplaced: set[str] = set()
        for argv in command_argvs:
            subject_index = self._command_subject_index_after_leading_env(argv)
            if subject_index is None or not self._looks_like_env_override_target(argv, subject_index):
                continue
            for token in argv[subject_index + 1:]:
                name = self._env_assignment_token_name(token)
                if name in env_names:
                    misplaced.add(name)
        return sorted(misplaced)

    def _validation_workspace_output_targets(self, raw_parts: list[str]) -> list[str]:
        targets: set[str] = set()
        for shell_text in self._shell_texts_for_static_check(raw_parts):
            targets.update(self._shell_workspace_output_targets(shell_text))
        return sorted(targets)

    def _looks_like_placeholder_validation_command(self, raw_parts: list[str]) -> bool:
        text = self._command_text_for_stateful_validation(raw_parts).lower()
        if not text:
            return False
        placeholder_markers = (
            "placeholder",
            "stub",
            "todo",
            "not implemented",
            "fake test",
            "dummy test",
        )
        pass_markers = (
            "passed",
            "success",
            "exit 0",
            "true",
            "print(",
            "echo ",
        )
        return any(marker in text for marker in placeholder_markers) and any(
            marker in text for marker in pass_markers
        )

    def _stateful_validation_runtime_state_finding(self, step: dict[str, Any], raw_parts: list[str]) -> str:
        if not self._step_mentions_runtime_state(step):
            return ""
        command_text = self._command_text_for_stateful_validation(raw_parts)
        if not command_text:
            return ""
        if self._command_text_runs_project_entrypoint_help_only(command_text):
            return ""
        if not self._command_text_runs_project_entrypoint(command_text):
            return ""
        if self._command_text_isolates_runtime_state(command_text):
            return ""
        return (
            "checks a stateful or resumable workflow but does not show isolated runtime state. "
            "Prefer running the validation from a trap-cleaned temporary working directory. "
            "Use an explicit temporary state/cache/checkpoint path only if the user-requested "
            "interface or existing project already exposes that path; do not add a public "
            "state-file/cache/checkpoint option solely for validation. Otherwise remove runtime "
            "state with status-safe cleanup so stale workspace state cannot affect later attempts."
        )

    def _stateful_validation_help_only_finding(self, step: dict[str, Any], raw_parts: list[str]) -> str:
        if not self._step_mentions_runtime_state(step):
            return ""
        command_text = self._command_text_for_stateful_validation(raw_parts)
        if not self._command_text_runs_project_entrypoint_help_only(command_text):
            return ""
        return (
            "only checks help or metadata for a stateful workflow. Add bounded behavioral validation "
            "that exercises the remembered position/checkpoint behavior with temporary input and isolated "
            "runtime state, or move the stateful acceptance criteria to a step whose validation script proves it."
        )

    def _step_mentions_runtime_state(self, step: dict[str, Any]) -> bool:
        payload = {
            "title": step.get("title", ""),
            "description": step.get("description", ""),
            "acceptance_criteria": step.get("acceptance_criteria", []),
        }
        text = json.dumps(payload, ensure_ascii=False).lower()
        markers = (
            ".watch_state",
            ".state",
            ".checkpoint",
            "state file",
            "runtime state",
            "checkpoint",
            "cursor",
            "last checked",
            "last processed",
            "remember the last",
            "remember last",
            "resumable",
            "resume",
            "offset",
            "pid file",
            "lock file",
            "cache file",
        )
        return any(marker in text for marker in markers)

    def _command_text_for_stateful_validation(self, raw_parts: list[str]) -> str:
        shell_texts = self._shell_texts_for_static_check(raw_parts)
        if shell_texts:
            return "\n".join(shell_texts)
        return " ".join(raw_parts)

    @staticmethod
    def _command_text_runs_project_entrypoint(command_text: str) -> bool:
        for segment in FeedbackLoopAgent._shell_command_segments_with_pipes(command_text):
            if FeedbackLoopAgent._argv_runs_project_entrypoint(
                FeedbackLoopAgent._safe_shell_split(segment)
            ):
                return True
        return False

    @classmethod
    def _argv_runs_project_entrypoint(cls, argv: list[str]) -> bool:
        if not argv or cls._argv_is_test_executable_probe(argv):
            return False
        subject_index = cls._command_subject_index_after_runtime_wrappers(argv)
        if subject_index is None or subject_index >= len(argv):
            return False
        candidate = _trim_reference_delimiters(argv[subject_index]).lower()
        if not candidate or candidate.startswith("$"):
            return False
        name = Path(candidate).name
        if not name.endswith((".sh", ".py", ".js", ".ts")):
            return False
        stem = name.rsplit(".", 1)[0]
        if any(marker in stem for marker in ("validate", "validator", "test", "check")):
            return False
        executable = Path(argv[0]).name.lower()
        if executable in {"echo", "printf", "grep", "rg", "test", "[", "cat", "ls", "stat", "wc"}:
            return False
        return True

    @classmethod
    def _argv_is_test_executable_probe(cls, argv: list[str]) -> bool:
        if len(argv) >= 3 and Path(argv[0]).name == "test" and argv[1] == "-x":
            return True
        if len(argv) >= 4 and Path(argv[0]).name == "[" and argv[1] == "-x":
            return True
        return False

    @classmethod
    def _command_subject_index_after_runtime_wrappers(cls, argv: list[str]) -> int | None:
        subject_index = cls._command_subject_index_after_leading_env(argv)
        if subject_index is None:
            return None
        executable = Path(argv[subject_index]).name.lower()
        if executable != "timeout":
            return subject_index
        index = subject_index + 1
        while index < len(argv) and argv[index].startswith("-"):
            option = argv[index]
            index += 1
            if option in {"-k", "--kill-after", "--foreground", "--preserve-status"}:
                if option in {"-k", "--kill-after"} and index < len(argv):
                    index += 1
                continue
        if index < len(argv) and re.fullmatch(r"\d+(?:\.\d+)?[smhd]?", argv[index]):
            index += 1
        if index >= len(argv):
            return None
        nested = cls._command_subject_index_after_leading_env(argv[index:])
        if nested is None:
            return index
        return index + nested

    @classmethod
    def _timeout_wraps_shell_builtin_wait(cls, raw_parts: list[str]) -> bool:
        if cls._argv_timeout_wraps_wait(raw_parts):
            return True
        for shell_text in cls._shell_texts_for_static_check(raw_parts):
            for segment in cls._shell_command_segments_with_pipes(shell_text):
                if cls._argv_timeout_wraps_wait(cls._safe_shell_split(segment)):
                    return True
        return False

    @classmethod
    def _argv_timeout_wraps_wait(cls, argv: list[str]) -> bool:
        subject_index = cls._command_subject_index_after_leading_env(argv)
        if subject_index is None or subject_index >= len(argv):
            return False
        if Path(argv[subject_index]).name.lower() != "timeout":
            return False
        wrapped_index = cls._command_subject_index_after_runtime_wrappers(argv)
        if wrapped_index is None or wrapped_index >= len(argv):
            return False
        return Path(argv[wrapped_index]).name == "wait"

    @classmethod
    def _command_text_runs_project_entrypoint_help_only(cls, command_text: str) -> bool:
        segments = [
            segment
            for segment in cls._shell_command_segments_with_pipes(command_text.lower())
            if cls._command_text_runs_project_entrypoint(segment)
        ]
        if not segments:
            return False
        help_markers = (" --help", " -h", " help", " --version", " version")
        return all(any(marker in f" {segment} " for marker in help_markers) for segment in segments)

    @classmethod
    def _command_text_isolates_runtime_state(cls, command_text: str) -> bool:
        lower = command_text.lower()
        temp_markers = ("$tmp", "${tmp", "$(mktemp", "/tmp/", "mktemp -d", "mktemp")
        state_markers = (
            "--state",
            "--state-file",
            "--cache",
            "--checkpoint",
            "state_file",
            "state_path",
            "statefile",
            "watch_state",
            "checkpoint",
            "cache_dir",
            "cache_path",
        )
        if any(marker in lower for marker in state_markers) and any(marker in lower for marker in temp_markers):
            return True
        if cls._command_text_changes_to_temp_dir(lower):
            return True
        runtime_paths = [re.escape(path) for path in RUNTIME_STATE_BASENAMES]
        cleanup_pattern = rf"\brm\s+-[^\n;&|]*[fr][^\n;&|]*\s+(?:\./)?(?:{'|'.join(runtime_paths)})\b"
        if re.search(cleanup_pattern, lower):
            return True
        return False

    @classmethod
    def _command_text_changes_to_temp_dir(cls, command_text: str) -> bool:
        temp_dir_vars = set(
            match.group(1)
            for match in re.finditer(
                r"\b([a-z_][a-z0-9_]*)=\$\(\s*mktemp\s+-d\b",
                command_text,
            )
        )
        for match in re.finditer(r"(?:^|[;&(]\s*)(?:cd|pushd)\s+([^;&|)]+)", command_text):
            parts = cls._safe_shell_split(match.group(1))
            if not parts:
                continue
            target = _normalize_workspace_path_text(_trim_reference_delimiters(parts[0])).lower()
            if not target:
                continue
            if target.startswith(("tmp/", "/tmp/", "var/tmp/", "/var/tmp/", "dev/shm/", "/dev/shm/")):
                return True
            if "mktemp" in target and "-d" in target:
                return True
            variable_match = re.fullmatch(r"\$\{?([a-z_][a-z0-9_]*)\}?", target)
            if variable_match and variable_match.group(1) in temp_dir_vars:
                return True
        return False

    @classmethod
    def _shell_workspace_output_targets(cls, shell_text: str) -> set[str]:
        targets: set[str] = set()
        tokens = cls._shell_tokens_with_spans(shell_text)
        control_tokens = {";", "&&", "||", "|", "&", "(", ")"}
        redirection_tokens = {">", ">>", "&>", ">|"}
        for index, (token, start, _end) in enumerate(tokens):
            if token not in redirection_tokens:
                continue
            if index + 1 >= len(tokens):
                continue
            raw_target = tokens[index + 1][0]
            if cls._shell_relative_write_is_in_temp_cwd(shell_text, start, raw_target):
                continue
            target = cls._workspace_output_target_from_shell_operand(raw_target)
            if target:
                targets.add(target)

        for index, (token, start, _end) in enumerate(tokens):
            if token != "tee":
                continue
            operand_index = index + 1
            while operand_index < len(tokens):
                operand, _operand_start, _operand_end = tokens[operand_index]
                if operand in control_tokens or operand in redirection_tokens:
                    break
                if operand.startswith("-"):
                    operand_index += 1
                    continue
                if cls._shell_relative_write_is_in_temp_cwd(shell_text, start, operand):
                    operand_index += 1
                    continue
                target = cls._workspace_output_target_from_shell_operand(operand)
                if target:
                    targets.add(target)
                operand_index += 1
        return targets

    @staticmethod
    def _shell_tokens_with_spans(shell_text: str) -> list[tuple[str, int, int]]:
        """Tokenize enough shell syntax to find unquoted redirects/tee targets."""
        tokens: list[tuple[str, int, int]] = []
        current: list[str] = []
        token_start: int | None = None
        quote: str | None = None
        punctuation = set("<>|&;()")
        index = 0

        def finish(end: int) -> None:
            nonlocal current, token_start
            if current:
                tokens.append(("".join(current), token_start if token_start is not None else end, end))
                current = []
                token_start = None

        while index < len(shell_text):
            char = shell_text[index]
            if quote:
                if char == quote:
                    quote = None
                    index += 1
                    continue
                if char == "\\" and quote == '"' and index + 1 < len(shell_text):
                    if token_start is None:
                        token_start = index
                    current.append(shell_text[index + 1])
                    index += 2
                    continue
                if token_start is None:
                    token_start = index
                current.append(char)
                index += 1
                continue

            if char in {"'", '"'}:
                if token_start is None:
                    token_start = index
                quote = char
                index += 1
                continue
            if char == "\\" and index + 1 < len(shell_text):
                if token_start is None:
                    token_start = index
                current.append(shell_text[index + 1])
                index += 2
                continue
            if char.isspace():
                finish(index)
                index += 1
                continue
            if char in punctuation:
                finish(index)
                punct_start = index
                punct = [char]
                index += 1
                while index < len(shell_text) and shell_text[index] in punctuation:
                    punct.append(shell_text[index])
                    index += 1
                tokens.append(("".join(punct), punct_start, index))
                continue
            if token_start is None:
                token_start = index
            current.append(char)
            index += 1

        finish(len(shell_text))
        return tokens

    @staticmethod
    def _workspace_output_target_from_shell_operand(operand: str) -> str | None:
        parts = FeedbackLoopAgent._safe_shell_split(operand.strip())
        if not parts:
            return None
        target = _normalize_workspace_path_text(_trim_reference_delimiters(parts[0]))
        if not target or target == ".":
            return None
        if target.startswith(("/", "$", "<", ">", "&", "(")):
            return None
        if target in {"-", "/dev/null"} or "://" in target:
            return None
        if any(char in target for char in "*?["):
            return None
        return target

    def _declared_environment_override_names(self) -> set[str]:
        text = "\n".join([
            self.config.project_design.prompt,
            json.dumps(self.requirements, sort_keys=True),
        ])
        return set(re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", text))

    @staticmethod
    def _env_assignment_token_name(token: str) -> str | None:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=", token)
        if not match:
            return None
        return match.group(1)

    @classmethod
    def _command_subject_index_after_leading_env(cls, argv: list[str]) -> int | None:
        index = 0
        while index < len(argv) and cls._env_assignment_token_name(argv[index]):
            index += 1
        if index >= len(argv):
            return None

        executable = Path(argv[index]).name
        if executable == "env":
            index += 1
            while index < len(argv):
                token = argv[index]
                if token.startswith("-"):
                    index += 1
                    continue
                if cls._env_assignment_token_name(token):
                    index += 1
                    continue
                break
            if index >= len(argv):
                return None
            executable = Path(argv[index]).name

        if executable in {"python", "python3", "python2", "pypy", "pypy3"}:
            scan = index + 1
            while scan < len(argv) and argv[scan].startswith("-") and argv[scan] not in {"-c", "-m"}:
                scan += 1
            if scan < len(argv) and argv[scan].endswith(".py"):
                return scan
            return index
        if executable in {"bash", "sh"}:
            if index + 1 < len(argv) and argv[index + 1] in {"-c", "-lc"}:
                return None
            if index + 1 < len(argv) and argv[index + 1].endswith(".sh"):
                return index + 1
            return index
        return index

    @staticmethod
    def _looks_like_env_override_target(argv: list[str], subject_index: int) -> bool:
        if subject_index < 0 or subject_index >= len(argv):
            return False
        subject = argv[subject_index]
        executable = Path(argv[0]).name if argv else ""
        return (
            subject.startswith(("./", "../", "/"))
            or "/" in subject
            or subject.endswith((".sh", ".py"))
            or executable in {"python", "python3", "python2", "pypy", "pypy3", "bash", "sh"}
        )

    @staticmethod
    def _shell_command_segments_with_pipes(shell_text: str) -> list[str]:
        return [
            segment.strip()
            for segment in re.split(r"(?:&&|\|\||[;|\n])", shell_text)
            if segment.strip()
        ]

    def _filtered_absence_check_finding(self, step: dict[str, Any] | None, raw_parts: list[str]) -> str | None:
        if not isinstance(step, dict):
            return None
        for shell_text in self._shell_texts_for_static_check(raw_parts):
            pattern = self._grep_invert_match_pattern_before_later_grep(shell_text)
            if not pattern:
                continue
            if not self._step_mentions_absence_of_pattern(step, pattern):
                continue
            display_pattern = pattern.strip() or "forbidden text"
            return (
                f"tries to prove `{display_pattern}` is absent by filtering it out with `grep -v` before "
                "checking another line. That can pass even when the forbidden output was present. Capture "
                "the full output, assert the forbidden text is absent from that complete output, and separately "
                "assert the required status line or count."
            )
        return None

    def _grep_invert_match_pattern_before_later_grep(self, shell_text: str) -> str | None:
        tokens = self._safe_shell_split(shell_text)
        for index, token in enumerate(tokens):
            if Path(token).name != "grep":
                continue
            pattern_index = self._grep_invert_pattern_index(tokens, index)
            if pattern_index is None:
                continue
            if "|" not in tokens[pattern_index + 1:]:
                continue
            pipe_index = tokens.index("|", pattern_index + 1)
            if any(Path(item).name == "grep" for item in tokens[pipe_index + 1:]):
                return tokens[pattern_index]

        # Fallback for compact shell snippets where `|` is glued to adjacent
        # tokens and shlex does not expose it as a separate token.
        match = re.search(
            r"\bgrep\b(?=[^|]*\s-(?:[A-Za-z]*v[A-Za-z]*|-invert-match)\b)[^|]*\s+"
            r"(?P<pattern>'[^']+'|\"[^\"]+\"|[^|;&\s]+)\s*\|\s*grep\b",
            shell_text,
        )
        if match:
            return match.group("pattern").strip("'\"")
        return None

    @staticmethod
    def _grep_invert_pattern_index(tokens: list[str], grep_index: int) -> int | None:
        invert = False
        index = grep_index + 1
        while index < len(tokens):
            token = tokens[index]
            if token == "--":
                index += 1
                break
            if token == "--invert-match":
                invert = True
                index += 1
                continue
            if token.startswith("--"):
                index += 1
                continue
            if token.startswith("-") and token != "-":
                if "v" in token[1:]:
                    invert = True
                index += 1
                continue
            break
        if invert and index < len(tokens):
            return index
        return None

    def _step_mentions_absence_of_pattern(self, step: dict[str, Any], pattern: str) -> bool:
        pattern_candidates = self._plain_pattern_candidates(pattern)
        if not pattern_candidates:
            return False
        text = json.dumps(
            {
                "prompt": self.config.project_design.prompt,
                "title": step.get("title", ""),
                "description": step.get("description", ""),
                "acceptance_criteria": step.get("acceptance_criteria", []),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).lower()
        absence_markers = (
            "absent",
            "absence",
            "avoid",
            "avoids",
            "without",
            "must not",
            "should not",
            "does not",
            "do not",
            "not contain",
            "not emit",
            "not output",
            "not trigger",
            "not triggered",
            "no ",
            "never",
            "forbidden",
        )
        for candidate in pattern_candidates:
            start = 0
            while True:
                index = text.find(candidate, start)
                if index < 0:
                    break
                window = text[max(0, index - 100): index + len(candidate) + 100]
                if any(marker in window for marker in absence_markers):
                    return True
                start = index + len(candidate)
        return False

    @staticmethod
    def _plain_pattern_candidates(pattern: str) -> list[str]:
        unquoted = pattern.strip().strip("'\"")
        if not unquoted:
            return []
        lowered = unquoted.lower()
        candidates = {lowered}
        simplified = re.sub(r"\\[bBwWdDsS]", "", lowered)
        simplified = simplified.strip("^$.*+?()[]{}|")
        simplified = re.sub(r"[^a-z0-9_ -]+", " ", simplified)
        simplified = re.sub(r"\s+", " ", simplified).strip()
        if simplified:
            candidates.add(simplified)
        return sorted(candidate for candidate in candidates if len(candidate) >= 3)

    def _printf_literal_percent_findings(self, raw_parts: list[str]) -> list[str]:
        findings: list[str] = []
        for shell_text in self._shell_texts_for_static_check(raw_parts):
            for segment in self._shell_command_segments_with_pipes(shell_text):
                argv = self._shell_argv_without_redirections(self._safe_shell_split(segment))
                subject_index = self._command_subject_index_after_leading_env(argv)
                if subject_index is None or subject_index >= len(argv):
                    continue
                if Path(argv[subject_index]).name != "printf":
                    continue
                format_index = subject_index + 1
                if format_index < len(argv) and argv[format_index] == "--":
                    format_index += 1
                if format_index >= len(argv):
                    continue
                if self._printf_format_has_invalid_literal_percent(argv[format_index]):
                    findings.append(
                        "contains a `printf` format string with a likely unescaped literal `%`. "
                        "Escape literal percent signs as `%%`, or use `printf '%s\\n' ...` with "
                        "the literal text supplied as an argument."
                    )
        return findings

    @staticmethod
    def _shell_argv_without_redirections(argv: list[str]) -> list[str]:
        cleaned: list[str] = []
        skip_next = False
        for token in argv:
            if skip_next:
                skip_next = False
                continue
            if re.fullmatch(r"\d*(?:<|>{1,2}|>&|<&)", token):
                skip_next = True
                continue
            if re.fullmatch(r"\d*(?:<|>{1,2}|>&|<&).+", token):
                continue
            cleaned.append(token)
        return cleaned

    @staticmethod
    def _printf_format_has_invalid_literal_percent(format_text: str) -> bool:
        conversion_chars = set("diouxXfFeEgGaAcCsSbq")
        index = 0
        while index < len(format_text):
            percent = format_text.find("%", index)
            if percent < 0:
                return False
            if percent + 1 < len(format_text) and format_text[percent + 1] == "%":
                index = percent + 2
                continue
            scan = percent + 1
            while scan < len(format_text) and format_text[scan] in "#0- +":
                scan += 1
            while scan < len(format_text) and (format_text[scan].isdigit() or format_text[scan] == "*"):
                scan += 1
            if scan < len(format_text) and format_text[scan] == ".":
                scan += 1
                while scan < len(format_text) and (format_text[scan].isdigit() or format_text[scan] == "*"):
                    scan += 1
            while scan < len(format_text) and format_text[scan] in "hlLjzt":
                scan += 1
            if scan >= len(format_text) or format_text[scan] not in conversion_chars:
                return True
            index = scan + 1
        return False

    @staticmethod
    def _shell_text_preserves_last_status_after_cleanup(lower_shell_text: str) -> bool:
        status_names = ("status", "rc", "code", "exit_code", "ret")
        if not any(re.search(rf"\b{name}\s*=\s*\$\?", lower_shell_text) for name in status_names):
            return False
        return any(re.search(rf"\bexit\s+\$\{{?{name}\}}?\b", lower_shell_text) for name in status_names)

    @classmethod
    def _shell_text_has_probe_then_trailing_status_mask(cls, lower_shell_text: str) -> bool:
        probe = (
            r"\b(?:python(?:\d+(?:\.\d+)?)?|pytest|node|npm|go|cargo|ruby|php|java|curl)\b"
            r"[^;&|()]*"
            r"(?:validate|validator|test|check|\.py|--tool|--count|--bad|wrong|invalid|malformed|mismatch)"
            r"[^;&|()]*"
        )
        trailing = r"(?:rm|rmdir|mv|cp|true|:|echo|printf|touch|mkdir)\b"
        for match in re.finditer(probe + r";\s*" + trailing, lower_shell_text, flags=re.IGNORECASE):
            probe_text = match.group(0).split(";", 1)[0]
            if cls._looks_like_validation_probe_text(probe_text):
                return True
        return False

    @classmethod
    def _shell_text_has_assertion_then_trailing_status_mask(cls, lower_shell_text: str) -> bool:
        trailing = r"(?:rm|rmdir|mv|cp|true|:|echo|printf|touch|mkdir)\b"
        assertions = (
            r"\[[^\]\n;]*(?:\$\?|-[a-z]\b|=|!=)[^\]\n;]*\]",
            r"\btest\b[^\n;]*",
            r"\bgrep\b[^\n;]*",
            r"\bdiff\b[^\n;]*",
            r"\bcmp\b[^\n;]*",
            r"\bpython(?:\d+(?:\.\d+)?)?\b[^\n;]*(?:assert|unittest|pytest|validate|check)",
        )
        for assertion in assertions:
            pattern = assertion + r"\s*;\s*" + trailing
            for match in re.finditer(pattern, lower_shell_text, flags=re.IGNORECASE):
                probe_text = lower_shell_text[: match.start()] + match.group(0).split(";", 1)[0]
                if cls._looks_like_validation_probe_text(probe_text):
                    return True
        return False

    def _looks_like_negative_path_pipeline_without_status_check(
        self,
        step: dict[str, Any],
        command: list[Any] | dict[str, Any],
        raw_parts: list[str],
    ) -> bool:
        if self._command_expected_returncode(command) != 0:
            return False
        step_text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", [])),
        ]).lower()
        if not self._text_requires_negative_path(step_text):
            return False
        for shell_text in self._shell_texts_for_static_check(raw_parts):
            lower = shell_text.lower()
            if "|" not in lower or "grep" not in lower:
                continue
            if not re.search(r"\b(?:python|node|npm|go|cargo|ruby|php|java|curl)\b[^|]*\|[^|]*\bgrep\b", lower):
                continue
            if self._shell_pipeline_checks_left_status(shell_text):
                continue
            if not any(marker in lower for marker in ("2>&1", "error", "invalid", "required", "fail", "exception", "missing")):
                continue
            return True
        return False

    def _looks_like_error_pipeline_without_status_check(
        self,
        command: list[Any] | dict[str, Any],
        raw_parts: list[str],
    ) -> bool:
        if self._command_expected_returncode(command) != 0:
            return False
        for shell_text in self._shell_texts_for_static_check(raw_parts):
            lower = shell_text.lower()
            if "|" not in lower or "grep" not in lower:
                continue
            if not re.search(r"\b(?:python|node|npm|go|cargo|ruby|php|java|curl)\b[^|]*\|[^|]*\bgrep\b", lower):
                continue
            if self._shell_pipeline_checks_left_status(shell_text):
                continue
            if any(marker in lower for marker in ("2>&1", "error", "invalid", "required", "exception", "missing")):
                return True
        return False

    @staticmethod
    def _shell_pipeline_checks_left_status(shell_text: str) -> bool:
        lower = shell_text.lower()
        status_markers = (
            "$?",
            "${?}",
            "pipestatus",
            "returncode",
            "status=",
            "status =",
            "rc=",
            "rc =",
            "exit_code=",
            "exit_code =",
            "ret=",
            "ret =",
        )
        return any(marker in lower for marker in status_markers)

    @staticmethod
    def _looks_like_validation_probe_text(text: str) -> bool:
        probe_markers = (
            "python",
            "pytest",
            "unittest",
            "validate",
            "validator",
            "check",
            "assert",
            "test",
            "grep",
            "diff",
            "cmp",
            "curl",
            "node",
            "npm",
            "go test",
            "cargo test",
            "expected",
            "invalid",
            "wrong",
            "bad",
            "mismatch",
        )
        return any(marker in text for marker in probe_markers)

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
        if "unittest" in lowered or "unittest" in joined or "pytest" in lowered or "pytest" in joined:
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
        if re.search(r"\b(?:all\s+tests\s+pass|no\s+failures?|without\s+failures?)\b", text):
            return False
        failure = r"(?:fail(?:s|ed|ing|ure|ures)?|non[-\s]?zero|error|exception)"
        intent = r"(?:expect(?:ed|s|ing)?|confirm(?:ed|s|ing)?|intend(?:ed|s)?|observe(?:d|s)?|expose(?:d|s)?|diagnos(?:e|ed|es|ing)|reproduc(?:e|ed|es|ing)|remain(?:s|ing)?)"
        if re.search(rf"\b{intent}\b[^.\n]{{0,100}}\b{failure}\b", text):
            return True
        if re.search(rf"\b{failure}\b[^.\n]{{0,100}}\b{intent}\b", text):
            return True
        return bool(re.search(r"\b(?:still|continues?\s+to|known|remaining)\s+fail(?:s|ure|ures|ing)?\b", text))

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
        applies = self._default_quality_policy_applies()
        return {
            "applies": applies,
            "requires_research_structure_step": self._default_quality_policy_requires_research_structure_step(),
            "explicit_artifact_only_constraint": explicit_artifact_only,
            "reason": self._default_quality_policy_reason(),
            "assumed_requirement": (
                "Use proportional quality policy. When the prompt explicitly asks for tests, documentation, design "
                "notes, structure, or a project/app/library-level build, require those requested deliverables. "
                "Only require a separate initial research/structure planning step when the prompt asks for "
                "explicit research, architecture, project-structure planning, or broader application-level scope. "
                "For bounded utility/script tasks that do not ask for extra "
                "deliverables, do not invent documentation, tests, or research files; keep the plan small and require "
                "direct validation evidence. Explicit output-only constraints override extra deliverables but never "
                "remove the need for validation evidence. Cited source URLs are required only when web research "
                "fetched sources."
            ),
        }

    def _default_quality_instruction(self) -> str:
        if not self._default_quality_policy_applies():
            if self._explicit_artifact_only_constraint():
                return (
                    "Use proportional quality policy for this output-only prompt: do not add unrequested "
                    "documentation, tests, research notes, architecture files, helper files, or scaffold steps. "
                    "Keep the requirements and plan focused on the explicitly requested deliverable(s), while still "
                    "using clear validation commands/evidence that prove those deliverables are correct."
                )
            return (
                "Use proportional quality policy for this prompt: do not add unrequested documentation, tests, "
                "research notes, architecture files, or scaffold steps. Keep the requirements and plan focused on "
                "the deliverables the user actually requested, while still using clear structure and validation "
                "commands/evidence that prove those deliverables are correct."
            )
        return (
            "Proportional quality policy applies because the prompt requests quality deliverables or project-level "
            "work: include the deliverables the user requested, such as tests, documentation, structure, or "
            "design/research notes. Do not add design/research notes unless the prompt requests them or the "
            "project-level scope genuinely needs them. "
            + (
                "Because requires_research_structure_step is true, the first implementation step, or the first part "
                "of the first step when the user requests very few steps, must research required patterns/knowledge, "
                "plan the project structure/architecture, and rewrite the remaining plan if that structure changes "
                "task order. "
                if self._default_quality_policy_requires_research_structure_step()
                else
                "Do not add a separate research/architecture step merely because tests or README documentation were "
                "requested; keep bounded tasks compact while still creating the requested tests/docs and validation. "
            )
            +
            "Only require cited source URLs when web research actually fetched source URLs; otherwise require "
            "available-knowledge notes. A short requested notes file can be produced with the relevant implementation "
            "step unless the prompt separately asks for research, architecture, or structure planning."
        )

    def _default_quality_policy_applies(self) -> bool:
        if not self.config.quality_policy.assume_code_quality_when_unspecified:
            return False
        if not self._legacy_semantic_phrase_checks_enabled():
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
        return not any(item in prompt for item in overrides) and self._prompt_requests_quality_scope(prompt)

    def _prompt_requests_quality_scope(self, prompt: str) -> bool:
        explicit_quality_markers = (
            "include tests",
            "include unit tests",
            "unittest",
            "test coverage",
            "readme documentation",
            "documented",
            "design notes",
            "architecture",
            "well structured",
            "well-tested",
            "well tested",
            "well documented",
            "easy to validate",
        )
        positive_file_markers = (
            r"\binclude\b[^.?!\n]{0,80}\b(?:readme|documentation|docs?|tests?|unit tests?)\b",
            r"\bcreate\b[^.?!\n]{0,80}\b(?:readme|documentation|docs?|tests?|unit tests?)\b",
            r"\badd\b[^.?!\n]{0,80}\b(?:readme|documentation|docs?|tests?|unit tests?)\b",
            r"\bwith\b[^.?!\n]{0,80}\b(?:readme|documentation|docs?|tests?|unit tests?)\b",
        )
        project_scope_patterns = (
            r"\bprojects?\b",
            r"\bapps?\b",
            r"\bapplications?\b",
            r"\bwebsites?\b",
            r"\bfrontends?\b",
            r"\bbrowser\b",
            r"\bgames?\b",
            r"\bplatformers?\b",
            r"\bpackages?\b",
            r"\blibrar(?:y|ies)\b",
            r"\bmodules?\b",
            r"\bservices?\b",
            r"\bapis?\b",
        )
        return (
            any(marker in prompt for marker in explicit_quality_markers)
            or any(re.search(pattern, prompt) for pattern in positive_file_markers)
            or any(re.search(pattern, prompt) for pattern in project_scope_patterns)
        )

    def _default_quality_policy_requires_research_structure_step(self) -> bool:
        if not (
            self.config.quality_policy.require_research_and_structure_step
            and self._default_quality_policy_applies()
        ):
            return False
        prompt = self.config.project_design.prompt.lower()
        research_structure_markers = (
            "architecture",
            "architectural",
            "design document",
            "project structure",
            "structure overview",
            "separation of concerns",
            "research",
            "professional",
            "application",
            "website",
            "browser",
            "frontend",
            "game",
            "platformer",
            "service",
            " api",
            "package",
            "library",
            "existing project",
        )
        return any(marker in prompt for marker in research_structure_markers)

    def _default_quality_policy_reason(self) -> str:
        if not self.config.quality_policy.assume_code_quality_when_unspecified:
            return "disabled in config"
        if not self._legacy_semantic_phrase_checks_enabled():
            return "keyword-based quality-scope classification disabled; model applies proportional quality from the prompt"
        if self._explicit_artifact_only_constraint():
            return "explicit artifact-only constraint"
        prompt = self.config.project_design.prompt.lower()
        if not self._prompt_requests_quality_scope(prompt):
            return "bounded utility/script prompt without requested extra quality deliverables"
        return "prompt requests quality deliverables or project-level scope"

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
                progress_callback=self._running_tool_progress_reviewer(source=source, context=context or {}),
                progress_interval_seconds=self.config.runtime.command_progress_review_interval_seconds,
                progress_min_interval_seconds=self.config.runtime.command_progress_review_min_interval_seconds,
            )
            for index, command, result in zip(runnable_indexes, runnable, executed):
                result["tool_verification"] = decisions.get(index, {"decision": "approved"})
                results[index] = result
        return [result for result in results if result is not None]

    def _running_tool_progress_reviewer(
        self,
        *,
        source: str,
        context: dict[str, Any],
    ):
        """Return a callback that lets the feedback model review a live command.

        The process runner owns draining and bounding stdout/stderr. This
        callback owns the model decision: continue or terminate based on the
        active request, plan state, previous repair history, and a compact live
        output snapshot. It deliberately does not encode task-specific answers.
        """
        if self.config.runtime.command_progress_review_interval_seconds <= 0:
            return None

        def review(snapshot: dict[str, Any]) -> dict[str, Any]:
            prompt = {
                "phase": "TOOL_PROGRESS_REVIEW_PHASE",
                "source": source,
                "context": context,
                "workflow_state": self._workflow_state_for_prompt(
                    context.get("step") if isinstance(context.get("step"), dict) else None
                ),
                "running_command": self._compact_running_tool_snapshot(snapshot),
                "runtime_policy": {
                    "workspace": str(self.workspace),
                    "configured_timeout_seconds": snapshot.get("timeout_seconds"),
                    "progress_review_interval_seconds": self.config.runtime.command_progress_review_interval_seconds,
                    "progress_review_min_interval_seconds": self.config.runtime.command_progress_review_min_interval_seconds,
                    "tool_output_max_chars": self.config.context_compaction.tool_output_max_chars,
                },
                "expected_json": {
                    "status": "continue|terminate",
                    "decision": "continue|terminate",
                    "summary": "why continue or terminate",
                    "evidence": ["specific observed fact"],
                    "risks": ["risk"],
                    "next_check_seconds": self.config.runtime.command_progress_review_interval_seconds,
                },
            }
            raw = self._feedback_chat(
                "TOOL_PROGRESS_REVIEW_PHASE\n"
                "A terminal command is still running. Decide whether it remains useful for the current task. "
                "Use the transcript, workflow state, and bounded live output snapshot. Do not stop it just "
                "because it has been running for a while, and do not continue it just because an earlier model "
                "asked for it. Stop only when the current evidence shows the command is wrong, unsafe, stuck in "
                "a hopeless loop, waiting for unavailable input, or no longer useful. Otherwise continue and set "
                "a sensible next_check_seconds. Heartbeats, health checks, elapsed time, and repeated generic "
                "log lines are observability, not task progress unless the user requested monitoring.\n"
                f"{_review_prompt_guidance()}\n"
                + json.dumps(prompt),
                temperature=0.0,
            )
            try:
                parsed = self._extract_json_or_retry(
                    raw,
                    phase="TOOL_PROGRESS_REVIEW_PHASE",
                    contract=TOOL_PROGRESS_REVIEW_CONTRACT,
                    feedback=True,
                )
            except Exception as exc:
                parsed = {
                    "status": "continue",
                    "decision": "continue",
                    "summary": f"Progress reviewer output was malformed; continued running command: {exc}",
                    "evidence": ["No parseable progress-review decision was available."],
                    "risks": ["A malformed review cannot safely justify terminating a previously approved command."],
                    "next_check_seconds": self.config.runtime.command_progress_review_interval_seconds,
                }
            normalized = self._normalize_running_tool_review(parsed)
            normalized["running_command"] = self._compact_running_tool_review_snapshot(prompt["running_command"])
            self.conversation.append(
                "user",
                "TOOL_PROGRESS_REVIEW_RESULT:\n"
                + json.dumps(self._compact_running_tool_review_for_transcript(normalized), indent=2),
            )
            return normalized

        return review

    def _compact_running_tool_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        limit = max(1000, min(self.config.context_compaction.tool_output_max_chars, 8000))
        stdout = str(snapshot.get("stdout", ""))
        stderr = str(snapshot.get("stderr", ""))
        return {
            "command_index": snapshot.get("command_index"),
            "command": snapshot.get("command"),
            "cwd": snapshot.get("cwd"),
            "elapsed_seconds": snapshot.get("elapsed_seconds"),
            "timeout_seconds": snapshot.get("timeout_seconds"),
            "hard_timeout_disabled": snapshot.get("hard_timeout_disabled"),
            "review_count": snapshot.get("review_count"),
            "returncode": snapshot.get("returncode"),
            "stdout_bytes": snapshot.get("stdout_bytes"),
            "stderr_bytes": snapshot.get("stderr_bytes"),
            "stdout_truncated": snapshot.get("stdout_truncated"),
            "stderr_truncated": snapshot.get("stderr_truncated"),
            "stdout": self._prompt_excerpt(stdout, limit),
            "stderr": self._prompt_excerpt(stderr, limit),
        }

    def _normalize_running_tool_review(self, review: dict[str, Any]) -> dict[str, Any]:
        review = dict(review)
        decision = str(review.get("decision") or review.get("status") or "continue").strip().lower()
        if decision in {"stop", "stopped", "terminate", "terminated", "kill", "cancel"}:
            decision = "terminate"
        elif decision != "continue":
            decision = "continue"
        review["decision"] = decision
        review["status"] = decision
        if not review.get("summary"):
            review["summary"] = (
                "Progress review terminated the running command."
                if decision == "terminate"
                else "Progress review allowed the running command to continue."
            )
        try:
            next_check = int(review.get("next_check_seconds", self.config.runtime.command_progress_review_interval_seconds))
        except (TypeError, ValueError):
            next_check = self.config.runtime.command_progress_review_interval_seconds
        review["next_check_seconds"] = max(
            self.config.runtime.command_progress_review_min_interval_seconds,
            next_check,
        )
        if not isinstance(review.get("evidence"), list):
            review["evidence"] = [str(review.get("evidence") or review.get("summary") or "")]
        if not isinstance(review.get("risks"), list):
            review["risks"] = [str(review.get("risks") or "")]
        return review

    def _compact_running_tool_review_for_transcript(self, review: dict[str, Any]) -> dict[str, Any]:
        running_command = review.get("running_command") if isinstance(review.get("running_command"), dict) else {}
        return {
            "status": review.get("status"),
            "decision": review.get("decision"),
            "summary": review.get("summary"),
            "running_command": running_command,
            "evidence": self._clip_list_for_transcript(review.get("evidence", [])),
            "risks": self._clip_list_for_transcript(review.get("risks", [])),
            "next_check_seconds": review.get("next_check_seconds"),
        }

    def _compact_running_tool_review_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "command": snapshot.get("command"),
            "cwd": snapshot.get("cwd"),
            "elapsed_seconds": snapshot.get("elapsed_seconds"),
            "timeout_seconds": snapshot.get("timeout_seconds"),
            "hard_timeout_disabled": snapshot.get("hard_timeout_disabled"),
            "review_count": snapshot.get("review_count"),
            "stdout_bytes": snapshot.get("stdout_bytes"),
            "stderr_bytes": snapshot.get("stderr_bytes"),
            "stdout_truncated": snapshot.get("stdout_truncated"),
            "stderr_truncated": snapshot.get("stderr_truncated"),
            "stdout_excerpt": self._prompt_excerpt(str(snapshot.get("stdout") or ""), 800),
            "stderr_excerpt": self._prompt_excerpt(str(snapshot.get("stderr") or ""), 800),
        }

    def _tool_call_verification_phase(
        self,
        commands: list[Any],
        *,
        source: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        deterministic = self._deterministic_tool_call_findings(commands, source=source, context=context)
        deterministic_indexes: set[int] = set()
        for finding in deterministic:
            try:
                index = int(finding.get("index", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(commands):
                deterministic_indexes.add(index)
        if deterministic and len(deterministic_indexes) == len(commands):
            review = self._normalize_tool_verification(
                {
                    "status": "blocked",
                    "summary": "Deterministic tool-call safety checks blocked one or more commands before model review.",
                    "commands": [],
                    "deterministic_only": True,
                },
                commands,
                deterministic,
                source=source,
                context=context,
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
                "timeout_zero_means": "no hard wall-clock deadline; progress review decides continuation",
                "progress_review_interval_seconds": self.config.runtime.command_progress_review_interval_seconds,
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
        verification_payload = json.dumps(prompt, ensure_ascii=False)
        raw = self._feedback_chat(
            "TOOL_CALL_VERIFICATION_PHASE\n"
            "Verify proposed terminal tool calls before execution. Use the whole transcript to understand intent. "
            "Approve commands that are correctly targeted, bounded, and useful for the current plan. Block commands "
            "that may destroy data, target the wrong path/device, depend on malformed quoting, run indefinitely "
            "without task justification or progress evidence, "
            "or fail to verify the intended behavior. Judge only the supplied command indexes; do not refer to or "
            "approve commands that are not present in the commands array. Return one explicit decision for every "
            "supplied command index; if any supplied command is missing or unclear, block it instead of reusing an "
            "older decision. Commands are argv arrays, not outer-shell command strings; for `bash -lc`, the script "
            "argument is evaluated by that bash process, not pre-expanded by the harness. Deterministic findings are authoritative "
            "safety signals. `timeout_seconds: 0` is not a shortcut for ordinary commands; approve it only when "
            "open-ended monitoring is justified and the command exposes enough bounded output for progress review. "
            "For literal stdout/file-content checks, block accidental regex validation: structured-output strings "
            "containing brackets or braces should use fixed-string matching, exact captured-output comparison, or "
            "a validator script. "
            "If context includes implementation-provided test_evidence, treat it as a model claim or intended "
            "validation unless matching command results already exist in the transcript; do not treat it as proof "
            "that the proposed command has already passed. Do not block a safe, bounded validation command merely "
            "because you suspect the proposed artifact or answer is wrong; approve the command so the harness can "
            "collect execution evidence, then the normal step review can reject the implementation if validation "
            "fails. Block validation commands for safety, targeting, quoting, boundedness, or coverage defects, "
            "not because you tried to solve the task yourself.\n"
            f"{_review_prompt_guidance(executable_deliverables=True)}\n"
            + verification_payload,
            temperature=0.0,
        )
        try:
            review = self._extract_json_or_retry(
                raw,
                phase="TOOL_CALL_VERIFICATION_PHASE",
                contract=TOOL_CALL_VERIFICATION_CONTRACT,
                feedback=True,
                current_question_context=(
                    "Tool-call verification payload. The `commands` array below is the authoritative set "
                    "of proposed commands for this review; judge only these indexes and do not substitute "
                    "older commands from workflow history.\n"
                    + verification_payload
                ),
            )
        except Exception as exc:
            review = {
                "status": "blocked",
                "summary": f"Tool verifier returned malformed or off-contract JSON: {exc}",
                "commands": [
                    {
                        "index": index,
                        "decision": "blocked",
                        "risk_level": "medium",
                        "reason": "Verifier output was malformed or off-contract; retry with clearer command intent.",
                    }
                    for index, _command in enumerate(commands)
                ],
                "repair_error": str(exc),
            }
        review = self._normalize_tool_verification(
            review,
            commands,
            deterministic,
            source=source,
            context=context,
        )
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
        *,
        source: str = "",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        review = dict(review)
        review_status = str(review.get("status") or "").strip().lower()
        approved_like_statuses = {"approved"}
        default_to_blocked = bool(review.get("needs_rework")) or review_status not in approved_like_statuses
        default_reason = str(review.get("summary") or "Verifier did not explicitly approve this command.")
        existing: dict[int, dict[str, Any]] = {}
        for item in review.get("commands", []):
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index", -1))
            except (TypeError, ValueError):
                continue
            existing[index] = dict(item)
        deterministic_by_index: dict[int, list[str]] = {}
        for finding in deterministic:
            try:
                index = int(finding.get("index", -1))
            except (TypeError, ValueError):
                continue
            deterministic_by_index.setdefault(index, []).append(str(finding.get("reason", "")))
        harness_generated_full_decisions = bool(review.get("deterministic_only"))
        missing_decisions = [
            index for index in range(len(commands))
            if index not in existing
        ]
        review_command_list = review.get("commands")
        has_partial_command_decisions = (
            isinstance(review_command_list, list)
            and bool(review_command_list)
            and bool(missing_decisions)
        )
        incomplete_approved_response = (
            not default_to_blocked
            and not harness_generated_full_decisions
            and has_partial_command_decisions
        )
        if incomplete_approved_response:
            default_reason = (
                "Tool verifier returned an approved or resolved status but omitted explicit decisions "
                f"for supplied command indexes {missing_decisions}; blocking all commands so the "
                "implementation can request a complete, current verification."
            )
            review["missing_command_decisions"] = missing_decisions
        normalized = []
        any_blocked = False
        for index, command in enumerate(commands):
            if incomplete_approved_response:
                item = {
                    "index": index,
                    "decision": "blocked",
                    "risk_level": "medium",
                    "reason": default_reason,
                }
            elif index in existing:
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
        unknown_indexes = sorted(index for index in existing if index < 0 or index >= len(commands))
        if unknown_indexes or self._summary_references_unknown_command_index(review.get("summary", ""), len(commands)):
            review["ignored_unknown_command_indexes"] = unknown_indexes
            review["summary"] = (
                "Tool calls verified for supplied command indexes only; any verifier text about unsupplied "
                "commands was ignored by harness normalization."
            )
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

    @staticmethod
    def _summary_references_unknown_command_index(summary: object, command_count: int) -> bool:
        text = str(summary or "").lower()
        for match in re.finditer(r"\bcommands?\s+((?:\d+\s*(?:,|and)?\s*)+)", text):
            for number in re.findall(r"\d+", match.group(1)):
                if int(number) >= command_count:
                    return True
        return False

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

    def _deterministic_tool_call_findings(
        self,
        commands: list[Any],
        *,
        source: str = "",
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        step = (context or {}).get("step") if isinstance(context, dict) else None
        for index, command in enumerate(commands):
            if isinstance(command, dict):
                command_value = command.get("cmd") or command.get("command") or []
                if isinstance(command_value, str):
                    findings.append({
                        "index": index,
                        "risk_level": "medium",
                        "reason": (
                            "Command object uses a string-valued `cmd`. Use an argv list such as "
                            "{\"cmd\": [\"bash\", \"-lc\", \"...\"]} so quoting and arguments can be verified."
                        ),
                    })
            elif isinstance(command, str):
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        "Command is a plain string. Use an argv list or a command object with list-valued `cmd` "
                        "so quoting and arguments can be verified."
                    ),
                })
            parts = self._command_parts_for_safety(command)
            if not parts:
                continue
            for finding in self._printf_literal_percent_findings(parts):
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": "Command " + finding,
                })
            if self._looks_like_placeholder_validation_command(parts):
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        "Command contains placeholder or stub test logic that can pass without exercising "
                        "the requested artifact. Replace it with assertions that run the implemented program "
                        "or inspect the requested deliverable behavior."
                    ),
                })
            grep_option_pattern_finding = self._grep_option_like_pattern_finding(parts)
            if grep_option_pattern_finding:
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": "Command " + grep_option_pattern_finding,
                })
            grep_literal_pattern_finding = self._grep_literal_regex_pattern_finding(parts)
            if grep_literal_pattern_finding:
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": "Command " + grep_literal_pattern_finding,
                })
            if self._timeout_wraps_shell_builtin_wait(parts):
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        "Command tries to run shell builtin `wait` through external `timeout`. `wait` is not "
                        "a standalone program, so `timeout ... wait $PID` fails before observing the child "
                        "process. Use a shell script that starts the child and waits inside the same shell, "
                        "or use a polling/communicate timeout pattern."
                    ),
                })
            findings.extend(self._validation_script_tool_findings(index, parts))
            expected_returncode = self._command_expected_returncode(command)
            requested_timeout = self._command_requested_timeout(command)
            executable = Path(parts[0]).name
            if (
                requested_timeout == 0
                and self.config.runtime.command_progress_review_interval_seconds <= 0
            ):
                findings.append({
                    "index": index,
                    "risk_level": "high",
                    "reason": (
                        "Command disables the hard wall-clock timeout with timeout_seconds=0, but progress "
                        "review is disabled. Use a positive timeout or enable command progress review so a "
                        "local model can periodically decide whether the command should continue."
                    ),
                })
            argv_metadata = self._looks_like_metadata_inside_argv(parts)
            if argv_metadata:
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        f"Command metadata `{argv_metadata}` appears inside the argv list. "
                        "Use a command object with a `cmd` argv list and metadata fields outside the argv array."
                    ),
                })
            if self._validation_command_appears_to_mutate_artifact(parts):
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        "Command appears to create or mutate the explicitly requested artifact. "
                        "For artifact-only prompts, the final artifact must be supplied in the JSON `files` "
                        "payload; commands should only verify it or write temporary evidence outside the workspace."
                    ),
                })
            source_mutation_target = self._workspace_source_mutation_target(parts)
            if source_mutation_target:
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        f"Command appears to write or mutate workspace source path `{source_mutation_target}`. "
                        "Generated terminal commands should validate or collect evidence; project source edits must "
                        "come from the JSON `files` payload, and negative-path checks should use /tmp fixtures, "
                        "wrapper commands, test doubles, or expected_returncode. If a file must be directly "
                        "executable, put the shebang in the files payload and validate with `test -x` or direct "
                        "invocation instead of `chmod`; do not infer that direct executability is required merely "
                        "because a Python file can run via `python script.py`."
                    ),
                })
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
            if expected_returncode == 0 and any(
                self._shell_swallows_failure_as_success(shell_text)
                for shell_text in self._shell_texts_for_static_check(parts)
            ):
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        "Shell command appears to mask a validation failure with `|| exit 0`, `|| true`, "
                        "or an equivalent always-success fallback. Use expected_returncode, or a wrapper that exits "
                        "1 if the command unexpectedly succeeds and exits 0 only after confirming the intended failure."
                    ),
                })
            if self._looks_like_expected_failure_status_masked_by_shell_tail(command, parts):
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        "Command declares an expected non-zero return code, but the shell script appears to run "
                        "cleanup or another trailing command after the command-under-test without preserving its "
                        "status. Capture `$?` immediately, run cleanup, then `exit $status`, or use `trap`."
                    ),
                })
            if expected_returncode == 0 and self._looks_like_validation_status_masked_by_shell_tail(command, parts):
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        "Shell command appears to run cleanup or another trailing command after a validation "
                        "assertion without preserving the assertion status. Use `trap`, capture `$?` and exit "
                        "with it after cleanup, or chain cleanup so a failed assertion cannot be hidden."
                    ),
                })
            workspace_output_targets = self._validation_workspace_output_targets(parts)
            if workspace_output_targets:
                targets = ", ".join(f"`{target}`" for target in workspace_output_targets)
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        f"Command writes temporary validation output to workspace path {targets}. Use `/tmp`, "
                        "`mktemp`, or another trap-cleaned temporary path so validation evidence does not become "
                        "an unrequested project artifact."
                    ),
                })
            if isinstance(step, dict):
                stateful_help_finding = self._stateful_validation_help_only_finding(step, parts)
                if stateful_help_finding:
                    findings.append({
                        "index": index,
                        "risk_level": "medium",
                        "reason": "Command " + stateful_help_finding,
                    })
                else:
                    runtime_state_finding = self._stateful_validation_runtime_state_finding(step, parts)
                    if runtime_state_finding:
                        findings.append({
                            "index": index,
                            "risk_level": "medium",
                        "reason": "Command " + runtime_state_finding,
                    })
                filtered_absence_finding = self._filtered_absence_check_finding(step, parts)
                if filtered_absence_finding:
                    findings.append({
                        "index": index,
                        "risk_level": "medium",
                        "reason": "Command " + filtered_absence_finding,
                    })
            misplaced_env_assignments = self._misplaced_environment_assignments_after_program(parts)
            if misplaced_env_assignments:
                names = ", ".join(f"`{name}=...`" for name in misplaced_env_assignments)
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        f"Command passes {names} after the script/program under test. Shell treats that as "
                        "an argument, not an environment override. Put environment assignments before the command "
                        "or use `env VAR=value command`."
                    ),
                })
            raw_numeric_comparison = self._raw_text_numeric_comparison_finding(parts)
            if raw_numeric_comparison:
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": "Command " + raw_numeric_comparison,
                })
            if expected_returncode == 0 and self._looks_like_error_pipeline_without_status_check(command, parts):
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        "Shell command pipes an expected failure-path command into grep without checking the "
                        "command-under-test exit status. Capture `$?`/a status variable and assert both the "
                        "non-zero code and error text, or use a command object with expected_returncode."
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
            inline_python_unreachable = self._inline_python_unreachable_after_return(parts)
            if inline_python_unreachable:
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        "Inline Python command contains unreachable statements after return before execution: "
                        + inline_python_unreachable
                    ),
                })
            heredoc_error = self._shell_heredoc_static_error(parts)
            if heredoc_error:
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        "Shell command contains malformed here-doc syntax: "
                        + heredoc_error
                    ),
                })
            shell_syntax_error = self._shell_static_syntax_error(parts)
            if shell_syntax_error:
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        "Shell command fails a static parse check before execution: "
                        + shell_syntax_error
                    ),
                })
            artifact_heredoc_error = self._artifact_only_heredoc_finding(parts)
            if artifact_heredoc_error:
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": "Command " + artifact_heredoc_error,
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

    def _step_feedback_tool_evidence(
        self,
        step: dict[str, Any],
        *,
        implementation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Collect the evidence the feedback agent can inspect directly.

        The feedback model is not trusted to merely believe the implementation
        model's report. Before each review, the harness takes a fresh workspace
        snapshot and re-runs the current plan step's validation commands. Those
        command results become the reviewer's evidence. When the current
        implementation attempt already produced passing validation-like
        commands, re-run those too so stale plan commands do not trap the step
        in repeated implementation-only repairs.
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
        accepted_validation_commands = self._accepted_validation_commands_from_implementation(implementation or {})
        planned_signatures = {self._command_signature(command) for command in validation_commands}
        accepted_validation_commands = [
            command
            for command in accepted_validation_commands
            if self._command_signature(command) not in planned_signatures
        ]
        accepted_validation_results: list[dict[str, Any]] = []
        if self.config.mcp_tools.terminal and accepted_validation_commands:
            accepted_validation_results = self._run_verified_commands(
                accepted_validation_commands,
                source="step_accepted_validation",
                context={
                    "step": step,
                    "purpose": "Reviewer-owned rerun of validation-like commands from the current implementation attempt.",
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
            "accepted_validation_commands": accepted_validation_commands,
            "accepted_validation_results": accepted_validation_results,
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

    def _accepted_validation_commands_from_implementation(self, implementation: dict[str, Any]) -> list[Any]:
        """Return passing validation-like commands from one implementation attempt."""
        commands: list[Any] = []
        seen: set[tuple[tuple[str, ...], int]] = set()
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
        return commands

    def _adopt_accepted_validation_commands_for_step(self, step: dict[str, Any], attempt: dict[str, Any]) -> None:
        """Update stale runbook validation commands after a step is accepted.

        A repair attempt may fix an invalid validator without needing a full
        requirements rewrite. Once that attempt is accepted, the runbook should
        carry the validation that actually proved the step so later reviews do
        not keep replaying obsolete commands.
        """
        implementation = attempt.get("implementation") or {}
        commands = self._accepted_validation_commands_from_implementation(implementation)
        if not commands:
            return
        current = list(step.get("validation_commands") or [])
        if [self._command_signature(command) for command in commands] == [
            self._command_signature(command) for command in current
        ]:
            return
        step["validation_commands"] = commands
        self.requirements["plan"] = self.plan_steps
        self._append_plan_note(
            f"[{step.get('id', 'step')}] updated validation commands from accepted implementation evidence."
        )

    def _looks_like_validation_evidence_command(self, result: dict[str, Any]) -> bool:
        command = [str(part) for part in (result.get("command") or [])]
        if not command:
            return False
        if command[0] in {"rm", "mv", "cp", "mkdir", "touch", "git", "sed", "tee", "cat"}:
            return False
        if int(result.get("expected_returncode", 0) or 0) != 0:
            return True
        if self._command_looks_like_executable_probe(command):
            return True
        text = " ".join(command).lower()
        if "http.server" in text and not any(marker in text for marker in ("test", "validate", "check")):
            return False
        validation_markers = (
            "unittest",
            "pytest",
            "npm test",
            "test_",
            "test ",
            "/test",
            "tests/",
            "validate",
            "check",
            "assert",
            "grep -q",
            "| grep",
            "playwright",
            "coverage",
        )
        return any(marker in text for marker in validation_markers)

    def _command_looks_like_executable_probe(self, command: Any) -> bool:
        argv = self._command_argv_for_static_check(command)
        if self._argv_looks_like_executable_probe(argv):
            return True
        return any(
            self._shell_text_looks_like_executable_probe(shell_text)
            for shell_text in self._shell_texts_for_static_check(argv)
        )

    @classmethod
    def _shell_text_looks_like_executable_probe(cls, shell_text: str) -> bool:
        return any(
            cls._argv_looks_like_executable_probe(cls._safe_shell_split(segment))
            for segment in cls._shell_command_segments(shell_text)
        )

    @classmethod
    def _argv_looks_like_executable_probe(cls, argv: list[str]) -> bool:
        if len(argv) >= 3 and Path(argv[0]).name == "test" and argv[1] == "-x":
            return Path(_trim_reference_delimiters(argv[2])).suffix.lower() in EXECUTABLE_DELIVERABLE_SUFFIXES
        if len(argv) >= 4 and Path(argv[0]).name == "[" and argv[1] == "-x":
            return Path(_trim_reference_delimiters(argv[2])).suffix.lower() in EXECUTABLE_DELIVERABLE_SUFFIXES
        subject_index = cls._command_subject_index_after_leading_env(argv)
        if subject_index is None:
            return False
        token = _trim_reference_delimiters(argv[subject_index])
        return (
            (token.startswith("./") or "/" in token)
            and Path(token).suffix.lower() in EXECUTABLE_DELIVERABLE_SUFFIXES
        )

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

    def _unrequested_runtime_state_artifacts(
        self,
        implementation: dict[str, Any],
        feedback_tool_evidence: dict[str, Any],
    ) -> list[str]:
        git = feedback_tool_evidence.get("git") or {}
        changed_paths = git.get("meaningful_changed_paths") or []
        written_paths = {
            _normalize_workspace_path_text(path).lower()
            for path in implementation.get("written", []) or []
        }
        harness_docs = {name.lower() for name in self._harness_doc_names()}
        artifacts: set[str] = set()
        for raw_path in changed_paths:
            runtime_path = self._runtime_state_artifact_path(raw_path)
            if not runtime_path:
                continue
            lowered = runtime_path.lower()
            if lowered in written_paths or lowered in harness_docs:
                continue
            artifacts.add(runtime_path)
        return sorted(artifacts)

    @staticmethod
    def _runtime_state_artifact_path(path: object) -> str:
        normalized = _normalize_workspace_path_text(_trim_reference_delimiters(path))
        if not normalized or normalized.startswith(("/", "$", "<", ">")):
            return ""
        parts = Path(normalized).parts
        if not parts:
            return ""
        lowered_parts = [part.lower() for part in parts]
        if any(part in RUNTIME_STATE_EXCLUDED_BASENAMES for part in lowered_parts):
            return ""
        basename = lowered_parts[-1]
        if basename in RUNTIME_STATE_BASENAMES:
            return normalized
        if basename.endswith(RUNTIME_STATE_SUFFIXES):
            return normalized
        if any(part in {".cache", ".checkpoint", ".state"} for part in lowered_parts[:-1]):
            return normalized
        return ""

    def _evidence_findings(
        self,
        step: dict[str, Any],
        implementation: dict[str, Any],
        feedback_tool_evidence: dict[str, Any] | None = None,
    ) -> list[str]:
        findings: list[str] = []
        semantic_phrase_checks = self._legacy_semantic_phrase_checks_enabled()
        implementation_commands = implementation.get("commands", [])
        feedback_results = (feedback_tool_evidence or {}).get("validation_results", [])
        accepted_feedback_results = (feedback_tool_evidence or {}).get("accepted_validation_results", [])
        skipped_harness_files = implementation.get("skipped_harness_files", [])
        if skipped_harness_files:
            findings.append(
                "Implementation attempted to write files blocked by harness-owned state or artifact-only policy: "
                + ", ".join(str(path) for path in skipped_harness_files)
                + ". Please keep project deliverables within the allowed workspace artifacts and use plan_note for progress."
            )
        runtime_state_artifacts = self._unrequested_runtime_state_artifacts(
            implementation,
            feedback_tool_evidence or {},
        )
        if runtime_state_artifacts:
            findings.append(
                "Validation or generated commands left runtime state artifact(s) in the workspace: "
                + ", ".join(f"`{path}`" for path in runtime_state_artifacts)
                + ". Prefer a trap-cleaned temporary working directory. Use a temporary state path only when "
                "the requested interface or existing project already exposes one; do not add a public state-file "
                "option solely for validation. Clean existing stale state safely and rerun validation so repair "
                "attempts do not depend on residue."
            )
        findings.extend(self._stale_validation_evidence_findings(feedback_tool_evidence or {}))
        findings.extend(self._delayed_resource_validation_findings(feedback_tool_evidence or {}))
        expected_validation = bool(step.get("validation_commands"))
        if expected_validation and not feedback_results:
            findings.append(f"{step.get('id', 'step')} has validation criteria but feedback tools produced no validation evidence.")
        feedback_validation_all_passed = self._feedback_validation_all_passed(step, feedback_results)
        for result in feedback_results:
            if result.get("timed_out"):
                findings.append(f"Feedback validation command timed out: {result.get('command')}")
            findings.extend(self._validation_result_integrity_findings(step, result, "Feedback validation"))
            if not self._command_returncode_matches_expected(result) and not self._is_failure_investigation_step(step):
                if self._plan_failure_is_superseded_by_accepted_validation(result, accepted_feedback_results):
                    continue
                if self._expected_nonzero_returncode_mismatch_is_scope_neutral(step, result):
                    continue
                findings.append(
                    f"Feedback validation command returned {result.get('returncode')} but expected "
                    f"{result.get('expected_returncode', 0)}: {result.get('command')}"
                    f"{self._command_failure_excerpt(result)}"
                )
                list_null_scope_gap = self._list_null_element_validation_scope_finding(result, "Feedback validation")
                if list_null_scope_gap:
                    findings.append(list_null_scope_gap)
                diagnostic_gap = (
                    self._silent_semantic_validation_failure_finding(step, result, "Feedback validation")
                    if semantic_phrase_checks
                    else ""
                )
                if diagnostic_gap:
                    findings.append(diagnostic_gap)
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
        for result in accepted_feedback_results:
            if result.get("timed_out"):
                findings.append(f"Accepted validation command timed out during step review: {result.get('command')}")
            findings.extend(self._validation_result_integrity_findings(step, result, "Accepted validation"))
            if not self._command_returncode_matches_expected(result) and not self._is_failure_investigation_step(step):
                if self._expected_nonzero_returncode_mismatch_is_scope_neutral(step, result):
                    continue
                findings.append(
                    f"Accepted validation command returned {result.get('returncode')} but expected "
                    f"{result.get('expected_returncode', 0)}: {result.get('command')}"
                    f"{self._command_failure_excerpt(result)}"
                )
                list_null_scope_gap = self._list_null_element_validation_scope_finding(result, "Accepted validation")
                if list_null_scope_gap:
                    findings.append(list_null_scope_gap)
        for result in implementation_commands:
            if result.get("timed_out"):
                findings.append(f"Implementation command timed out: {result.get('command')}")
            findings.extend(self._validation_result_integrity_findings(step, result, "Implementation"))
            if not self._command_returncode_matches_expected(result) and not self._is_failure_investigation_step(step):
                if self._implementation_self_check_is_reviewer_discretion(
                    result,
                    feedback_validation_all_passed=feedback_validation_all_passed,
                ):
                    continue
                if self._expected_nonzero_returncode_mismatch_is_scope_neutral(step, result):
                    continue
                findings.append(
                    f"Implementation command returned {result.get('returncode')} but expected "
                    f"{result.get('expected_returncode', 0)}: {result.get('command')}"
                    f"{self._command_failure_excerpt(result)}"
                )
                list_null_scope_gap = self._list_null_element_validation_scope_finding(result, "Implementation")
                if list_null_scope_gap:
                    findings.append(list_null_scope_gap)
                diagnostic_gap = (
                    self._silent_semantic_validation_failure_finding(step, result, "Implementation")
                    if semantic_phrase_checks
                    else ""
                )
                if diagnostic_gap:
                    findings.append(diagnostic_gap)
        findings.extend(
            self._documentation_content_evidence_findings(
                step,
                feedback_results + accepted_feedback_results + implementation_commands,
                "Step review",
            )
        )
        if semantic_phrase_checks:
            findings.extend(self._stdout_json_format_implementation_findings(feedback_tool_evidence or {}))
            findings.extend(
                self._negative_path_evidence_findings(
                    step,
                    feedback_results + accepted_feedback_results + implementation_commands,
                    "Step review",
                )
            )
            findings.extend(
                self._executable_deliverable_evidence_findings(
                    step,
                    feedback_results + accepted_feedback_results + implementation_commands,
                    feedback_tool_evidence or {},
                    "Step review",
                )
            )
            findings.extend(self._public_api_implementation_shape_findings(feedback_tool_evidence or {}))
        findings.extend(self._git_diff_findings(step, implementation, feedback_tool_evidence or {}))
        findings.extend(
            self._workspace_reference_findings(
                feedback_tool_evidence or {},
                allow_planned_future_refs=True,
            )
        )
        return findings

    def _stale_validation_evidence_findings(self, feedback_tool_evidence: dict[str, Any]) -> list[str]:
        """Find validators that can pass by re-reading old success output.

        This is intentionally a validation-integrity heuristic, not a
        task-specific solver. It catches a common weak-test shape: assert an
        output capture contains a success marker, change the system under test,
        then assert the same capture contains that marker again without clearing
        the capture or comparing only newly produced evidence.
        """
        findings: list[str] = []
        for item in feedback_tool_evidence.get("workspace_files", []) or []:
            path = str(item.get("path") or "")
            if not self._workspace_path_looks_like_validation_source(path):
                continue
            content = str(item.get("content") or "")
            repeated = self._reused_output_assertions_after_state_change(content)
            for output_path in repeated:
                findings.append(
                    f"Validation script `{path}` reuses `{output_path}` for more than one success assertion after "
                    "changing the tested state without clearing that capture or proving only new output was counted. "
                    "This can pass on stale evidence; revise validation so each phase uses fresh output or compares "
                    "a before/after event count."
                )
        return findings

    def _delayed_resource_validation_findings(self, feedback_tool_evidence: dict[str, Any]) -> list[str]:
        """Find delayed-event tests where the watched resource already exists.

        Watcher and polling tasks often need evidence that a file, log line, or
        trigger appears after the process starts. A common weak validator creates
        a temporary file with `NamedTemporaryFile(delete=False)` or `mkstemp`,
        then launches the watcher against that already-existing path and starts a
        delayed writer thread. That proves the success path only, not the delayed
        appearance behavior the test claims to cover.
        """
        findings: list[str] = []
        for item in feedback_tool_evidence.get("workspace_files", []) or []:
            path = str(item.get("path") or "")
            if not self._workspace_path_looks_like_validation_source(path):
                continue
            content = str(item.get("content") or "")
            if not self._validation_claims_delayed_resource(content):
                continue
            if not self._validation_creates_existing_temp_resource_before_watch(content):
                continue
            if self._validation_removes_placeholder_before_watch(content):
                continue
            findings.append(
                f"Validation script `{path}` claims to prove a resource appears after the watcher starts, "
                "but it creates an already-existing temporary file/path before launching the watched command. "
                "That can pass without proving delayed detection. Use a target path that does not exist yet, "
                "or unlink the placeholder and assert it is absent before starting the watcher."
            )
        return findings

    @staticmethod
    def _validation_claims_delayed_resource(content: str) -> bool:
        lower = content.lower()
        delayed_phrases = (
            "appears after",
            "appears during",
            "appears later",
            "created after",
            "created later",
            "after startup",
            "during wait",
            "during polling",
            "wait_for_file",
            "watcher starts",
        )
        if any(phrase in lower for phrase in delayed_phrases):
            return True
        return bool(
            re.search(
                r"\bdef\s+test_[^\n]*(?:appear|later|startup|during|delay|poll|watch)",
                lower,
            )
        )

    @staticmethod
    def _validation_creates_existing_temp_resource_before_watch(content: str) -> bool:
        creation_matches = list(
            re.finditer(
                r"tempfile\.(?:NamedTemporaryFile\s*\([^)]*delete\s*=\s*False|mkstemp)\b",
                content,
                flags=re.IGNORECASE,
            )
        )
        if not creation_matches:
            return False
        watch_match = re.search(
            r"\bsubprocess\.(?:run|Popen|call|check_call|check_output)\s*\(",
            content,
        )
        if watch_match is None:
            return False
        return any(match.start() < watch_match.start() for match in creation_matches)

    @staticmethod
    def _validation_removes_placeholder_before_watch(content: str) -> bool:
        watch_match = re.search(
            r"\bsubprocess\.(?:run|Popen|call|check_call|check_output)\s*\(",
            content,
        )
        prefix = content[: watch_match.start()] if watch_match else content
        return bool(
            re.search(r"\bos\.(?:remove|unlink)\s*\(", prefix)
            or re.search(r"\bPath\s*\([^)]*\)\.unlink\s*\(", prefix)
            or re.search(r"\.unlink\s*\(", prefix)
        )

    @staticmethod
    def _workspace_path_looks_like_validation_source(path: str) -> bool:
        lowered = path.lower()
        basename = Path(lowered).name
        if basename in {"plan.md", "requirements.md", "research.md", "readme.md"}:
            return False
        if any(part in basename for part in ("test", "validate", "verify", "check")):
            return True
        return lowered.endswith((".test.js", ".spec.js", "_test.py", ".bats"))

    def _reused_output_assertions_after_state_change(self, content: str) -> list[str]:
        assertions: dict[str, list[tuple[int, str]]] = {}
        lines = content.splitlines()
        for index, line in enumerate(lines):
            output_path = self._grep_quiet_output_path(line)
            if output_path:
                assertions.setdefault(output_path, []).append((index, line))
        stale: set[str] = set()
        for output_path, occurrences in assertions.items():
            if len(occurrences) < 2:
                continue
            for (first_index, _first_line), (second_index, _second_line) in zip(occurrences, occurrences[1:]):
                between = "\n".join(lines[first_index + 1:second_index])
                if not self._validation_state_changes_between_assertions(between):
                    continue
                if self._validation_output_reset_or_baselined(output_path, between):
                    continue
                stale.add(output_path)
                break
        return sorted(stale)

    @staticmethod
    def _grep_quiet_output_path(line: str) -> str:
        if "grep" not in line or "-q" not in line:
            return ""
        candidate = re.sub(r"^\s*(?:if|while|until)\s+", "", line.strip())
        candidate = candidate.split(";", 1)[0].strip()
        try:
            tokens = shlex.split(candidate)
        except ValueError:
            return ""
        if "grep" in tokens:
            tokens = tokens[tokens.index("grep"):]
        if not tokens or tokens[0] != "grep":
            return ""
        has_quiet = any(token.startswith("-") and "q" in token for token in tokens[1:])
        if not has_quiet:
            return ""
        positional = [
            token
            for token in tokens[1:]
            if not token.startswith("-")
            and not re.match(r"^\d*>", token)
            and token not in {"then", "do"}
        ]
        if len(positional) < 2:
            return ""
        for token in reversed(positional[1:]):
            normalized = _trim_reference_delimiters(token)
            if FeedbackLoopAgent._path_looks_like_captured_output(normalized):
                return normalized
        return ""

    @staticmethod
    def _path_looks_like_captured_output(path: str) -> bool:
        if not path or path.startswith("$"):
            return False
        lowered = Path(path).name.lower()
        if any(marker in lowered for marker in ("output", "stdout", "stderr", "result", "capture")):
            return True
        return lowered.endswith((".out", ".stdout", ".stderr"))

    @staticmethod
    def _validation_state_changes_between_assertions(text: str) -> bool:
        lowered = text.lower()
        return any(
            marker in lowered
            for marker in (
                "restart",
                "resume",
                "truncat",
                "append",
                "kill ",
                "pkill",
                "rm -f",
                "mv ",
                "cp ",
                "touch ",
                "printf ",
                "echo ",
                "sleep ",
                " > ",
                " >> ",
            )
        )

    @staticmethod
    def _validation_output_reset_or_baselined(output_path: str, text: str) -> bool:
        lowered = text.lower()
        escaped = re.escape(output_path)
        reset_patterns = (
            rf"\brm\s+-f\s+['\"]?{escaped}['\"]?",
            rf"\btruncate\s+-s\s+0\s+['\"]?{escaped}['\"]?",
            rf"(?:^|[\n;&]\s*):\s*>\s*['\"]?{escaped}['\"]?",
            rf"(?:^|[\n;&]\s*)>\s*['\"]?{escaped}['\"]?",
            rf"\bcp\s+/dev/null\s+['\"]?{escaped}['\"]?",
        )
        if any(re.search(pattern, text) for pattern in reset_patterns):
            return True
        baseline_patterns = (
            rf"\bgrep\s+[^;\n]*-c[^;\n]*['\"]?{escaped}['\"]?",
            rf"\bwc\s+-l[^;\n]*['\"]?{escaped}['\"]?",
        )
        return any(re.search(pattern, lowered) for pattern in baseline_patterns)

    def _validation_script_tool_findings(self, index: int, parts: list[str]) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for script_ref in self._validation_script_refs_from_command(parts):
            path = self._workspace_local_script_path(script_ref)
            if path is None or not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            evidence = {
                "workspace_files": [{
                    "path": str(path.relative_to(self.workspace)),
                    "content": content,
                }]
            }
            for finding in self._stale_validation_evidence_findings(evidence):
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": finding,
                })
            for finding in self._delayed_resource_validation_findings(evidence):
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": finding,
                })
            for finding in self._blocking_validation_subprocess_findings(content):
                findings.append({
                    "index": index,
                    "risk_level": "high",
                    "reason": finding,
                })
        return findings

    @staticmethod
    def _blocking_validation_subprocess_findings(content: str) -> list[str]:
        if "subprocess.Popen" not in content or not re.search(r"stdout\s*=\s*subprocess\.PIPE", content):
            return []
        if ".stdout.readline()" not in content and ".stdout.readline(" not in content:
            return []
        lowered = content.lower()
        bounded_read_markers = (
            ".communicate(timeout=",
            "select.select",
            "selectors.",
            "fcntl.",
            "threading.",
            "asyncio.",
            "queue.",
        )
        if any(marker in lowered for marker in bounded_read_markers):
            return []
        if not re.search(r"\bwhile\b[\s\S]{0,600}\.stdout\.readline\(", content):
            return []
        return [
            "Validation script uses blocking `stdout.readline()` on a subprocess with piped stdout "
            "inside a loop. For long-running or watch-style commands this can hang after the last "
            "line of output; use `communicate(timeout=...)`, selectors/nonblocking IO, a helper "
            "thread/queue, or terminate after a bounded condition before reading."
        ]

    @classmethod
    def _validation_script_refs_from_command(cls, parts: list[str]) -> list[str]:
        refs: list[str] = []
        for token in parts[1:]:
            if cls._token_looks_like_validation_script(token):
                refs.append(_trim_reference_delimiters(token))
        for shell_text in cls._shell_texts_for_static_check(parts):
            try:
                shell_tokens = shlex.split(shell_text)
            except ValueError:
                shell_tokens = []
            for token in shell_tokens:
                if cls._token_looks_like_validation_script(token):
                    refs.append(_trim_reference_delimiters(token))
            for match in re.finditer(
                r"(?:^|[\s;&|])(?:bash|sh|source|\.)\s+([./A-Za-z0-9_-]*(?:validate|verify|check|test)[A-Za-z0-9_.-]*\.sh)\b",
                shell_text,
                flags=re.IGNORECASE,
            ):
                refs.append(_trim_reference_delimiters(match.group(1)))
        return list(dict.fromkeys(refs))

    @staticmethod
    def _token_looks_like_validation_script(token: object) -> bool:
        normalized = _trim_reference_delimiters(token)
        if not normalized or normalized.startswith("-"):
            return False
        lowered = Path(normalized).name.lower()
        return lowered.endswith((".sh", ".py")) and any(
            marker in lowered for marker in ("validate", "verify", "check", "test")
        )

    def _workspace_local_script_path(self, script_ref: str) -> Path | None:
        normalized = Path(_normalize_workspace_path_text(script_ref))
        if normalized.is_absolute() or ".." in normalized.parts:
            return None
        candidate = (self.workspace / normalized).resolve()
        try:
            candidate.relative_to(self.workspace.resolve())
        except ValueError:
            return None
        return candidate

    def _public_api_implementation_shape_findings(self, feedback_tool_evidence: dict[str, Any]) -> list[str]:
        """Catch implementation-time drift back into unrequested API shapes."""
        entrypoints = self._prompt_public_entrypoints(self.config.project_design.prompt)
        if not entrypoints:
            return []
        if self._prompt_explicitly_names_output_representation(self.config.project_design.prompt):
            return []
        files = [
            item
            for item in feedback_tool_evidence.get("workspace_files", [])
            if item.get("path") not in self._harness_doc_names()
        ]
        if not files:
            return []
        body = "\n".join(
            f"\n### {item.get('path')}\n{item.get('content', '')}"
            for item in files
        )
        implementation_body = "\n".join(
            f"\n### {item.get('path')}\n{item.get('content', '')}"
            for item in files
            if self._workspace_path_looks_like_implementation_source(str(item.get("path") or ""))
        )
        lower = body.lower()
        implementation_lower = implementation_body.lower()
        if not self._pair_like_public_api_context(lower):
            return []
        findings: list[str] = []
        requirements_text = json.dumps(self.requirements, sort_keys=True)
        for entrypoint in entrypoints:
            entry_lower = entrypoint.lower()
            if f"{entry_lower}(" not in lower:
                continue
            if self._requirements_signature_names_public_output_representation(requirements_text, entrypoint):
                continue
            if not self._canonical_pair_shape_conversion_detected(lower, entry_lower) and not (
                implementation_lower and self._implementation_source_converts_pair_shape(implementation_lower)
            ):
                continue
            if self._shape_preservation_evidence_present(lower, entry_lower):
                continue
            if self._documented_canonical_shape_evidence_present(files, entry_lower):
                continue
            findings.append(
                f"Implementation evidence for public API `{entrypoint}` appears to convert flexible pair-like "
                "caller inputs into one canonical output representation even though the user did not request "
                "that representation. Either make the chosen representation explicit in public documentation "
                "or source-level API documentation, then validate two separate caller shapes against that same "
                "chosen output representation: one call whose input is a list of tuple pairs, such as "
                f"`{entrypoint}([(1, 3), (2, 4)])`, and one call whose input is a list of list pairs, such as "
                f"`{entrypoint}([[1, 3], [2, 4]])`. Mixed inputs or a tuple containing nested interval tuples "
                "do not replace those two cases. Or request a requirements/plan clarification. Do not switch "
                "to same-input-type preservation merely to satisfy this review unless the original user request "
                "named that behavior."
        )
        return findings

    @classmethod
    def _requirements_signature_names_public_output_representation(cls, requirements_text: str, entrypoint: str) -> bool:
        """Return true when accepted requirements already specify an entrypoint return shape."""
        lower = requirements_text.lower()
        entry = re.escape(entrypoint.lower())
        shape_markers = (
            "list[",
            "tuple[",
            "dict[",
            "set[",
            "list of",
            "tuple of",
            "dict of",
            "set of",
            "json",
            "str",
            "string",
        )
        for match in re.finditer(rf"\b{entry}\s*\([^)]*\)\s*->\s*[^`\".\n,;]{{0,160}}", lower):
            window = match.group(0)
            if any(marker in window for marker in shape_markers):
                return True
        return False

    def _pair_like_public_api_context(self, implementation_text: str) -> bool:
        combined = " ".join([
            self.config.project_design.prompt,
            json.dumps(self.requirements, sort_keys=True),
            implementation_text,
        ]).lower()
        pair_markers = ("pair", "pairs", "interval", "intervals", "record", "records")
        flexible_markers = ("tuple", "tuples", "list", "lists", "iterable", "sequence", "container")
        return any(marker in combined for marker in pair_markers) and any(
            marker in implementation_text for marker in flexible_markers
        )

    @staticmethod
    def _workspace_path_looks_like_implementation_source(path: str) -> bool:
        normalized = path.replace("\\", "/").lower()
        name = normalized.rsplit("/", 1)[-1]
        if not normalized.endswith(".py"):
            return False
        if normalized.startswith("tests/") or "/tests/" in normalized:
            return False
        if name.startswith("test_") or name.endswith("_test.py"):
            return False
        return True

    def _canonical_pair_shape_conversion_detected(self, implementation_text: str, entrypoint: str) -> bool:
        conversion_markers = (
            "as lists",
            "as tuples",
            "canonical output",
            "canonical representation",
            "consistent return type",
            "convert to list",
            "convert to tuple",
            "converted to list",
            "converted to tuple",
            "returns lists",
            "returns tuples",
            "return lists",
            "return tuples",
            "regardless of whether input",
            "even if the input",
        )
        if any(marker in implementation_text for marker in conversion_markers):
            return True
        return self._entrypoint_call_converts_pair_shape(
            implementation_text,
            entrypoint,
            input_shape="tuple",
            output_shape="list",
        ) or self._entrypoint_call_converts_pair_shape(
            implementation_text,
            entrypoint,
            input_shape="list",
            output_shape="tuple",
        )

    @staticmethod
    def _implementation_source_converts_pair_shape(implementation_text: str) -> bool:
        constructor_patterns = (
            r"\bappend\s*\(\s*(?:list|tuple)\s*\(",
            r"\bextend\s*\(\s*(?:list|tuple)\s*\(",
            r"\breturn\s+\[\s*(?:list|tuple)\s*\(",
            r"\[\s*(?:list|tuple)\s*\([^)]+\)\s+for\s+\w+\s+in\s+",
        )
        return any(re.search(pattern, implementation_text) for pattern in constructor_patterns)

    def _shape_preservation_evidence_present(self, implementation_text: str, entrypoint: str) -> bool:
        tuple_preserved = self._entrypoint_call_preserves_pair_shape(
            implementation_text,
            entrypoint,
            shape="tuple",
        )
        list_preserved = self._entrypoint_call_preserves_pair_shape(
            implementation_text,
            entrypoint,
            shape="list",
        )
        return tuple_preserved and list_preserved

    def _documented_canonical_shape_evidence_present(
        self,
        files: list[dict[str, Any]],
        entrypoint: str,
    ) -> bool:
        """Accept a deliberate canonical pair shape only when it is public and tested.

        A reviewer may decide that normalizing flexible pair inputs to one
        concrete pair representation is reasonable, but only if that behavior is
        not hidden. Documentation must name the chosen representation, and tests
        must cover both tuple-pair and list-pair callers against that shape.
        """
        doc_text = "\n".join(
            str(item.get("content") or "")
            for item in files
            if (
                self._workspace_path_looks_like_public_documentation(str(item.get("path") or ""))
                or self._workspace_path_looks_like_implementation_source(str(item.get("path") or ""))
            )
        ).lower()
        if not doc_text:
            return False
        output_shape = self._documented_canonical_pair_output_shape(doc_text)
        if output_shape is None:
            return False
        test_text = "\n".join(
            str(item.get("content") or "")
            for item in files
            if self._workspace_path_looks_like_test_source(str(item.get("path") or ""))
        ).lower()
        if not test_text:
            return False
        return (
            self._entrypoint_call_validates_output_shape(
                test_text,
                entrypoint,
                input_shape="tuple",
                output_shape=output_shape,
            )
            and self._entrypoint_call_validates_output_shape(
                test_text,
                entrypoint,
                input_shape="list",
                output_shape=output_shape,
            )
        )

    @staticmethod
    def _workspace_path_looks_like_public_documentation(path: str) -> bool:
        normalized = path.replace("\\", "/").lower()
        name = normalized.rsplit("/", 1)[-1]
        return (
            name in {"readme.md", "readme.rst", "readme.txt"}
            or normalized.startswith("docs/")
            or "/docs/" in normalized
            or normalized.endswith((".md", ".rst", ".txt"))
        )

    @classmethod
    def _workspace_path_looks_like_test_source(cls, path: str) -> bool:
        normalized = path.replace("\\", "/").lower()
        name = normalized.rsplit("/", 1)[-1]
        return (
            normalized.endswith(".py")
            and (
                normalized.startswith("tests/")
                or "/tests/" in normalized
                or name.startswith("test_")
                or name.endswith("_test.py")
            )
        )

    @staticmethod
    def _documented_canonical_pair_output_shape(doc_text: str) -> str | None:
        list_patterns = (
            r"\breturns?\s*:\s*(?:\n\s*)?(?:a\s+)?list\s+of\s+lists\b",
            r"\breturns?\s*:\s*(?:\n\s*)?(?:a\s+)?list\b[^.\n]{0,180}\bas\s+lists?\b",
            r"\breturns?\s*:\s*(?:\n\s*)?(?:a\s+)?list\b[^.\n]{0,180}\b(?:where\s+)?(?:each|every)\s+(?:interval|pair|element|item)\b[^.\n]{0,100}\b(?:is\s+)?(?:a\s+)?list\b",
            r"\breturns?\b[^.\n]{0,160}\blist\s+of\s+lists\b",
            r"\breturns?\b[^.\n]{0,180}\blist\b[^.\n]{0,100}\bas\s+lists?\b",
            r"\boutput\b[^.\n]{0,160}\blist\s+of\s+lists\b",
            r"\boutput\b[^.\n]{0,180}\blist\b[^.\n]{0,100}\bas\s+lists?\b",
            r"\bresult\b[^.\n]{0,160}\blist\s+of\s+lists\b",
            r"\bresult\b[^.\n]{0,180}\blist\b[^.\n]{0,100}\bas\s+lists?\b",
            r"\breturn\s+format\b[^.\n]{0,160}\blist\s+of\s+lists\b",
            r"\binner\s+list\b",
            r"\boutput\s+elements?\b[^.\n]{0,120}\blists?\b",
            r"\bmerged\s+integer\s+pairs?\s*\(as\s+lists?\)",
        )
        tuple_patterns = (
            r"\breturns?\s*:\s*(?:\n\s*)?(?:a\s+)?list\s+of\s+tuples\b",
            r"\breturns?\s*:\s*(?:\n\s*)?(?:a\s+)?list\b[^.\n]{0,180}\bas\s+tuples?\b",
            r"\breturns?\s*:\s*(?:\n\s*)?(?:a\s+)?list\b[^.\n]{0,180}\b(?:where\s+)?(?:each|every)\s+(?:interval|pair|element|item)\b[^.\n]{0,100}\b(?:is\s+)?(?:a\s+)?tuple\b",
            r"\breturns?\b[^.\n]{0,160}\blist\s+of\s+tuples\b",
            r"\breturns?\b[^.\n]{0,180}\blist\b[^.\n]{0,100}\bas\s+tuples?\b",
            r"\boutput\b[^.\n]{0,160}\blist\s+of\s+tuples\b",
            r"\boutput\b[^.\n]{0,180}\blist\b[^.\n]{0,100}\bas\s+tuples?\b",
            r"\bresult\b[^.\n]{0,160}\blist\s+of\s+tuples\b",
            r"\bresult\b[^.\n]{0,180}\blist\b[^.\n]{0,100}\bas\s+tuples?\b",
            r"\breturn\s+format\b[^.\n]{0,160}\blist\s+of\s+tuples\b",
            r"\binner\s+tuple\b",
            r"\boutput\s+elements?\b[^.\n]{0,120}\btuples?\b",
            r"\bmerged\s+integer\s+pairs?\s*\(as\s+tuples?\)",
        )
        list_doc = any(re.search(pattern, doc_text) for pattern in list_patterns)
        tuple_doc = any(re.search(pattern, doc_text) for pattern in tuple_patterns)
        if list_doc == tuple_doc:
            return None
        return "list" if list_doc else "tuple"

    def _entrypoint_call_validates_output_shape(
        self,
        implementation_text: str,
        entrypoint: str,
        *,
        input_shape: str,
        output_shape: str,
    ) -> bool:
        if input_shape == output_shape:
            direct = self._entrypoint_call_preserves_pair_shape(
                implementation_text,
                entrypoint,
                shape=input_shape,
            )
        else:
            direct = self._entrypoint_call_converts_pair_shape(
                implementation_text,
                entrypoint,
                input_shape=input_shape,
                output_shape=output_shape,
            )
        return direct or self._entrypoint_call_type_asserts_pair_shape(
            implementation_text,
            entrypoint,
            input_shape=input_shape,
            output_shape=output_shape,
        ) or self._entrypoint_call_compares_to_shaped_expected_variable(
            implementation_text,
            entrypoint,
            input_shape=input_shape,
            output_shape=output_shape,
        )

    def _entrypoint_call_compares_to_shaped_expected_variable(
        self,
        implementation_text: str,
        entrypoint: str,
        *,
        input_shape: str,
        output_shape: str,
    ) -> bool:
        input_pattern = self._entrypoint_pair_input_pattern(entrypoint, input_shape)
        output_literal = self._pair_literal_prefix_pattern(output_shape)
        variable_name = r"[a-z_][a-z0-9_]*"
        for match in re.finditer(rf"\b({variable_name})\s*=\s*{output_literal}", implementation_text):
            variable = re.escape(match.group(1))
            window = implementation_text[match.end(): match.end() + 1800]
            patterns = (
                rf"assertequal\s*\(\s*{input_pattern}[^\n]{{0,280}},\s*{variable}\s*\)",
                rf"assertequal\s*\(\s*{variable}\s*,\s*{input_pattern}[^\n]{{0,280}}\)",
                rf"assert\s+{input_pattern}[^\n]{{0,280}}==\s*{variable}\b",
                rf"assert\s+{variable}\s*==\s*{input_pattern}[^\n]{{0,280}}",
            )
            if any(re.search(pattern, window) for pattern in patterns):
                return True
        return False

    def _entrypoint_call_type_asserts_pair_shape(
        self,
        implementation_text: str,
        entrypoint: str,
        *,
        input_shape: str,
        output_shape: str,
    ) -> bool:
        input_pattern = self._entrypoint_pair_input_pattern(entrypoint, input_shape)
        type_name = "list" if output_shape == "list" else "tuple"
        variable_name = r"[a-z_][a-z0-9_]*"

        def output_shape_asserted_for_variable(window: str, variable: str) -> bool:
            variable = re.escape(variable)
            return bool(
                re.search(rf"assertisinstance\s*\(\s*{variable}\s*\[\s*0\s*\]\s*,\s*{type_name}\s*\)", window)
                or re.search(rf"isinstance\s*\(\s*{variable}\s*\[\s*0\s*\]\s*,\s*{type_name}\s*\)", window)
                or re.search(rf"type\s*\(\s*{variable}\s*\[\s*0\s*\]\s*\)\s*(?:is|==)\s*{type_name}\b", window)
                or re.search(
                    rf"all\s*\(\s*isinstance\s*\([^)]*,\s*{type_name}\s*\)\s+for\s+[^)]*\s+in\s+{variable}\s*\)",
                    window,
                )
            )

        direct_call_patterns = (
            rf"isinstance\s*\(\s*{input_pattern}[^;\n]{{0,240}}\[\s*0\s*\]\s*,\s*{type_name}\s*\)",
            rf"assertisinstance\s*\(\s*{input_pattern}[^;\n]{{0,240}}\[\s*0\s*\]\s*,\s*{type_name}\s*\)",
        )
        if any(re.search(pattern, implementation_text) for pattern in direct_call_patterns):
            return True
        assignment_pattern = rf"\b({variable_name})\s*=\s*{input_pattern}"
        for match in re.finditer(assignment_pattern, implementation_text):
            window = implementation_text[match.start(): match.start() + 900]
            if output_shape_asserted_for_variable(window, match.group(1)):
                return True
        input_literal = r"\(" if input_shape == "tuple" else r"\["
        input_assignment_pattern = rf"\b({variable_name})\s*=\s*\[\s*{input_literal}"
        for input_match in re.finditer(input_assignment_pattern, implementation_text):
            input_variable = re.escape(input_match.group(1))
            window = implementation_text[input_match.start(): input_match.start() + 1400]
            call_assignment_pattern = rf"\b({variable_name})\s*=\s*{re.escape(entrypoint)}\s*\(\s*{input_variable}\s*\)"
            for call_match in re.finditer(call_assignment_pattern, window):
                if output_shape_asserted_for_variable(window[call_match.start():], call_match.group(1)):
                    return True
        return False

    def _entrypoint_call_converts_pair_shape(
        self,
        implementation_text: str,
        entrypoint: str,
        *,
        input_shape: str,
        output_shape: str,
    ) -> bool:
        input_pattern = self._entrypoint_pair_input_pattern(entrypoint, input_shape)
        output_pattern = self._pair_output_pattern(output_shape)
        for match in re.finditer(input_pattern, implementation_text):
            window = implementation_text[match.start(): match.start() + 500]
            if re.search(output_pattern, window):
                return True
        return False

    def _entrypoint_call_preserves_pair_shape(self, implementation_text: str, entrypoint: str, *, shape: str) -> bool:
        input_pattern = self._entrypoint_pair_input_pattern(entrypoint, shape)
        output_pattern = self._pair_output_pattern(shape)
        for match in re.finditer(input_pattern, implementation_text):
            window = implementation_text[match.start(): match.start() + 500]
            if re.search(output_pattern, window):
                return True
        return False

    @staticmethod
    def _entrypoint_pair_input_pattern(entrypoint: str, shape: str) -> str:
        if shape == "tuple":
            return rf"\b{re.escape(entrypoint)}\s*\(\s*\[\s*\("
        return rf"\b{re.escape(entrypoint)}\s*\(\s*\[\s*\["

    @staticmethod
    def _pair_output_pattern(shape: str) -> str:
        if shape == "tuple":
            return r"(?:==|,\s*)\s*\[\s*\("
        return r"(?:==|,\s*)\s*\[\s*\["

    @staticmethod
    def _pair_literal_prefix_pattern(shape: str) -> str:
        if shape == "tuple":
            return r"\[\s*\("
        return r"\[\s*\["

    def _silent_semantic_validation_failure_finding(
        self,
        step: dict[str, Any],
        result: dict[str, Any],
        source: str,
    ) -> str | None:
        """Ask for repair-useful diagnostics when a semantic check fails silently."""
        if not (
            self._computed_answer_prompt_requires_semantic_validation()
            or self._looks_like_validation_evidence_command(result)
        ):
            return None
        if result.get("timed_out"):
            return None
        if self._command_returncode_matches_expected(result):
            return None
        stdout = str(result.get("stdout") or "").strip()
        stderr = str(result.get("stderr") or "").strip()
        if stdout or stderr:
            return None
        step_id = str(step.get("id") or "step")
        return (
            f"{source} for {step_id} failed without stdout or stderr. For semantic-output or behavior "
            "validation, the next command should print concise mismatch diagnostics such as expected value, "
            "actual artifact value or command output, the failing sub-check, or a representative failing case "
            "while still exiting non-zero on failure."
        )

    def _feedback_validation_all_passed(self, step: dict[str, Any], feedback_results: list[dict[str, Any]]) -> bool:
        if not step.get("validation_commands") or not feedback_results:
            return False
        return all(
            not result.get("timed_out")
            and self._command_returncode_matches_expected(result)
            and not self._validation_result_integrity_findings(step, result, "Feedback validation")
            for result in feedback_results
        )

    def _implementation_self_check_is_reviewer_discretion(
        self,
        result: dict[str, Any],
        *,
        feedback_validation_all_passed: bool,
    ) -> bool:
        """Avoid turning every failed model-side self-check into a hard gate.

        Implementation-requested commands are useful evidence, but they can be
        over-specific, stale, or badly quoted. When the reviewer already has
        fresh passing validation for the planned step, the feedback model should
        judge a failed implementation self-check in context instead of the
        deterministic gate forcing another attempt. Reviewer-owned validation
        failures remain hard findings above.
        """
        if not feedback_validation_all_passed:
            return False
        if result.get("timed_out"):
            return False
        if self._validation_result_integrity_findings({}, result, "Implementation"):
            return False
        return self._looks_like_validation_evidence_command(result)

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
        if self._result_blocked_by_tool_verifier(result):
            return True
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
                    "mathematically incorrect",
                    "operator precedence",
                    "does not calculate",
                    "logic error in the validation command",
                    "validation command is flawed",
                    "missing the '--' separator",
                    "missing the -- separator",
                    "end-of-options separator",
                    "will cause it to fail",
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

    def _expected_nonzero_returncode_mismatch_is_scope_neutral(
        self,
        step: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:
        """Avoid turning an over-specific validator exit code into task scope.

        Many requirements only ask that an invalid-input command fail with a
        non-zero status. If a generated validator guessed `expected_returncode:
        1` but the program returns another non-zero code, such as argparse's
        conventional 2, the command still proves the user-visible failure
        behavior. Exact exit codes remain strict when the accepted requirements
        or acceptance criteria explicitly name a non-zero code.
        """
        try:
            expected = int(result.get("expected_returncode", 0))
            actual = int(result.get("returncode", 0))
        except (TypeError, ValueError):
            return False
        if expected <= 0 or actual <= 0 or actual == expected:
            return False
        scope = self._step_behavior_scope_text(step)
        if not self._scope_accepts_unspecified_nonzero_exit(scope):
            return False
        if self._scope_requires_exact_nonzero_exit_code(scope):
            return False
        return self._result_proves_negative_path(result)

    def _step_behavior_scope_text(self, step: dict[str, Any]) -> str:
        chunks = [
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", []) or []),
        ]
        if isinstance(self.requirements, dict):
            chunks.append(str(self.requirements.get("project_summary") or self.requirements.get("summary") or ""))
            for key in ("refined_requirements", "assumptions"):
                chunks.extend(str(item) for item in self.requirements.get(key, []) or [])
            for item in self.requirements.get("open_questions", []) or []:
                if isinstance(item, dict):
                    chunks.append(str(item.get("decision", "")))
                else:
                    chunks.append(str(item))
        return re.sub(r"\s+", " ", " ".join(chunks)).strip().lower()

    @staticmethod
    def _scope_accepts_unspecified_nonzero_exit(scope: str) -> bool:
        return bool(
            re.search(r"\b(?:exit|exits|return|returns)\b[^.\n;]{0,100}\bnon[- ]?zero\b", scope)
            or re.search(r"\bnon[- ]?zero\b[^.\n;]{0,60}\b(?:exit|status|code|returncode|return\s+code)\b", scope)
        )

    @staticmethod
    def _scope_requires_exact_nonzero_exit_code(scope: str) -> bool:
        return bool(
            re.search(
                r"\b(?:exit|exits|return|returns|returned)\b[^.\n;]{0,80}"
                r"\b(?:code|status|returncode|return\s+code)\s+[1-9]\d*\b",
                scope,
            )
            or re.search(
                r"\b(?:exit|status|returncode|return\s+code)\s+[1-9]\d*\b",
                scope,
            )
        )

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

    def _grep_option_like_pattern_finding(self, command: Any) -> str:
        """Detect grep checks where an option-looking pattern lacks `--`/`-e`.

        A common local-model quoting failure is `grep -q '--flag text' file`.
        Shell quoting keeps the pattern as one argv token, but grep still treats
        a token beginning with `--` as an option unless the command uses `--` or
        `-e`. This check is about command shape, not task content.
        """
        tokens = self._validation_command_tokens_for_similarity(command)
        if not tokens:
            return ""
        separators = {"&&", "||", "|", ";", "then", "else", "elif", "fi", "do", "done"}
        index = 0
        while index < len(tokens):
            if Path(str(tokens[index])).name != "grep":
                index += 1
                continue
            index += 1
            while index < len(tokens):
                token = str(tokens[index])
                if token in separators:
                    break
                if token == "--":
                    break
                if self._grep_option_consumes_pattern_argument(token):
                    index += 2 if token in {"-e", "--regexp"} else 1
                    continue
                if self._grep_option_has_embedded_argument(token):
                    index += 1
                    continue
                option_arg_count = self._grep_option_argument_count(token)
                if option_arg_count:
                    index += 1 + option_arg_count
                    continue
                if self._grep_known_no_arg_option(token):
                    index += 1
                    continue
                if token.startswith("--"):
                    return (
                        "uses a `grep` pattern or argument that starts with `--` before an end-of-options "
                        "separator. Use `grep -q -- PATTERN FILE` or `grep -q -e PATTERN FILE` so grep "
                        "does not parse the pattern as an option."
                    )
                if token.startswith("-") and " " in token:
                    return (
                        "uses a `grep` pattern that starts with `-` before an end-of-options separator. "
                        "Use `grep -q -- PATTERN FILE` or `grep -q -e PATTERN FILE` so grep does not "
                        "parse the pattern as an option."
                    )
                break
        return ""

    def _grep_literal_regex_pattern_finding(self, command: Any) -> str:
        """Detect likely literal structured output matched as accidental regex."""
        tokens = self._validation_command_tokens_for_similarity(command)
        if not tokens:
            return ""
        separators = {"&&", "||", "|", ";", "then", "else", "elif", "fi", "do", "done"}
        index = 0
        while index < len(tokens):
            if Path(str(tokens[index])).name != "grep":
                index += 1
                continue
            command_tokens: list[str] = []
            index += 1
            while index < len(tokens) and str(tokens[index]) not in separators:
                command_tokens.append(str(tokens[index]))
                index += 1
            if self._grep_command_uses_fixed_or_explicit_regex(command_tokens):
                continue
            patterns = self._grep_pattern_tokens(command_tokens)
            if any(self._looks_like_literal_structured_grep_pattern(pattern) for pattern in patterns):
                return (
                    "uses `grep` regex matching for a literal structured-output pattern containing regex "
                    "metacharacters. Use `grep -Fq`, an exact captured-output comparison, or a small validator "
                    "script so brackets/braces are treated literally."
                )
        return ""

    @classmethod
    def _grep_command_uses_fixed_or_explicit_regex(cls, tokens: list[str]) -> bool:
        for token in tokens:
            if token in {"-F", "--fixed-strings", "-E", "--extended-regexp", "-P", "--perl-regexp"}:
                return True
            if token.startswith("--regexp="):
                continue
            if not token.startswith("-") or token == "-" or token == "--":
                continue
            option_letters = token[1:]
            if option_letters and all(char.isalpha() for char in option_letters):
                if any(char in option_letters for char in "FEP"):
                    return True
        return False

    def _grep_pattern_tokens(self, tokens: list[str]) -> list[str]:
        patterns: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "--":
                if index + 1 < len(tokens):
                    patterns.append(tokens[index + 1])
                break
            if token in {"-e", "--regexp"}:
                if index + 1 < len(tokens):
                    patterns.append(tokens[index + 1])
                index += 2
                continue
            if token.startswith("--regexp="):
                patterns.append(token.split("=", 1)[1])
                index += 1
                continue
            option_arg_count = self._grep_option_argument_count(token)
            if option_arg_count:
                index += 1 + option_arg_count
                continue
            if self._grep_known_no_arg_option(token):
                index += 1
                continue
            if token.startswith("-") and token != "-":
                index += 1
                continue
            patterns.append(token)
            break
        return patterns

    @staticmethod
    def _looks_like_literal_structured_grep_pattern(pattern: str) -> bool:
        stripped = pattern.strip()
        if not stripped:
            return False
        starts_like_structured = stripped[0] in "{["
        contains_json_quote = '"' in stripped and (":" in stripped or "," in stripped)
        if not (starts_like_structured or contains_json_quote):
            return False
        return FeedbackLoopAgent._has_unescaped_regex_bracket(stripped)

    @staticmethod
    def _has_unescaped_regex_bracket(text: str) -> bool:
        escaped = False
        for char in text:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char in "[]":
                return True
        return False

    @staticmethod
    def _grep_option_consumes_pattern_argument(token: str) -> bool:
        return token in {"-e", "--regexp"} or token.startswith("--regexp=")

    @staticmethod
    def _grep_option_argument_count(token: str) -> int:
        options_with_values = {
            "-A",
            "-B",
            "-C",
            "-m",
            "--after-context",
            "--before-context",
            "--context",
            "--max-count",
            "--binary-files",
            "--devices",
            "--directories",
            "--exclude",
            "--exclude-dir",
            "--exclude-from",
            "--include",
            "--label",
        }
        if token in {"-f", "--file"}:
            return 1
        if token in options_with_values:
            return 1
        if re.fullmatch(r"-[ABCm]\d+", token):
            return 0
        return 0

    @staticmethod
    def _grep_option_has_embedded_argument(token: str) -> bool:
        options_with_values = {
            "--after-context",
            "--before-context",
            "--binary-files",
            "--context",
            "--devices",
            "--directories",
            "--exclude",
            "--exclude-dir",
            "--exclude-from",
            "--file",
            "--include",
            "--label",
            "--max-count",
        }
        return any(token.startswith(option + "=") for option in options_with_values)

    @staticmethod
    def _grep_known_no_arg_option(token: str) -> bool:
        known_long = {
            "--basic-regexp",
            "--extended-regexp",
            "--fixed-strings",
            "--perl-regexp",
            "--ignore-case",
            "--no-ignore-case",
            "--word-regexp",
            "--line-regexp",
            "--null-data",
            "--no-messages",
            "--invert-match",
            "--version",
            "--help",
            "--quiet",
            "--silent",
            "--files-with-matches",
            "--files-without-match",
            "--count",
            "--line-number",
            "--with-filename",
            "--no-filename",
            "--only-matching",
            "--recursive",
            "--dereference-recursive",
            "--text",
            "--binary",
            "--unix-byte-offsets",
            "--null",
        }
        if token in known_long:
            return True
        if not token.startswith("-") or token.startswith("--") or token == "-":
            return False
        return all(char in "EFGPhiIlnqRrsvwxcHLoUZa" for char in token[1:])

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

    def _negative_path_evidence_findings(
        self,
        step: dict[str, Any],
        results: list[dict[str, Any]],
        evidence_label: str,
    ) -> list[str]:
        if not self._step_requires_negative_path_evidence(step):
            return []
        if any(self._result_proves_negative_path(result) for result in results):
            return []
        return [
            (
                f"{evidence_label} for {step.get('id', 'this step')} has acceptance criteria for an error, "
                "invalid-input, or non-zero/failure path, but the validation evidence only proves the success path. "
                "Add a bounded command with expected_returncode, a generated test/validation script, a copied "
                "temporary workspace, or a wired temporary fixture that the program under validation actually "
                "consumes. Require error-text assertions only when the requirements name the text or a wrapper "
                "hides the child status."
            )
        ]

    def _validation_commands_include_negative_path(self, commands: list[Any], *, step: dict[str, Any] | None = None) -> bool:
        for command in commands:
            if self._command_expected_returncode(command) != 0:
                return True
            argv = self._command_argv_for_static_check(command)
            texts = [" ".join(argv).lower(), json.dumps(command, sort_keys=True).lower()]
            texts.extend(shell_text.lower() for shell_text in self._shell_texts_for_static_check(argv))
            combined = "\n".join(texts)
            markers = (
                "returncode",
                "$?",
                "-ne 0",
                "!= 0",
                "exit 1",
                "if !",
                "pytest.raises",
                "assertraises",
                "assert_raises",
                "raises(",
                "except ",
                "try:",
                "grep -q",
                "error:",
                "invalid",
                "bad_",
                "bad-",
                "wrong",
                "mismatch",
            )
            if any(marker in combined for marker in markers):
                return True
        if step is not None and self._test_runner_covers_declared_negative_path(commands, step):
            return True
        return False

    def _test_runner_covers_declared_negative_path(self, commands: list[Any], step: dict[str, Any]) -> bool:
        if not any(self._command_is_test_runner(command) for command in commands):
            return False
        step_text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", [])),
        ]).lower()
        if not any(marker in step_text for marker in ("test", "tests", "unittest", "pytest")):
            return False
        if self._text_requires_negative_path(step_text):
            return True
        negative_markers = (
            "raises",
            "raise ",
            "valueerror",
            "invalid",
            "non-zero",
            "nonzero",
            "non zero",
            "error path",
            "error handling",
            "failure path",
            "rejects",
            "rejection",
        )
        return any(marker in step_text for marker in negative_markers)

    def _command_is_test_runner(self, command: Any) -> bool:
        argv = [part.lower() for part in self._command_argv_for_static_check(command)]
        if not argv:
            return False
        if "pytest" in argv[0] or any(part in ("pytest", "py.test") for part in argv):
            return True
        return any(part == "unittest" for part in argv)

    def _documentation_content_validation_findings(
        self,
        step: dict[str, Any],
        commands: list[Any],
        source_label: str,
    ) -> list[str]:
        targets = self._documentation_content_targets(step)
        if not targets or self._commands_may_be_dedicated_documentation_validator(commands):
            return []
        missing = [
            target
            for target in targets.values()
            if not self._commands_inspect_documentation_content(
                commands,
                str(target["path"]),
                needs_semantic=bool(target.get("semantic")),
            )
        ]
        if not missing:
            return []
        step_id = str(step.get("id") or "step")
        return [
            (
                f"{source_label} for {step_id} requires content evidence for `{target['path']}` "
                f"({target['reason']}), but validation commands do not inspect that file's content. "
                "Add a bounded content check such as grep/rg against the required section or phrase, "
                "`test -s` for explicit non-empty-only requirements, or a dedicated validation script."
            )
            for target in missing
        ]

    def _documentation_content_evidence_findings(
        self,
        step: dict[str, Any],
        results: list[dict[str, Any]],
        source_label: str,
    ) -> list[str]:
        commands = [result.get("command") or [] for result in results]
        return self._documentation_content_validation_findings(step, commands, source_label)

    def _documentation_content_tool_findings(
        self,
        step: dict[str, Any],
        commands: list[Any],
    ) -> list[dict[str, Any]]:
        targets = self._documentation_content_targets(step)
        if not targets or self._commands_may_be_dedicated_documentation_validator(commands):
            return []
        missing = [
            target
            for target in targets.values()
            if not self._commands_inspect_documentation_content(
                commands,
                str(target["path"]),
                needs_semantic=bool(target.get("semantic")),
            )
        ]
        findings: list[dict[str, Any]] = []
        for target in missing:
            affected = [
                index
                for index, command in enumerate(commands)
                if self._command_mentions_documentation_path(command, str(target["path"]))
            ]
            if not affected:
                continue
            for index in affected:
                findings.append({
                    "index": index,
                    "risk_level": "medium",
                    "reason": (
                        f"The current step requires content evidence for `{target['path']}` "
                        f"({target['reason']}), but the proposed command batch does not inspect that "
                        "file's content. Use grep/rg/assert-style content validation, `test -s` for "
                        "explicit non-empty-only requirements, or a dedicated validation script."
                    ),
                })
        return findings

    def _documentation_content_targets(self, step: dict[str, Any]) -> dict[str, dict[str, Any]]:
        doc_paths = self._documentation_paths_in_text(json.dumps(step, ensure_ascii=False))
        if not doc_paths:
            return {}
        targets: dict[str, dict[str, Any]] = {}
        for path in doc_paths:
            self._merge_documentation_content_target(
                targets,
                path,
                reason="documentation file is a planned deliverable",
                semantic=False,
            )
        criteria = [str(item) for item in step.get("acceptance_criteria", []) or []]
        criteria.append(str(step.get("description", "")))
        for text in criteria:
            mentioned = self._documentation_paths_in_text(text)
            if self._text_has_non_empty_documentation_requirement(text):
                affected = mentioned or doc_paths
                for path in affected:
                    self._merge_documentation_content_target(
                        targets,
                        path,
                        reason="file must be non-empty",
                        semantic=False,
                    )
        return targets

    def _merge_documentation_content_target(
        self,
        targets: dict[str, dict[str, Any]],
        path: str,
        *,
        reason: str,
        semantic: bool,
    ) -> None:
        key = self._normalize_documentation_path(path)
        if not key:
            return
        existing = targets.setdefault(
            key,
            {"path": path.strip("`'\""), "reason": reason, "semantic": False},
        )
        existing["semantic"] = bool(existing.get("semantic")) or semantic
        if semantic:
            existing["reason"] = reason

    @staticmethod
    def _documentation_paths_in_text(text: str) -> list[str]:
        seen: set[str] = set()
        paths: list[str] = []
        for match in re.finditer(r"`?((?:[\w.-]+/)*[\w.-]+\.(?:md|rst|txt))`?", text, flags=re.IGNORECASE):
            path = match.group(1).strip("`'\"")
            if not FeedbackLoopAgent._path_looks_like_documentation_file(path):
                continue
            key = path.replace("\\", "/").lstrip("./").lower()
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
        return paths

    @staticmethod
    def _path_looks_like_documentation_file(path: str) -> bool:
        normalized = path.replace("\\", "/").lstrip("./").lower()
        name = Path(normalized).name
        suffix = Path(name).suffix
        if suffix in {".md", ".markdown", ".rst"}:
            return True
        if suffix != ".txt":
            return False
        stem = Path(name).stem
        doc_tokens = ("readme", "doc", "docs", "note", "notes", "guide", "manual", "changelog")
        return any(token in stem for token in doc_tokens)

    @staticmethod
    def _normalize_documentation_path(path: str) -> str:
        return path.strip("`'\"").replace("\\", "/").lstrip("./").lower()

    @staticmethod
    def _text_has_non_empty_documentation_requirement(text: str) -> bool:
        lower = text.lower()
        return any(marker in lower for marker in ("not empty", "non-empty", "nonempty"))

    def _commands_may_be_dedicated_documentation_validator(self, commands: list[Any]) -> bool:
        return any(self._command_may_be_dedicated_documentation_validator(command) for command in commands)

    def _command_may_be_dedicated_documentation_validator(self, command: Any) -> bool:
        parts = [part.lower() for part in self._command_parts_for_safety(command)]
        if not parts:
            return False
        executable = Path(parts[0]).name
        script_parts = [
            part
            for part in parts[1:]
            if part.endswith((".py", ".sh", ".js", ".mjs", ".cjs"))
        ]
        if any(re.search(r"(?:^|/)(?:validate|check)[\w.-]*\.(?:py|sh|js|mjs|cjs)$", part) for part in script_parts):
            return True
        if any("doc" in part or "readme" in part for part in script_parts):
            return any(marker in " ".join(parts) for marker in ("test", "unittest", "pytest", "validate", "check"))
        if executable in {"pytest", "py.test"}:
            return any("doc" in part or "readme" in part for part in parts[1:])
        if len(parts) >= 3 and executable.endswith("python") and parts[1] == "-m" and parts[2] == "unittest":
            return any("doc" in part or "readme" in part for part in parts[3:])
        return False

    def _commands_inspect_documentation_content(
        self,
        commands: list[Any],
        path: str,
        *,
        needs_semantic: bool,
    ) -> bool:
        return any(
            self._command_inspects_documentation_content(command, path, needs_semantic=needs_semantic)
            for command in commands
        )

    def _command_mentions_documentation_path(self, command: Any, path: str) -> bool:
        normalized = self._normalize_documentation_path(path)
        return any(
            normalized in text.replace("\\", "/").lstrip("./").lower()
            for text in self._command_texts_for_static_check(command)
        )

    def _command_inspects_documentation_content(
        self,
        command: Any,
        path: str,
        *,
        needs_semantic: bool,
    ) -> bool:
        if self._command_may_be_dedicated_documentation_validator(command):
            return True
        return any(
            self._text_inspects_documentation_path(text, path, needs_semantic=needs_semantic)
            for text in self._command_texts_for_static_check(command)
        )

    def _command_texts_for_static_check(self, command: Any) -> list[str]:
        parts = self._command_parts_for_safety(command)
        texts = [json.dumps(command, ensure_ascii=False), " ".join(parts)]
        texts.extend(self._shell_texts_for_static_check(parts))
        return texts

    def _text_inspects_documentation_path(self, text: str, path: str, *, needs_semantic: bool) -> bool:
        lower = text.replace("\\", "/").lower()
        normalized = self._normalize_documentation_path(path)
        if normalized not in lower:
            return False
        quoted_path = rf"['\"]?(?:\./)?{re.escape(normalized)}['\"]?"
        # A grep pattern may legitimately contain `|` for alternation, for example
        # `grep -qE 'Usage|Arguments' README.md`. Treat the path-bearing grep
        # segment as content inspection even when the pattern itself contains `|`.
        grep_or_rg = rf"\b(?:grep|rg)\b[^\n;&]*{quoted_path}"
        python_read = (
            re.search(quoted_path + r"(?:(?![;&|]).){0,260}\b(?:read_text|read_bytes|read\()", lower)
            or re.search(r"\bopen\s*\([^)]*" + quoted_path + r"[^)]*\)", lower)
        )
        python_asserts_content = python_read and any(
            marker in lower for marker in ("assert", "sys.exit", "raise systemexit", " in ", "len(")
        )
        semantic_check = bool(re.search(grep_or_rg, lower)) or bool(python_asserts_content)
        if semantic_check:
            return True
        if needs_semantic:
            return False
        non_empty_patterns = (
            rf"\btest\b(?:(?![;&|]).)*\s-s\s+{quoted_path}",
            rf"\[\s+-s\s+{quoted_path}\s*\]",
            rf"\b(?:wc|stat)\b(?:(?![;&|]).)*{quoted_path}(?:(?![;&|]).)*(?:-gt|>|assert|test\b)",
            rf"\b(?:grep|rg)\b\s+-q\s+['\"]?\.\*?['\"]?\s+{quoted_path}",
        )
        return any(re.search(pattern, lower) for pattern in non_empty_patterns)

    def _step_requires_negative_path_evidence(self, step: dict[str, Any]) -> bool:
        text = " ".join([
            str(step.get("title", "")),
            str(step.get("description", "")),
            " ".join(str(item) for item in step.get("acceptance_criteria", [])),
        ]).lower()
        return self._text_requires_negative_path(text)

    @staticmethod
    def _text_requires_negative_path(text: str) -> bool:
        text = text.lower()
        cli_arity_markers = (
            "exactly one argument",
            "exactly one positional argument",
            "takes exactly one argument",
            "takes exactly one positional argument",
            "take exactly one argument",
            "take exactly one positional argument",
        )
        markers = (
            "non-zero",
            "nonzero",
            "non zero",
            "returns non-zero",
            "raises",
            "raise ",
            "throws",
            "invalid",
            "incorrect",
            "wrong format",
            "wrong count",
            "error message",
            "failure path",
        )
        explicit_nonzero_status = bool(
            re.search(
                r"\b(?:exit|exits|return|returns)\s+(?:with\s+)?(?:code|status|return\s+code)?\s*(?:[1-9]\d*|non[- ]?zero)\b",
                text,
            )
        )
        return (
            explicit_nonzero_status
            or any(marker in text for marker in markers)
            or any(marker in text for marker in cli_arity_markers)
        )

    def _result_proves_negative_path(self, result: dict[str, Any]) -> bool:
        if int(result.get("expected_returncode", 0)) != 0:
            return True
        command_text = " ".join(str(part) for part in (result.get("command") or [])).lower()
        evidence_text = " ".join([
            command_text,
            str(result.get("stdout") or "").lower(),
            str(result.get("stderr") or "").lower(),
        ])
        markers = (
            "returncode",
            "$?",
            "-ne 0",
            "!= 0",
            "nonzero",
            "non-zero",
            "assert_raises",
            "assertraises",
            "pytest.raises",
            "raises(",
            "except ",
            "try:",
            "subprocess.run",
        )
        if any(marker in evidence_text for marker in markers):
            return True
        return any(marker in command_text for marker in ("unittest", "pytest", "test_"))

    def _project_evidence_findings(
        self,
        step_results: list[dict[str, Any]],
        feedback_tool_evidence: dict[str, Any] | None = None,
    ) -> list[str]:
        findings: list[str] = []
        semantic_phrase_checks = self._legacy_semantic_phrase_checks_enabled()
        if semantic_phrase_checks:
            findings.extend(
                self._unrequested_scope_expansion_findings(
                    self.requirements,
                    source_label="Final requirements",
                )
            )
        findings.extend(self._stale_validation_evidence_findings(feedback_tool_evidence or {}))
        findings.extend(self._delayed_resource_validation_findings(feedback_tool_evidence or {}))
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
                        if self._expected_nonzero_returncode_mismatch_is_scope_neutral(step, result):
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
                        if self._expected_nonzero_returncode_mismatch_is_scope_neutral(step, result):
                            continue
                        findings.append(
                            f"Step {step_id} accepted validation returned {result.get('returncode')} "
                            f"but expected {result.get('expected_returncode', 0)}: {result.get('command')}"
                        )
                findings.extend(
                    self._documentation_content_evidence_findings(
                        step,
                        results + accepted_results,
                        f"Step {step_id} final evidence",
                    )
                )
                if semantic_phrase_checks:
                    findings.extend(
                        self._negative_path_evidence_findings(
                            step,
                            results + accepted_results,
                            f"Step {step_id} final evidence",
                        )
                    )
                    findings.extend(
                        self._executable_deliverable_evidence_findings(
                            step,
                            results + accepted_results,
                            feedback_tool_evidence or {},
                            f"Step {step_id} final evidence",
                        )
                    )
                continue
            if not attempts:
                continue
            implementation = attempts[-1].get("implementation", {})
            commands = implementation.get("commands", [])
            step = next((item for item in self.plan_steps if str(item.get("id")) == step_id), {})
            if not commands:
                findings.append(f"Step {step_id} final attempt has no command evidence.")
            for result in commands:
                if result.get("timed_out") or (
                    not self._command_returncode_matches_expected(result)
                    and not self._expected_nonzero_returncode_mismatch_is_scope_neutral(step, result)
                ):
                    findings.append(f"Step {step_id} final attempt has failing evidence: {result.get('command')}")
                findings.extend(self._validation_result_integrity_findings({}, result, f"Step {step_id} final attempt"))
            findings.extend(
                self._documentation_content_evidence_findings(
                    step,
                    commands,
                    f"Step {step_id} final attempt",
                )
            )
            if semantic_phrase_checks:
                findings.extend(
                    self._negative_path_evidence_findings(
                        step,
                        commands,
                        f"Step {step_id} final attempt",
                    )
                )
                findings.extend(
                    self._executable_deliverable_evidence_findings(
                        step,
                        commands,
                        feedback_tool_evidence or {},
                        f"Step {step_id} final attempt",
                    )
                )
        findings.extend(
            self._workspace_reference_findings(
                feedback_tool_evidence or {},
                allow_planned_future_refs=False,
            )
        )
        if semantic_phrase_checks:
            findings.extend(self._public_api_implementation_shape_findings(feedback_tool_evidence or {}))
            findings.extend(self._stdout_json_format_implementation_findings(feedback_tool_evidence or {}))
            findings.extend(self._artifact_only_workspace_findings(feedback_tool_evidence or {}))
        return findings

    def _stdout_json_format_implementation_findings(self, feedback_tool_evidence: dict[str, Any]) -> list[str]:
        """Catch source that pretty-prints JSON to stdout when compact machine output was requested."""
        if not self._prompt_requests_machine_json_stdout():
            return []
        findings: list[str] = []
        for item in feedback_tool_evidence.get("workspace_files", []) or []:
            path = str(item.get("path") or "")
            if not self._workspace_path_looks_like_implementation_source(path):
                continue
            content = str(item.get("content") or "")
            if self._source_pretty_prints_json_stdout(content):
                findings.append(
                    f"{path} appears to pretty-print or indent JSON stdout even though the prompt requested "
                    "machine-readable normalized JSON output without presentation formatting. Use compact "
                    "deterministic JSON for stdout unless the user explicitly asks for pretty output."
                )
            if self._source_uses_default_json_separators_for_stdout(content):
                findings.append(
                    f"{path} appears to emit JSON stdout with Python's default separators even though the "
                    "prompt requests machine-readable compact JSON. Use compact separators such as "
                    "`separators=(',', ':')` for caller-visible stdout unless the user explicitly asks for spaces."
                )
        return findings

    @staticmethod
    def _source_pretty_prints_json_stdout(content: str) -> bool:
        lower = content.lower()
        if "json.dump" not in lower and "json.dumps" not in lower:
            return False
        if "indent" not in lower:
            return False
        stdout_markers = (
            "print(",
            "sys.stdout",
            "stdout",
        )
        return any(marker in lower for marker in stdout_markers)

    @staticmethod
    def _source_uses_default_json_separators_for_stdout(content: str) -> bool:
        if "json.dumps" not in content and "json.dump" not in content:
            return False
        stdout_patterns = (
            r"print\s*\(\s*json\.dumps\s*\(",
            r"sys\.stdout\.write\s*\(\s*json\.dumps\s*\(",
            r"json\.dump\s*\([^,\n]+,\s*sys\.stdout\b",
        )
        if not any(re.search(pattern, content) for pattern in stdout_patterns):
            return False
        compact_markers = (
            "separators=",
            "indent=",
        )
        return not any(marker in content for marker in compact_markers)

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
        if self._accepted_validation_is_same_grep_check_with_ignore_case(plan_result, accepted_results):
            return True
        if self._result_blocked_by_tool_verifier(plan_result):
            return True
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
        if (
            "tool call blocked before execution by verification step" in text
            and any(
                marker in text
                for marker in (
                    "logically flawed",
                    "does not actually verify",
                    "logic error in the validation command",
                    "validation command is flawed",
                    "missing the '--' separator",
                    "missing the -- separator",
                    "end-of-options separator",
                    "will cause it to fail",
                    "stale",
                    "misaligned",
                )
            )
        ):
            return True
        if self._accepted_validation_uses_argument_separator_for_same_command(plan_result, accepted_results):
            return True
        if self._accepted_validation_is_same_doc_grep_check_with_heading_punctuation(plan_result, accepted_results):
            return True
        accepted_command_text = json.dumps(
            [item.get("command") for item in accepted_results],
            ensure_ascii=False,
        ).lower()
        return (
            "returned non-zero exit status 2" in text
            and any(marker in accepted_command_text for marker in (" -- ", "'--'", '"--"'))
        )

    @staticmethod
    def _result_blocked_by_tool_verifier(result: dict[str, Any]) -> bool:
        """Return True when a command never ran because tool verification blocked it."""
        if result.get("blocked_by_tool_verifier"):
            return True
        try:
            returncode = int(result.get("returncode", 0))
        except (TypeError, ValueError):
            returncode = 0
        if returncode != 126:
            return False
        text = f"{result.get('stdout') or ''}\n{result.get('stderr') or ''}".lower()
        return "tool call blocked before execution by verification step" in text

    def _accepted_validation_uses_argument_separator_for_same_command(
        self,
        plan_result: dict[str, Any],
        accepted_results: list[dict[str, Any]],
    ) -> bool:
        stderr = str(plan_result.get("stderr") or "").lower()
        stdout = str(plan_result.get("stdout") or "").lower()
        if not any(
            marker in f"{stdout}\n{stderr}"
            for marker in (
                "the following arguments are required",
                "unrecognized arguments",
                "unrecognized option",
                "unknown option",
            )
        ):
            return False
        plan_tokens = self._validation_command_tokens_for_similarity(plan_result.get("command"))
        if not plan_tokens or "--" in plan_tokens:
            return False
        if not any(str(token).startswith("-") and token != "-" for token in plan_tokens):
            return False
        for accepted in accepted_results:
            accepted_tokens = self._validation_command_tokens_for_similarity(accepted.get("command"))
            if not accepted_tokens or "--" not in accepted_tokens:
                continue
            without_separator = [token for token in accepted_tokens if token != "--"]
            if without_separator == plan_tokens:
                return True
        return False

    def _accepted_validation_is_same_doc_grep_check_with_heading_punctuation(
        self,
        plan_result: dict[str, Any],
        accepted_results: list[dict[str, Any]],
    ) -> bool:
        plan_pairs = self._grep_pattern_file_pairs(plan_result.get("command"))
        if not plan_pairs or not all(self._grep_pair_path_is_doc(path) for _pattern, path in plan_pairs):
            return False
        normalized_plan = {
            (self._normalize_doc_grep_pattern(pattern), _normalize_workspace_path_text(path).lower())
            for pattern, path in plan_pairs
        }
        if any(not pattern for pattern, _path in normalized_plan):
            return False
        for accepted in accepted_results:
            accepted_pairs = self._grep_pattern_file_pairs(accepted.get("command"))
            if not accepted_pairs:
                continue
            normalized_accepted = {
                (self._normalize_doc_grep_pattern(pattern), _normalize_workspace_path_text(path).lower())
                for pattern, path in accepted_pairs
                if self._grep_pair_path_is_doc(path)
            }
            if normalized_plan <= normalized_accepted:
                return True
        return False

    def _grep_pattern_file_pairs(self, command: Any) -> list[tuple[str, str]]:
        tokens = self._validation_command_tokens_for_similarity(command)
        pairs: list[tuple[str, str]] = []
        index = 0
        while index < len(tokens):
            if tokens[index] != "grep":
                index += 1
                continue
            index += 1
            while index < len(tokens):
                option = tokens[index]
                if option in {"&&", "||", ";", "|"}:
                    break
                if option == "--":
                    index += 1
                    break
                if not option.startswith("-") or option == "-":
                    break
                index += 1
            if index + 1 < len(tokens):
                pairs.append((tokens[index], tokens[index + 1]))
                index += 2
                continue
            break
        return pairs

    @staticmethod
    def _grep_pair_path_is_doc(path: str) -> bool:
        normalized = _normalize_workspace_path_text(path).lower()
        return normalized.endswith((".md", ".markdown", ".txt")) or Path(normalized).name.startswith("readme")

    @staticmethod
    def _normalize_doc_grep_pattern(pattern: str) -> str:
        text = str(pattern).strip().strip('"').strip("'").lower()
        return re.sub(r"[:：]+$", "", text)

    def _accepted_validation_is_same_grep_check_with_ignore_case(
        self,
        plan_result: dict[str, Any],
        accepted_results: list[dict[str, Any]],
    ) -> bool:
        plan_normalized = self._normalize_grep_command_for_case_equivalence(plan_result.get("command"))
        if not plan_normalized:
            return False
        plan_text, plan_removed_ignore_case = plan_normalized
        for accepted in accepted_results:
            accepted_normalized = self._normalize_grep_command_for_case_equivalence(accepted.get("command"))
            if not accepted_normalized:
                continue
            accepted_text, accepted_removed_ignore_case = accepted_normalized
            if accepted_text == plan_text and accepted_removed_ignore_case and not plan_removed_ignore_case:
                return True
        return False

    def _normalize_grep_command_for_case_equivalence(self, command: Any) -> tuple[str, bool] | None:
        """Normalize a grep-only validation command while ignoring `grep -i`.

        This is intentionally narrow. It lets a fresh accepted validator such as
        `grep -qi usage README.md` supersede a stale case-sensitive plan command
        `grep -q usage README.md`, without allowing arbitrary weaker checks to
        bypass a failed reviewer-owned validation command.
        """
        command_text = self._validation_command_text_for_similarity(command)
        if not command_text:
            return None
        try:
            tokens = shlex.split(command_text)
        except ValueError:
            return None
        if not tokens:
            return None
        normalized: list[str] = []
        saw_grep = False
        removed_ignore_case = False
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token != "grep":
                normalized.append(token)
                index += 1
                continue
            saw_grep = True
            normalized.append(token)
            index += 1
            while index < len(tokens):
                option = tokens[index]
                if option in {"&&", "||", ";", "--", "-"} or not option.startswith("-"):
                    break
                if option == "--ignore-case":
                    removed_ignore_case = True
                    index += 1
                    continue
                if option.startswith("--"):
                    normalized.append(option)
                    index += 1
                    continue
                flags = option[1:]
                flags_without_ignore_case = flags.replace("i", "")
                if flags_without_ignore_case != flags:
                    removed_ignore_case = True
                if flags_without_ignore_case:
                    normalized.append("-" + flags_without_ignore_case)
                index += 1
        if not saw_grep:
            return None
        return (" ".join(normalized), removed_ignore_case)

    def _validation_command_text_for_similarity(self, command: Any) -> str:
        if isinstance(command, dict):
            command = command.get("cmd") or command.get("command")
        if isinstance(command, list):
            parts = [str(part) for part in command]
            if len(parts) >= 3 and parts[0] in {"bash", "sh"} and parts[1] in {"-c", "-lc"}:
                return parts[2]
            return " ".join(shlex.quote(part) for part in parts)
        return str(command or "")

    def _validation_command_tokens_for_similarity(self, command: Any) -> list[str]:
        text = self._validation_command_text_for_similarity(command)
        if not text:
            return []
        try:
            return shlex.split(text)
        except ValueError:
            return []

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
                "Please revise the plan validation command: reviewer-owned evidence shows the command is malformed, stale, or misaligned."
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
        for key in (
            "required_changes",
            "cross_check_questions",
            "verification_evidence",
            "evidence_reviewed",
            "runbook_updates",
        ):
            review[key] = self._as_list_field(review.get(key))
        return review

    def _suppress_unsupported_validation_syntax_objection(
        self,
        review: dict[str, Any],
        *,
        scope: str,
    ) -> dict[str, Any]:
        """Ignore syntax-only reviewer objections contradicted by deterministic checks.

        The model reviewer may still reject weak semantic validation. What it
        must not do is invent Python/shell syntax failures after the deterministic
        command checks found none; that creates tunnel-vision loops where valid
        validators are repeatedly rewritten for no evidence-backed reason.
        """
        if self._status(review) not in {"needs_rework", "needs_plan_change", "needs_requirements_change"}:
            return review
        summary = str(review.get("summary") or "")
        changes = "\n".join(str(item) for item in review.get("required_changes", []) or [])
        text = f"{summary}\n{changes}".lower()
        syntax_markers = (
            "python -c",
            "syntax",
            "compound statement",
            "generator expression",
            "comprehension",
            "semicolon",
            "too complex",
            "parser",
            "single-line expression",
        )
        if not any(marker in text for marker in syntax_markers):
            return review
        semantic_or_scope_markers = (
            "does not compare",
            "not compare",
            "missing",
            "no semantic",
            "not semantic",
            "does not perform",
            "does not verify",
            "only checks",
            "file existence",
            "wrong",
            "unsafe",
            "destructive",
            "not bounded",
            "starts a server",
        )
        if any(marker in text for marker in semantic_or_scope_markers):
            return review

        suppressed = dict(review)
        suppressed["status"] = "resolved"
        suppressed["needs_rework"] = False
        suppressed["summary"] = (
            f"{scope} accepted: deterministic command checks found no syntax issue, "
            "so a syntax-only reviewer objection was ignored."
        )
        suppressed["required_changes"] = []
        suppressed["suppressed_reviewer_findings"] = [
            {
                "reason": "unsupported_validation_syntax_objection",
                "original_status": review.get("status"),
                "original_summary": review.get("summary"),
                "original_required_changes": review.get("required_changes", []),
            }
        ]
        return suppressed

    def _suppress_unsupported_negative_path_shell_objection(
        self,
        review: dict[str, Any],
        *,
        feedback_tool_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Ignore a narrow false objection to the standard expected-failure shell wrapper.

        Small reviewers sometimes misread ``cmd && exit 1 || exit 0`` and claim
        it fails when ``cmd`` exits non-zero. In shell control flow, the opposite
        is true: the right-hand ``exit 0`` runs when the command under test
        fails, so the wrapper is a valid way to prove an expected failure. Only
        suppress this objection when reviewer-owned validation actually passed.
        """
        if self._status(review) not in {"needs_rework", "needs_plan_change", "needs_requirements_change"}:
            return review
        summary = str(review.get("summary") or "")
        changes = "\n".join(str(item) for item in review.get("required_changes", []) or [])
        questions = "\n".join(str(item) for item in review.get("cross_check_questions", []) or [])
        evidence = "\n".join(str(item) for item in review.get("verification_evidence", []) or [])
        text = f"{summary}\n{changes}\n{questions}\n{evidence}".lower()
        shell_wrapper_markers = (
            "&& exit 1 || exit 0",
            "exit 1 || exit 0",
            "subshell logic",
            "non-zero exit status",
            "non-zero exit requirement",
        )
        false_objection_markers = (
            "logically flawed",
            "contradictory",
            "should have failed",
            "actually return 0",
            "returns 1 if",
            "fix the validation command",
        )
        if not any(marker in text for marker in shell_wrapper_markers):
            return review
        if not any(marker in text for marker in false_objection_markers):
            return review

        results = feedback_tool_evidence.get("validation_results") or []
        matching_results = [
            result
            for result in results
            if self._command_returncode_matches_expected(result)
            and not result.get("timed_out")
            and "&& exit 1 || exit 0" in " ".join(str(part) for part in (result.get("command") or []))
        ]
        if not matching_results:
            return review

        suppressed = dict(review)
        suppressed["status"] = "resolved"
        suppressed["needs_rework"] = False
        suppressed["summary"] = (
            "Step accepted: reviewer-owned validation passed, and the expected-failure "
            "shell wrapper objection contradicted normal shell control flow."
        )
        suppressed["required_changes"] = []
        suppressed["suppressed_reviewer_findings"] = [
            {
                "reason": "unsupported_negative_path_shell_objection",
                "original_status": review.get("status"),
                "original_summary": review.get("summary"),
                "original_required_changes": review.get("required_changes", []),
            }
        ]
        return suppressed

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
            iterations = (final_review or {}).get("iterations") or []
            last_review = iterations[-1].get("review", {}) if iterations else {}
            if not last_review.get("deterministic_evidence_findings"):
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
