# Deferred Validation: Cautious Harness Fixes

Date: 2026-08-22

This change set was prepared from the August publication logs. It is intentionally
limited to generic control-flow and prompt defects with direct evidence. No local
model task, calibration run, or publication benchmark has been executed for it.
The worktree must remain uncommitted until the validation below is complete.

## Changes To Validate

### Progress review can end a successful observation

- `TOOL_PROGRESS_REVIEW_PHASE` now has three decisions: `continue`,
  `stop_satisfied`, and `terminate`.
- `stop_satisfied` ends a running process only when its intended observation is
  already present. It records `ended_by_progress_review=true` and
  `satisfied_by_progress_review=true`; it is not recorded as the existing
  unsuccessful `stopped_by_progress_review` outcome.
- The process exit remains the synthetic boundary code 125 for audit clarity.
  Command evidence separately records that the expected outcome was satisfied,
  so step and final reviewers still see that the process did not exit naturally.
- `terminate` and malformed-review fallback behavior are unchanged: unsafe or
  hopeless work fails, while an unparseable progress decision conservatively
  continues.

### Workspace and retained-path instructions are explicit

- Planning and review payloads now expose the exact command cwd and say that
  model-authored project paths are relative to it.
- Requirements now state that a deliverable list is not an exclusive path
  inventory. `restrict` is appropriate only when an explicit request or
  workspace rule forbids additional retained paths; otherwise the model should
  choose `allow`.
- The valid-looking `validation_method` prose placeholder was replaced with an
  empty string so it cannot be copied into an accepted plan as fake evidence.

### Rejected protocol output is audit-only

- Any response rejected by a phase contract is replaced in active model context
  by a harness-owned omission receipt. The raw response remains in the append-only
  full transcript.
- A bounded tail may still be included in the immediate format-repair prompt.
  Rejected first and second repair responses are also removed from later semantic
  context, preventing a reviewer from accepting the rejected object while the
  harness validates a fallback object.

### Small review and plan-state clarifications

- Evidence guidance now forbids claiming that a command ran or passed when no
  matching supplied result exists.
- Rewording only a plan step title no longer reopens resolved work. Changes to
  descriptions, dependencies, paths, acceptance criteria, or validation still
  reset the step to pending.

## Deliberately Deferred

These findings require broader state or protocol redesign and are not part of
this low-risk pass:

- evidence-ID enforcement and persistent evidence obligations after a validator
  is removed;
- patch-based plan refinement and explicit invalidation of resolved steps;
- cross-scope no-progress detection for plan and requirements loops;
- component-budgeted context compaction and plan-note summarization;
- phase-specific critical reasoning budgets and a separate decision-only turn;
- tool-verification result caching or a smaller status vocabulary.

## Required Unit And Integration Checks

1. Run `python3 -m unittest discover -s tests` and require all tests to pass.
2. Confirm an open-ended command that prints a requested marker and then waits is
   ended by `stop_satisfied` after one review, retains its stdout, is not reported
   as `stopped_by_progress_review`, and creates no deterministic evidence failure.
3. Confirm `terminate` still produces code 125, an unsuccessful stop marker, and
   a deterministic validation finding.
4. Confirm a malformed or missing progress decision continues with explicit
   protocol-error provenance and a bounded next-review interval.
5. Confirm a hard timeout racing a progress response cannot be converted into a
   satisfied result.
6. Confirm a fenced but otherwise useful JSON response is present in
   `conversation.full.jsonl`, absent from active `conversation.jsonl`, and can be
   repaired conversationally without inferred phrase matching.
7. Confirm a title-only plan refinement preserves `resolved`, while changing a
   description, criterion, path, dependency, or validator resets it to `pending`.
8. Confirm execution-environment payloads use `/workspace/project` in the normal
   agent container and that generated plans use relative paths from that cwd.
9. Confirm a request for a script plus a runtime log does not infer an exclusive
   one-file final state. Separately confirm an explicit "only this path" request
   still produces and enforces `restrict`.
10. Build the agent image and run a smoke invocation through the documented
    model-server plus agent-container topology.

## Later Model Calibration

Use five new tasks disjoint from `publication-30`, one model at a time:

1. A no-deadline observer that emits decisive output and then continues emitting
   heartbeats. It should choose `stop_satisfied`, not loop indefinitely.
2. A command waiting for unavailable interactive input. It should choose
   `terminate` and repair the plan rather than accepting the command.
3. A small multi-file task where requested deliverables need a conventional
   supporting file. It should not invent an exclusive path boundary.
4. A task with an explicit exact final-path inventory. The restriction must
   remain enforced.
5. A malformed structured-response scenario followed by a substantive repair.
   Rejected content must not become accepted control state.

For each task inspect the full transcript, active compacted transcript, runbook,
tool results, progress-review decisions, and external grade. Test at least one
strong 27B profile and one protocol-fragile smaller profile. Only generic defects
seen across these disjoint cases may justify another edit.

After calibration, freeze source and run the full publication suite without code
changes. Compare pass count, protocol-error count, progress-review count, task
hours, repeated-step count, and path-policy failures against the August matrix.

## Build-Only Checks For This Pass

- `python3 -m compileall -q feedback_agent scripts tests`: passed.
- `git diff --check`: passed.
- `docker build -t agentic-feedback:review-20260822 .`: passed; local image
  manifest list `sha256:cff022b2c35754d4d5470060af7589ca72524f4588aa0247d8539f2949a73512`.
- Unit tests, model calls, harness tasks, calibration, and publication benchmarks:
  deliberately not run in this pass.

## Validation Outcome

The deferred unit and live integration checks were completed on 2026-08-24.
Results, model matrix, raw-evidence locations, fixes discovered during live
testing, and remaining limits are recorded in
`docs/context-compaction-validation-20260824.md` and
`docs/non-compaction-validation-20260825.md`. The publication benchmark was not
rerun and no commit was created.
