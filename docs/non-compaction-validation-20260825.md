# Non-Compaction Harness Validation

Date: 2026-08-25

Status: completed on the uncommitted development worktree. This is calibration
evidence, not a publication benchmark. No benchmark answer or task-specific
solver behavior was added to the harness.

## Scope

This exercise completed the non-compaction checks deferred in
`docs/deferred-validation-20260822.md`. It focused on:

- progress review ending a successful open-ended observation;
- progress review terminating an unsuccessful blocked-input command;
- permissive and exact retained-path policies;
- audit-only handling and conversational repair of malformed model output;
- plan-state preservation and reviewer evidence visibility;
- the documented two-container topology, with one local model server and one
  disposable harness container at a time.

The strong local profiles used for cross-model behavior were
`gemma4-31b-qat-mtp` and `qwen3.8-27b`. The models were never resident
simultaneously.

## Final Current-Build Runs

These runs used agent image `agentic-feedback-coding:local`, manifest list
`sha256:7e30bdadd0642969ec11df0e4aad8a5de8890cf1837d049d95fac26da7ced408`.
The source was frozen during every listed run.

| Model | Task | Grade | Seconds | Attempts | Key control evidence |
|---|---|---:|---:|---:|---|
| Gemma 31B | exact final inventory | pass | 1095.3 | 1 + 1 | `restrict`; only `result.txt` retained |
| Gemma 31B | successful open-ended observation | pass | 1052.4 | 1 + 1 | one command run; one `stop_satisfied` review |
| Gemma 31B | unsuccessful blocked input | pass | 2262.4 | 1 + 2 | one terminated run; no satisfied result; plan repaired |
| Qwen 3.8 27B | successful open-ended observation | pass | 2735.9 | 1 + 1 | one command run; one `stop_satisfied` review |

Raw results:

- `runs/deferred-validation-20260824/gemma31-fix4/results.json`
- `runs/deferred-validation-20260824/gemma31-fix5/results.json`
- `runs/deferred-validation-20260824/qwen38-fix4/results.json`

Additional retained-path and protocol coverage ran immediately before the final
narrow review-evidence change. That change did not alter path enforcement,
response omission, or command execution:

| Model | Task | Grade | Seconds | Attempts | Evidence |
|---|---|---:|---:|---:|---|
| Gemma 31B | supporting artifacts allowed | pass | 2107.9 | 1 + 1 + 1 | no write failure or plan repair |
| Qwen 3.8 27B | supporting artifacts allowed | pass | 4808.2 | 1 + 1 + 1 + 1 | no write failure or protocol repair |
| Qwen 3.8 27B | exact final inventory | pass | 1370.5 | 1 + 1 | external inventory grader passed |
| Qwen 3.8 27B | unsuccessful blocked input | pass | 5839.3 | 1 + 2 | terminated result retained |

Raw results:

- `runs/deferred-validation-20260824/gemma31-fix3/results.json`
- `runs/deferred-validation-20260824/qwen38-fix3/results.json`
- `runs/deferred-validation-20260824/qwen38/results.json`

## Failure Found

The first Gemma rerun after the replay fix exposed a generic, high-impact review
defect:

1. The accepted requirements explicitly required `["bash",
   "signal_watch.sh"]` with `timeout_seconds=0`.
2. The embedded plan replaced it with `timeout 3s bash signal_watch.sh | ...`.
3. Requirements review, plan review, lifecycle review, tool review, both step
   reviews, final review, and approach review all accepted the substitute.
4. The harness returned `resolved`, but the external grader failed because no
   progress-reviewed command had run.

Evidence:

- result: `runs/deferred-validation-20260824/gemma31-fix3/results.json`;
- workspace:
  `workspaces/benchmarks/20260825T034303Z/harness/deferred-progress-stop-satisfied`;
- append-only transcript:
  `.agent_state/conversation.full.jsonl` under that workspace;
- accepted contradictory state: `.agent_state/summary.json` and `PLAN.md` under
  that workspace.

The root cause was not missing original-request text. The request and refined
requirement were present. The small reviewer followed repeated accepted claims,
while compact final-review payloads reduced successful command evidence to
counts and represented skipped final-state commands only by counts. The exact
substitute command remained mainly inside a longer runbook artifact.

## Generic Corrections

The correction is deliberately domain-neutral:

1. The central original-request fit check now states that an explicitly
   prescribed invocation, sequence, timeout behavior, or verification process
   is a requirement. A supplemental check may coexist but cannot replace it.
2. The existing critical plan lifecycle review first checks those explicit
   process constraints, then checks final replay lifecycle. No new phase or
   task-specific parser was added.
3. Compact final-review plans now include bounded validation command specs, not
   only command counts.
4. Compact step results now include bounded successful command results, with
   explicit total and omitted counts.
5. Compact final evidence now includes declared commands even when final replay
   correctly skips them because `final_state=false`.

The corrected Gemma run then planned and executed the exact no-deadline command,
received one `stop_satisfied` decision, skipped final replay, and passed the
same external grader. Qwen followed the same lifecycle and also passed.

## Protocol Evidence

Qwen produced two reasoning-only/off-contract reviewer turns in the final
current-build run. For each event:

- the raw response remains in `conversation.full.jsonl`;
- active context contains a `HARNESS_RESPONSE_OMISSION` instead of the rejected
  model speech;
- the bounded repair request asks the same contextual question again;
- a validated receipt records the repaired decision;
- no phrase-list intent inference or synthesized model answer is used.

The Qwen active transcript has two omission receipts and no rejected reasoning
turn. Both steps resolved on their first implementation attempt.

## Automated Verification

- `python -m unittest tests.test_feedback_agent`: 556 tests passed in 123.328s.
- `python -m unittest discover -s tests`: 566 tests passed in 122.380s.
- Focused compact-evidence and lifecycle tests: 4 passed in 0.731s.
- `python -m compileall -q feedback_agent scripts tests`: passed.
- `git diff --check`: passed.
- Agent Docker image build: passed.
- Every live task above used an external post-validation grader.
- Successful-observation final runs each recorded exactly one progress-review
  request and one actual no-deadline command result.
- The unsuccessful-observation final run recorded one progress review,
  `ended_by_progress_review=true`, `satisfied_by_progress_review=false`,
  `stopped_by_progress_review=true`, and no satisfied result anywhere in the
  summary.

## Remaining Limits

- Qwen occasionally emits all useful content inside visible reasoning tags.
  Conversational repair worked in every observed case, so no model-specific
  parsing was added.
- Critical tool-verification and final-review turns dominate elapsed time. This
  is the configured quality tradeoff rather than repair looping.
- Qwen approved one cheap final replay of `cat README.md`. Avoiding all such
  harmless rechecks would require broader verifier-decision caching and is not
  justified by this evidence.
- A local model can still make a weak task decision. The harness now keeps the
  relevant original constraint and exact command evidence close to reviewers,
  but it does not encode task semantics or benchmark answers.
