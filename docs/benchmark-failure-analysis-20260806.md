# Benchmark Failure Analysis - 2026-08-06 Model Refresh

This note records failure and odd-behavior findings from the 2026-08-06
publication refresh. It is intentionally separate from the README so the README
can stay readable while preserving enough detail for the next harness fix pass.

## Evidence

- Refreshed zero-shot baselines:
  `runs/publication-20260806-model-refresh/<profile>/zero-shot/results.json`
- New full-harness runs:
  `runs/publication-20260806-model-refresh/{devstral-small-2512,qwen36-fable-fusion-mtp,kat-coder-v2.5-dev}/harness/results.json`
- Incomplete Qwythos harness run:
  `runs/publication-20260806-model-refresh/qwythos-27b-mtp/harness/INCOMPLETE.md`
- Older full-harness comparison matrix:
  `runs/publication-20260801-full-matrix/<profile>/harness/results.json`

Completed new harness results:

| Profile | Harness result | Harness time | Notes |
|---|---:|---:|---|
| `qwen36-fable-fusion-mtp` | 28/30 | 15.60h | Highest new pass rate, very expensive. |
| `kat-coder-v2.5-dev` | 22/30 | 4.95h | Large harness lift, many protocol repairs. |
| `devstral-small-2512` | 2/30 | 4.87h | Major regression versus zero-shot. |
| `qwythos-27b-mtp` | incomplete | >64m on task 1 | Repeated `S1` repair attempts without a terminal row. |

## Per-Profile Findings

### `devstral-small-2512`

Zero-shot passed 18/30, but the harness passed only 2/30. This is a real
regression, not just a time-cost issue.

Observed signals:

- 242 `JSON_REPAIR` or `MINIMAL_JSON_REPAIR` phases across 30 task logs.
- 282 critical phases despite a very low final pass rate.
- `algo-002-nested-parity` failed after 51.3m and reached attempt 7.
- `hist-006-dotnet-dependency` failed after 45.5m.
- `hist-003-real-existing-invoice-bugfix` failed after 19.3m and reached
  attempt 19 in the log scan.

Likely root cause:

Devstral Small 2 can answer many tasks in zero-shot, but under the harness it
often fails the control protocol, then spends most of its budget repairing
protocol shape instead of solving the user task. The prompt burden and JSON
contract appear too heavy for this profile in full-harness mode.

Minimal generic fix direction:

- Add a profile capability flag or benchmark profile that can reduce protocol
  depth for non-reasoning instruction models.
- Keep the machine protocol exact, but reduce how often full critical review is
  invoked after repeated protocol-shape failures.
- Do not special-case Devstral tasks or answers.

### `qwen36-fable-fusion-mtp`

Zero-shot passed 24/30. The harness passed 28/30, failing only:

- `tool-002-log-watch`
- `hist-006-dotnet-dependency`

Observed signals:

- Only 9 JSON repair phases across 30 tasks.
- 921 critical phases across 30 tasks.
- Several successful tasks were very expensive:
  - `web-002-browser-interaction`: pass in 99.5m
  - `hist-002-real-jsonl-stats`: pass in 88.1m
  - `algo-001-balanced-grid`: pass in 64.3m
  - `hist-003-real-existing-invoice-bugfix`: pass in 64.0m
- `hist-006-dotnet-dependency` failed after 53.8m with final and approach
  statuses recorded as `cannot_resolve`.

Likely root cause:

This profile mostly follows protocol and benefits from repair, but the harness
uses high-cost critical review very often. Some long successes justify the
repair mechanism, but the run also shows repeated deep attempts where the
review loop may be spending more than the task deserves.

Minimal generic fix direction:

- Keep the repair loop for capable models, because it is clearly improving pass
  count.
- Add a progress-yield gate: after repeated attempts on the same step, require a
  short decision on whether there is new evidence, a changed strategy, or only
  repetition.
- Escalate to the larger critical thinking budget only when the previous attempt
  produced concrete failing evidence or the next tool call is risky.

### `kat-coder-v2.5-dev`

Zero-shot passed 7/30. The harness passed 22/30, a large improvement.

Harness failures:

- `algo-003-multiset-path`
- `algo-005-state-machine`
- `code-003-interval-merge`
- `code-005-existing-bugfix`
- `tool-001-disk-monitor`
- `workflow-002-autonomous-repair`
- `planning-002-plan-update`
- `hist-006-dotnet-dependency`

Observed signals:

- 76 JSON repair phases and 330 critical phases across 30 tasks.
- `code-004-config-normalizer` passed after 27.6m and reached attempt 7.
- `hist-006-dotnet-dependency` failed after 23.7m, still in repeated `S1`
  repair, and reached attempt 7.
- `hist-002-real-jsonl-stats` passed after 22.5m.
- `data-001-csv-window` and `data-002-dedupe` both passed, but each took about
  18m after repeated step repairs.

Likely root cause:

The harness strongly helps KAT, but KAT is protocol-fragile. It often produces
useful work only after JSON repair and critical review. The weak point is not
one benchmark task; it is repeated expensive repair without an early question:
"What changed since the last attempt?"

Minimal generic fix direction:

- Add a harness-owned attempt summary after repeated failures of the same step:
  last command/tool result, last reviewer objection, what changed, and what
  remains unresolved.
- Ask the model to choose a next route from a small protocol set:
  continue with a changed tactic, update the plan, accept a documented
  compromise, or fail with evidence.
- Avoid phrase-list parsing. If the answer does not match the protocol, ask the
  same contextual question again.

### `qwythos-27b-mtp`

Zero-shot passed 20/30, but the full-harness run did not complete task 1.

Observed state when stopped:

- Active task: `algo-001-balanced-grid`
- Elapsed time: more than 64 minutes.
- No `results.json` row had been written.
- The model was still repeating `IMPLEMENT_PLAN_STEP_PHASE step_id=S1` critical
  attempts.
- Last observed attempt: `S1 attempt=7/critical`.
- Server health checks stayed OK (`http=200`).
- The log already had 4 JSON repair phases and 71 critical-phase markers from a
  single task.

Likely root cause:

This is a harness-control problem exposed by a slow model. The workflow did not
have an effective low-yield repair ceiling for a single step. Because the model
remained responsive, ordinary process health checks were not enough to stop the
run. This is distinct from a hard timeout: the problem is repeated repair
without meaningful progress.

Minimal generic fix direction:

- Add a model-mediated progress supervisor for repeated same-step repairs. The
  supervisor should see compact history, the repeated objections, and the last
  concrete evidence, then decide whether continued repair is useful.
- The decision should be harness-owned state after validation. The model should
  not mark work complete by editing plan text.
- The supervisor should not use task-specific thresholds or benchmark answers.

## Cross-Cutting Root Causes

### 1. Low-yield same-step repair loops

The strongest pattern is repeated attempts on one step after repeated review
failure. Qwythos reached `S1` attempt 7 on task 1 before any grade. KAT reached
attempt 7 on several tasks. Fable reached attempt 7 on long successful tasks.

This is useful when attempts are genuinely changing strategy, but harmful when
the same correction cycle repeats. The harness needs to distinguish progress
from activity.

### 2. Critical review is too frequent for some profiles

Critical phases are expensive:

- Fable: 921 critical markers across 30 logs.
- KAT: 330 critical markers across 30 logs.
- Devstral Small 2: 282 critical markers across 30 logs despite 2/30 pass rate.
- Qwythos: 71 critical markers in one incomplete task.

The critical budget should be reserved for high-value moments: risky tool
calls, concrete failing test evidence, unresolved plan conflicts, or an explicit
decision to reframe.

### 3. Protocol repair can dominate task solving

Devstral Small 2 and KAT both produced many JSON repairs. Protocol repair is
needed, but too many repair turns can bury the actual task under control chatter.
The harness should keep the protocol exact and small, then ask again
conversationally when the model misses it.

### 4. Health checks are not progress checks

Qwythos had healthy HTTP responses while making no useful benchmark progress.
Process liveness is not enough. The harness needs a separate progress question:
is this path still improving the solution, or only repeating the same loop?

### 5. Strong models still benefit from the harness

The failures should not hide the positive evidence. KAT improved from 7/30
zero-shot to 22/30 harness. Fable improved from 24/30 to 28/30. Older matrix
rows show large gains for Qwen3.6 27B MTP, Qwopus, Gemma 26B, and
Qwen3-Coder-Next. The right fix is not to remove repair; it is to make repair
more selective and better supervised.

## Recommended Next Fixes

These are intentionally generic and do not encode benchmark answers.

1. Add same-step progress supervision.

   After repeated failed reviews for the same step, summarize the last evidence,
   last objection, and changed tactic. Ask the model to select one validated
   route: continue with a materially different tactic, update the plan, accept a
   documented compromise, or fail with evidence.

2. Gate critical-budget escalation.

   Use the larger critical thinking budget only after concrete failed evidence,
   risky tool calls, unresolved plan conflicts, or a supervisor decision that a
   deeper review is warranted.

3. Preserve compact evidence for progress decisions.

   The progress supervisor must see enough recent tool output and reviewer
   objections to decide whether anything changed. Do not provide whole logs;
   provide bounded evidence plus raw-log paths.

4. Fail fast on repeated protocol-shape failure within the same phase.

   If a model repeatedly misses the same small protocol, ask the same question
   again with the expected schema. After repeated misses, record a
   harness-owned protocol failure and move to a validated fallback rather than
   spending unbounded turns.

5. Keep profile-specific tuning limited to model behavior, not task content.

   It is acceptable to use lower protocol depth or different budgets for a
   model profile. It is not acceptable to add task-specific hints, benchmark
   answer logic, or phrase-list intent inference.
