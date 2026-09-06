# Context Compaction Validation

Date: 2026-08-24

This report validates context compaction in isolation and through the normal
model-server plus agent-container workflow. It is development evidence, not a
publication benchmark and not evidence that the local models solve every task.
No commit was created.

## Scope

- Production compaction path: `maybe_compact`, stage selection, summary repair,
  active-context assembly, and audit receipts.
- Exact states: 240 first-compaction states extracted from prior benchmark
  transcripts.
- Stress states: long initial requests at roughly 8k, 24k, and 36k tokens;
  provenance conflicts; repeated compaction; and bulky unvalidated noise.
- Models, one at a time: Gemma 4 26B A4B QAT MTP, Gemma 4 31B QAT MTP,
  Qwen3.8 27B, and Devstral Small 2 2512.
- Semantic audit: one stress case from each category for each model, reviewed
  from the complete assembled active context by a separate Devstral invocation.

The exact corpus is
`runs/context-compaction-20260822/final-v2/corpus.jsonl.gz`. The 300-case stress
corpus is `runs/context-compaction-20260823/stress-300.jsonl.gz`. Final raw
results and summaries are under `runs/context-compaction-20260823/final/`.

## Production-Flow Matrix

Each model ran 240 exact states and 60 balanced stress states, for 1,200 local
model compaction cases in total.

| Model | Exact proxy pass | Stress proxy pass | Exact mean score | Stress mean score | Repairs | Errors/fallbacks | Fit/request loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B A4B QAT MTP | 237/240 | 47/60 | 0.812 | 0.757 | 42 | 0/0 | 0/0 |
| Gemma 4 31B QAT MTP | 238/240 | 50/60 | 0.817 | 0.769 | 10 | 0/0 | 0/0 |
| Devstral Small 2 2512 | 240/240 | 59/60 | 0.837 | 0.810 | 1 | 0/0 | 0/0 |
| Qwen3.8 27B | 239/240 | 56/60 | 0.840 | 0.809 | 44 | 0/0 | 0/0 |

`Fit/request loss` reports post-compaction context-fit failures followed by
authoritative initial-request preservation failures. All 1,200 calls completed;
no deterministic fallback was needed. Exact cases selected the conservative
stage. Across the 240 stress cases, stage selection was 200 conservative, 16
broad, and 24 emergency.

The lexical/structural proxy is intentionally conservative and is not a task
correctness score. The 24-case semantic audit passed 24/24, with mean scores of
92.5 for Gemma 26, 90.8 for Gemma 31, 89.2 for Devstral, and 92.0 for Qwen3.8.
All judge responses initially used fenced JSON and recovered through the strict
conversational protocol repair. The Devstral candidate subset was judged by a
separate invocation of the same model family, so that subset is not an
independent cross-model judgment.

## Live Integration

All live scenarios used the documented `agentic-feedback-net` topology with
`agentic-devstral-small-2512-server` and a separate disposable agent container.

| Scenario | Result | Evidence |
|---|---|---|
| Forced compaction | Transport passed; task unresolved for unrelated model/plan defects | Repeated broad and conservative receipts preserved the initial request, current step, repair history, and context fit. |
| Rejected structured output | Passed after transport fix | Fenced raw responses remained in `conversation.full.jsonl`, were absent from final active context, and conversational repair continued without a llama.cpp role-template error. |
| Open-ended successful observation | Resolved | Three executions ended with code 125, `satisfied_by_progress_review=true`, `stopped_by_progress_review=false`, and retained `READY` output. |
| Unavailable-input failure | Correctly unresolved after prompt clarification | Every observed blocked process ended with code 125, `stopped_by_progress_review=true`, `satisfied_by_progress_review=false`, and retained `WAITING_FOR_INPUT` evidence. Two later conservative compactions retained that failure meaning. |

The first forced-compaction attempt exposed a generic transport defect: a late
system receipt could appear after ordinary user/assistant turns, which violates
some llama.cpp templates. Model-bound messages now preserve that receipt as a
labelled user-context update while keeping its audit role unchanged. The first
unavailable-input run exposed an ambiguous progress prompt in which the same
state could satisfy both `stop_satisfied` and `terminate`; the prompt now gives
the original request's explicit success/failure meaning precedence.

## Verification

- `python3 -m unittest discover -s tests`: 559 tests passed in 123.957s.
- Focused role-normalization, protocol-omission, progress-decision, output-drain,
  and timeout-race tests passed.
- `docker build -t agentic-feedback-coding:local .`: passed after each live fix.
- A late `stop_satisfied` response cannot override an already-fired hard
  timeout: the result remains code 124 and is not marked progress-satisfied.

## Remaining Limits

- Small models can still produce semantically bad artifacts while satisfying
  JSON structure. In the forced-compaction task, Devstral double-encoded a file
  body and accepted empty stdout as proof. Compaction retained the relevant
  state; it did not cause or conceal this defect.
- A stale plan validation command consumed repair effort after the model had
  proposed a better temporary-fixture check. This is a broader plan-validation
  lifecycle issue and was not folded into compaction changes.
- An intentionally failing command was rerun by step, reviewer, and final-review
  phases even after equivalent failure evidence existed. Result semantics were
  correct, but evidence reuse across review scopes remains an efficiency issue.
- Proxy scores are useful for corpus-scale regression detection, not a substitute
  for downstream task outcomes. Future publication benchmarks should remain
  frozen and independently graded.
