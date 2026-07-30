# July 2026 Harness Validation

This record covers the generic harness changes validated before the July 29
publication reruns. It supplements the normalized tables in `README.md`; raw
run directories are intentionally ignored by git and remain local.

## Changes Exercised

1. Explicit unlimited command timeouts survive evidence replay as
   `timeout_seconds: 0` rather than failing on the runtime `None` value.
2. Step and final reviewers may request one bounded independent validation
   round through the existing tool-verification boundary.
3. Exhausted JSON repair is harness-owned `protocol_error`, separate from the
   model's verdict on task feasibility. Protocol repair has a 512-token
   reasoning-budget cap and cannot forge harness provenance.
4. Repeated observable failure evidence triggers one reassessment before the
   no-progress guard stops further repair. Materially changed evidence resets
   the guard.
5. Plan validation rejects statically invalid direct or shell-wrapped inline
   Python and shell programs without interpreting task semantics.
6. Requirements review now owns requirements, assumptions, feasibility, and
   verification strategy; detailed step and command adequacy remains in plan
   validation.

## Automated Checks

- `python3 -m unittest discover -s tests`: 508 passed in 124.907 seconds.
- `python3 -m compileall -q feedback_agent scripts tests`: passed.
- `git diff --check`: passed before the frozen runs.
- All 15 checked-in configs loaded through production validation.
- The 44-task corpus and the 30-task publication, 5-task calibration, and
  3-task grader-correction suites dry-loaded successfully.
- Host and frozen image hashes matched for `agent.py`, `compaction.py`, and
  `protocol.py`.

## Calibration

`development-watch-5` is disjoint from `publication-30`. Each task used one
model-server container and one isolated agent container, with one model loaded
at a time.

| Model | Result | Local evidence |
|---|---:|---|
| Qwen3-Coder-Next Q5_K_M | 4/5 | `runs/calibration-20260728-qwen3-coder-next-pass3-*` |
| Qwen3.6 27B MTP | 5/5 | `runs/calibration-20260728-qwen36-27b-pass1/` |

Coder Next computed the correct value for `dev-001` but violated the requested
deliverable scope by retaining an extra file. This was left as a failure rather
than adding a task- or filename-specific rule.

## Frozen Publication Runs

No harness, runner, or grader source changed during these runs. Both used
`--task-timeout-seconds 0`; model health and phase progress remained visible.

| Model | Pass | Fail | Average | Total | Local evidence |
|---|---:|---:|---:|---:|---|
| Qwen3.6 27B MTP | 28 | 2 | 2294.9s | 19.12h | `runs/publication-20260729-qwen36-27b-frozen/` |
| Qwen3-Coder-Next Q5_K_M | 19 | 11 | 751.5s | 6.26h | `runs/publication-20260729-qwen3-coder-next-frozen/` |

Each `results.json` contains exactly 30 unique task IDs. The tracked diff stayed
at `76e7623abdf7a51604959d26ef24d72e5bd873f4d417ce0f7694ad90b37106e9`.

## Observations For A Future Calibration Cycle

- Coder Next is still poorly aligned with the exact control protocol. Fourteen
  tasks ended as `protocol_error`; ten non-resolved tasks nevertheless passed
  external grading, while three internally resolved tasks failed. Any future
  adjustment should remain a small conversational protocol recovery mechanism,
  not phrase matching or inferred task intent.
- Review and artifact quality must remain separate metrics. A model-format
  failure should stop trusted control flow, but should not be reported as proof
  that an already-created artifact is wrong.
- Fine-grained plans amplify review cost. Coder Next's .NET task expanded to
  eight steps and seven attempts on one step before an overall approach retry.
  A future calibration should test a generic request for the smallest complete
  plan, while preserving decomposition for genuinely complex tasks.
- Qwen3.6's two failures were ordinary task mistakes accepted by its reviewers,
  not harness exceptions: one output-shape mismatch and one missing negative
  input contract. Reviewer evidence challenges should be tested against similar
  but disjoint cases before any further prompt change.
- The unlimited-timeout replay defect did not recur. Long commands and long
  tasks remained observable and model-reviewed without a hard goal deadline.

Do not use the publication tasks to tune these points. Start with new disjoint
calibration tasks, freeze source again, and rerun publication only after those
generic behaviors are satisfactory.
