# Qwen3.8 27B Benchmark Notes, 2026-08-17 to 2026-08-19

## Model Artifact

- Profile: `qwen3.8-27b`
- Repository: `unsloth/Qwen3.8-27B-GGUF`
- Artifact: `Qwen3.8-27B-UD-Q4_K_XL.gguf`
- Local path: `/mnt/hf/models/qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf`
- Size: 17,923,394,624 bytes
- SHA-256: `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372`
- Sampling: `temperature=1.0`, `top_p=0.95`, `top_k=20`
- Server profile: port `8177`, context `262144`, `spec_type=draft-mtp`
- High-thinking profile: `reasoning_budget_tokens=8192`, automatic critical
  budget (`30720` tokens in this benchmark), and `--reasoning-preserve`

## Publication-30 Results

| Mode | Result | Runtime | Raw Evidence |
|---|---:|---:|---|
| Zero-shot, original profile (`--mode single-shot`) | 22/30 | 2.25h | `runs/publication-20260815-qwen38-27b/zero-shot/` |
| Full harness, original profile (`--mode harness`) | 27/30 | 29.24h | `runs/publication-20260815-qwen38-27b/harness/` |
| Zero-shot, high-thinking profile (`--mode single-shot`) | 24/30 | 3.03h | `runs/publication-20260817-qwen38-27b-high-thinking/zero-shot/` |
| Full harness, high-thinking profile (`--mode harness`) | 28/30 | 35.85h | `runs/publication-20260817-qwen38-27b-high-thinking/harness/` |

## Observed Issues For Later Review

- The high-thinking profile improved the Qwen3.8 zero-shot result from `22/30`
  to `24/30` and the harness result from `27/30` to `28/30`, but increased
  harness runtime from 29.24h to 35.85h.
- Qwen3.8 gained four passes over high-thinking zero-shot under the harness, so
  repair cycles helped, but the runtime cost was high. The high-thinking
  `hist-006-dotnet-dependency` harness task failed after 10,283.5s with a
  protocol error.
- Long-running tasks often spent more time in verifier/reviewer phases than in
  terminal execution. Critical review and repair phases used large reasoning
  budgets and frequently ran for several minutes per call.
- Several phases needed protocol repair after malformed JSON or missing
  decisions, especially in long-context critical repair paths. This remained
  true with the larger reasoning budget.
- The high-thinking harness recovered `tool-002-log-watch` and
  `tool-003-output-truncation`, both failures in the original Qwen3.8 harness
  run. The remaining high-thinking harness failures were
  `code-003-interval-merge` and `hist-006-dotnet-dependency`.
- `integration-001-mini-package` passed, but only after 16,156.5s. The task
  reopened plan refinement late and repeated several critical implementation
  and review loops. This is useful evidence for a future generic
  progress-management improvement, not for task-specific prompting.
