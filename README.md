# agenticFeedbackCoding

`agenticFeedbackCoding` runs a local AI coding workflow where one agent implements a project and a second agent reviews every step with tests, git diffs, command output, and evidence before accepting it. Normal work runs inside Docker for safety: only the generated project workspace is mounted out to the host, while the local model server stays outside and is reached through an OpenAI-compatible API.

The project is intentionally config-driven. One JSON file defines the model endpoint, workspace, review strictness, allowed tools, web/offline mode, and the project prompt.

## Quick Start

Start the local model server, then run a real benchmark through Docker:

```bash
MODEL_ROOT=$HOME/hf/models bash scripts/start_default_model_server.sh
bash scripts/build_and_run.sh --config config.real-palindrome.json
```

That run builds the agent container, mounts only the configured workspace, asks the local model to build the project, and stores the full transcript plus review evidence under `workspaces/real-palindrome/.agent_state/`.

## One Config File

Copy a real config and edit the prompt/workspace:

```bash
cp config.example.json config.my-project.json
```

The important fields are usually enough:

```json
{
  "implementation_model": {
    "base_url": "http://127.0.0.1:8161/v1",
    "model": "local-gguf",
    "context_window": 76800,
    "max_tokens": 2048,
    "temperature": 0.25,
    "request_timeout_seconds": 21600
  },
  "feedback_model": null,
  "mcp_tools": {
    "terminal": true,
    "web_scraping": false,
    "web_interaction": true
  },
  "runtime": {
    "docker_isolation": true,
    "workspace": "workspaces/my-new-project",
    "command_timeout_seconds": 120,
    "max_command_timeout_seconds": 21600
  },
  "web_research": { "enabled": false },
  "project_design": {
    "title": "My new project",
    "prompt": "Build a small browser game with tests and documentation."
  }
}
```

`command_timeout_seconds` is only the default timeout for one terminal command. It is not the model response timeout. If a generated test or build step needs longer, the agent can request it per command:

```json
{"cmd": ["python", "long_running_check.py"], "timeout_seconds": 7200}
```

That request is clamped by `runtime.max_command_timeout_seconds`. Model calls use `implementation_model.request_timeout_seconds`, which is set high by default for long local-model runs.

## Safety Model

Normal agentic work runs inside Docker. `scripts/run_agent.sh` refuses to run the workflow directly on the host unless `ALLOW_HOST_AGENT_RUN=1` is explicitly set for harness development.

The container gets one writable mount: the configured `runtime.workspace`, mapped to `/workspace/project`. The config file is mounted read-only. The Docker socket is not mounted. Host networking is used only so the agent container can reach a local OpenAI-compatible model server such as `127.0.0.1:8161`.

The agent container includes Python, Chromium, Playwright, `pytest`, `curl`, `git`, `jq`, `requests`, and `beautifulsoup4`, so generated projects can run tests, browser checks, and scraping-style tasks without installing those tools into the host project folder.

## Install And Model Setup

Clone and enter the repo:

```bash
git clone git@github.com:YOUR_USER/agenticFeedbackCoding.git
cd agenticFeedbackCoding
```

Install host dependencies, Docker, Python requirements, Ubuntu/Mesa Vulkan packages, and build the agent container image:

```bash
bash scripts/bootstrap_ubuntu.sh
```

For the tested Qwen3.6 profile, download/build the default model tooling:

```bash
HF_TOKEN_FILE=$HOME/hf.key MODEL_ROOT=$HOME/hf/models \
  bash scripts/bootstrap_ubuntu.sh --download-model --build-llama-vulkan
```

`hf.key` is a plain text Hugging Face token outside this repo. Create it only if you need authenticated Hugging Face access:

```bash
printf '%s' 'hf_your_token_here' > "$HOME/hf.key"
chmod 600 "$HOME/hf.key"
```

Default model paths are defined in `scripts/env.sh`:

```bash
HF_ROOT=$HOME/hf
MODEL_ROOT=$HF_ROOT/models
HF_TOKEN_FILE=$HOME/hf.key
```

Start the default llama.cpp/Vulkan server:

```bash
MODEL_ROOT=$HOME/hf/models bash scripts/start_default_model_server.sh
```

The default configs expect:

```text
http://127.0.0.1:8161/v1
```

## AMD And Driver Notes

The validated local path for this project is Vulkan, not ROCm. On the AMD Ryzen AI Max+ 395 / Strix Halo machine used during development, llama.cpp with Vulkan was more reliable than ROCm for GGUF serving.

The Ubuntu bootstrap installs these AMD-relevant packages:

```bash
libvulkan1 mesa-vulkan-drivers vulkan-tools clinfo
```

Useful checks:

```bash
vulkaninfo --summary
ls -l /dev/dri
clinfo | head -80
```

If Vulkan causes instability, try CPU fallback for the model server:

```bash
USE_DRI=0 GPU_LAYERS=0 MODEL_ROOT=$HOME/hf/models bash scripts/start_default_model_server.sh
```

That is much slower, but it avoids GPU-driver paths. ROCm is not required and is intentionally not automated here.

## Workflow

The workflow is deliberately more structured than one-pass code generation:

1. Requirements refinement fills gaps in the user prompt, records assumptions, and drafts a plan.
2. Plan validation checks whether the plan is feasible, clear, ordered, and verifiable before implementation starts.
3. Per-step implementation loops run one plan item at a time.
4. Feedback reviews inspect requirements, code, files, command output, reports, screenshots, git diffs, and previous critique before accepting a step.
5. A workspace-local git repository records the accepted baseline and each accepted plan step.
6. A final whole-project review checks the complete result after all individual steps are done.
7. Context compaction preserves durable memory when the transcript approaches the configured context window.

Both agents share one durable chat history. New feedback is appended at the end; previous requirements, implementation attempts, reviews, and correction requests stay visible until compaction is needed.

## Feedback Review Tools

The feedback agent does not only read the implementation agent's claims. Before each step review, the harness gives the feedback phase its own evidence:

- a fresh snapshot of generated workspace files
- an independent run of the current plan step's `validation_commands`
- return codes, stdout/stderr tails, and timeout flags from those commands
- `git status --short`
- meaningful changed paths, ignoring harness bookkeeping files such as `PLAN.md` and `.agent_state/`
- `git diff --stat`
- a truncated `git diff`

The automatic evidence gate uses that feedback-side evidence first. In hard-pushback mode it rejects a step if validation is missing, fails, times out, or if the implementation claims completion without meaningful git changes.

## Git Checkpointing

When `git_policy.enabled=true`, the generated workspace is initialized as a git repository. After requirements and plan validation, the harness creates a baseline commit. Accepted steps are committed only after feedback returns a resolved status. The final whole-project review also creates an acceptance commit.

If you want the final project left as uncommitted changes for manual inspection, set:

```json
"git_policy": {
  "enabled": true,
  "commit_completed_steps": true,
  "require_step_diff": true,
  "leave_final_changes_uncommitted": true,
  "final_reset_mode": "soft"
}
```

With that mode, the harness still uses commits internally during review, then resets to the baseline at the end so the final project appears in git as uncommitted/staged changes.

## Web Research And Offline Mode

Web research is optional. The project can run fully locally/offline if you disable it:

```json
"mcp_tools": {
  "web_scraping": false
},
"web_research": {
  "enabled": false
}
```

When enabled, web research only runs if the prompt explicitly asks to search/research/browse, look up current/latest information, or includes source URLs. The harness then fetches pages, writes `RESEARCH.md`, appends the research result to the transcript, injects compact research notes into later prompts, and asks the generated project to cite/apply source URLs when sources were actually fetched.

## Output Files

Each run creates a generated workspace under `workspaces/` and writes:

- a workspace-local `.git/` repository with baseline, accepted-step, and final-review commits when `git_policy.enabled=true`
- `RESEARCH.md` when web research was requested and enabled
- `REQUIREMENTS.md` with refined requirements and assumptions
- `PLAN.md` with ordered tasks, acceptance criteria, validation commands, and status
- `.agent_state/conversation.jsonl` with the full machine-readable agent chat
- `.agent_state/conversation.md` with the same transcript in readable Markdown
- `.agent_state/summary.json` with step results, review statuses, and feedback evidence

Generated workspaces, logs, reports, transcripts, and test evidence are ignored by git. They are useful locally, but they should not be published by accident.

## Configuration Knobs

| Field | Purpose | Typical values |
|---|---|---|
| `implementation_model.name` | Human-readable model profile name. | `qwen3.6-27b-q4km` |
| `implementation_model.base_url` | OpenAI-compatible endpoint used by the implementation agent. | `http://127.0.0.1:8161/v1` |
| `implementation_model.model` | Model id sent to the endpoint. llama.cpp accepts `local-gguf`. | `local-gguf` |
| `implementation_model.context_window` | Context budget used by compaction logic. | `32768`, `76800` |
| `implementation_model.max_tokens` | Max response length per model call. | `1024` to `4096` |
| `implementation_model.temperature` | Generation randomness. Lower is usually better for coding. | `0.1` to `0.3` |
| `implementation_model.request_timeout_seconds` | HTTP timeout for one model response. This is separate from terminal command timeouts. | `21600` |
| `feedback_model` | Optional separate reviewer model. `null` reuses the implementation model. | `null` or another model block |
| `mcp_tools.terminal` | Allows command execution for implementation and reviewer validation. | `true` |
| `mcp_tools.web_scraping` | Allows web research/scraping when a task asks for it. | `true` or `false` |
| `mcp_tools.web_interaction` | Allows browser-style validation. | `true` or `false` |
| `runtime.docker_isolation` | Runs generated project work in a container. Normal use should keep this true. | `true` |
| `runtime.docker_image` | Agent container image tag. | `agentic-feedback-coding:local` |
| `runtime.workspace` | Host-visible output folder for generated project files. | `workspaces/my-task` |
| `runtime.command_timeout_seconds` | Default timeout for one terminal command. Commands can override it with `{"cmd": [...], "timeout_seconds": N}`. | `60` to `300` |
| `runtime.max_command_timeout_seconds` | Maximum accepted per-command override. Prevents accidental unbounded terminal commands. | `3600` to `21600` |
| `runtime.print_transcript` | Prints the live agent conversation. | `true` for debugging |
| `context_compaction.enabled` | Enables transcript compaction near context limits. | `true` |
| `context_compaction.threshold_ratio` | Trigger compaction at this fraction of context. | `0.8` |
| `context_compaction.keep_recent_turns` | Recent turns kept verbatim during compaction. | `6` to `12` |
| `phases.requirements_refinement.max_iterations` | Requirement refinement retry budget. | `2` |
| `phases.plan_validation.max_iterations` | Plan validation retry budget. | `2` |
| `phases.implementation.max_iterations` | Per-step implementation retry budget. | `7` |
| `review_policy.hard_pushback_iterations` | Strict review attempts before compromise. | `3` |
| `review_policy.compromise_iterations` | Bounded compromise attempts after strict review. | `4` |
| `review_policy.final_review_iterations` | Whole-project review attempts. | `1` or `2` |
| `quality_policy.assume_code_quality_when_unspecified` | Adds default structure/tests/docs requirement unless prompt overrides it. | `true` |
| `quality_policy.require_research_and_structure_step` | Requires a first research/architecture step. | `true` |
| `web_research.enabled` | Enables harness-owned web research before requirements refinement. | `true` or `false` |
| `git_policy.enabled` | Initializes a workspace-local git repository and records git evidence. | `true` |
| `git_policy.commit_completed_steps` | Commits each accepted plan step after feedback resolves it. | `true` |
| `git_policy.require_step_diff` | Rejects step acceptance when there are no meaningful implementation changes to review. | `true` |
| `project_design.title` | Short task title. | Any string |
| `project_design.prompt` | Actual task prompt. | Detailed project brief |

## Real Example Configs

These configs are intended to run against a real local model endpoint. `config.real-palindrome.json` and `config.real-arithmetic.json` are the compact verified evidence runs documented below; the others are reusable starting points for larger tasks.

- `config.example.json` - starter task tracker project.
- `config.real-palindrome.json` - small verified CLI benchmark used as the current evidence run.
- `config.real-arithmetic.json` - small verified arithmetic package task.
- `config.real-website.json` - static website plus browser interaction task.
- `config.real-city-research.json` - web-research manifest task.
- `config.real-platformer.json` - browser platformer task with Playwright validation requirements.
- `config.gpx-editor.json` - GPX editor task with browser/map-style interaction requirements.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/bootstrap_ubuntu.sh` | Main Ubuntu setup script for Docker, Python env, optional model download, optional llama.cpp/Vulkan image build. |
| `scripts/install_ubuntu.sh` | Compatibility wrapper around `scripts/bootstrap_ubuntu.sh`. |
| `scripts/download_default_model.sh` | Downloads and verifies the default Qwen3.6 GGUF model and mmproj files. |
| `scripts/start_default_model_server.sh` | Builds if needed and starts the default llama.cpp/Vulkan model server on port `8161`. |
| `scripts/build_and_run.sh` | Convenience wrapper to build/run the agent harness from a config. |
| `scripts/run_agent.sh` | Lower-level runner that re-enters Docker when `runtime.docker_isolation=true`. |
| `scripts/env.sh` | Shared path/model defaults. Override values in the shell. |

## Verified Real Runs

Two real Docker-isolated Qwen3.6-27B Q4_K_M runs were executed with:

```bash
HF_ROOT=/mnt/hf MODEL_ROOT=/mnt/hf/models bash scripts/start_default_model_server.sh
bash scripts/build_and_run.sh --config config.real-palindrome.json
bash scripts/build_and_run.sh --config config.real-arithmetic.json
```

Observed `config.real-palindrome.json` result:

- The run entered Docker via `scripts/run_agent.sh` because `runtime.docker_isolation=true`.
- The generated project was written to `workspaces/real-palindrome` through the `/workspace/project` mount.
- The local model created `DESIGN_NOTES.md`, `palindrome.py`, `cli.py`, `test_palindrome.py`, `test_cli.py`, and `README.md`.
- Requirements refinement was challenged once because the first draft skipped the required research/structure-planning step and included a redundant end-to-end step.
- Plan validation then accepted a three-step plan: core module/tests, CLI/tests, and documentation.
- Feedback-side validation independently ran `python -m unittest test_palindrome -v`, `python -m unittest discover -v`, `python cli.py racecar`, `python cli.py hello`, `test -f PLAN.md`, `test -f palindrome.py`, `test -f README.md`, and `test -f DESIGN_NOTES.md`.
- The generated project passed 20 `unittest` cases, including core palindrome behavior, CLI integration, missing-argument handling, case-insensitivity, punctuation handling, non-palindromes, empty strings, and Unicode-aware alphanumeric filtering.
- Workspace git recorded a baseline commit, one accepted commit per completed plan step, and a final review commit.
- Final whole-project review resolved with a clean git state after rerunning all plan validation commands.

Observed `config.real-arithmetic.json` result:

- The run entered Docker via the same isolated `/workspace/project` mount and wrote `workspaces/real-arithmetic`.
- The fresh Docker build path was exercised for both the llama.cpp/Vulkan server image and the agent image.
- The first plan was rejected by the feedback agent because it skipped the required research/structure-planning step and did not independently verify README contents.
- The revised plan added `S0` for available-knowledge notes and project structure planning, then `S1` for code/tests, then `S2` for docs.
- The local model created `PROJECT_NOTES.md`, `arithmetic_box.py`, `test_arithmetic_box.py`, and `README.md`.
- Feedback-side validation independently ran the `PROJECT_NOTES.md` assertion, `python -m unittest discover -v`, and README content assertions for `add`, `multiply`, `mean`, and test instructions.
- The generated project passed 22 `unittest` cases for numeric happy paths, mixed int/float behavior, negative/zero cases, generator input for `mean`, `TypeError` cases, and `ValueError` for empty mean input.
- Workspace git recorded a baseline commit, one accepted commit per completed plan step, and a final review commit.
- Final whole-project review resolved with a clean git state.

The evidence is stored locally in ignored generated workspaces:

```text
workspaces/real-palindrome/.agent_state/summary.json
workspaces/real-palindrome/.agent_state/conversation.md
workspaces/real-arithmetic/.agent_state/summary.json
workspaces/real-arithmetic/.agent_state/conversation.md
```

## Tests

Run the harness unit tests without Docker:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -v
```

Run a real Docker-isolated benchmark:

```bash
MODEL_ROOT=$HOME/hf/models bash scripts/start_default_model_server.sh
bash scripts/build_and_run.sh --config config.real-palindrome.json
```

If your model cache lives outside `$HOME/hf`, override both roots:

```bash
HF_ROOT=/mnt/hf MODEL_ROOT=/mnt/hf/models bash scripts/start_default_model_server.sh
```
