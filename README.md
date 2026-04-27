# agenticFeedbackCoding

`agenticFeedbackCoding` runs a local AI coding workflow where one agent implements a project and a second agent reviews every step with tests, git diffs, and evidence before accepting it. It is config-driven, Docker-isolated by default, can run fully offline, and keeps long agent conversations coherent with a shared transcript, PLAN.md, git checkpoints, and context compaction.

Fast local smoke test after cloning:

```bash
bash scripts/bootstrap_ubuntu.sh
bash scripts/build_and_run.sh --config config.example.json --mock
```

The mock run does not need a model or Hugging Face token. It builds the Docker harness, writes a tiny project to `workspaces/demo`, and proves the planning, review, test-evidence, and git-checkpoint flow works.

## One Config File

A run is controlled from one JSON file. Copy `config.example.json`, edit the task prompt, choose the workspace, and decide whether web research is allowed:

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

The full config has more knobs for retry budgets, strict review, git policy, and context compaction. The important idea is simple: project intent and harness behavior live together in one file.

`command_timeout_seconds` is only the default timeout for one terminal command. It is not the model response timeout. If a generated test or build step needs longer, the agent can request it per command:

```json
{"cmd": ["python", "long_running_check.py"], "timeout_seconds": 7200}
```

That request is clamped by `runtime.max_command_timeout_seconds`. Model calls use `implementation_model.request_timeout_seconds`, which is set high by default for long local-model runs.

## Safety Model

Normal agentic work runs inside Docker. `scripts/run_agent.sh` refuses to run the workflow on the host unless you explicitly set `ALLOW_HOST_AGENT_RUN=1`, which is intended only for harness development.

The container gets one writable mount: the configured `runtime.workspace`, mapped to `/workspace/project`, so generated output is visible on the host. The config file is mounted read-only. The Docker socket is not mounted. Host networking is used only so the container can reach a local OpenAI-compatible model server such as `127.0.0.1:8161`.

The agent container includes Python, Chromium, Playwright, `curl`, `git`, `jq`, `requests`, and `beautifulsoup4`, so generated projects can run small tests, browser checks, and scraping-style tasks without installing those tools directly into the host project folder.

## Quick Start Details

Clone and enter the repo:

```bash
git clone git@github.com:YOUR_USER/agenticFeedbackCoding.git
cd agenticFeedbackCoding
```

Install host dependencies, Docker, Python requirements, Ubuntu/Mesa Vulkan packages, and build the agent container image:

```bash
bash scripts/bootstrap_ubuntu.sh
```

Run the deterministic smoke scenario:

```bash
bash scripts/build_and_run.sh --config config.example.json --mock
```

That is enough to validate the harness itself. It does not download a model.

## Building A New Project

1. Copy a config:

```bash
cp config.example.json config.my-project.json
```

2. Edit these fields in `config.my-project.json`:

```json
"runtime": {
  "docker_isolation": true,
  "workspace": "workspaces/my-project"
},
"project_design": {
  "title": "My project",
  "prompt": "Describe exactly what the agents should build."
}
```

3. If you want a fully offline run, disable web research in both places:

```json
"mcp_tools": {
  "web_scraping": false
},
"web_research": {
  "enabled": false
}
```

4. Run it:

```bash
bash scripts/build_and_run.sh --config config.my-project.json --real
```

Use `--mock` instead of `--real` only for deterministic harness tests.

## Local Model Setup

The harness talks to any OpenAI-compatible chat endpoint. The tested local profile uses llama.cpp/Vulkan serving `Qwen3.6-27B Q4_K_M` on an AMD Ryzen AI Max+ 395 / Strix Halo machine with 96GB unified memory. Other GPUs, CPUs, cloud endpoints, and model servers should work if they expose an OpenAI-compatible `/v1/chat/completions` API, but they were not validated here.

For the tested Qwen3.6 profile, run:

```bash
HF_TOKEN_FILE=$HOME/hf.key MODEL_ROOT=$HOME/hf/models \
  bash scripts/bootstrap_ubuntu.sh --download-model --build-llama-vulkan
```

What is `hf.key`?

- It is a plain text file outside this repo containing a Hugging Face access token.
- You only need it for downloads that require authentication or when you prefer authenticated Hugging Face requests.
- Create it like this after generating a token on Hugging Face:

```bash
printf '%s' 'hf_your_token_here' > "$HOME/hf.key"
chmod 600 "$HOME/hf.key"
```

The model download is large. Keep models outside the repo. Default paths are defined in `scripts/env.sh`:

```bash
HF_ROOT=$HOME/hf
MODEL_ROOT=$HF_ROOT/models
HF_TOKEN_FILE=$HOME/hf.key
```

Start the default llama.cpp/Vulkan server:

```bash
MODEL_ROOT=$HOME/hf/models bash scripts/start_default_model_server.sh
```

Then run the real-model smoke config:

```bash
bash scripts/build_and_run.sh --config config.qwen36-smoke.json --real
```

The default config expects:

```text
http://127.0.0.1:8161/v1
```

## AMD And Driver Notes

The validated local path for this project is Vulkan, not ROCm. On the Strix Halo machine used during development, llama.cpp with Vulkan was more reliable than ROCm for GGUF serving.

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

References:

- Docker Ubuntu install: https://docs.docker.com/engine/install/ubuntu/
- Ubuntu `mesa-vulkan-drivers`: https://packages.ubuntu.com/mesa-vulkan-drivers
- AMD ROCm Linux install docs: https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html

## What The Harness Does

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

The deterministic evidence gate uses that feedback-side evidence first. In hard-pushback mode it rejects a step if validation is missing, fails, times out, or if the implementation claims completion without meaningful git changes.

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

When enabled, web research only runs if the prompt explicitly asks to search/research/browse, look up current/latest information, or includes source URLs. The harness then:

- fetches source URLs directly, or performs a small best-effort DuckDuckGo HTML search when no URL was provided
- extracts page title and readable text excerpts
- writes the evidence to `RESEARCH.md`
- appends `WEB_RESEARCH_TOOL_RESULT` to the durable transcript
- injects compact research notes into requirements, planning, and implementation prompts
- requires generated project work to cite/apply fetched source URLs outside tool-generated `RESEARCH.md`

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

The main settings live in JSON config files.

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
| `web_research.max_search_results` | Search results to collect when no URL is provided. | `3` |
| `web_research.max_pages` | Maximum pages fetched per run. | `3` |
| `web_research.timeout_seconds` | Per-page network timeout. | `15` |
| `web_research.max_page_bytes` | Maximum bytes read from one page. | `1000000` |
| `web_research.excerpt_chars` | Characters kept per source excerpt. | `3000` |
| `web_research.user_agent` | User-Agent for fetch/search requests. | project default |
| `git_policy.enabled` | Initializes a workspace-local git repository and records git evidence. | `true` |
| `git_policy.commit_completed_steps` | Commits each accepted plan step after feedback resolves it. | `true` |
| `git_policy.require_step_diff` | Rejects step acceptance when there are no meaningful implementation changes to review. | `true` |
| `git_policy.leave_final_changes_uncommitted` | Soft-resets accepted commits at the end so the final project is visible as uncommitted changes. | `false` |
| `git_policy.final_reset_mode` | Reset mode used when leaving changes uncommitted. | `soft` or `mixed` |
| `git_policy.commit_user_name` | Local git author name used inside generated workspaces. | `agenticFeedbackCoding` |
| `git_policy.commit_user_email` | Local git author email used inside generated workspaces. | `agentic-feedback@example.local` |
| `project_design.title` | Short task title. | Any string |
| `project_design.prompt` | Actual task prompt. | Detailed project brief |

## Default Quality Policy

Unless the user prompt explicitly says otherwise, requirements refinement assumes the project should be well structured, well tested, and well documented. When that default applies, the first implementation step must research needed patterns/knowledge, plan the project structure/architecture, and rewrite or confirm the remaining task order.

Disable that behavior only when you really want a throwaway run:

```json
"quality_policy": {
  "assume_code_quality_when_unspecified": false,
  "require_research_and_structure_step": false
}
```

## Critical Review Policy

Feedback is strict first, then bounded so it cannot loop forever:

```json
"review_policy": {
  "hard_pushback_iterations": 3,
  "compromise_iterations": 4,
  "final_review_iterations": 1
}
```

During hard-pushback iterations, feedback rejects missing or inconsistent evidence. During compromise iterations, it may accept a diluted requirement or `skipped_with_note`, but it must record the limitation.

## Config Files

- `config.example.json` - default deterministic task-tracker project.
- `config.mock-website.json` - static multi-page website and clicker interaction.
- `config.mock-cities.json` - non-development city image manifest workflow.
- `config.mock-emptydiff.json` - no-change attempt exercise proving git-diff feedback pushback.
- `config.mock-platformer.json` - browser platformer stress scenario with Playwright screenshot validation.
- `config.qwen36-smoke.json` - small real-model profile for Qwen3.6 via a local OpenAI-compatible endpoint.

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

## Running Mock Scenarios

```bash
bash scripts/build_and_run.sh --config config.mock-website.json --mock
bash scripts/build_and_run.sh --config config.mock-cities.json --mock
bash scripts/build_and_run.sh --config config.mock-emptydiff.json --mock
bash scripts/build_and_run.sh --config config.mock-platformer.json --mock
```

Mock mode uses deterministic local responses, so these runs validate the harness itself without needing a local model server.

## Tests

Run the Python unit/integration tests without Docker:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -v
```

Run Docker mock harness checks:

```bash
bash scripts/build_and_run.sh --config config.example.json --mock
bash scripts/build_and_run.sh --config config.mock-emptydiff.json --mock
bash scripts/build_and_run.sh --config config.mock-platformer.json --mock
```
