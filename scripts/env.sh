#!/usr/bin/env bash
# Shared defaults for local setup scripts. Override these in the environment
# instead of editing scripts, especially before publishing configs.

_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ENV_REPO_ROOT="$(cd "$_ENV_SCRIPT_DIR/.." && pwd)"

REPO_ROOT="${REPO_ROOT:-$_ENV_REPO_ROOT}"
HF_ROOT="${HF_ROOT:-$HOME/hf}"
MODEL_ROOT="${MODEL_ROOT:-$HF_ROOT/models}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/hf.key}"

MODEL_PROFILE="${MODEL_PROFILE:-gemma4-26b-a4b-qat-mtp}"

QWEN36_REPO_ID="${QWEN36_REPO_ID:-batiai/Qwen3.6-27B-GGUF}"
QWEN36_MODEL_FILE="${QWEN36_MODEL_FILE:-Qwen-Qwen3.6-27B-Q4_K_M.gguf}"
QWEN36_MMPROJ_FILE="${QWEN36_MMPROJ_FILE:-mmproj-Qwen-Qwen3.6-27B-BF16.gguf}"
QWEN36_LOCAL_DIR="${QWEN36_LOCAL_DIR:-$MODEL_ROOT/qwen3.6-27b-gguf}"

export REPO_ROOT HF_ROOT MODEL_ROOT HF_TOKEN_FILE MODEL_PROFILE
export QWEN36_REPO_ID QWEN36_MODEL_FILE QWEN36_MMPROJ_FILE QWEN36_LOCAL_DIR
