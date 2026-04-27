#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  exec /usr/local/bin/llama-server "$@"
fi

MODEL="${MODEL:-}"
if [ -z "$MODEL" ]; then
  echo "MODEL is not set. Provide MODEL=/path/to/model.gguf" >&2
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "MODEL file not found: $MODEL" >&2
  exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8161}"
CTX_SIZE="${CTX_SIZE:-32768}"
GPU_LAYERS="${GPU_LAYERS:-999}"
THREADS="${THREADS:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

EXTRA_ARR=()
if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
fi

exec /usr/local/bin/llama-server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --ctx-size "$CTX_SIZE" \
  --n-gpu-layers "$GPU_LAYERS" \
  --threads "$THREADS" \
  "${EXTRA_ARR[@]}"
