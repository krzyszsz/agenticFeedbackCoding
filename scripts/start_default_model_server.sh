#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_CPP_PIN="876a4321163249c43ca4e986818fab5ab081f282"
source "$REPO_ROOT/scripts/env.sh"

if [ -n "${MODEL_PROFILE:-}" ]; then
  eval "$(PYTHONPATH="$REPO_ROOT" python3 -m feedback_agent.model_profiles env "$MODEL_PROFILE")"
fi

SERVER_IMAGE="${SERVER_IMAGE:-agentic-feedback-llama-vulkan:local}"
CONTAINER="${CONTAINER:-agentic-gemma4-26b-mtp-server}"
MODEL_PATH="${MODEL_PATH:-$QWEN36_LOCAL_DIR/$QWEN36_MODEL_FILE}"
MODEL_DRAFT_PATH="${MODEL_DRAFT_PATH:-}"
# Profiles use an explicit empty value to mean that no projector belongs to the
# model. Fall back only when the variable is truly unset.
MMPROJ_PATH="${MMPROJ_PATH-$QWEN36_LOCAL_DIR/$QWEN36_MMPROJ_FILE}"
PORT="${PORT:-8161}"
PUBLISH_HOST="${PUBLISH_HOST:-127.0.0.1}"
PUBLISH_PORT="${PUBLISH_PORT:-$PORT}"
DOCKER_NETWORK="${DOCKER_NETWORK:-agentic-feedback-net}"
CTX_SIZE="${CTX_SIZE:-76800}"
GPU_LAYERS="${GPU_LAYERS:-999}"
THREADS="${THREADS:-8}"
PARALLEL="${PARALLEL:-1}"
MEM_LIMIT="${MEM_LIMIT:-75g}"
MEMORY_SWAP="${MEMORY_SWAP:-75g}"
MEM_RESERVATION="${MEM_RESERVATION:-67g}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
USE_DRI="${USE_DRI:-1}"
LLAMA_DEVICE="${LLAMA_DEVICE:-Vulkan0}"
REASONING_MODE="${REASONING_MODE:-on}"
REASONING_BUDGET="${REASONING_BUDGET:-4096}"
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"
SPEC_TYPE="${SPEC_TYPE:-}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-0}"
PROFILE_LLAMA_EXTRA_ARGS="${PROFILE_LLAMA_EXTRA_ARGS:-}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Missing model: $MODEL_PATH" >&2
  echo "Run: bash scripts/download_default_model.sh" >&2
  exit 1
fi

EXTRA_ARGS="--jinja --reasoning $REASONING_MODE --reasoning-budget $REASONING_BUDGET --reasoning-format $REASONING_FORMAT --no-context-shift"
if [ -f "$MMPROJ_PATH" ]; then
  EXTRA_ARGS="--mmproj $MMPROJ_PATH $EXTRA_ARGS"
fi
if [ -n "$MODEL_DRAFT_PATH" ]; then
  if [ ! -f "$MODEL_DRAFT_PATH" ]; then
    echo "Missing MTP draft model: $MODEL_DRAFT_PATH" >&2
    exit 1
  fi
  EXTRA_ARGS="$EXTRA_ARGS --model-draft $MODEL_DRAFT_PATH"
fi
if [ -n "$SPEC_TYPE" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --spec-type $SPEC_TYPE"
fi
if [ "$SPEC_DRAFT_N_MAX" -gt 0 ]; then
  EXTRA_ARGS="$EXTRA_ARGS --spec-draft-n-max $SPEC_DRAFT_N_MAX"
fi
if [ -n "$PROFILE_LLAMA_EXTRA_ARGS" ]; then
  EXTRA_ARGS="$EXTRA_ARGS $PROFILE_LLAMA_EXTRA_ARGS"
fi
if [ -n "$LLAMA_DEVICE" ] && [ "$USE_DRI" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --device $LLAMA_DEVICE"
fi
if [ -n "${LLAMA_EXTRA_ARGS:-}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS $LLAMA_EXTRA_ARGS"
fi

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
fi

if [ "${REBUILD_SERVER_IMAGE:-0}" = "1" ] || ! "${DOCKER[@]}" image inspect "$SERVER_IMAGE" >/dev/null 2>&1; then
  LLAMA_CPP_CACHEBUST_VALUE="${LLAMA_CPP_CACHEBUST:-0}"
  if [ "${REBUILD_SERVER_IMAGE:-0}" = "1" ] && [ -z "${LLAMA_CPP_CACHEBUST:-}" ]; then
    LLAMA_CPP_CACHEBUST_VALUE="$(date +%s)"
  fi
  "${DOCKER[@]}" build \
    -f "$REPO_ROOT/docker/llama-cpp-vulkan.Dockerfile" \
    --build-arg "LLAMA_CPP_REF=${LLAMA_CPP_REF:-$LLAMA_CPP_PIN}" \
    --build-arg "LLAMA_CPP_CACHEBUST=$LLAMA_CPP_CACHEBUST_VALUE" \
    -t "$SERVER_IMAGE" \
    "$REPO_ROOT"
fi

"${DOCKER[@]}" rm -f "$CONTAINER" >/dev/null 2>&1 || true

DEVICE_ARGS=()
if [ "$USE_DRI" = "1" ]; then
  DEVICE_ARGS+=(--device=/dev/dri)
fi

VOLUME_ARGS=(-v "$HF_ROOT:$HF_ROOT:ro")
case "$MODEL_ROOT" in
  "$HF_ROOT"|"$HF_ROOT"/*) ;;
  *) VOLUME_ARGS+=(-v "$MODEL_ROOT:$MODEL_ROOT:ro") ;;
esac
if [ -n "${MODEL_LOCAL_DIR:-}" ]; then
  case "$MODEL_LOCAL_DIR" in
    "$HF_ROOT"|"$HF_ROOT"/*|"$MODEL_ROOT"|"$MODEL_ROOT"/*) ;;
    *) VOLUME_ARGS+=(-v "$MODEL_LOCAL_DIR:$MODEL_LOCAL_DIR:ro") ;;
  esac
fi

NETWORK_ARGS=()
if [ "$DOCKER_NETWORK" = "host" ]; then
  NETWORK_ARGS=(--network=host)
else
  "${DOCKER[@]}" network inspect "$DOCKER_NETWORK" >/dev/null 2>&1 || \
    "${DOCKER[@]}" network create "$DOCKER_NETWORK" >/dev/null
  NETWORK_ARGS=(--network "$DOCKER_NETWORK" --network-alias "$CONTAINER" -p "$PUBLISH_HOST:$PUBLISH_PORT:$PORT")
fi

"${DOCKER[@]}" run -d --name "$CONTAINER" \
  --memory="$MEM_LIMIT" \
  --memory-swap="$MEMORY_SWAP" \
  --memory-reservation="$MEM_RESERVATION" \
  --oom-score-adj="$OOM_SCORE_ADJ" \
  "${DEVICE_ARGS[@]}" \
  --security-opt label=disable \
  --ipc=host \
  "${NETWORK_ARGS[@]}" \
  "${VOLUME_ARGS[@]}" \
  -e MODEL="$MODEL_PATH" \
  -e PORT="$PORT" \
  -e CTX_SIZE="$CTX_SIZE" \
  -e GPU_LAYERS="$GPU_LAYERS" \
  -e THREADS="$THREADS" \
  -e PARALLEL="$PARALLEL" \
  -e EXTRA_ARGS="$EXTRA_ARGS" \
  "$SERVER_IMAGE" >/dev/null

for _ in $(seq 1 "${READY_ATTEMPTS:-60}"); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PUBLISH_PORT/v1/models" || true)"
  if [ "$code" = "200" ]; then
    echo "Model server ready on host: http://127.0.0.1:$PUBLISH_PORT/v1"
    echo "Model profile: ${MODEL_PROFILE:-manual}"
    echo "Model path: $MODEL_PATH"
    if [ -n "$MODEL_DRAFT_PATH" ]; then
      echo "MTP draft path: $MODEL_DRAFT_PATH"
    fi
    if [ "$DOCKER_NETWORK" != "host" ]; then
      echo "Model server ready on Docker network '$DOCKER_NETWORK': http://$CONTAINER:$PORT/v1"
    fi
    exit 0
  fi
  sleep "${READY_SLEEP_SECONDS:-2}"
done

echo "Model server did not become ready on host port $PUBLISH_PORT." >&2
"${DOCKER[@]}" logs --tail 200 "$CONTAINER" || true
exit 1
