#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$PROJECT_DIR"

CONFIG_PATH="$PROJECT_DIR/config.example.json"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    -*)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
    *)
      if [ "$CONFIG_PATH" != "$PROJECT_DIR/config.example.json" ]; then
        echo "Only one config path can be provided." >&2
        exit 2
      fi
      CONFIG_PATH="$1"
      shift
      ;;
  esac
done

CONFIG_ABS="$(realpath "$CONFIG_PATH")"

docker_enabled="$(python3 - "$CONFIG_ABS" "$REPO_ROOT" <<'PY'
import json
import pathlib
import sys
cfg = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("1" if cfg.get("runtime", {}).get("docker_isolation", True) else "0")
PY
)"

if [ "$docker_enabled" = "1" ] && [ "${AGENT_IN_CONTAINER:-0}" != "1" ]; then
  image="$(python3 - "$CONFIG_ABS" <<'PY'
import json
import pathlib
import sys
cfg = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(cfg.get("runtime", {}).get("docker_image", "agentic-feedback-coding:local"))
PY
)"
  workspace="$(python3 - "$CONFIG_ABS" "$REPO_ROOT" <<'PY'
import json
import pathlib
import sys
cfg = json.loads(pathlib.Path(sys.argv[1]).read_text())
repo = pathlib.Path(sys.argv[2])
workspace = pathlib.Path(cfg["runtime"]["workspace"])
if not workspace.is_absolute():
    workspace = repo / workspace
print(workspace.resolve())
PY
)"
  docker_user="$(python3 - "$CONFIG_ABS" <<'PY'
import json
import pathlib
import sys
cfg = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(cfg.get("runtime", {}).get("docker_user", "host"))
PY
)"
  mkdir -p "$workspace"
  docker_cmd=(docker)
  if ! docker info >/dev/null 2>&1; then
    docker_cmd=(sudo docker)
  fi
  if [ "${REBUILD_AGENT_IMAGE:-0}" = "1" ] || ! "${docker_cmd[@]}" image inspect "$image" >/dev/null 2>&1; then
    echo "Building agent image: $image" >&2
    "${docker_cmd[@]}" build -t "$image" "$PROJECT_DIR"
  else
    echo "Using existing agent image: $image (set REBUILD_AGENT_IMAGE=1 to rebuild)" >&2
  fi

  agent_network="${AGENT_DOCKER_NETWORK:-${DOCKER_NETWORK:-agentic-feedback-net}}"
  model_server_container="${MODEL_SERVER_CONTAINER:-agentic-qwen36-server}"
  model_server_port="${MODEL_SERVER_PORT:-8161}"
  network_args=()
  env_args=(
    -e AGENT_IN_CONTAINER=1
    -e HOME=/tmp
    -e DOTNET_ROOT=/tmp/.dotnet
    -e PATH=/tmp/.dotnet:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    -e AGENT_WORKSPACE=/workspace/project
    -e REPO_ROOT=/app
  )

  case "$agent_network" in
    host)
      network_args=(--network=host)
      ;;
    none|"")
      ;;
    *)
      "${docker_cmd[@]}" network inspect "$agent_network" >/dev/null 2>&1 || \
        "${docker_cmd[@]}" network create "$agent_network" >/dev/null
      network_args=(--network "$agent_network")
      if [ -z "${AGENT_IMPLEMENTATION_BASE_URL:-}" ]; then
        AGENT_IMPLEMENTATION_BASE_URL="http://$model_server_container:$model_server_port/v1"
      fi
      ;;
  esac

  if [ -n "${AGENT_IMPLEMENTATION_BASE_URL:-}" ]; then
    env_args+=(-e "AGENT_IMPLEMENTATION_BASE_URL=$AGENT_IMPLEMENTATION_BASE_URL")
  fi
  if [ -n "${AGENT_FEEDBACK_BASE_URL:-}" ]; then
    env_args+=(-e "AGENT_FEEDBACK_BASE_URL=$AGENT_FEEDBACK_BASE_URL")
  fi

  user_args=()
  case "$docker_user" in
    host)
      user_args=(--user "$(id -u):$(id -g)")
      ;;
    root)
      user_args=(--user "0:0")
      ;;
    none|"")
      user_args=()
      ;;
    *)
      user_args=(--user "$docker_user")
      ;;
  esac
  "${docker_cmd[@]}" run --rm "${network_args[@]}" --security-opt label=disable \
    "${user_args[@]}" \
    "${env_args[@]}" \
    -v "$workspace:/workspace/project" \
    -v "$CONFIG_ABS:/app/config.json:ro" \
    "$image" --config /app/config.json
  exit 0
fi

if [ "$docker_enabled" != "1" ] && [ "${AGENT_IN_CONTAINER:-0}" != "1" ] && [ "${ALLOW_HOST_AGENT_RUN:-0}" != "1" ]; then
  cat >&2 <<'ERROR'
Refusing to run the agent workflow directly on the host.

Set runtime.docker_isolation=true in the config for normal use. If you are
deliberately doing harness development outside Docker, rerun with:
  ALLOW_HOST_AGENT_RUN=1
ERROR
  exit 2
fi

cd "$REPO_ROOT"
PYTHONPATH="$PROJECT_DIR" python3 -m feedback_agent.cli --config "$CONFIG_ABS"
