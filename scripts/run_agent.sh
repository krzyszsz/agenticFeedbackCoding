#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$PROJECT_DIR"
source "$REPO_ROOT/scripts/env.sh"

if [ -n "${MODEL_PROFILE:-}" ]; then
  eval "$(PYTHONPATH="$REPO_ROOT" python3 -m feedback_agent.model_profiles env "$MODEL_PROFILE")"
fi

CONFIG_PATH="$PROJECT_DIR/config.example.json"
WORKSPACE_OVERRIDE=""
CONTAINER_CLI_ARGS=()
HOST_CLI_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      CONTAINER_CLI_ARGS+=(--config /app/config.json)
      shift 2
      ;;
    --workspace)
      WORKSPACE_OVERRIDE="$2"
      # The host uses this path for the bind mount. Inside Docker the mounted
      # workspace is always /workspace/project via AGENT_WORKSPACE, so forwarding
      # the host-relative path would send the agent back into /app/workspaces.
      HOST_CLI_ARGS+=(--workspace "$2")
      shift 2
      ;;
    --title)
      CONTAINER_CLI_ARGS+=(--title "$2")
      HOST_CLI_ARGS+=(--title "$2")
      shift 2
      ;;
    --prompt)
      CONTAINER_CLI_ARGS+=(--prompt "$2")
      HOST_CLI_ARGS+=(--prompt "$2")
      shift 2
      ;;
    --prompt-file)
      prompt_content="$(cat "$2")"
      CONTAINER_CLI_ARGS+=(--prompt "$prompt_content")
      HOST_CLI_ARGS+=(--prompt "$prompt_content")
      shift 2
      ;;
    --offline)
      CONTAINER_CLI_ARGS+=(--offline)
      HOST_CLI_ARGS+=(--offline)
      shift
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
if [ "${#CONTAINER_CLI_ARGS[@]}" -eq 0 ]; then
  CONTAINER_CLI_ARGS=(--config /app/config.json)
fi

config_field() {
  local field="$1"
  PYTHONPATH="$PROJECT_DIR" python3 - "$CONFIG_ABS" "$REPO_ROOT" "$WORKSPACE_OVERRIDE" "$field" <<'PY'
import sys
from pathlib import Path
from dataclasses import replace
from feedback_agent.config import load_config

cfg = load_config(sys.argv[1], repo_root=Path(sys.argv[2]))
workspace_override = sys.argv[3]
if workspace_override:
    workspace = Path(workspace_override)
    if not workspace.is_absolute():
        workspace = (Path(sys.argv[2]) / workspace).resolve()
    cfg = replace(cfg, runtime=replace(cfg.runtime, workspace=workspace))
field = sys.argv[4]
if field == "docker_enabled":
    print("1" if cfg.runtime.docker_isolation else "0")
elif field == "image":
    print(cfg.runtime.docker_image)
elif field == "workspace":
    print(cfg.runtime.workspace)
elif field == "docker_user":
    print(cfg.runtime.docker_user)
else:
    raise SystemExit(f"unknown field: {field}")
PY
}

docker_enabled="$(config_field docker_enabled)"

if [ "$docker_enabled" = "1" ] && [ "${AGENT_IN_CONTAINER:-0}" != "1" ]; then
  image="${AGENT_IMAGE:-$(config_field image)}"
  workspace="$(config_field workspace)"
  docker_user="$(config_field docker_user)"
  mkdir -p "$workspace"
  docker_cmd=(docker)
  if ! docker info >/dev/null 2>&1; then
    docker_cmd=(sudo docker)
  fi
  if [ "${REBUILD_AGENT_IMAGE:-0}" = "1" ] || ! "${docker_cmd[@]}" image inspect "$image" >/dev/null 2>&1; then
    if [ "${SKIP_AGENT_IMAGE_BUILD:-0}" = "1" ]; then
      echo "Agent image not found locally and SKIP_AGENT_IMAGE_BUILD=1: $image" >&2
      exit 2
    fi
    echo "Building agent image: $image" >&2
    "${docker_cmd[@]}" build -t "$image" "$PROJECT_DIR"
  else
    echo "Using existing agent image: $image (set REBUILD_AGENT_IMAGE=1 to rebuild)" >&2
  fi

  agent_network="${AGENT_DOCKER_NETWORK:-${DOCKER_NETWORK:-agentic-feedback-net}}"
  model_server_container="${MODEL_SERVER_CONTAINER:-${CONTAINER:-agentic-gemma4-26b-mtp-server}}"
  model_server_port="${MODEL_SERVER_PORT:-${PORT:-8161}}"
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
    "$image" "${CONTAINER_CLI_ARGS[@]}"
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
PYTHONPATH="$PROJECT_DIR" python3 -m feedback_agent.cli --config "$CONFIG_ABS" "${HOST_CLI_ARGS[@]}"
