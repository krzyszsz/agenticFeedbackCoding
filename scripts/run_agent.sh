#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$PROJECT_DIR"

CONFIG_PATH="$PROJECT_DIR/config.example.json"
ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      ARGS+=("$1")
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
  mkdir -p "$workspace"
  docker_cmd=(docker)
  if ! docker info >/dev/null 2>&1; then
    docker_cmd=(sudo docker)
  fi
  "${docker_cmd[@]}" build -t "$image" "$PROJECT_DIR"
  "${docker_cmd[@]}" run --rm --network=host --security-opt label=disable \
    --user "$(id -u):$(id -g)" \
    -e AGENT_IN_CONTAINER=1 \
    -e HOME=/tmp \
    -e AGENT_WORKSPACE=/workspace/project \
    -e REPO_ROOT=/app \
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
