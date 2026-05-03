#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/config.example.json"
RUN_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      RUN_ARGS+=(--config "$2")
      shift 2
      ;;
    --workspace|--title|--prompt|--prompt-file)
      RUN_ARGS+=("$1" "$2")
      shift 2
      ;;
    --offline)
      RUN_ARGS+=("$1")
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ "${#RUN_ARGS[@]}" -eq 0 ]; then
  RUN_ARGS=(--config "$CONFIG_PATH")
fi

exec "$PROJECT_DIR/scripts/run_agent.sh" "${RUN_ARGS[@]}"
