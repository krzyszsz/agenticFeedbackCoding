#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/config.example.json"
MOCK=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --real)
      MOCK=0
      shift
      ;;
    --mock)
      MOCK=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

args=(--config "$CONFIG_PATH")
if [ "$MOCK" = "1" ]; then
  args+=(--mock)
fi

exec "$PROJECT_DIR/scripts/run_agent.sh" "${args[@]}"
