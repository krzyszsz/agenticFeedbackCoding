#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

if [ -n "${MODEL_PROFILE:-}" ]; then
  eval "$(PYTHONPATH="$REPO_ROOT" python3 -m feedback_agent.model_profiles env "$MODEL_PROFILE")"
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import huggingface_hub
PY
then
  echo "Missing Python package: huggingface_hub. Run scripts/bootstrap_ubuntu.sh or install requirements.txt." >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
fi

DOWNLOAD_REPO_ID="${MODEL_REPO_ID:-$QWEN36_REPO_ID}"
DOWNLOAD_LOCAL_DIR="${MODEL_LOCAL_DIR:-$QWEN36_LOCAL_DIR}"
DOWNLOAD_FILES=()
if [ -n "${MODEL_FILE:-}" ]; then
  DOWNLOAD_FILES+=("$MODEL_FILE")
elif [ -n "${MODEL_PATH:-}" ]; then
  DOWNLOAD_FILES+=("$(realpath --relative-to="$DOWNLOAD_LOCAL_DIR" "$MODEL_PATH" 2>/dev/null || basename "$MODEL_PATH")")
else
  DOWNLOAD_FILES+=("$QWEN36_MODEL_FILE")
fi
if [ -n "${MODEL_DRAFT_FILE:-}" ]; then
  DOWNLOAD_FILES+=("$MODEL_DRAFT_FILE")
elif [ -n "${MODEL_DRAFT_PATH:-}" ]; then
  DOWNLOAD_FILES+=("$(realpath --relative-to="$DOWNLOAD_LOCAL_DIR" "$MODEL_DRAFT_PATH" 2>/dev/null || basename "$MODEL_DRAFT_PATH")")
fi
if [ -n "${MMPROJ_FILE:-}" ]; then
  DOWNLOAD_FILES+=("$MMPROJ_FILE")
elif [ -n "${MMPROJ_PATH:-}" ]; then
  DOWNLOAD_FILES+=("$(realpath --relative-to="$DOWNLOAD_LOCAL_DIR" "$MMPROJ_PATH" 2>/dev/null || basename "$MMPROJ_PATH")")
elif [ -z "${MODEL_PATH:-}" ]; then
  DOWNLOAD_FILES+=("$QWEN36_MMPROJ_FILE")
fi

mkdir -p "$DOWNLOAD_LOCAL_DIR"
VERIFY_JSON="${VERIFY_JSON:-$DOWNLOAD_LOCAL_DIR/model_verify.json}"
export VERIFY_JSON
export DOWNLOAD_REPO_ID DOWNLOAD_LOCAL_DIR
DOWNLOAD_FILES_JSON="$("$PYTHON_BIN" - "${DOWNLOAD_FILES[@]}" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
)"
export DOWNLOAD_FILES_JSON

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from huggingface_hub import HfApi, hf_hub_download

repo_id = os.environ["DOWNLOAD_REPO_ID"]
local_dir = Path(os.environ["DOWNLOAD_LOCAL_DIR"]).expanduser().resolve()
files = json.loads(os.environ["DOWNLOAD_FILES_JSON"])
token = os.environ.get("HF_TOKEN") or None
verify_json = Path(os.environ["VERIFY_JSON"]).expanduser().resolve()

api = HfApi(token=token)
tree = {item.path: item for item in api.list_repo_tree(repo_id, repo_type="model", recursive=True, expand=True)}

# A profile names the first GGUF shard used to launch llama.cpp. Discover the
# remaining shards from that standard filename so a fresh download is complete.
shard_pattern = re.compile(r"^(.*)-(\d{5})-of-(\d{5})(\.gguf)$")
expanded_files = list(files)
for rel in files:
    match = shard_pattern.match(rel)
    if not match or int(match.group(2)) != 1:
        continue
    prefix, _, count, suffix = match.groups()
    for index in range(2, int(count) + 1):
        sibling = f"{prefix}-{index:05d}-of-{count}{suffix}"
        if sibling not in tree:
            raise SystemExit(f"Missing expected GGUF shard in {repo_id}: {sibling}")
        if sibling not in expanded_files:
            expanded_files.append(sibling)
files = expanded_files

def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def remote_sha256(node):
    lfs = getattr(node, "lfs", None)
    if isinstance(lfs, dict):
        for key in ("sha256", "oid"):
            value = lfs.get(key)
            if value and len(str(value)) == 64:
                return str(value)
    if lfs is not None:
        for key in ("sha256", "oid"):
            value = getattr(lfs, key, None)
            if value and len(str(value)) == 64:
                return str(value)
    return None

plan = []
for rel in files:
    node = tree.get(rel)
    if node is None:
        raise SystemExit(f"Missing expected file in {repo_id}: {rel}")
    plan.append({
        "path": rel,
        "size_bytes": int(getattr(node, "size", 0) or 0),
        "remote_sha256": remote_sha256(node),
    })

print(f"Download plan for {repo_id} -> {local_dir}")
for item in plan:
    print(f"  - {item['path']}: {item['size_bytes'] / (1024 ** 3):.2f} GiB")
print(f"  total: {sum(item['size_bytes'] for item in plan) / (1024 ** 3):.2f} GiB")

local_dir.mkdir(parents=True, exist_ok=True)
results = []
for item in plan:
    path = Path(hf_hub_download(
        repo_id=repo_id,
        filename=item["path"],
        repo_type="model",
        local_dir=local_dir,
        token=token,
    ))
    digest = sha256_file(path)
    expected = item["remote_sha256"]
    ok = expected is None or digest == expected
    if not ok:
        raise SystemExit(f"SHA256 mismatch for {path}: expected {expected}, got {digest}")
    results.append({
        "repo_id": repo_id,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "remote_sha256": expected,
        "verified": ok,
    })

verify_json.parent.mkdir(parents=True, exist_ok=True)
verify_json.write_text(json.dumps({"files": results}, indent=2) + "\n", encoding="utf-8")
print(f"Downloaded and verified {len(results)} files.")
print(f"Verification report: {verify_json}")
PY
