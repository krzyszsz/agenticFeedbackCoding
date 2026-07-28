#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

INSTALL_HOST=1
INSTALL_AMD_VULKAN=1
DOWNLOAD_MODEL=0
BUILD_AGENT_IMAGE=1
BUILD_LLAMA_VULKAN=0
PRINT_ROCM_NOTES=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/bootstrap_ubuntu.sh [options]

Default: install Ubuntu host dependencies, Docker Engine, AMD/Mesa Vulkan packages,
Python venv requirements, and build the agent container image.

Options:
  --download-model       Download the selected MODEL_PROFILE target and MTP files.
  --build-llama-vulkan   Build the optional llama.cpp Vulkan server image.
  --skip-host            Do not install apt/Docker/Vulkan host packages.
  --skip-amd-vulkan      Do not install libvulkan1/mesa-vulkan-drivers/vulkan-tools.
  --skip-agent-image     Do not build agentic-feedback-coding:local.
  --with-rocm-notes      Print ROCm guidance. ROCm install is intentionally not automated.
  -h, --help             Show this help.

Common full setup:
  HF_TOKEN_FILE=$HOME/hf.key MODEL_ROOT=$HOME/hf/models bash scripts/bootstrap_ubuntu.sh --download-model --build-llama-vulkan
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --download-model) DOWNLOAD_MODEL=1; shift ;;
    --build-llama-vulkan) BUILD_LLAMA_VULKAN=1; shift ;;
    --skip-host) INSTALL_HOST=0; shift ;;
    --skip-amd-vulkan) INSTALL_AMD_VULKAN=0; shift ;;
    --skip-agent-image) BUILD_AGENT_IMAGE=0; shift ;;
    --with-rocm-notes) PRINT_ROCM_NOTES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

if [ "$INSTALL_HOST" = "1" ]; then
  if [ -r /etc/os-release ]; then
    . /etc/os-release
    if [ "${ID:-}" != "ubuntu" ]; then
      echo "Warning: this bootstrap script is written for Ubuntu; detected ${PRETTY_NAME:-unknown}." >&2
    fi
    UBUNTU_CODENAME="${VERSION_CODENAME:-noble}"
  else
    UBUNTU_CODENAME="noble"
  fi

  as_root apt-get update
  as_root apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg git jq python3 python3-pip python3-venv

  if ! command -v docker >/dev/null 2>&1; then
    as_root install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | as_root gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    as_root chmod a+r /etc/apt/keyrings/docker.gpg
    ARCH="$(dpkg --print-architecture)"
    echo "deb [arch=$ARCH signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $UBUNTU_CODENAME stable" \
      | as_root tee /etc/apt/sources.list.d/docker.list >/dev/null
    as_root apt-get update
    as_root apt-get install -y --no-install-recommends \
      docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  fi

  if [ "$INSTALL_AMD_VULKAN" = "1" ]; then
    as_root apt-get install -y --no-install-recommends \
      libvulkan1 mesa-vulkan-drivers vulkan-tools clinfo
  fi

  if getent group docker >/dev/null 2>&1; then
    as_root usermod -aG docker "$USER" || true
  fi
fi

python3 -m venv "$REPO_ROOT/.venv"
# shellcheck disable=SC1091
. "$REPO_ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements.txt"

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
fi

if [ "$BUILD_AGENT_IMAGE" = "1" ]; then
  "${DOCKER[@]}" build -t agentic-feedback-coding:local "$REPO_ROOT"
fi

if [ "$BUILD_LLAMA_VULKAN" = "1" ]; then
  "${DOCKER[@]}" build \
    -f "$REPO_ROOT/docker/llama-cpp-vulkan.Dockerfile" \
    --build-arg "LLAMA_CPP_REF=${LLAMA_CPP_REF:-master}" \
    -t agentic-feedback-llama-vulkan:local \
    "$REPO_ROOT"
fi

if [ "$DOWNLOAD_MODEL" = "1" ]; then
  bash "$REPO_ROOT/scripts/download_default_model.sh"
fi

if [ "$PRINT_ROCM_NOTES" = "1" ]; then
  cat <<'NOTES'
ROCm is not required for the default path in this project. The tested local path
uses llama.cpp with Vulkan/RADV/Mesa. If you want ROCm anyway, use AMD's current
ROCm Ubuntu package-manager documentation for your exact Ubuntu/driver version.
This repository does not automate ROCm because the Strix Halo validation that led
to this project found Vulkan more reliable than ROCm on that host.
NOTES
fi

echo "Bootstrap complete. If Docker group membership was added, log out/in or run: newgrp docker"
