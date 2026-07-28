from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
import json
from pathlib import Path
import shlex


@dataclass(frozen=True)
class ModelProfile:
    name: str
    role: str
    repo_id: str
    local_dir: str
    model_file: str
    draft_file: str | None
    mmproj_file: str | None
    container_name: str
    port: int
    context_window: int
    memory_limit: str
    memory_reservation: str
    temperature: float
    top_p: float | None
    top_k: int | None
    reasoning_mode: str
    reasoning_format: str
    reasoning_budget_tokens: int
    spec_type: str
    spec_draft_n_max: int
    notes: str

    @property
    def model_path(self) -> str:
        return str(Path(self.local_dir) / self.model_file)

    @property
    def draft_path(self) -> str:
        if not self.draft_file:
            return ""
        return str(Path(self.local_dir) / self.draft_file)

    @property
    def mmproj_path(self) -> str:
        if not self.mmproj_file:
            return ""
        return str(Path(self.local_dir) / self.mmproj_file)


MODEL_PROFILES: dict[str, ModelProfile] = {
    "gemma4-26b-a4b-qat-mtp": ModelProfile(
        name="gemma4-26b-a4b-qat-mtp",
        role="weak_fast",
        repo_id="unsloth/gemma-4-26B-A4B-it-qat-GGUF",
        local_dir="/mnt/hf/models/gemma4-26b-a4b-it-qat-q4_0-gguf",
        model_file="gemma-4-26B_q4_0-it.gguf",
        draft_file="MTP/gemma-4-26B-A4B-it-Q4_0-MTP.gguf",
        mmproj_file="gemma-4-26B-it-mmproj.gguf",
        container_name="agentic-gemma4-26b-mtp-server",
        port=8161,
        context_window=131072,
        memory_limit="75g",
        memory_reservation="67g",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        reasoning_mode="on",
        reasoning_format="deepseek",
        reasoning_budget_tokens=4096,
        spec_type="draft-mtp",
        spec_draft_n_max=4,
        notes="Fast MoE Gemma 4 QAT target with local MTP draft head.",
    ),
    "gemma4-31b-qat-mtp": ModelProfile(
        name="gemma4-31b-qat-mtp",
        role="strong_dense",
        repo_id="unsloth/gemma-4-31B-it-qat-GGUF",
        local_dir="/mnt/hf/models/gemma4-31b-it-qat-gguf",
        model_file="gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        draft_file="MTP/gemma-4-31B-it-Q4_0-MTP.gguf",
        mmproj_file=None,
        container_name="agentic-gemma4-31b-mtp-server",
        port=8162,
        context_window=131072,
        memory_limit="75g",
        memory_reservation="67g",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        reasoning_mode="on",
        reasoning_format="deepseek",
        reasoning_budget_tokens=4096,
        spec_type="draft-mtp",
        spec_draft_n_max=4,
        notes="Stronger dense Gemma 4 QAT target with local MTP draft head.",
    ),
    "qwen3.6-27b-mtp": ModelProfile(
        name="qwen3.6-27b-mtp",
        role="strong_dense",
        repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        local_dir="/mnt/hf/models/qwen3.6-27b-mtp-gguf",
        model_file="Qwen3.6-27B-UD-Q4_K_XL.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-qwen36-27b-mtp-server",
        port=8163,
        context_window=131072,
        memory_limit="75g",
        memory_reservation="67g",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        reasoning_mode="on",
        reasoning_format="deepseek",
        reasoning_budget_tokens=4096,
        spec_type="draft-mtp",
        spec_draft_n_max=2,
        notes=(
            "Qwen3.6 27B MTP profile. This replaces the requested 'Qwen 26B QAT MTP' "
            "name because the public/local MTP artifact is Qwen3.6-27B-MTP-GGUF."
        ),
    ),
    "qwen3-coder-next": ModelProfile(
        name="qwen3-coder-next",
        role="strong_coding_moe",
        repo_id="Qwen/Qwen3-Coder-Next-GGUF",
        local_dir="/mnt/hf/models/qwen3-coder-next-gguf",
        model_file="Qwen3-Coder-Next-Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00001-of-00004.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-qwen3-coder-next-server",
        port=8164,
        context_window=131072,
        memory_limit="88g",
        memory_reservation="80g",
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        reasoning_mode="off",
        reasoning_format="deepseek",
        reasoning_budget_tokens=0,
        spec_type="",
        spec_draft_n_max=0,
        notes=(
            "Coding-specialized Qwen3-Coder-Next Q5_K_M. The official model is an "
            "80B-total, 3B-active MoE and supports only non-thinking mode; it is "
            "not a 32B dense model."
        ),
    ),
    "deepseek-r1-distill-qwen-7b": ModelProfile(
        name="deepseek-r1-distill-qwen-7b",
        role="weak_fast_reasoning",
        repo_id="unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        local_dir="/mnt/hf/models/deepseek-r1-distill-qwen-7b-gguf",
        model_file="DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-deepseek-r1-distill-qwen-7b-server",
        port=8165,
        context_window=131072,
        memory_limit="20g",
        memory_reservation="16g",
        temperature=0.6,
        top_p=0.95,
        top_k=None,
        reasoning_mode="on",
        reasoning_format="deepseek",
        reasoning_budget_tokens=4096,
        spec_type="",
        spec_draft_n_max=0,
        notes=(
            "Fast DeepSeek-R1 reasoning distill on Qwen2.5-Math-7B. DeepSeek does "
            "not publish an 8B Qwen distill; its official 8B distill is Llama-based."
        ),
    ),
}

ALIASES = {
    "fast": "gemma4-26b-a4b-qat-mtp",
    "weak": "gemma4-26b-a4b-qat-mtp",
    "gemma-fast": "gemma4-26b-a4b-qat-mtp",
    "gemma31": "gemma4-31b-qat-mtp",
    "qwen26b-qat-mtp": "qwen3.6-27b-mtp",
    "qwen-26b-qat-mtp": "qwen3.6-27b-mtp",
    "qwen27": "qwen3.6-27b-mtp",
    "qwen3-coder-next-32b-dense": "qwen3-coder-next",
    "qwen-coder-next": "qwen3-coder-next",
    "deepseek-r1-distill-qwen-8b": "deepseek-r1-distill-qwen-7b",
    "deepseek-qwen-8b": "deepseek-r1-distill-qwen-7b",
    "deepseek7": "deepseek-r1-distill-qwen-7b",
}


def resolve_profile(name: str) -> ModelProfile:
    key = ALIASES.get(name, name)
    try:
        return MODEL_PROFILES[key]
    except KeyError as exc:
        known = ", ".join(sorted([*MODEL_PROFILES, *ALIASES]))
        raise SystemExit(f"Unknown model profile '{name}'. Known profiles/aliases: {known}") from exc


def profile_to_env(profile: ModelProfile) -> dict[str, str]:
    return {
        "MODEL_PROFILE": profile.name,
        "MODEL_REPO_ID": profile.repo_id,
        "MODEL_LOCAL_DIR": profile.local_dir,
        "MODEL_FILE": profile.model_file,
        "MODEL_DRAFT_FILE": profile.draft_file or "",
        "MMPROJ_FILE": profile.mmproj_file or "",
        "MODEL_PATH": profile.model_path,
        "MODEL_DRAFT_PATH": profile.draft_path,
        "MMPROJ_PATH": profile.mmproj_path,
        "CONTAINER": profile.container_name,
        "PORT": str(profile.port),
        "PUBLISH_PORT": str(profile.port),
        "CTX_SIZE": str(profile.context_window),
        "MEM_LIMIT": profile.memory_limit,
        "MEMORY_SWAP": profile.memory_limit,
        "MEM_RESERVATION": profile.memory_reservation,
        "REASONING_MODE": profile.reasoning_mode,
        "REASONING_FORMAT": profile.reasoning_format,
        "REASONING_BUDGET": str(profile.reasoning_budget_tokens),
        "SPEC_TYPE": profile.spec_type,
        "SPEC_DRAFT_N_MAX": str(profile.spec_draft_n_max),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or export local model profile metadata.")
    parser.add_argument("command", choices=["list", "json", "env"])
    parser.add_argument("profile", nargs="?", default="gemma4-26b-a4b-qat-mtp")
    args = parser.parse_args()

    if args.command == "list":
        print(json.dumps({name: asdict(profile) for name, profile in MODEL_PROFILES.items()}, indent=2))
        return 0

    profile = resolve_profile(args.profile)
    if args.command == "json":
        print(json.dumps(asdict(profile), indent=2))
        return 0

    for key, value in profile_to_env(profile).items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
