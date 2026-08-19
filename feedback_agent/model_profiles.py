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
    server_extra_args: str = ""
    min_p: float | None = 0.0
    presence_penalty: float | None = 0.0
    repeat_penalty: float | None = 1.0
    system_prompt_as_user: bool = False

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
        min_p=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.0,
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
        context_window=76800,
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
            "not a 32B dense model. The local 76,800-token server context leaves "
            "enough memory headroom for this 56 GB quantization."
        ),
        min_p=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.0,
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
        top_k=0,
        reasoning_mode="on",
        reasoning_format="deepseek",
        reasoning_budget_tokens=4096,
        spec_type="",
        spec_draft_n_max=0,
        notes=(
            "Fast DeepSeek-R1 reasoning distill on Qwen2.5-Math-7B. DeepSeek does "
            "not publish an 8B Qwen distill; its official 8B distill is Llama-based."
        ),
        system_prompt_as_user=True,
    ),
    "devstral-small-2507": ModelProfile(
        name="devstral-small-2507",
        role="coding_dense",
        repo_id="mistralai/Devstral-Small-2507_gguf",
        local_dir="/mnt/hf/models/devstral-small-2507-gguf",
        model_file="Devstral-Small-2507-Q4_K_M.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-devstral-small-2507-server",
        port=8166,
        context_window=131072,
        memory_limit="48g",
        memory_reservation="40g",
        temperature=0.15,
        top_p=1.0,
        top_k=0,
        reasoning_mode="off",
        reasoning_format="none",
        reasoning_budget_tokens=0,
        spec_type="",
        spec_draft_n_max=0,
        notes="Devstral Small 1.1 (24B dense) Q4_K_M coding profile.",
        min_p=0.01,
    ),
    "deepseek-coder-v2-lite-instruct": ModelProfile(
        name="deepseek-coder-v2-lite-instruct",
        role="coding_moe",
        repo_id="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        local_dir="/mnt/hf/models/deepseek-coder-v2-lite-instruct-gguf",
        model_file="DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-deepseek-coder-v2-lite-server",
        port=8167,
        context_window=131072,
        memory_limit="48g",
        memory_reservation="40g",
        temperature=0.3,
        top_p=0.95,
        top_k=0,
        reasoning_mode="off",
        reasoning_format="none",
        reasoning_budget_tokens=0,
        spec_type="",
        spec_draft_n_max=0,
        notes="DeepSeek-Coder-V2-Lite-Instruct (16B total, 2.4B active MoE) Q4_K_M.",
    ),
    "deepseek-r1-0528-qwen3-8b": ModelProfile(
        name="deepseek-r1-0528-qwen3-8b",
        role="reasoning_dense",
        repo_id="unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
        local_dir="/mnt/hf/models/deepseek-r1-0528-qwen3-8b-gguf",
        model_file="DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-deepseek-r1-0528-qwen3-8b-server",
        port=8168,
        context_window=131072,
        memory_limit="32g",
        memory_reservation="24g",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        reasoning_mode="on",
        reasoning_format="deepseek",
        reasoning_budget_tokens=4096,
        spec_type="",
        spec_draft_n_max=0,
        notes="DeepSeek-R1-0528 reasoning distill on Qwen3-8B, Q4_K_M.",
        system_prompt_as_user=False,
    ),
    "qwen3-8b": ModelProfile(
        name="qwen3-8b",
        role="reasoning_dense",
        repo_id="Qwen/Qwen3-8B-GGUF",
        local_dir="/mnt/hf/models/qwen3-8b-gguf",
        model_file="Qwen3-8B-Q4_K_M.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-qwen3-8b-server",
        port=8169,
        context_window=40960,
        memory_limit="32g",
        memory_reservation="24g",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        reasoning_mode="on",
        reasoning_format="deepseek",
        reasoning_budget_tokens=4096,
        spec_type="",
        spec_draft_n_max=0,
        notes="Qwen3-8B Q4_K_M in thinking mode at its recommended normal context.",
    ),
    "deepseek-r1-distill-llama-8b": ModelProfile(
        name="deepseek-r1-distill-llama-8b",
        role="reasoning_dense",
        repo_id="unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
        local_dir="/mnt/hf/models/deepseek-r1-distill-llama-8b-gguf",
        model_file="DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-deepseek-r1-distill-llama-8b-server",
        port=8170,
        context_window=131072,
        memory_limit="32g",
        memory_reservation="24g",
        temperature=0.6,
        top_p=0.95,
        top_k=0,
        reasoning_mode="on",
        reasoning_format="deepseek",
        reasoning_budget_tokens=4096,
        spec_type="",
        spec_draft_n_max=0,
        notes="DeepSeek-R1 reasoning distill on Llama-3.1-8B, Q4_K_M.",
        system_prompt_as_user=True,
    ),
    "qwen2.5-coder-7b-instruct": ModelProfile(
        name="qwen2.5-coder-7b-instruct",
        role="coding_dense",
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        local_dir="/mnt/hf/models/qwen2.5-coder-7b-instruct-gguf",
        model_file="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-qwen25-coder-7b-server",
        port=8171,
        context_window=131072,
        memory_limit="32g",
        memory_reservation="24g",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        reasoning_mode="off",
        reasoning_format="none",
        reasoning_budget_tokens=0,
        spec_type="",
        spec_draft_n_max=0,
        notes="Qwen2.5-Coder-7B-Instruct Q4_K_M coding profile.",
        repeat_penalty=1.1,
    ),
    "qwopus3.6-27b-coder": ModelProfile(
        name="qwopus3.6-27b-coder",
        role="coding_dense",
        repo_id="Jackrong/Qwopus3.6-27B-Coder-Compat-MTP-GGUF",
        local_dir="/mnt/hf/models/qwopus3.6-27b-coder-compat-mtp-gguf",
        model_file="Qwopus3.6-27B-Coder-Compat-MTP-Q5_K_M.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-qwopus36-27b-coder-server",
        port=8172,
        context_window=32768,
        memory_limit="75g",
        memory_reservation="67g",
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        reasoning_mode="off",
        reasoning_format="none",
        reasoning_budget_tokens=0,
        spec_type="draft-mtp",
        spec_draft_n_max=2,
        notes=(
            "Qwopus3.6-27B-Coder compatibility GGUF with corrected llama.cpp tool-history "
            "templates and a bundled MTP head. The fine-tune is evaluated in non-thinking mode."
        ),
    ),
    "devstral-small-2512": ModelProfile(
        name="devstral-small-2512",
        role="coding_dense",
        repo_id="unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",
        local_dir="/mnt/hf/models/devstral-small-2-24b-instruct-2512-gguf",
        model_file="Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-devstral-small-2512-server",
        port=8173,
        context_window=131072,
        memory_limit="48g",
        memory_reservation="40g",
        temperature=0.15,
        top_p=1.0,
        top_k=0,
        reasoning_mode="off",
        reasoning_format="none",
        reasoning_budget_tokens=0,
        spec_type="",
        spec_draft_n_max=0,
        notes="Devstral Small 2 24B Instruct 2512 using Unsloth's recommended UD-Q4_K_XL GGUF.",
        min_p=0.01,
    ),
    "qwen36-fable-fusion-mtp": ModelProfile(
        name="qwen36-fable-fusion-mtp",
        role="strong_dense",
        repo_id="DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF",
        local_dir="/mnt/hf/models/qwen36-fable-fusion-711-heretic-neo-max-mtp-gguf",
        model_file="Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-qwen36-fable-fusion-mtp-server",
        port=8174,
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
        notes="DavidAU Qwen3.6 27B Fable Fusion 711 Heretic NEO MAX MTP Q4_K_M.",
        server_extra_args="--reasoning-preserve",
        min_p=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.0,
    ),
    "kat-coder-v2.5-dev": ModelProfile(
        name="kat-coder-v2.5-dev",
        role="coding_moe",
        repo_id="bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF",
        local_dir="/mnt/hf/models/kat-coder-v2.5-dev-gguf",
        model_file="Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_M.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-kat-coder-v25-dev-server",
        port=8175,
        context_window=131072,
        memory_limit="75g",
        memory_reservation="67g",
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        reasoning_mode="on",
        reasoning_format="deepseek",
        reasoning_budget_tokens=4096,
        spec_type="",
        spec_draft_n_max=0,
        notes="Kwaipilot KAT-Coder-V2.5-Dev 35B/3B-active MoE using bartowski's recommended Q5_K_M GGUF.",
        server_extra_args="--reasoning-preserve",
        min_p=0.0,
        presence_penalty=1.5,
        repeat_penalty=1.0,
    ),
    "qwythos-27b-mtp": ModelProfile(
        name="qwythos-27b-mtp",
        role="reasoning_dense",
        repo_id="empero-ai/Qwythos-27B-v1-GGUF",
        local_dir="/mnt/hf/models/qwythos-27b-v1-gguf",
        model_file="Qwythos-27B-MTP-Q4_K_M.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-qwythos-27b-mtp-server",
        port=8176,
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
        notes="Qwythos 27B v1 MTP Q4_K_M using Empero's recommended agentic/tool settings.",
        server_extra_args="--reasoning-preserve",
        min_p=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.05,
    ),
    "qwen3.8-27b": ModelProfile(
        name="qwen3.8-27b",
        role="strong_dense",
        repo_id="unsloth/Qwen3.8-27B-GGUF",
        local_dir="/mnt/hf/models/qwen3.8-27b-gguf",
        model_file="Qwen3.8-27B-UD-Q4_K_XL.gguf",
        draft_file=None,
        mmproj_file=None,
        container_name="agentic-qwen38-27b-server",
        port=8177,
        context_window=262144,
        memory_limit="75g",
        memory_reservation="67g",
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        reasoning_mode="on",
        reasoning_format="deepseek",
        reasoning_budget_tokens=8192,
        spec_type="draft-mtp",
        spec_draft_n_max=2,
        notes=(
            "Qwen3.8 27B Unsloth Dynamic V3.0 UD-Q4_K_XL GGUF profile with "
            "native 262K context, preserved thinking, and a larger reasoning budget."
        ),
        server_extra_args="--reasoning-preserve",
        min_p=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.0,
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
    "devstral-small": "devstral-small-2507",
    "devstral-24b": "devstral-small-2507",
    "deepseek-coder-v2-lite": "deepseek-coder-v2-lite-instruct",
    "deepseek-coder-v2-lite-16b": "deepseek-coder-v2-lite-instruct",
    "deepseek-r1-distill-8b": "deepseek-r1-0528-qwen3-8b",
    "deepseek-r1-qwen3-8b": "deepseek-r1-0528-qwen3-8b",
    "qwen2.5-coder-7b": "qwen2.5-coder-7b-instruct",
    "qwopus": "qwopus3.6-27b-coder",
    "qwopus3.6": "qwopus3.6-27b-coder",
    "devstral-small-2": "devstral-small-2512",
    "devstral-small-2-2512": "devstral-small-2512",
    "devstral-2512": "devstral-small-2512",
    "fable-fusion": "qwen36-fable-fusion-mtp",
    "qwen-fable-fusion": "qwen36-fable-fusion-mtp",
    "qwen36-heretic": "qwen36-fable-fusion-mtp",
    "kat-coder": "kat-coder-v2.5-dev",
    "kat-v2.5-dev": "kat-coder-v2.5-dev",
    "kat-coder-v25-dev": "kat-coder-v2.5-dev",
    "qwythos": "qwythos-27b-mtp",
    "qwythos-27b": "qwythos-27b-mtp",
    "qwythos-27b-v1": "qwythos-27b-mtp",
    "qwen38": "qwen3.8-27b",
    "qwen3.8": "qwen3.8-27b",
    "qwen38-27b": "qwen3.8-27b",
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
        "PROFILE_LLAMA_EXTRA_ARGS": profile.server_extra_args,
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
