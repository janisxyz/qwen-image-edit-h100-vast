from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace"))


def detect_gpu() -> tuple[str, int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        first_line = result.stdout.strip().splitlines()[0]
        name, memory_mb = first_line.rsplit(",", 1)
        return name.strip(), int(float(memory_mb.strip()))
    except Exception:
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.name, int(props.total_memory / 1024 / 1024)
        except Exception:
            pass
    return "Unknown NVIDIA GPU", 0


def env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


def main() -> None:
    gpu_name, memory_mb = detect_gpu()
    memory_gb = round(memory_mb / 1024, 1) if memory_mb else 0.0
    requested_profile = (os.getenv("PROFILE", "auto").strip() or "auto").lower()
    upper_name = gpu_name.upper()

    is_rtx6000_ada = "RTX 6000 ADA" in upper_name
    name_only_h200_fallback = memory_mb == 0 and "H200" in upper_name
    name_only_h100_fallback = memory_mb == 0 and ("H100" in upper_name or "A100" in upper_name)
    name_only_rtx6000_ada_fallback = memory_mb == 0 and is_rtx6000_ada

    if memory_mb >= 120 * 1024 or name_only_h200_fallback:
        tier = "h200"
        auto_profile = "vast-h100"
        model_mode = "bf16"
        default_candidates = 4
        max_candidates = 8
        comfy_gpu_mode = "gpu-only"
        cache_lru = 6
        reserve_vram = 2.0
    elif memory_mb >= 70 * 1024 or name_only_h100_fallback:
        tier = "h100-class"
        auto_profile = "vast-h100"
        model_mode = "bf16"
        default_candidates = 2
        max_candidates = 4
        comfy_gpu_mode = "highvram"
        cache_lru = 3
        reserve_vram = 1.5
    elif (
        is_rtx6000_ada and memory_mb >= 44 * 1024
    ) or name_only_rtx6000_ada_fallback:
        tier = "rtx6000-ada"
        auto_profile = "vast-rtx6000-ada"
        model_mode = "bf16"
        default_candidates = 1
        max_candidates = 2
        comfy_gpu_mode = "normalvram"
        cache_lru = 1
        reserve_vram = 2.0
    elif memory_mb >= 40 * 1024:
        tier = "large-consumer"
        auto_profile = "local-4060"
        model_mode = "gguf"
        default_candidates = 2
        max_candidates = 3
        comfy_gpu_mode = "lowvram"
        cache_lru = 0
        reserve_vram = 1.5
    elif memory_mb >= 20 * 1024:
        tier = "midrange"
        auto_profile = "local-4060"
        model_mode = "gguf"
        default_candidates = 1
        max_candidates = 2
        comfy_gpu_mode = "lowvram"
        cache_lru = 0
        reserve_vram = 1.25
    else:
        tier = "low-vram"
        auto_profile = "local-4060"
        model_mode = "gguf"
        default_candidates = 1
        max_candidates = 1
        comfy_gpu_mode = "lowvram"
        cache_lru = 0
        reserve_vram = 1.0

    profile = auto_profile if requested_profile == "auto" else requested_profile
    supported_profiles = {"vast-h100", "vast-rtx6000-ada", "local-4060"}
    if profile not in supported_profiles:
        raise SystemExit(
            f"Unsupported PROFILE={profile!r}; use auto, vast-h100, vast-rtx6000-ada, or local-4060"
        )

    if profile == "vast-h100":
        model_mode = "bf16"
        if comfy_gpu_mode in {"lowvram", "normalvram"}:
            comfy_gpu_mode = "highvram"
    elif profile == "vast-rtx6000-ada":
        model_mode = "bf16"
        comfy_gpu_mode = "normalvram"
        default_candidates = min(default_candidates, 1)
        max_candidates = min(max_candidates, 2)
        cache_lru = min(cache_lru, 1)
        reserve_vram = max(reserve_vram, 2.0)
    else:
        model_mode = "gguf"
        comfy_gpu_mode = "lowvram"

    if memory_mb <= 10 * 1024:
        quant = "qwen-image-edit-2511-Q2_K.gguf"
    elif memory_mb <= 16 * 1024:
        quant = "qwen-image-edit-2511-Q3_K_M.gguf"
    else:
        quant = "qwen-image-edit-2511-Q4_K_M.gguf"

    values: dict[str, str] = {
        "REQUESTED_PROFILE": requested_profile,
        "PROFILE": profile,
        "GPU_NAME": gpu_name,
        "GPU_VRAM_MB": str(memory_mb),
        "GPU_VRAM_GB": str(memory_gb),
        "GPU_TIER": tier,
        "MODEL_MODE": model_mode,
        "DEFAULT_CANDIDATES": str(env_int("DEFAULT_CANDIDATES", default_candidates)),
        "MAX_CANDIDATES": str(env_int("MAX_CANDIDATES", max_candidates)),
        "LOCAL_GGUF_FILENAME": os.getenv("LOCAL_GGUF_FILENAME", "").strip() or quant,
        "COMFY_GPU_MODE": os.getenv("COMFY_GPU_MODE", "").strip() or comfy_gpu_mode,
        "COMFY_CACHE_LRU": str(env_int("COMFY_CACHE_LRU", cache_lru)),
        "RESERVE_VRAM_GB": str(env_float("RESERVE_VRAM_GB", reserve_vram)),
    }

    if int(values["DEFAULT_CANDIDATES"]) > int(values["MAX_CANDIDATES"]):
        values["DEFAULT_CANDIDATES"] = values["MAX_CANDIDATES"]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "hardware.json").write_text(json.dumps(values, indent=2), encoding="utf-8")

    for key, value in values.items():
        print(f"export {key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
