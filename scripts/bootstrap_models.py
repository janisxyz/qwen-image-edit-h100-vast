from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace"))
PROFILE = os.getenv("PROFILE", "vast-h100")
HF_TOKEN = os.getenv("HF_TOKEN") or None
DOWNLOAD_LIGHTNING = os.getenv("DOWNLOAD_LIGHTNING_LORA", "0") == "1"
LOCAL_GGUF_FILENAME = os.getenv("LOCAL_GGUF_FILENAME", "qwen-image-edit-2511-Q2_K.gguf")

COMMON = [
    {
        "repo_id": "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "filename": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "target": DATA_DIR / "models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    },
    {
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "filename": "split_files/vae/qwen_image_vae.safetensors",
        "target": DATA_DIR / "models/vae/qwen_image_vae.safetensors",
    },
]

if PROFILE == "vast-h100":
    MODELS = COMMON + [
        {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
            "target": DATA_DIR / "models/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
        }
    ]
elif PROFILE == "local-4060":
    allowed = {
        "qwen-image-edit-2511-Q2_K.gguf",
        "qwen-image-edit-2511-Q3_K_M.gguf",
        "qwen-image-edit-2511-Q4_K_M.gguf",
    }
    if LOCAL_GGUF_FILENAME not in allowed:
        raise SystemExit(
            f"Unsupported LOCAL_GGUF_FILENAME={LOCAL_GGUF_FILENAME!r}; choose one of {sorted(allowed)}"
        )
    MODELS = COMMON + [
        {
            "repo_id": "unsloth/Qwen-Image-Edit-2511-GGUF",
            "filename": LOCAL_GGUF_FILENAME,
            "target": DATA_DIR / f"models/unet/{LOCAL_GGUF_FILENAME}",
        }
    ]
else:
    raise SystemExit(f"Unknown PROFILE={PROFILE}")

if DOWNLOAD_LIGHTNING and PROFILE == "vast-h100":
    MODELS.append(
        {
            "repo_id": "lightx2v/Qwen-Image-Edit-2511-Lightning",
            "filename": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "target": DATA_DIR / "models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        }
    )


def materialize(item: dict[str, object]) -> str:
    target = Path(item["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 1_000_000:
        return f"cached: {target}"

    downloaded = Path(
        hf_hub_download(
            repo_id=str(item["repo_id"]),
            filename=str(item["filename"]),
            token=HF_TOKEN,
            cache_dir=str(DATA_DIR / "cache/huggingface"),
        )
    )
    temp = target.with_suffix(target.suffix + ".partial")
    temp.unlink(missing_ok=True)
    try:
        os.link(downloaded, temp)
    except OSError:
        shutil.copy2(downloaded, temp)
    temp.replace(target)

    if target.stat().st_size <= 1_000_000:
        raise RuntimeError(f"Downloaded model is unexpectedly small: {target}")
    return f"ready: {target}"


def main() -> None:
    with ThreadPoolExecutor(max_workers=min(3, len(MODELS))) as pool:
        futures = [pool.submit(materialize, item) for item in MODELS]
        for future in as_completed(futures):
            print(f"[models] {future.result()}", flush=True)


if __name__ == "__main__":
    main()
