# Qwen Image Edit 2511 — Adaptive ComfyUI API

Authenticated ComfyUI API for the full Qwen-Image-Edit-2511 base model. The container detects the NVIDIA GPU and VRAM at startup, selects BF16 or GGUF, chooses a safe ComfyUI memory mode, downloads missing model files into `/workspace`, and exposes a small FastAPI service.

## Quality policy

This project intentionally uses the **full base model**, not Turbo or the optional Lightning acceleration LoRA.

The workflow follows the official high-quality Qwen/ComfyUI values:

| Setting | Value |
|---|---:|
| Steps | 40 |
| CFG / true CFG | 4.0 |
| Sampler | Euler |
| Scheduler | Simple |
| Denoise | 1.0 |
| Model sampling shift | 3.1 |
| CFGNorm strength | 1.0 |
| Reference latent method | `index_timestep_zero` |
| Turbo | Disabled |
| Lightning LoRA | Not attached |

The API allows 20–100 steps, but defaults to 40. Forty steps are the official Qwen quality setting; ComfyUI documents 20 as a faster compromise.

## Automatic hardware profiles

Set:

```env
PROFILE=auto
```

| Detected hardware | Model | ComfyUI runtime | Candidate default / max |
|---|---|---|---:|
| RTX 4060 or up to 10GB | Q2_K GGUF | `--lowvram --cpu-vae` | 1 / 1 |
| 11–16GB | Q3_K_M GGUF | `--lowvram --cpu-vae` | 1 / 1 |
| 20–39GB | Q4_K_M GGUF | `--lowvram --cpu-vae` | 1 / 2 |
| Generic 40–69GB | Q4_K_M GGUF | `--lowvram --cpu-vae` | 2 / 3 |
| RTX 6000 Ada 48GB | BF16 | default DynamicVRAM + RAM offload + 2GB reserve | 1 / 1 |
| H100/A100 70–119GB | BF16 | `--highvram` | 2 / 4 |
| H200 NVL 120GB+ | BF16 | `--gpu-only` | 4 / 8 |

For the RTX 6000 Ada, `normalvram` is an internal profile label. ComfyUI's normal/default VRAM mode has **no `--normalvram` CLI flag**, so the entrypoint intentionally passes no VRAM-mode flag. It keeps DynamicVRAM, asynchronous offloading, pinned RAM, and the default RAM-pressure cache enabled. This is preferable to disk-backed offloading on hosts with sufficient RAM.

Detected values are written to:

```text
/workspace/hardware.json
```

They are also returned by:

```bash
curl http://localhost:8000/health
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8000/v1/capabilities
```

## Published images

```text
ghcr.io/janisxyz/qwen-image-edit-h100-vast:h100
ghcr.io/janisxyz/qwen-image-edit-h100-vast:h200-nvl
ghcr.io/janisxyz/qwen-image-edit-h100-vast:rtx6000-ada
ghcr.io/janisxyz/qwen-image-edit-h100-vast:4060
```

The three cloud tags use the same adaptive cloud image. Their separate names make Vast.ai templates easier to understand.

## Vast.ai — RTX 6000 Ada

Use:

```text
Image: ghcr.io/janisxyz/qwen-image-edit-h100-vast:rtx6000-ada
Launch mode: Docker ENTRYPOINT
Entrypoint arguments: empty
Port: 8000/tcp
Volume mount: /workspace
Persistent storage: 180GB recommended
```

Docker options:

```text
-p 8000:8000 -e PROFILE=auto -e API_KEY=REPLACE_WITH_LONG_RANDOM_SECRET -e COMFY_LISTEN_HOST=127.0.0.1 -e DOWNLOAD_LOG_INTERVAL_SECONDS=5 -e MODEL_DOWNLOAD_WORKERS=3 -e MODEL_DOWNLOAD_RETRIES=5 -e RESERVE_VRAM_GB=2.0
```

The expected startup detection is:

```text
GPU_TIER=rtx6000-ada
PROFILE=vast-h100
MODEL_MODE=bf16
COMFY_GPU_MODE=normalvram
COMFY_CACHE_LRU=0
DEFAULT_CANDIDATES=1
MAX_CANDIDATES=1
```

The generated ComfyUI command should contain:

```text
--reserve-vram 2.0
```

It must **not** contain `--normalvram`, `--highvram`, `--gpu-only`, `--lowvram`, or `--fast-disk` for this profile.

See also:

```text
vast-template.rtx6000-ada.example.json
docker-compose.rtx6000-ada.yml
```

## Local RTX 4060

Requirements:

- Docker Desktop with WSL2 backend
- Current NVIDIA driver and Docker GPU support
- 8GB dedicated VRAM
- 32GB system RAM recommended
- At least 40GB free storage

```powershell
Copy-Item .env.example .env -ErrorAction SilentlyContinue; docker compose -f docker-compose.4060.yml up --build
```

ComfyUI is exposed locally at `http://localhost:8188`; the authenticated API is at `http://localhost:8000`.

## Model files

Cloud BF16 profiles download:

```text
models/diffusion_models/qwen_image_edit_2511_bf16.safetensors
models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
models/vae/qwen_image_vae.safetensors
```

Low-VRAM profiles download one of:

```text
models/unet/qwen-image-edit-2511-Q2_K.gguf
models/unet/qwen-image-edit-2511-Q3_K_M.gguf
models/unet/qwen-image-edit-2511-Q4_K_M.gguf
```

All downloads support progress logging, retries, partial-file resume, and atomic completion. Reusing the same `/workspace` volume prevents redownloads.

## API use

### Upload references

```bash
curl -X POST http://HOST:PORT/v1/assets \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@front.jpg"
```

Repeat for up to three references. Each response returns an `asset` path.

### Submit a full-quality job

```bash
curl -X POST http://HOST:PORT/v1/jobs \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a premium studio product photograph while preserving the exact subject identity.",
    "references": ["assets/FIRST.jpg", "assets/SECOND.jpg", "assets/THIRD.jpg"],
    "seed": 10001,
    "steps": 40,
    "cfg": 4.0,
    "output_prefix": "product/shot_01"
  }'
```

Omit `steps`, `cfg`, and `candidates` to use the detected defaults.

### Poll

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://HOST:PORT/v1/jobs/PROMPT_ID
```

### Download

Use the authenticated relative URL returned in `outputs[].download`.

## Batch and benchmark tools

```bash
python tools/submit_manifest.py --api http://HOST:PORT --token YOUR_API_KEY --manifest manifest.example.json
```

```bash
python tools/benchmark.py \
  --api http://HOST:PORT \
  --token YOUR_API_KEY \
  --reference front.jpg \
  --reference face.jpg \
  --reference three_quarter.jpg \
  --prompt "Preserve the exact subject identity while changing the camera angle." \
  --candidate-sizes 1,2
```

The benchmark defaults to the official 40-step quality mode.

## Validation safeguards

CI validates:

- Python, shell, JSON, and Compose syntax
- RTX 4060, RTX 6000 Ada, H100, and H200 detector outputs
- Exact 40-step / CFG 4 / Euler / Simple workflow settings
- Absence of Turbo, Lightning, and LoRA loader nodes
- Absence of the invalid `--normalvram` flag
- Dependency consistency and transformer import smoke tests

Docker builds also run shell/Python syntax checks and compare runtime flags against the pinned ComfyUI CLI.

## Security

The public FastAPI service requires a bearer token. Cloud profiles bind raw ComfyUI to `127.0.0.1`; expose only `8000/tcp` unless temporary ComfyUI debugging is required.
