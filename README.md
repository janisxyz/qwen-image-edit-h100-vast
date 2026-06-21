# Qwen Image Edit 2511 — Adaptive ComfyUI API

A reusable ComfyUI container and authenticated API for Qwen-Image-Edit-2511. The container detects the NVIDIA GPU and available VRAM at startup, selects the appropriate model format, tunes ComfyUI memory mode, and chooses safe candidate-batch defaults.

| Detected hardware | Model path | ComfyUI mode | Automatic candidates |
|---|---|---|---|
| RTX 4060 / up to 10GB | Q2_K GGUF | low VRAM + CPU VAE | default 1, max 1 |
| 11–16GB | Q3_K_M GGUF | low VRAM + CPU VAE | default 1, max 1 |
| 20–39GB | Q4_K_M GGUF | low VRAM + CPU VAE | default 1, max 2 |
| Generic 40–69GB GPUs | Q4_K_M GGUF | low VRAM + CPU VAE | default 2, max 3 |
| RTX 6000 Ada 48GB | BF16 diffusion + FP8 text encoder | normal VRAM + smart offload | default 1, max 1 |
| H100/A100 class, 70–119GB | BF16 diffusion + FP8 text encoder | high VRAM | default 2, max 4 |
| H200 NVL class, 120GB+ | BF16 diffusion + FP8 text encoder | GPU only | default 4, max 8 |

The production workflow uses the **full BF16 model at 20 steps and CFG 4**. Turbo/Lightning is not attached to the workflow.

Both profiles automatically download missing models into `/workspace/models`. Reattaching the same persistent volume prevents redownloads.

## Automatic adaptation

Leave this in the environment:

```env
PROFILE=auto
```

At startup, `scripts/detect_hardware.py` reads the GPU name and total VRAM through `nvidia-smi`, then exports:

```text
PROFILE
GPU_NAME
GPU_VRAM_GB
GPU_TIER
MODEL_MODE
COMFY_GPU_MODE
DEFAULT_CANDIDATES
MAX_CANDIDATES
LOCAL_GGUF_FILENAME
```

The detected values are also stored at:

```text
/workspace/hardware.json
```

Check them through:

```bash
curl http://localhost:8000/health
curl -H "Authorization: Bearer local-test-key" http://localhost:8000/v1/capabilities
```

Omit `candidates` in an API request to use the detected automatic default. Every value can still be overridden through environment variables for benchmarking.

## Local RTX 4060 test

Requirements:

- Docker Desktop with WSL2 backend
- Current NVIDIA driver and Docker GPU support
- 8GB VRAM
- 32GB system RAM recommended
- At least 40GB free storage

```powershell
Copy-Item .env.example .env -ErrorAction SilentlyContinue; docker compose -f docker-compose.4060.yml up --build
```

First startup downloads the automatically selected GGUF model, FP8 Qwen 2.5 VL encoder and Qwen VAE.

```powershell
curl http://localhost:8000/health
curl -H "Authorization: Bearer local-test-key" http://localhost:8000/v1/models
```

ComfyUI is available locally at `http://localhost:8188`.

## Cloud images

GitHub Actions builds and publishes:

```text
ghcr.io/janisxyz/qwen-image-edit-h100-vast:h100
ghcr.io/janisxyz/qwen-image-edit-h100-vast:h200-nvl
ghcr.io/janisxyz/qwen-image-edit-h100-vast:rtx6000-ada
ghcr.io/janisxyz/qwen-image-edit-h100-vast:4060
```

The H100, H200 and RTX 6000 Ada tags use the same adaptive BF16 cloud image. With `PROFILE=auto`:

- H200 NVL selects BF16, GPU-only mode, default batch 4 and maximum batch 8.
- H100/A100 class selects BF16, high-VRAM mode, default batch 2 and maximum batch 4.
- RTX 6000 Ada selects BF16, normal-VRAM smart offloading, a 2GB VRAM reserve and one candidate per inference.

The RTX 6000 Ada profile is deliberately conservative because the BF16 diffusion model, FP8 text encoder, VAE and runtime buffers do not all fit simultaneously in 48GB VRAM. Smart offloading keeps the full-quality BF16 workflow while avoiding an unsafe GPU-only configuration.

For Vast.ai:

1. Create a persistent volume of at least 180GB and mount it at `/workspace`.
2. Choose the matching image tag for the rented GPU.
3. Expose TCP port `8000`.
4. Set `API_KEY`.
5. Set `PROFILE=auto`.
6. Keep raw ComfyUI internal unless you specifically need port `8188/tcp` for debugging.

Example RTX 6000 Ada image:

```text
ghcr.io/janisxyz/qwen-image-edit-h100-vast:rtx6000-ada
```

Example Docker options:

```text
-p 8000:8000 -e PROFILE=auto -e API_KEY=REPLACE_WITH_A_LONG_RANDOM_SECRET -e COMFY_LISTEN_HOST=127.0.0.1 -e DOWNLOAD_LOG_INTERVAL_SECONDS=5
```

The first boot downloads the required model set. Later boots reuse the persistent volume.

## Reference selection

Qwen-Image-Edit-2511 receives up to three references in this workflow. Keep all 4–6 photos per doll, but choose the best three for each shot:

- Front shot: full frontal, face close-up, body/clothing detail
- Three-quarter shot: three-quarter, face, full body
- Rear shot: rear/side, proportions, hair/material detail

## Upload references

```bash
curl -X POST http://localhost:8000/v1/assets \
  -H "Authorization: Bearer local-test-key" \
  -F "file=@front.jpg"
```

Repeat for the other references. The response returns an `asset` path.

## Submit an automatically sized job

Leave out `candidates`:

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H "Authorization: Bearer local-test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a premium studio product photograph. Preserve the exact doll identity, face, hair, makeup, material and body proportions. Change only pose, camera and lighting.",
    "references": ["assets/FIRST.jpg", "assets/SECOND.jpg", "assets/THIRD.jpg"],
    "seed": 10001,
    "steps": 20,
    "cfg": 4.0,
    "output_prefix": "doll_001/shot_01"
  }'
```

Poll:

```bash
curl -H "Authorization: Bearer local-test-key" http://localhost:8000/v1/jobs/PROMPT_ID
```

## Queue a manifest

```bash
python -m pip install httpx
python tools/submit_manifest.py \
  --api http://localhost:8000 \
  --token local-test-key \
  --manifest manifest.example.json
```

## Benchmark exact throughput

Hardware detection chooses safe defaults, but the exact fastest batch depends on reference resolution and the host. Benchmark the allowed range once:

```bash
python tools/benchmark.py \
  --api https://YOUR-VAST-ENDPOINT \
  --token YOUR_API_KEY \
  --reference front.jpg \
  --reference face.jpg \
  --reference three_quarter.jpg \
  --prompt "Create a premium studio product photograph while preserving the exact doll identity." \
  --candidate-sizes 1,2,4,6,8
```

Use the batch size with the lowest `seconds_per_image`, not the highest momentary GPU utilization.

## Automatic model downloads

`scripts/bootstrap_models.py` is idempotent:

1. Checks the detected target model path.
2. Reuses existing model files.
3. Downloads only missing files.
4. Stores the Hugging Face cache under `/workspace/cache/huggingface`.
5. Atomically moves completed files into ComfyUI's model folders.

Destroy the GPU instance only when `/workspace` is persistent. Keep another copy of references and outputs.

## Security

The public API requires a bearer token. Cloud profiles bind raw ComfyUI to localhost. The local 4060 compose file exposes port 8188 for debugging only.
