# Qwen Image Edit 2511 — ComfyUI API

Two profiles with the same API and workflow semantics:

| Profile | Intended hardware | Model | Candidate batching |
|---|---|---|---|
| `vast-h100` | H100/A100 80GB | BF16 diffusion + FP8 text encoder | 1–4, benchmark it |
| `local-4060` | RTX 4060 8GB test machine | Q2_K GGUF + CPU text encoder/VAE | 1 |

Both profiles automatically download missing models into `/workspace/models`. Reattaching the same volume prevents redownloads.

## Why the profiles differ

The cloud profile keeps the diffusion model in BF16 for final quality and uses candidate batching to improve throughput. The 4060 profile is a functional test environment: Q2_K GGUF, aggressive offloading, CPU VAE, one candidate. It verifies references, prompts, API jobs, filenames and output handling; it is not a speed or final-quality benchmark.

Qwen-Image-Edit-2511 currently accepts up to three references in the native ComfyUI `TextEncodeQwenImageEditPlus` node. Keep all 4–6 photos per doll, but choose the best three per shot.

## Local RTX 4060 test

Requirements:

- Docker Desktop with WSL2 backend
- Current NVIDIA driver and Docker GPU support
- 8GB VRAM
- 32GB system RAM recommended; 16GB plus a large pagefile may work very slowly
- At least 40GB free storage

```bash
cp .env.example .env
docker compose -f docker-compose.4060.yml up --build
```

First startup downloads the Q2_K GGUF, FP8 Qwen 2.5 VL encoder and Qwen VAE.

```bash
curl http://localhost:8000/health
curl -H "Authorization: Bearer local-test-key" http://localhost:8000/v1/models
```

ComfyUI is available locally at `http://localhost:8188`.

To test a larger quant later:

```env
LOCAL_GGUF_FILENAME=qwen-image-edit-2511-Q3_K_M.gguf
```

or:

```env
LOCAL_GGUF_FILENAME=qwen-image-edit-2511-Q4_K_M.gguf
```

Those require more RAM/offloading and are unnecessary merely to validate the API.

## H100 production

GitHub Actions builds:

```text
ghcr.io/janisxyz/qwen-image-edit-h100-vast:h100
ghcr.io/janisxyz/qwen-image-edit-h100-vast:4060
```

For Vast.ai:

1. Create a persistent volume of at least 180GB and mount it at `/workspace`.
2. Use `ghcr.io/janisxyz/qwen-image-edit-h100-vast:h100`.
3. Expose HTTP port `8000`.
4. Set `API_KEY`.
5. Select an H100 80GB offer.
6. Use `vast-template.example.json` as the field reference.

The first boot downloads the BF16 model set. Later boots reuse the persistent volume.

## Upload references

```bash
curl -X POST http://localhost:8000/v1/assets \
  -H "Authorization: Bearer local-test-key" \
  -F "file=@front.jpg"
```

Repeat for the other references. The response returns an `asset` path.

## Submit one job

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H "Authorization: Bearer local-test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a premium studio product photograph. Preserve the exact doll identity, face, hair, makeup, material and body proportions. Change only pose, camera and lighting.",
    "references": ["assets/FIRST.jpg", "assets/SECOND.jpg", "assets/THIRD.jpg"],
    "candidates": 1,
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

## Benchmark H100 batching

```bash
python tools/benchmark.py \
  --api https://YOUR-VAST-ENDPOINT \
  --token YOUR_API_KEY \
  --reference front.jpg \
  --reference face.jpg \
  --reference three_quarter.jpg \
  --prompt "Create a premium studio product photograph while preserving the exact doll identity." \
  --candidate-sizes 1,2,3,4
```

Use the batch size with the lowest `seconds_per_image`, not the one with the highest momentary utilization.

## Automatic model downloads

`scripts/bootstrap_models.py` is idempotent:

1. Checks the target path.
2. Reuses existing model files.
3. Downloads only missing files.
4. Stores the Hugging Face cache under `/workspace/cache/huggingface`.
5. Atomically moves completed files into ComfyUI's model folders.

Destroy the GPU instance only when `/workspace` is persistent. Keep another copy of references and outputs.

## Security

The public API requires a bearer token. The H100 profile keeps raw ComfyUI bound to localhost inside the container. The local 4060 compose file exposes port 8188 for debugging only.
