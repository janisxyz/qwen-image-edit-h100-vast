from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

VERSION = "1.1.0"
PROFILE = os.getenv("PROFILE", "local-4060")
REQUESTED_PROFILE = os.getenv("REQUESTED_PROFILE", PROFILE)
GPU_NAME = os.getenv("GPU_NAME", "Unknown NVIDIA GPU")
GPU_VRAM_GB = float(os.getenv("GPU_VRAM_GB", "0") or 0)
GPU_TIER = os.getenv("GPU_TIER", "unknown")
MODEL_MODE = os.getenv("MODEL_MODE", "gguf")
COMFY_GPU_MODE = os.getenv("COMFY_GPU_MODE", "lowvram")
DEFAULT_CANDIDATES = int(os.getenv("DEFAULT_CANDIDATES", "1"))
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "1"))
LOCAL_GGUF_FILENAME = os.getenv("LOCAL_GGUF_FILENAME", "qwen-image-edit-2511-Q2_K.gguf")
COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
WORKFLOW_DIR = Path(os.getenv("WORKFLOW_DIR", "/workspace/workflows"))
INPUT_DIR = Path(os.getenv("COMFY_INPUT_DIR", "/workspace/input")).resolve()
OUTPUT_DIR = Path(os.getenv("COMFY_OUTPUT_DIR", "/workspace/output")).resolve()
API_KEY = os.getenv("API_KEY", "")

app = FastAPI(
    title="Qwen Image Edit ComfyUI API",
    version=VERSION,
    description=f"Resolved profile: {PROFILE}; GPU: {GPU_NAME}",
)


def auth(authorization: str | None = Header(default=None)) -> None:
    expected = f"Bearer {API_KEY}"
    if not API_KEY:
        raise HTTPException(500, "API_KEY is not configured")
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(401, "Invalid bearer token")


def clean_relative(value: str) -> str:
    value = value.replace("\\", "/")
    cleaned = re.sub(r"[^A-Za-z0-9._/-]+", "_", value).strip("._/")
    if not cleaned or ".." in cleaned.split("/"):
        raise ValueError("Unsafe relative path")
    return cleaned[:220]


def child_path(root: Path, relative: str) -> Path:
    candidate = (root / clean_relative(relative)).resolve()
    if candidate != root and root not in candidate.parents:
        raise HTTPException(400, "Unsafe path")
    return candidate


class JobRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=12_000)
    references: list[str] = Field(min_length=1, max_length=3)
    candidates: int | None = Field(default=None, ge=1)
    seed: int = Field(default=1, ge=0, le=2**64 - 1)
    steps: int = Field(default=20, ge=1, le=100)
    cfg: float = Field(default=4.0, ge=0.0, le=20.0)
    output_prefix: str = "qwen-edit/result"

    @field_validator("references")
    @classmethod
    def refs_are_safe(cls, refs: list[str]) -> list[str]:
        return [clean_relative(ref) for ref in refs]

    @field_validator("output_prefix")
    @classmethod
    def prefix_is_safe(cls, value: str) -> str:
        return clean_relative(value)


class BatchRequest(BaseModel):
    jobs: list[JobRequest] = Field(min_length=1, max_length=1000)


async def get_json(path: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{COMFY_URL}{path}")
        response.raise_for_status()
        return response.json()


async def post_json(path: str, body: dict) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{COMFY_URL}{path}", json=body)
        if response.status_code >= 400:
            raise HTTPException(response.status_code, response.text)
        return response.json()


def workflow_name() -> str:
    if PROFILE == "vast-h100":
        return "qwen_edit_h100.json"
    if PROFILE == "local-4060":
        return "qwen_edit_4060.json"
    raise HTTPException(500, f"Unknown PROFILE={PROFILE}")


def resolve_candidates(job: JobRequest) -> int:
    candidates = job.candidates if job.candidates is not None else DEFAULT_CANDIDATES
    if candidates > MAX_CANDIDATES:
        raise HTTPException(
            400,
            f"Detected {GPU_NAME} allows at most {MAX_CANDIDATES} candidates per inference; "
            f"omit candidates to use the automatic default of {DEFAULT_CANDIDATES}.",
        )
    return candidates


def build_workflow(job: JobRequest) -> tuple[dict, int]:
    candidates = resolve_candidates(job)
    path = WORKFLOW_DIR / workflow_name()
    if not path.is_file():
        raise HTTPException(500, f"Workflow missing: {path}")
    workflow = json.loads(path.read_text(encoding="utf-8"))

    refs = list(job.references)
    while len(refs) < 3:
        refs.append(refs[-1])

    for node_id, ref in zip(("1", "2", "3"), refs):
        if not child_path(INPUT_DIR, ref).is_file():
            raise HTTPException(400, f"Reference does not exist: {ref}")
        workflow[node_id]["inputs"]["image"] = ref

    if PROFILE == "local-4060":
        workflow["5"]["inputs"]["unet_name"] = LOCAL_GGUF_FILENAME

    workflow["10"]["inputs"]["prompt"] = job.prompt
    workflow["15"]["inputs"]["amount"] = candidates
    workflow["16"]["inputs"]["seed"] = job.seed
    workflow["16"]["inputs"]["steps"] = job.steps
    workflow["16"]["inputs"]["cfg"] = job.cfg
    workflow["18"]["inputs"]["filename_prefix"] = job.output_prefix
    return workflow, candidates


def capabilities_payload() -> dict:
    return {
        "requested_profile": REQUESTED_PROFILE,
        "resolved_profile": PROFILE,
        "gpu_name": GPU_NAME,
        "gpu_vram_gb": GPU_VRAM_GB,
        "gpu_tier": GPU_TIER,
        "model_mode": MODEL_MODE,
        "comfy_gpu_mode": COMFY_GPU_MODE,
        "default_candidates": DEFAULT_CANDIDATES,
        "max_candidates": MAX_CANDIDATES,
        "gguf_filename": LOCAL_GGUF_FILENAME if MODEL_MODE == "gguf" else None,
        "turbo": False,
        "steps_default": 20,
        "cfg_default": 4.0,
    }


@app.get("/health")
async def health() -> dict:
    try:
        stats = await get_json("/system_stats")
    except Exception as exc:
        raise HTTPException(503, f"ComfyUI unavailable: {exc}") from exc
    return {
        "ok": True,
        "version": VERSION,
        "capabilities": capabilities_payload(),
        "comfyui": stats,
    }


@app.get("/v1/capabilities", dependencies=[Depends(auth)])
async def capabilities() -> dict:
    return capabilities_payload()


@app.get("/v1/models", dependencies=[Depends(auth)])
async def models() -> dict:
    files = {
        "bf16": Path("/workspace/models/diffusion_models/qwen_image_edit_2511_bf16.safetensors"),
        "gguf_q2": Path("/workspace/models/unet/qwen-image-edit-2511-Q2_K.gguf"),
        "gguf_q3": Path("/workspace/models/unet/qwen-image-edit-2511-Q3_K_M.gguf"),
        "gguf_q4": Path("/workspace/models/unet/qwen-image-edit-2511-Q4_K_M.gguf"),
        "text_encoder": Path("/workspace/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
        "vae": Path("/workspace/models/vae/qwen_image_vae.safetensors"),
    }
    return {
        "capabilities": capabilities_payload(),
        "files": {
            name: {
                "present": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
            }
            for name, path in files.items()
        },
    }


@app.post("/v1/assets", dependencies=[Depends(auth)])
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(400, "Allowed formats: PNG, JPG, JPEG, WEBP")
    relative = f"assets/{int(time.time())}_{secrets.token_hex(6)}{suffix}"
    target = child_path(INPUT_DIR, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as destination:
        shutil.copyfileobj(file.file, destination)
    return {"asset": relative, "bytes": target.stat().st_size}


@app.post("/v1/jobs", dependencies=[Depends(auth)])
async def create_job(job: JobRequest) -> dict:
    workflow, candidates = build_workflow(job)
    result = await post_json("/prompt", {"prompt": workflow})
    return {
        "prompt_id": result.get("prompt_id"),
        "number": result.get("number"),
        "candidates": candidates,
        "node_errors": result.get("node_errors", {}),
    }


@app.post("/v1/batches", dependencies=[Depends(auth)])
async def create_batch(batch: BatchRequest) -> dict:
    queued = []
    for job in batch.jobs:
        workflow, candidates = build_workflow(job)
        result = await post_json("/prompt", {"prompt": workflow})
        queued.append({
            "prompt_id": result.get("prompt_id"),
            "number": result.get("number"),
            "candidates": candidates,
            "output_prefix": job.output_prefix,
        })
    return {"count": len(queued), "queued": queued}


@app.get("/v1/jobs/{prompt_id}", dependencies=[Depends(auth)])
async def job(prompt_id: str) -> dict:
    history = await get_json(f"/history/{prompt_id}")
    if prompt_id not in history:
        return {"prompt_id": prompt_id, "status": "queued_or_running"}

    item = history[prompt_id]
    outputs = []
    for node in item.get("outputs", {}).values():
        for image in node.get("images", []):
            outputs.append({
                **image,
                "download": (
                    f"/v1/files?filename={image.get('filename','')}"
                    f"&subfolder={image.get('subfolder','')}"
                    f"&type={image.get('type','output')}"
                ),
            })
    status = item.get("status", {})
    return {
        "prompt_id": prompt_id,
        "completed": status.get("completed", True),
        "status": status.get("status_str", "completed"),
        "outputs": outputs,
        "raw_status": status,
    }


@app.get("/v1/files", dependencies=[Depends(auth)])
async def file(filename: str, subfolder: str = "", type: Literal["output"] = "output"):
    del type
    relative = f"{subfolder}/{filename}" if subfolder else filename
    target = child_path(OUTPUT_DIR, relative)
    if not target.is_file():
        raise HTTPException(404, "Output file not found")
    return FileResponse(target)


@app.get("/v1/queue", dependencies=[Depends(auth)])
async def queue() -> dict:
    return await get_json("/queue")


@app.post("/v1/interrupt", dependencies=[Depends(auth)])
async def interrupt() -> dict:
    return await post_json("/interrupt", {})
