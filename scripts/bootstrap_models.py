from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace"))
PROFILE = os.getenv("PROFILE", "vast-h100")
HF_TOKEN = os.getenv("HF_TOKEN") or None
DOWNLOAD_LIGHTNING = os.getenv("DOWNLOAD_LIGHTNING_LORA", "0") == "1"
LOCAL_GGUF_FILENAME = os.getenv("LOCAL_GGUF_FILENAME", "qwen-image-edit-2511-Q2_K.gguf")
LOG_INTERVAL = max(float(os.getenv("DOWNLOAD_LOG_INTERVAL_SECONDS", "5")), 1.0)
MAX_RETRIES = max(int(os.getenv("MODEL_DOWNLOAD_RETRIES", "5")), 1)
WORKERS = max(int(os.getenv("MODEL_DOWNLOAD_WORKERS", "3")), 1)
CHUNK_SIZE = max(int(os.getenv("MODEL_DOWNLOAD_CHUNK_MB", "8")), 1) * 1024 * 1024

COMMON = [
    {
        "label": "text-encoder",
        "repo_id": "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "filename": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "target": DATA_DIR / "models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    },
    {
        "label": "vae",
        "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
        "filename": "split_files/vae/qwen_image_vae.safetensors",
        "target": DATA_DIR / "models/vae/qwen_image_vae.safetensors",
    },
]

if PROFILE == "vast-h100":
    MODELS = COMMON + [
        {
            "label": "diffusion-bf16",
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
            "label": "diffusion-gguf",
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
            "label": "lightning-lora",
            "repo_id": "lightx2v/Qwen-Image-Edit-2511-Lightning",
            "filename": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "target": DATA_DIR / "models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        }
    )


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(label: str, message: str) -> None:
    print(f"{timestamp()} [models:{label}] {message}", flush=True)


def human_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    return f"{value:.2f} {units[index]}"


def human_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "unknown"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def auth_headers() -> dict[str, str]:
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "qwen-image-edit-bootstrap/1.2",
    }
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def remote_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{quote(filename, safe='/')}"


def remote_size(client: httpx.Client, url: str, label: str) -> int:
    log(label, "requesting remote file metadata")
    response = client.head(url)
    response.raise_for_status()
    raw = response.headers.get("x-linked-size") or response.headers.get("content-length") or "0"
    try:
        size = int(raw)
    except ValueError:
        size = 0
    if size:
        log(label, f"remote size={human_bytes(size)}")
    else:
        log(label, "remote size unavailable; progress will show bytes and speed without percentage")
    return size


def stream_download(item: dict[str, object]) -> str:
    label = str(item["label"])
    repo_id = str(item["repo_id"])
    filename = str(item["filename"])
    target = Path(item["target"])
    partial = target.with_suffix(target.suffix + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)

    log(label, f"checking target={target}")
    if target.is_file() and target.stat().st_size > 1_000_000:
        size = target.stat().st_size
        log(label, f"cache hit; existing model is ready ({human_bytes(size)})")
        return f"cached {target} ({human_bytes(size)})"

    url = remote_url(repo_id, filename)
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=30.0)
    started_all = time.monotonic()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(headers=auth_headers(), follow_redirects=True, timeout=timeout) as client:
                expected_total = remote_size(client, url, label)
                existing = partial.stat().st_size if partial.is_file() else 0

                if expected_total and existing > expected_total:
                    log(label, "partial file is larger than remote file; discarding invalid partial")
                    partial.unlink(missing_ok=True)
                    existing = 0

                request_headers: dict[str, str] = {}
                if existing:
                    request_headers["Range"] = f"bytes={existing}-"
                    log(
                        label,
                        f"attempt {attempt}/{MAX_RETRIES}; resuming at {human_bytes(existing)}",
                    )
                else:
                    log(label, f"attempt {attempt}/{MAX_RETRIES}; starting download")

                with client.stream("GET", url, headers=request_headers) as response:
                    if response.status_code == 416 and expected_total and existing == expected_total:
                        log(label, "partial file already contains the complete remote file")
                    else:
                        response.raise_for_status()
                        resumed = existing > 0 and response.status_code == 206
                        if existing and not resumed:
                            log(label, "server did not accept resume request; restarting from byte zero")
                            existing = 0

                        content_length = int(response.headers.get("content-length") or 0)
                        total = expected_total or (existing + content_length if resumed else content_length)
                        mode = "ab" if resumed else "wb"
                        current = existing
                        attempt_start = time.monotonic()
                        last_log_at = attempt_start
                        last_log_bytes = current

                        with partial.open(mode) as destination:
                            for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                                if not chunk:
                                    continue
                                destination.write(chunk)
                                current += len(chunk)
                                now = time.monotonic()
                                if now - last_log_at >= LOG_INTERVAL:
                                    interval = max(now - last_log_at, 0.001)
                                    speed = (current - last_log_bytes) / interval
                                    elapsed = max(now - attempt_start, 0.001)
                                    average_speed = max((current - existing) / elapsed, 0.001)
                                    if total:
                                        percent = min(current / total * 100, 100.0)
                                        remaining = max(total - current, 0)
                                        eta = remaining / average_speed
                                        progress = (
                                            f"{percent:6.2f}%  {human_bytes(current)} / {human_bytes(total)}"
                                        )
                                    else:
                                        eta = None
                                        progress = human_bytes(current)
                                    log(
                                        label,
                                        f"download {progress} | speed={human_bytes(speed)}/s "
                                        f"| avg={human_bytes(average_speed)}/s | eta={human_duration(eta)}",
                                    )
                                    last_log_at = now
                                    last_log_bytes = current
                            destination.flush()
                            os.fsync(destination.fileno())

                        if total and current != total:
                            raise RuntimeError(
                                f"incomplete download: received {current} bytes, expected {total} bytes"
                            )

                final_size = partial.stat().st_size
                if final_size <= 1_000_000:
                    raise RuntimeError(
                        f"downloaded model is unexpectedly small: {final_size} bytes"
                    )

                log(label, f"download complete; moving partial into place ({human_bytes(final_size)})")
                partial.replace(target)
                elapsed_all = time.monotonic() - started_all
                log(
                    label,
                    f"ready target={target} size={human_bytes(final_size)} elapsed={human_duration(elapsed_all)}",
                )
                return f"ready {target} ({human_bytes(final_size)})"
        except Exception as exc:
            if attempt >= MAX_RETRIES:
                log(label, f"FAILED after {attempt} attempts: {type(exc).__name__}: {exc}")
                raise
            delay = min(2 ** attempt, 30)
            partial_size = partial.stat().st_size if partial.is_file() else 0
            log(
                label,
                f"attempt {attempt} failed: {type(exc).__name__}: {exc}; "
                f"partial={human_bytes(partial_size)}; retrying in {delay}s",
            )
            time.sleep(delay)

    raise RuntimeError("unreachable download state")


def main() -> None:
    started = time.monotonic()
    workers = min(WORKERS, len(MODELS))
    log("bootstrap", f"profile={PROFILE} models={len(MODELS)} parallel_workers={workers}")
    log("bootstrap", f"data_dir={DATA_DIR} progress_interval={LOG_INTERVAL:.1f}s retries={MAX_RETRIES}")
    log("bootstrap", f"huggingface_auth={'configured' if HF_TOKEN else 'anonymous'}")

    for index, item in enumerate(MODELS, start=1):
        log(
            "bootstrap",
            f"plan {index}/{len(MODELS)} label={item['label']} repo={item['repo_id']} "
            f"file={item['filename']} target={item['target']}",
        )

    results: list[str] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="model-download") as pool:
        futures = {pool.submit(stream_download, item): str(item["label"]) for item in MODELS}
        for future in as_completed(futures):
            label = futures[future]
            result = future.result()
            results.append(result)
            log("bootstrap", f"completed label={label}: {result}")

    elapsed = time.monotonic() - started
    log("bootstrap", f"all required models ready; count={len(results)} elapsed={human_duration(elapsed)}")


if __name__ == "__main__":
    main()
