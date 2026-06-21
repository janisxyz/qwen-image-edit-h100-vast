#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${PROFILE:-auto}"
DATA_DIR="${DATA_DIR:-/workspace}"
COMFYUI_DIR="${COMFYUI_DIR:-/opt/ComfyUI}"
APP_DIR="${APP_DIR:-/opt/qwen-worker}"
COMFY_PORT="${COMFY_PORT:-8188}"
COMFY_LISTEN_HOST="${COMFY_LISTEN_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
STARTED_AT="$(date +%s)"

log() {
  printf '%s [startup] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

on_error() {
  local exit_code=$?
  log "FAILED exit_code=$exit_code line=${BASH_LINENO[0]:-unknown} command=${BASH_COMMAND:-unknown}"
  exit "$exit_code"
}
trap on_error ERR

log "============================================================"
log "Qwen Image Edit container boot started"
log "container_user=$(id -u):$(id -g) hostname=$(hostname)"
log "requested_profile=$PROFILE data_dir=$DATA_DIR"
log "============================================================"

log "phase 1/8: preparing persistent directories"
mkdir -p \
  "$DATA_DIR/models/diffusion_models" \
  "$DATA_DIR/models/text_encoders" \
  "$DATA_DIR/models/vae" \
  "$DATA_DIR/models/loras" \
  "$DATA_DIR/models/unet" \
  "$DATA_DIR/input" \
  "$DATA_DIR/output" \
  "$DATA_DIR/workflows" \
  "$DATA_DIR/cache/huggingface" \
  "$DATA_DIR/logs"
log "persistent directories ready"
df -h "$DATA_DIR" 2>/dev/null | sed 's/^/[disk] /' || true

log "phase 2/8: linking ComfyUI data directories"
for name in models input output; do
  rm -rf "$COMFYUI_DIR/$name"
  ln -s "$DATA_DIR/$name" "$COMFYUI_DIR/$name"
  log "linked $COMFYUI_DIR/$name -> $DATA_DIR/$name"
done
cp -fv "$APP_DIR"/workflows/*.json "$DATA_DIR/workflows/" | sed 's/^/[workflow] /'
log "workflow templates synchronized"

log "phase 3/8: detecting NVIDIA hardware and selecting runtime profile"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>&1 \
    | sed 's/^/[gpu] /' || true
else
  log "WARNING: nvidia-smi is unavailable; detector will try PyTorch CUDA"
fi
python -u "$APP_DIR/scripts/detect_hardware.py" > /tmp/qwen-hardware.env
while IFS= read -r line; do
  log "detector_output=$line"
done < /tmp/qwen-hardware.env
# shellcheck disable=SC1091
source /tmp/qwen-hardware.env
log "GPU=$GPU_NAME VRAM=${GPU_VRAM_GB}GB tier=$GPU_TIER"
log "resolved_profile=$PROFILE model_mode=$MODEL_MODE comfy_gpu_mode=$COMFY_GPU_MODE"
log "default_candidates=$DEFAULT_CANDIDATES max_candidates=$MAX_CANDIDATES gguf=$LOCAL_GGUF_FILENAME"
log "hardware manifest written to $DATA_DIR/hardware.json"

log "phase 4/8: preparing API authentication"
if [[ -z "${API_KEY:-}" ]]; then
  if [[ -s "$DATA_DIR/API_KEY.txt" ]]; then
    export API_KEY="$(cat "$DATA_DIR/API_KEY.txt")"
    log "reused API key from $DATA_DIR/API_KEY.txt"
  else
    export API_KEY="$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
)"
    printf '%s\n' "$API_KEY" > "$DATA_DIR/API_KEY.txt"
    chmod 600 "$DATA_DIR/API_KEY.txt"
    log "generated a new API key at $DATA_DIR/API_KEY.txt"
  fi
else
  log "using API key supplied through environment"
fi
log "API key is configured; secret value is intentionally not printed"

log "phase 5/8: verifying and downloading required models"
MODEL_STARTED_AT="$(date +%s)"
log "model bootstrap starting; existing complete files will be reused"
log "download_progress_interval=${DOWNLOAD_LOG_INTERVAL_SECONDS:-5}s workers=${MODEL_DOWNLOAD_WORKERS:-3} retries=${MODEL_DOWNLOAD_RETRIES:-5}"
python -u "$APP_DIR/scripts/bootstrap_models.py"
MODEL_ELAPSED=$(( $(date +%s) - MODEL_STARTED_AT ))
log "model bootstrap completed in ${MODEL_ELAPSED}s"
find "$DATA_DIR/models" -type f -maxdepth 3 -printf '%p %s bytes\n' 2>/dev/null \
  | sort | sed 's/^/[model-file] /' || true

declare -a COMFY_ARGS=(
  --listen "$COMFY_LISTEN_HOST"
  --port "$COMFY_PORT"
  --preview-method none
)

case "$PROFILE" in
  vast-h100)
    if [[ "$COMFY_GPU_MODE" == "gpu-only" ]]; then
      COMFY_ARGS+=(--gpu-only)
    else
      COMFY_ARGS+=(--highvram)
    fi
    if [[ "${COMFY_CACHE_LRU:-0}" -gt 0 ]]; then
      COMFY_ARGS+=(--cache-lru "$COMFY_CACHE_LRU")
    fi
    ;;
  local-4060)
    COMFY_ARGS+=(
      --lowvram
      --cpu-vae
      --reserve-vram "${RESERVE_VRAM_GB:-1.0}"
      --cache-none
      --fast-disk
    )
    ;;
  *)
    log "unknown resolved PROFILE=$PROFILE"
    exit 2
    ;;
esac

if [[ "${DISABLE_SMART_MEMORY:-0}" == "1" ]]; then
  COMFY_ARGS+=(--disable-smart-memory)
fi

if [[ -n "${COMFY_ARGS_EXTRA:-}" ]]; then
  read -r -a EXTRA <<< "$COMFY_ARGS_EXTRA"
  COMFY_ARGS+=("${EXTRA[@]}")
fi

log "phase 6/8: starting ComfyUI"
printf -v COMFY_COMMAND '%q ' python -u main.py "${COMFY_ARGS[@]}"
log "command=$COMFY_COMMAND"
log "ComfyUI output is streamed live below and copied to $DATA_DIR/logs/comfyui.log"
cd "$COMFYUI_DIR"
python -u main.py "${COMFY_ARGS[@]}" > >(tee -a "$DATA_DIR/logs/comfyui.log") 2>&1 &
COMFY_PID=$!
log "ComfyUI process started pid=$COMFY_PID"

cleanup() {
  local reason="${1:-container shutdown}"
  log "cleanup requested: $reason"
  if kill -0 "$COMFY_PID" 2>/dev/null; then
    log "stopping ComfyUI pid=$COMFY_PID"
    kill "$COMFY_PID" 2>/dev/null || true
    wait "$COMFY_PID" 2>/dev/null || true
  fi
}
trap 'cleanup "signal received"; exit 0' TERM INT
trap 'cleanup "entrypoint exit"' EXIT

log "phase 7/8: waiting for ComfyUI health endpoint"
COMFY_READY=0
for attempt in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null 2>&1; then
    COMFY_READY=1
    WAIT_SECONDS=$((attempt * 2))
    log "ComfyUI is healthy after approximately ${WAIT_SECONDS}s"
    break
  fi
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    log "ComfyUI exited unexpectedly before becoming healthy"
    tail -n 250 "$DATA_DIR/logs/comfyui.log" | sed 's/^/[comfyui-tail] /' || true
    exit 1
  fi
  if (( attempt == 1 || attempt % 5 == 0 )); then
    log "waiting for ComfyUI attempt=$attempt/240 elapsed=$((attempt * 2))s"
  fi
  sleep 2
done

if [[ "$COMFY_READY" != "1" ]]; then
  log "ComfyUI failed to become healthy within 480 seconds"
  tail -n 250 "$DATA_DIR/logs/comfyui.log" | sed 's/^/[comfyui-tail] /' || true
  exit 1
fi

curl -fsS "http://127.0.0.1:${COMFY_PORT}/system_stats" \
  | python -m json.tool 2>/dev/null | sed 's/^/[comfyui-health] /' || true

export COMFY_URL="http://127.0.0.1:${COMFY_PORT}"
export WORKFLOW_DIR="$DATA_DIR/workflows"
export COMFY_INPUT_DIR="$DATA_DIR/input"
export COMFY_OUTPUT_DIR="$DATA_DIR/output"

log "phase 8/8: starting authenticated FastAPI service"
TOTAL_ELAPSED=$(( $(date +%s) - STARTED_AT ))
log "boot preparation completed in ${TOTAL_ELAPSED}s"
log "API listening on 0.0.0.0:${API_PORT}"
log "health endpoint: http://HOST:${API_PORT}/health"
log "capabilities endpoint: http://HOST:${API_PORT}/v1/capabilities"
log "generation and sampler progress will continue streaming from ComfyUI"
cd "$APP_DIR"
exec python -u -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "$API_PORT" \
  --workers 1 \
  --log-level "${UVICORN_LOG_LEVEL:-info}" \
  --access-log
