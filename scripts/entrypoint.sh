#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${PROFILE:-vast-h100}"
DATA_DIR="${DATA_DIR:-/workspace}"
COMFYUI_DIR="${COMFYUI_DIR:-/opt/ComfyUI}"
APP_DIR="${APP_DIR:-/opt/qwen-worker}"
COMFY_PORT="${COMFY_PORT:-8188}"
COMFY_LISTEN_HOST="${COMFY_LISTEN_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"

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

for name in models input output; do
  rm -rf "$COMFYUI_DIR/$name"
  ln -s "$DATA_DIR/$name" "$COMFYUI_DIR/$name"
done

cp -f "$APP_DIR"/workflows/*.json "$DATA_DIR/workflows/"

if [[ -z "${API_KEY:-}" ]]; then
  if [[ -s "$DATA_DIR/API_KEY.txt" ]]; then
    export API_KEY="$(cat "$DATA_DIR/API_KEY.txt")"
  else
    export API_KEY="$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
)"
    printf '%s\n' "$API_KEY" > "$DATA_DIR/API_KEY.txt"
    chmod 600 "$DATA_DIR/API_KEY.txt"
  fi
fi

echo "[startup] profile=$PROFILE"
echo "[startup] API key file: $DATA_DIR/API_KEY.txt"
echo "[startup] checking models (existing files are reused)"
python "$APP_DIR/scripts/bootstrap_models.py"

COMFY_ARGS=(
  --listen "$COMFY_LISTEN_HOST"
  --port "$COMFY_PORT"
  --preview-method none
)

case "$PROFILE" in
  vast-h100)
    COMFY_ARGS+=(--highvram --cache-lru 3)
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
    echo "Unknown PROFILE=$PROFILE" >&2
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

cd "$COMFYUI_DIR"
python main.py "${COMFY_ARGS[@]}" >"$DATA_DIR/logs/comfyui.log" 2>&1 &
COMFY_PID=$!

cleanup() {
  kill "$COMFY_PID" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    echo "[startup] ComfyUI exited unexpectedly" >&2
    tail -n 250 "$DATA_DIR/logs/comfyui.log" || true
    exit 1
  fi
  sleep 2
done

curl -fsS "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null

export COMFY_URL="http://127.0.0.1:${COMFY_PORT}"
export WORKFLOW_DIR="$DATA_DIR/workflows"
export COMFY_INPUT_DIR="$DATA_DIR/input"
export COMFY_OUTPUT_DIR="$DATA_DIR/output"

cd "$APP_DIR"
exec uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT" --workers 1
