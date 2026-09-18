#!/usr/bin/env bash
# Production entrypoint for the supertonic streaming TTS server.
#
# Honours SUPERTONIC_MODEL_PATH (path to a mounted model directory),
# SUPERTONIC_TRT_CACHE (path to a persistent volume for TRT engines), and
# the rest of the SUPERTONIC_* env vars consumed by server.py.
#
# Dependency resolution is entirely a build-time concern: the image builds
# /opt/supertonic/.venv with uv from pyproject.toml + uv.lock, and the uv
# binary itself is torn down in the same layer. At run time we just exec that
# venv's python directly — no network call ever happens, and no package
# manager is present in the container at all.

set -euo pipefail

VENV_PYTHON="/opt/supertonic/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[entrypoint] venv interpreter not found at $VENV_PYTHON" >&2
  echo "[entrypoint] image was built incorrectly — the uv install did not populate the venv." >&2
  exit 2
fi

MODEL_PATH="${SUPERTONIC_MODEL_PATH:-}"
AUTO_DL="${SUPERTONIC_AUTO_DOWNLOAD:-0}"

if [[ -n "$MODEL_PATH" ]]; then
  required=(
    "$MODEL_PATH/onnx/duration_predictor.onnx"
    "$MODEL_PATH/onnx/text_encoder.onnx"
    "$MODEL_PATH/onnx/vector_estimator.onnx"
    "$MODEL_PATH/onnx/vocoder.onnx"
    "$MODEL_PATH/onnx/tts.json"
    "$MODEL_PATH/onnx/unicode_indexer.json"
  )
  for f in "${required[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "[entrypoint] missing required model file: $f" >&2
      echo "[entrypoint] mount the supertonic-3 weights at SUPERTONIC_MODEL_PATH" >&2
      echo "             or unset SUPERTONIC_MODEL_PATH and set SUPERTONIC_AUTO_DOWNLOAD=1 for dev." >&2
      exit 2
    fi
  done
  echo "[entrypoint] using model at $MODEL_PATH"
elif [[ "$AUTO_DL" == "1" ]]; then
  echo "[entrypoint] SUPERTONIC_AUTO_DOWNLOAD=1 — supertonic will fetch weights from HuggingFace."
else
  echo "[entrypoint] no SUPERTONIC_MODEL_PATH and SUPERTONIC_AUTO_DOWNLOAD!=1 — refusing to start." >&2
  exit 2
fi

# TRT engine cache directory must be writable.
TRT_CACHE="${SUPERTONIC_TRT_CACHE:-/var/cache/supertonic/trt}"
mkdir -p "$TRT_CACHE" || {
  echo "[entrypoint] cannot create TRT cache dir $TRT_CACHE" >&2
  exit 2
}

cd /opt/supertonic
exec "$VENV_PYTHON" -u src/server.py "$@"
