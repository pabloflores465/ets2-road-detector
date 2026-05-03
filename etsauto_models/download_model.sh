#!/bin/bash
# Download ETSAuto bevlanedet model

MODEL_URL="https://github.com/Lyric0620/ETSAuto/releases/download/v2.2.2/bevlanedet.onnx"
MODEL_FILE="etsauto_models/bevlanedet.onnx"

if [ -f "$MODEL_FILE" ]; then
    echo "[INFO] ETSAuto model already exists"
    exit 0
fi

echo "[INFO] Downloading ETSAuto bevlanedet.onnx (~126MB)..."
curl -L -o "$MODEL_FILE" "$MODEL_URL"
echo "[INFO] Download complete"
