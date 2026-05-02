#!/bin/bash
set -e
cd "$(dirname "$0")"

# Crear venv si no existe
if [ ! -d "venv" ]; then
    echo "[INFO] Creando entorno virtual de Python..."
    python3 -m venv venv
fi

echo "[INFO] Activando entorno virtual..."
source venv/bin/activate

echo "[INFO] Instalando dependencias si es necesario..."
pip install -q -r requirements.txt

echo "[INFO] Iniciando ETS2 Road Detector (YOLOP ONNX)..."
python3 road_detector.py
