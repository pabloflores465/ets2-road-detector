#!/bin/bash
set -e
cd "$(dirname "$0")"

# Coral TPU necesita que el loader encuentre libedgetpu.1.dylib
export DYLD_LIBRARY_PATH="/usr/local/lib:${DYLD_LIBRARY_PATH}"

# Crear entorno uv si no existe
if [ ! -d ".venv" ]; then
    echo "[INFO] Creando entorno uv con Python 3.9..."
    uv venv --python 3.9 .venv
fi

echo "[INFO] Activando entorno uv..."

# Verificar que todo esta instalado
echo "[INFO] Verificando dependencias..."
uv run python3 -c "
import onnxruntime as ort
from pycoral.utils.edgetpu import list_edge_tpus
print('ONNX:', ort.get_available_providers())
print('Coral:', list_edge_tpus())
print('OK')
" 2>/dev/null || {
    echo "[INFO] Instalando dependencias..."
    uv pip install onnxruntime opencv-python "numpy<2" Pillow mss pyobjc-framework-Quartz pyobjc-framework-ApplicationServices
    uv pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-macosx_12_0_arm64.whl
    uv pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp39-cp39-macosx_12_0_arm64.whl
}

# Descargar modelo ETSAuto si no existe
if [ ! -f "etsauto_models/bevlanedet.onnx" ]; then
    echo "[INFO] Descargando modelo ETSAuto bevlanedet.onnx (~126MB)..."
    bash etsauto_models/download_model.sh
fi

echo "[INFO] Iniciando detector unificado (ETSAuto + Coral TPU)..."
uv run python3 detector.py
