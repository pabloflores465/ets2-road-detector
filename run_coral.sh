#!/bin/bash
set -e
cd "$(dirname "$0")"

# libedgetpu esta en /usr/local/lib pero el loader no lo encuentra automaticamente
export DYLD_LIBRARY_PATH="/usr/local/lib:${DYLD_LIBRARY_PATH}"

# El detector Coral necesita Python 3.9 + pycoral + tflite-runtime
# El venv39 ya esta configurado con estas dependencias
if [ ! -d "venv39" ]; then
    echo "[ERROR] No se encuentra venv39. Ejecuta primero:"
    echo "  /usr/bin/python3 -m venv venv39"
    echo "  source venv39/bin/activate"
    echo "  pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral"
    echo "  pip install 'numpy<2' Pillow opencv-python"
    exit 1
fi

echo "[INFO] Activando entorno Coral (Python 3.9)..."
source venv39/bin/activate

echo "[INFO] Verificando Coral TPU..."
python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print('TPUs:', list_edge_tpus())"

echo "[INFO] Iniciando detector Coral..."
python3 coral_detector.py
