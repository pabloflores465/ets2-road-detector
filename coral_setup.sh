#!/bin/bash
# Setup script para Google Coral Edge TPU
# IMPORTANTE: macOS Apple Silicon (M1/M2/M3) NO tiene soporte oficial de Coral.
#             Este script solo funciona en macOS Intel, Linux o Windows.

set -e
echo "=============================================="
echo "  Coral TPU Setup"
echo "=============================================="
echo ""

OS=$(uname -s)
ARCH=$(uname -m)

echo "[INFO] OS detectado: $OS"
echo "[INFO] Arquitectura: $ARCH"

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo ""
    echo "[ERROR] macOS Apple Silicon (M1/M2/M3) NO soportado oficialmente."
    echo "        Google no provee libedgetpu para ARM64 en macOS."
    echo ""
    echo "Alternativas:"
    echo "  1. Usar una VM Ubuntu (UTM/Parallels) con USB passthrough"
    echo "  2. Usar Docker con --privileged y /dev/bus/usb"
    echo "  3. Usar una Raspberry Pi 4/5 o PC Linux"
    echo ""
    exit 1
fi

if [[ "$OS" == "Darwin" && "$ARCH" == "x86_64" ]]; then
    echo "[INFO] macOS Intel detectado."
    echo "[INFO] Descargando libedgetpu..."
    
    # Crear directorio para librerias
    mkdir -p lib
    
    # Descargar libedgetpu universal (incluye x86_64)
    curl -L -o lib/libedgetpu.tar.gz \
        "https://github.com/google-coral/libedgetpu/releases/download/release-frogfish/libedgetpu-direct-20210202-darwin-x86_64.tar.gz" \
        2>/dev/null || echo "[WARN] No se pudo descargar libedgetpu automaticamente."
    
    if [ -f lib/libedgetpu.tar.gz ]; then
        tar -xzf lib/libedgetpu.tar.gz -C lib/ --strip-components=1 2>/dev/null || true
        echo "[INFO] libedgetpu descargado en lib/"
    fi
    
    echo "[INFO] Instalando pycoral y tflite-runtime..."
    pip install pycoral tflite-runtime || {
        echo "[WARN] Fallo instalacion via pip. Intentando con tflite_runtime alternativo..."
        pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0-cp39-cp39-macosx_10_15_x86_64.whl 2>/dev/null || true
        pip install pycoral 2>/dev/null || true
    }
fi

if [[ "$OS" == "Linux" ]]; then
    echo "[INFO] Linux detectado."
    echo "[INFO] Instalando libedgetpu desde apt..."
    
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install -y libedgetpu1-std python3-pycoral
    
    echo "[INFO] Instalando dependencias Python..."
    pip install tflite-runtime numpy opencv-python Pillow
fi

echo ""
echo "=============================================="
echo "  Setup completado"
echo "=============================================="
echo ""
echo "Prueba con:"
echo "  python3 -c 'from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())'"
echo ""
echo "Luego ejecuta el detector Coral:"
echo "  python3 coral_detector.py"
