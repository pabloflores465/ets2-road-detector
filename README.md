# ETS2 Road Detector Overlay (macOS)

Ventana flotante sin bordes que captura **Euro Truck Simulator 2** usando la API nativa **Quartz** de macOS, detecta en tiempo real y muestra el resultado siempre encima de todo.

## Que detecta

### Carretera y carriles (YOLOP ONNX 640x640)
- **Area transitable** → verde
- **Lineas de carril** → rojo

### Objetos (Coral Edge TPU — mucho mas rapido)
- **Vehiculos** → `car`, `truck`, `bus`
- **Peatones** → `person`, `rider`
- **Senales** → `traffic light`, `stop sign`
- **Otros** → `bicycle`, `motorcycle`

Si no tienes Coral TPU, los objetos se detectan con YOLOP (mas lento).

## Requisitos

- **macOS** con **Apple Silicon (M1/M2/M3)** o Intel
- **Euro Truck Simulator 2 abierto**
- **Permiso de Grabacion de pantalla** para Terminal
- **Google Coral USB Accelerator** (opcional pero recomendado)
- `uv` instalado (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Instalacion

```bash
cd ets2_road_detector
```

### 1. Instalar `uv` (si no lo tienes)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Correr el detector

```bash
./run.sh
```

Esto hace todo automaticamente:
- Crea entorno `uv` con Python 3.9
- Instala ONNX Runtime, pycoral, tflite-runtime, OpenCV, etc.
- Descarga YOLOP 640x640 (~34 MB) si no existe
- Detecta la ventana de ETS2 via Quartz
- Lanza la ventana flotante con deteccion en tiempo real

## Uso

1. Abre **Euro Truck Simulator 2**.
2. Ejecuta `./run.sh`.
3. El script detecta automaticamente la ventana del juego.
4. Aparece ventana **flotante sin bordes** siempre encima.
5. **Arrastra** para moverla donde quieras.
6. Presiona **X** rojo para cerrar.

## Permisos en macOS (obligatorio)

### Grabacion de pantalla

1. **Preferencias del Sistema > Privacidad y Seguridad > Grabacion de pantalla**
2. Activa tu terminal (**Terminal**, **iTerm2**, **VS Code**)
3. **Reinicia la terminal**

Sin esto la ventana sale negra.

## Coral TPU

### Verificar que funciona

```bash
export DYLD_LIBRARY_PATH="/usr/local/lib:${DYLD_LIBRARY_PATH}"
uv run python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"
```

Debe mostrar algo como:
```python
[{'type': 'usb', 'path': '/sys/bus/usb/devices/1-2'}]
```

### Problemas comunes con Coral

| Problema | Solucion |
|----------|----------|
| `dlopen(libedgetpu.1.dylib)` | `export DYLD_LIBRARY_PATH=/usr/local/lib` (ya incluido en `run.sh`) |
| `SystemError: initialization of _pywrap_coral` | Downgrade numpy: `uv pip install "numpy<2"` |
| No detecta USB | Usa **hub USB con alimentacion externa**. Coral consume mucha corriente. |

## Modos de visualizacion

Edita `DISPLAY_MODE` en `detector.py`:

| Modo | Descripcion |
|------|-------------|
| `"overlay"` | Original + carretera verde + carriles rojos + objetos (default) |
| `"mask"` | Solo segmentacion (negro + verde/rojo) |
| `"split"` | Mitad original, mitad deteccion |
| `"debug"` | Heatmap de probabilidades crudas |

## Configuracion

Edita constantes al inicio de `detector.py`:

| Parametro | Descripcion | Default |
|-----------|-------------|---------|
| `MODEL_RES` | 320 (rapido) o 640 (preciso) | `640` |
| `FRAME_SKIP` | Procesar 1 de cada N frames | `2` |
| `CAPTURE_MAX_H` | Altura maxima captura | `480` |
| `USE_CORAL` | Usar Coral TPU para objetos | `True` |
| `SHOW_LANES` | Mostrar lineas de carril | `True` |
| `ROAD_ALPHA` / `LANE_ALPHA` | Intensidad colores | `1.0` |

## Solucion de problemas

| Problema | Solucion |
|----------|----------|
| Ventana negra | Falta permiso de Grabacion de pantalla |
| "No se detecto la ventana" | ETS2 no esta abierto |
| FPS muy bajos | Baja `MODEL_RES` a 320, sube `FRAME_SKIP`, baja `CAPTURE_MAX_H` |
| `macOS 26 required, have 16` | Python 3.9 del sistema tiene tkinter roto. `run.sh` usa `uv` con Python 3.9.25 que funciona. |
| CPU al 90% | Normal. CoreML ayuda. Coral TPU reduce carga masivamente. |

## Comandos utiles

```bash
# Correr detector (todo automatico)
./run.sh

# Verificar Coral TPU
export DYLD_LIBRARY_PATH="/usr/local/lib:${DYLD_LIBRARY_PATH}"
uv run python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"

# Verificar ONNX Runtime
uv run python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Instalar dependencias manualmente
uv pip install onnxruntime opencv-python "numpy<2" Pillow mss pyobjc-framework-Quartz
uv pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-macosx_12_0_arm64.whl
uv pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp39-cp39-macosx_12_0_arm64.whl

# Recrear entorno desde cero
rm -rf .venv
uv venv --python 3.9 .venv
./run.sh
```

## Estructura

```
ets2_road_detector/
├── run.sh                   # Comando principal: ./run.sh
├── detector.py              # Detector unificado YOLOP + Coral
├── ets2_capture.py          # Captura de ventana via Quartz
├── road_detector.py         # YOLOP puro (legacy)
├── coral_detector.py        # Coral puro (legacy)
├── coral_setup.sh           # Script de setup manual para Coral
├── coral_models/
│   ├── ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
│   └── coco_labels.txt
├── weights/
│   └── yolop-640-640.onnx   # Se descarga solo al ejecutar
└── .venv/                   # Entorno uv (no se versiona)
```

## Creditos

- Modelo [YOLOP](https://github.com/hustvl/YOLOP) por hustvl
- ONNX Runtime con [CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [Google Coral](https://coral.ai/) Edge TPU
