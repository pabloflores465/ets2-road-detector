# ETS2 Road Detector Overlay (macOS)

Ventana flotante sin bordes que captura **Euro Truck Simulator 2** usando la API nativa **Quartz** de macOS, detecta en tiempo real con **YOLOP ONNX** y muestra el resultado siempre encima.

## Que detecta

YOLOP es un modelo multi-task que detecta simultaneamente:

- **Carretera transitable** → verde
- **Lineas de carril** → rojo
- **Obstaculos / objetos** → bounding boxes con etiquetas:
  - `car`, `truck`, `bus` — vehiculos
  - `person`, `rider`, `bike`, `motor` — peatones y ciclistas
  - `traffic light`, `traffic sign` — senales de transito
  - `train` — trenes

## Modos de visualizacion

Edita `DISPLAY_MODE` en `road_detector.py`:

| Modo | Descripcion |
|------|-------------|
| `"overlay"` | Original + deteccion coloreada + objetos (default) |
| `"mask"` | Solo segmentacion (negro + verde/rojo) |
| `"split"` | Mitad original, mitad deteccion |
| `"debug"` | Heatmap de probabilidades crudas del modelo |

## Configuracion de deteccion de objetos

Edita estas constantes en `road_detector.py`:

| Parametro | Descripcion | Default |
|-----------|-------------|---------|
| `SHOW_OBJECTS` | Activar/detecar obstaculos | `True` |
| `CONF_THRESHOLD` | Confianza minima para mostrar objeto | `0.4` |
| `NMS_IOU_THRESHOLD` | Umbral de supresion de solapamiento | `0.5` |

## Requisitos

- **macOS** (Intel o Apple Silicon)
- Python 3.9+
- **Euro Truck Simulator 2 abierto**
- **Permiso de Grabacion de pantalla** para Terminal

## Instalacion rapida

```bash
cd ets2_road_detector
chmod +x run.sh
./run.sh
```

La primera vez descarga el modelo YOLOP 640x640 (~34 MB).

## Uso

1. Abre **Euro Truck Simulator 2**.
2. Ejecuta `./run.sh`.
3. El script detecta automaticamente la ventana del juego via Quartz.
4. Aparece ventana flotante sin bordes siempre encima.
5. **Arrastra** para moverla.
6. Presiona **X** rojo para cerrar.

## Permisos en macOS

### Grabacion de pantalla (obligatorio)

1. **Preferencias del Sistema > Privacidad y Seguridad > Grabacion de pantalla**
2. Activa tu terminal (**Terminal**, **iTerm2**, **VS Code**)
3. **Reinicia la terminal**

Sin esto la ventana sale negra.

## Coral TPU — ¿Acalaria el proceso?

**Si. Masivamente.** Un Google Coral USB Accelerator puede llevar la inferencia de **~10 FPS en CPU a 30-60+ FPS** con latencia mucho menor.

### Problema

YOLOP ONNX **no corre directamente** en Coral. Necesitas:
1. Convertir ONNX → TensorFlow Lite (TFLite)
2. Cuantizar a INT8 (requerido por Edge TPU)
3. Compilar con `edgetpu_compiler` para generar el modelo `.tflite` con ops aceleradas

### Paso a paso (avanzado)

```bash
# 1. Instalar dependencias
pip install onnx tf-keras onnx-tensorflow

# 2. ONNX -> TensorFlow SavedModel
python -c "
import onnx
from onnx_tf.backend import prepare
model = onnx.load('weights/yolop-640-640.onnx')
tf_rep = prepare(model)
tf_rep.export_graph('yolop_tf')
"

# 3. TensorFlow -> TFLite con cuantizacion INT8
# (necesitas un dataset representativo para calibracion)
# Ver: https://coral.ai/docs/edgetpu/models-on-edge/

# 4. Compilar para Edge TPU
edgetpu_compiler yolop_quantized.tflite
```

**Alternativa mas simple:** Si tienes un Mac con **Apple Silicon (M1/M2/M3)**, CoreML ya esta activado y usa el **Neural Engine** interno. No necesitas Coral. Si aun es lento:
- Baja `MODEL_RES` a `320`
- Sube `FRAME_SKIP` a `2` o `3`
- Baja `CAPTURE_MAX_H` a `360`

## Configuracion de rendimiento

| Parametro | Descripcion | Default |
|-----------|-------------|---------|
| `MODEL_RES` | 320 (rapido) o 640 (preciso) | `640` |
| `FRAME_SKIP` | Procesar 1 de cada N frames | `2` |
| `CAPTURE_MAX_H` | Altura maxima de captura | `480` |
| `FPS_LIMIT` | Maximo FPS | `30` |
| `SHOW_LANES` | Mostrar lineas de carril | `True` |
| `ROAD_ALPHA` / `LANE_ALPHA` | Intensidad de colores | `1.0` |

## Solucion de problemas

| Problema | Solucion |
|----------|----------|
| Ventana negra | Falta permiso de Grabacion de pantalla |
| "No se detecto la ventana" | ETS2 no esta abierto o usa nombre raro |
| FPS muy bajos | Baja `MODEL_RES` a 320, sube `FRAME_SKIP`, baja `CAPTURE_MAX_H` |
| No detecta carretera | Prueba modo `debug` para ver heatmap |
| CPU al 90% | Normal en CPU. CoreML ayuda en Apple Silicon. Coral TPU es lo ideal. |

## Estructura

```
ets2_road_detector/
├── run.sh                  # Lanzador
├── requirements.txt        # Dependencias
├── road_detector.py        # Detector principal
├── weights/
│   └── yolop-640-640.onnx # Modelo (~34 MB, se descarga solo)
└── venv/                   # Entorno virtual Python
```
