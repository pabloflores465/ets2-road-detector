# ETS2 Road Detector Overlay (macOS Optimizado)

Ventana flotante sin bordes que captura **Euro Truck Simulator 2** usando la API nativa **Quartz** de macOS, detecta la carretera en tiempo real con **YOLOP ONNX** optimizado para **Apple Silicon** y muestra el resultado siempre encima.

## Optimizaciones incluidas

| Optimizacion | Impacto |
|-------------|---------|
| **CoreML Execution Provider** | Usa Apple Neural Engine (ANE) en vez de CPU puro |
| **Modelo 320x320** | ~3x mas rapido que 640x640 con perdida minima de precision |
| **Frame skipping** | Procesa 1 de cada N frames, reutiliza mascara anterior |
| **Resize previo** | Reduce captura antes de inferencia |

## Que detecta

- **Carretera transitable** → verde
- **Lineas de carril** → rojo

## Modos de visualizacion

Edita `DISPLAY_MODE` en `road_detector.py`:

| Modo | Descripcion |
|------|-------------|
| `"overlay"` | Original + deteccion coloreada (default) |
| `"mask"` | Solo segmentacion (negro + verde/rojo) |
| `"split"` | Mitad original, mitad deteccion |

## Configuracion de rendimiento

Edita estas constantes al inicio de `road_detector.py`:

| Parametro | Descripcion | Default | Opciones |
|-----------|-------------|---------|----------|
| `MODEL_RES` | Resolucion del modelo | `320` | `320` (rapido) o `640` (preciso) |
| `FRAME_SKIP` | Procesar 1 de cada N frames | `2` | `1` (todos), `2`, `3`... |
| `CAPTURE_MAX_H` | Altura maxima de captura | `480` | Bajar = mas rapido |
| `ROAD_ALPHA` | Intensidad verde | `0.7` | `0.0` - `1.0` |
| `LANE_ALPHA` | Intensidad rojo | `0.9` | `0.0` - `1.0` |

**Recomendacion para maximo FPS:**
```python
MODEL_RES = 320
FRAME_SKIP = 2
CAPTURE_MAX_H = 360
```

## Requisitos

- **macOS** con Apple Silicon (M1/M2/M3) o Intel
- Python 3.9+
- **Euro Truck Simulator 2 abierto**
- **Permiso de Grabacion de pantalla** para Terminal

## Instalacion

```bash
cd ets2_road_detector
chmod +x run.sh
./run.sh
```

## Permisos en macOS

### Grabacion de pantalla (obligatorio)

1. **Preferencias del Sistema > Privacidad y Seguridad > Grabacion de pantalla**
2. Activa tu terminal (**Terminal**, **iTerm2**, **VS Code**)
3. **Reinicia la terminal**

Sin este permiso la ventana sale negra.

## Solucion de problemas

| Problema | Solucion |
|----------|----------|
| CPU al 90% | Verifica que diga `CoreMLExecutionProvider` al iniciar. Si dice `CPUExecutionProvider`, reinstala: `pip install --upgrade --force-reinstall onnxruntime` |
| FPS muy bajos | Baja `MODEL_RES` a `320`, sube `FRAME_SKIP` a `2` o `3`, baja `CAPTURE_MAX_H` a `360` |
| Ventana negra | Falta permiso de Grabacion de pantalla |
| Colores tenues | Sube `ROAD_ALPHA` y `LANE_ALPHA` a `1.0` |
| No detecta ventana | Edita `WINDOW_NAMES` en el codigo |

## Estructura

```
ets2_road_detector/
├── run.sh              # Lanzador
├── requirements.txt    # Dependencias
├── road_detector.py    # Detector principal
├── .gitignore
├── README.md
└── weights/
    └── yolop-320-320.onnx   # Se descarga solo al ejecutar
```

## Creditos

- Modelo [YOLOP](https://github.com/hustvl/YOLOP) por hustvl
- ONNX Runtime con [CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
