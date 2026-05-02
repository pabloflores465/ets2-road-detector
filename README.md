# ETS2 Road Detector Overlay (macOS Quartz)

Ventana flotante sin bordes que captura **Euro Truck Simulator 2** usando la API nativa **Quartz** de macOS, detecta la carretera en tiempo real con **YOLOP ONNX** y muestra el resultado siempre encima de todo.

![ETS2 Road Detector concepto](https://i.imgur.com/placeholder.png)

## Que detecta el modelo

- **Area transitable (carretera)** → **verde brillante**
- **Lineas de carril** → **rojo**

## Modos de visualizacion

Edita la constante `DISPLAY_MODE` al inicio de `road_detector.py`:

| Modo | Descripcion |
|------|-------------|
| `"overlay"` | Imagen original + deteccion coloreada encima (default) |
| `"mask"` | Solo la segmentacion: negro con verde/rojo |
| `"split"` | Mitad original (izq) + mitad con deteccion (der) |

## Requisitos

- **macOS** (Intel o Apple Silicon)
- Python 3.9+ (preinstalado en macOS)
- **Euro Truck Simulator 2 abierto**
- **Permiso de Grabacion de pantalla** para la Terminal

## Instalacion rapida

```bash
cd ets2_road_detector
chmod +x run.sh
./run.sh
```

El script crea automaticamente un entorno virtual, instala dependencias, descarga el modelo YOLOP (~36 MB) y lanza el detector.

## Uso

1. Abre **Euro Truck Simulator 2**.
2. Ejecuta `./run.sh`.
3. El script **detecta automaticamente** la ventana del juego via Quartz.
4. Aparece una ventana **flotante y sin bordes** siempre encima.
5. **Arrastra** la ventana para moverla donde quieras.
6. Presiona el boton **X** rojo para cerrar.

### Si falla la auto-deteccion

Si Quartz no encuentra el juego, el script muestra la captura de pantalla completa y te pide que **arrastres** para seleccionar la ventana de ETS2 manualmente.

## Como funciona la captura

El script usa **Quartz** (API nativa de macOS) para:
1. Listar todas las ventanas del sistema
2. Encontrar la de ETS2 por nombre (`Euro Truck`, `eurotrucks2`, `ETS2`, `Steam`)
3. Capturar **solo esa ventana** por su CGWindowID
4. Filtrar ventanas fantasmas (tamaño 0x0) y quedarse con la mas grande

## Permisos en macOS (muy importante)

### Grabacion de pantalla

La primera vez que ejecutes, macOS bloqueara la captura.

1. Ve a **Preferencias del Sistema > Privacidad y Seguridad > Grabacion de pantalla**.
2. Activa el interruptor junto a tu terminal (**Terminal**, **iTerm2**, **VS Code**, etc.).
3. **Reinicia la terminal** para que surta efecto.

Sin este permiso, la ventana saldra negra o vacia.

## Configuracion

Edita `road_detector.py` para ajustar:

| Parametro | Descripcion | Default |
|-----------|-------------|---------|
| `DISPLAY_MODE` | `"overlay"`, `"mask"` o `"split"` | `"overlay"` |
| `ROAD_ALPHA` | Intensidad del area verde (0.0-1.0) | `0.7` |
| `LANE_ALPHA` | Intensidad de las lineas rojas (0.0-1.0) | `0.9` |
| `FPS_LIMIT` | Maximo FPS para no saturar CPU | `25` |
| `SHOW_LANES` | Mostrar lineas de carril | `True` |
| `WINDOW_NAMES` | Nombres de proceso a buscar | `["Euro Truck", ...]` |

## Solucion de problemas

| Problema | Causa | Solucion |
|----------|-------|----------|
| Ventana negra / vacia | Sin permiso de Grabacion de pantalla | Ve a Preferencias del Sistema > Privacidad > Grabacion de pantalla |
| "No se detecto la ventana" | El juego usa nombre de proceso raro | Edita `WINDOW_NAMES` en el codigo |
| FPS muy bajos (~1-5) | YOLOP ONNX es pesado en CPU | Cierra apps pesadas, baja resolucion del juego |
| Colores muy tenues | Alpha muy bajo | Sube `ROAD_ALPHA` y `LANE_ALPHA` a `1.0` |
| Captura descuadrada | Wine/Steam con bordes invisibles | Usa seleccion manual al iniciar |

## Estructura del proyecto

```
ets2_road_detector/
├── run.sh                      # Lanzador automatico
├── requirements.txt            # Dependencias
├── road_detector.py            # Detector principal
├── weights/
│   └── yolop-640-640.onnx     # Modelo YOLOP (~34 MB)
└── venv/                       # Entorno virtual Python
```

## Desinstalacion

Borra la carpeta completa:
```bash
rm -rf /Users/pabloflores/Documents/robotics/ets2_road_detector
```

## Notas tecnicas

- El modelo YOLOP ONNX corre en **CPU** (no requiere GPU).
- La captura via Quartz es **nativa de macOS**, no usa librerias de terceros para screenshot.
- La ventana overlay usa `tkinter` con `-topmost` para estar siempre encima.
