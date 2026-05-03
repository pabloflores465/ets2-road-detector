# Alternativas para Conducción Autónoma en ETS2

Después de múltiples intentos con YOLOP + control por teclado, aquí están las alternativas reales ordenadas por viabilidad.

---

## Opción 1: Mejorar Lane Detection (YOLOPv2)

**Problema actual:** YOLOP original tiene lane detection débil en cabin view.

**Solución:** Upgradear a **YOLOPv2**
- Repo: https://github.com/CAIC-AD/YOLOPv2
- Mejor precisión en lane detection
- Mejor segmentación de área transitable
- Soporte ONNX disponible

**Esfuerzo:** Medio (cambiar modelo, ajustar post-processing)
**Ventaja:** Mantiene la arquitectura actual, solo mejora la percepción

---

## Opción 2: europilot - CNN End-to-End (Más Prometedor)

**Repo:** https://github.com/marsauto/europilot

**Qué hace:**
- Estilo NVIDIA PilotNet
- Toma la imagen de la pantalla → predice ángulo de steering directamente
- No necesita lane detection explícito
- Entrenado con behavioral cloning (imitación)

**Arquitectura:**
```
Input: 320x160 RGB frame
  → Conv2D(24, 5x5, stride=2)
  → Conv2D(36, 5x5, stride=2)
  → Conv2D(48, 5x5, stride=2)
  → Conv2D(64, 3x3)
  → Conv2D(64, 3x3)
  → Flatten
  → Dense(1164) → Dense(100) → Dense(50) → Dense(10)
  → Output: steering angle (-1 a 1)
```

**Cómo implementar:**
1. Grabar dataset: mientras tú conduces, capturar (frame, steering, throttle)
2. Entrenar CNN por ~2-4 horas en GPU
3. Reemplazar todo el pipeline de lane detection por la CNN

**Esfuerzo:** Alto (necesita dataset + entrenamiento)
**Ventaja:** No depende de lane lines visibles. Funciona en ciudad, autopista, curvas.

---

## Opción 3: ChosunTruck - OpenCV Puro (Más Robusto)

**Repo:** https://github.com/bethesirius/ChosunTruck

**Qué hace:**
- No usa ML para lane detection
- **Perspective transform** (bird's eye view) de la región de la carretera
- **Sliding window** para encontrar líneas
- **Histogram** de píxeles blancos en la parte inferior
- Curva de segundo grado (polyfit) para cada línea

**Por qué es más robusto:**
- Transforma a vista de pájaro → las líneas son paralelas
- Ignora el dashboard completamente
- Usa historial de frames (si no detecta en uno, usa el anterior)

**Esquema:**
```
Frame → ROI (región de interés) → Perspective Transform
→ Threshold HLS (S-channel) → Sliding Windows
→ Polyfit left/right → Compute center → Steering PID
```

**Esfuerzo:** Medio (implementar perspective transform + sliding windows)
**Ventaja:** Muy robusto, no depende de un modelo de ML

---

## Opción 4: vJoy Virtual Joystick (Mejor Control)

**Problema actual:** Las flechas de teclado son digitales (0 o 1). ETS2 maneja mal steering digital.

**Solución:** Emular un joystick virtual con control analógico

**Opciones en macOS:**
1. **vJoy** (no disponible en macOS nativamente)
2. **ControllerMate** (macOS, puede enviar eventos de joystick)
3. **BetterTouchTool** + scripting
4. **Python + pygame** → emular joystick HID

**Implementación:**
```python
# En vez de pynput (teclado), usar pygame joystick
import pygame
pygame.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Steering analógico: -1.0 a 1.0
joystick.set_axis(0, steer_value)  # X axis
# Throttle analógico: 0.0 a 1.0
joystick.set_axis(2, throttle_value)  # Z axis
```

**Esfuerzo:** Medio-Alto (necesita driver HID virtual en macOS)
**Ventaja:** Control suave como volante real

---

## Opción 5: ETS2 Telemetry + SCS SDK

**ETS2 tiene un SDK oficial** que expone:
- Posición exacta del camión
- Velocidad, RPM, marcha
- Ángulo de steering actual
- Coordenadas GPS del mundo

**Cómo usar:**
1. Activar `developer mode` en ETS2
2. Leer memoria compartida (`/dev/shm/...` en macOS)
3. El autopilot lee posición real y corrige

**Ventaja:** No depende de visión por computadora para saber dónde está el camión.
**Desventaja:** Requiere modificar archivos del juego.

---

## Opción 6: Reinforcement Learning (Más Complejo)

**Repo:** https://github.com/aleju/self-driving-truck

**Qué hace:**
- DQN (Deep Q-Network) entrenado dentro del juego
- Reward function: stay in lane + speed + no crash
- Entrena por días/semanas

**No recomendado** para este proyecto (demasiado complejo, requiere GPU potente).

---

## Recomendación Final

Si quieres que funcione **esta semana**:

1. **Corto plazo (2-3 días):** Implementar **ChosunTruck** (OpenCV puro)
   - Bird's eye view perspective transform
   - Sliding window lane detection
   - Es más robusto que YOLOP para ETS2 cabin view

2. **Medio plazo (1-2 semanas):** Entrenar **europilot CNN**
   - Grabar 30-60 minutos de conducción tuya
   - Entrenar modelo en Google Colab (GPU gratis)
   - Reemplazar todo el pipeline por la CNN

3. **Si nada funciona:** Cambiar a **cámara exterior** en ETS2 (tecla `1` o `2`)
   - La referencia `stefanos50` usaba cámara exterior
   - Sin dashboard bloqueando la vista
   - YOLOP funciona mucho mejor

---

## Recursos

| Proyecto | URL | Tipo |
|----------|-----|------|
| YOLOPv2 | https://github.com/CAIC-AD/YOLOPv2 | Lane detection mejorado |
| europilot | https://github.com/marsauto/europilot | CNN end-to-end |
| ChosunTruck | https://github.com/bethesirius/ChosunTruck | OpenCV puro |
| self-driving-truck | https://github.com/aleju/self-driving-truck | Reinforcement Learning |
| ETS2-Driving-AI | https://github.com/daviddelarocha/ETS2-Driving-AI | Multimodal DL |
| ETS2-Self-Driving-AI | https://github.com/Dodecahedrane/ETS2-Self-Driving-AI | Lane keeping |
