#!/usr/bin/env python3
"""
Test script: verifica que pynput pueda controlar ETS2.
Instrucciones:
1. Abre ETS2, pon el camión en marcha (modo automático o neutral)
2. Corre este script
3. Debería acelerar 2s, girar izquierda 2s, girar derecha 2s, frenar 1s
"""
import time
from pynput.keyboard import Controller, Key

kb = Controller()

print("[TEST] Enfoca ETS2 (click en la ventana) en 3 segundos...")
time.sleep(3)

print("[TEST] Presionando UP (acelerar) por 2 segundos...")
kb.press(Key.up)
time.sleep(2.0)
kb.release(Key.up)
print("[TEST] UP liberado")

time.sleep(0.5)

print("[TEST] Presionando LEFT por 2 segundos...")
kb.press(Key.left)
time.sleep(2.0)
kb.release(Key.left)
print("[TEST] LEFT liberado")

time.sleep(0.5)

print("[TEST] Presionando RIGHT por 2 segundos...")
kb.press(Key.right)
time.sleep(2.0)
kb.release(Key.right)
print("[TEST] RIGHT liberado")

time.sleep(0.5)

print("[TEST] Presionando DOWN (frenar) por 1 segundo...")
kb.press(Key.down)
time.sleep(1.0)
kb.release(Key.down)
print("[TEST] DOWN liberado")

print("[TEST] Terminado. El camión giró/aceleró/frenó?")
