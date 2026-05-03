#!/usr/bin/env python3
"""
test_controls.py
Prueba que pynput pueda enviar teclas al sistema (requiere permiso de Accesibilidad).

Si ves las flechas moviéndose en algún campo de texto (ej: Notes, TextEdit),
los permisos funcionan correctamente.
"""
import time
from pynput.keyboard import Controller, Key

print("=" * 50)
print("TEST DE PERMISOS PYNPUT")
print("=" * 50)
print("\n1. Abre TextEdit o Notes y deja el cursor en un documento vacío.")
print("2. En 3 segundos empezaré a enviar flechas y letras.")
print("3. Si ves caracteres aparecer, los permisos funcionan.\n")

time.sleep(3)

kb = Controller()

try:
    for i in range(5):
        print(f"  -> Enviando UP")
        kb.press(Key.up)
        time.sleep(0.1)
        kb.release(Key.up)
        time.sleep(0.3)

        print(f"  -> Enviando RIGHT")
        kb.press(Key.right)
        time.sleep(0.1)
        kb.release(Key.right)
        time.sleep(0.3)

        print(f"  -> Enviando 'a'")
        kb.type("a")
        time.sleep(0.5)

    print("\n✅ TEST COMPLETADO — Si viste letras/flechas, pynput funciona.")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("   Pynput no tiene permisos. Ve a:")
    print("   System Settings -> Privacy & Security -> Accessibility")
    print("   Añade la ruta exacta de Python:")
    import sys
    print(f"   {sys.executable}")
