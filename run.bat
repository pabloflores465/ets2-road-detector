@echo off
cd /d "%~dp0"
echo [INFO] Instalando dependencias si es necesario...
pip install -q -r requirements.txt
echo [INFO] Iniciando ETS2 Road Detector...
python road_detector.py
pause
