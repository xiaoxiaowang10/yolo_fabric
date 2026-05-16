@echo off
title Fabric-Classifier

if not exist ".venv\Scripts\python.exe" (
    echo [*] Setting up venv...
    python -m venv .venv 2>nul
    if not exist ".venv\Scripts\python.exe" (
        echo [X] Python not found - install Python 3.12+
        pause & exit /b 1
    )
    .venv\Scripts\python -m pip install -q --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
    .venv\Scripts\pip install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo [*] Ready.
)

if not exist "config.yaml"   copy "..\config.yaml" "config.yaml" >nul 2>&1
if not exist "models\*.onnx" (echo [X] models\*.onnx missing & pause & exit /b 1)

echo [*] Starting server - open http://localhost:8564
echo.
.venv\Scripts\python server.py
pause
