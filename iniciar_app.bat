@echo off
rem Abre la interfaz grafica sin necesidad de activar el entorno virtual.
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No existe el entorno virtual .venv. Siga los pasos de GUIA_DE_USO.md.
    pause
    popd
    exit /b 1
)
".venv\Scripts\python.exe" gui_app.py
if errorlevel 1 pause
popd
