# Guía de uso rápida

Pasos para ejecutar el proyecto de reconocimiento facial en Windows. Para la teoría y las limitaciones, vea el [README](README.md).

## Requisitos

- Windows con una cámara web que funcione.
- [Python 3.9 o superior](https://www.python.org/downloads/) (al instalar, marque **Add Python to PATH**).
- Visual Studio Code (opcional, pero recomendado).

## 1. Preparar el entorno (solo la primera vez)

Abra la carpeta `reconocimiento_facial_opencv` en VS Code, abra una terminal (`Ctrl + ñ`) y ejecute:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Si PowerShell no deja activar el entorno

Error típico: *"No se puede cargar el archivo ...\Activate.ps1. El archivo no está firmado digitalmente"*.

**Causa:** PowerShell bloquea scripts sin firma. Esto ocurre sobre todo cuando el proyecto está en una **unidad de red** (por ejemplo `Z:` que apunta a `\\servidor\carpeta`), porque Windows trata esos archivos como remotos y `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` **no basta**.

Use cualquiera de estas opciones (de la más recomendada a la menos):

1. **Permitir scripts solo en la terminal actual** (no cambia nada permanente; repetir en cada terminal nueva):

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

2. **No activar el entorno** y llamar directamente a su Python (siempre funciona):

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe capture_faces.py
   .\.venv\Scripts\python.exe train_model.py
   .\.venv\Scripts\python.exe recognize_faces.py
   ```

3. **Usar el activador de CMD** (no depende de la política de PowerShell):

   ```powershell
   .\.venv\Scripts\activate.bat
   ```

4. **Mover el proyecto a un disco local** (por ejemplo `C:\proyectos\`). Es lo ideal: el entorno virtual en red es lento y puede dar errores. Si lo hace, borre la carpeta `.venv` y créela de nuevo, porque los entornos no se pueden copiar entre ubicaciones.

> Cada vez que abra una terminal nueva, active el entorno de nuevo con `.\.venv\Scripts\Activate.ps1` (o use la opción 2). Verá `(.venv)` al inicio de la línea.

## Interfaz gráfica (recomendado)

Reúne todo el flujo en una sola ventana con tema verde/negro tipo Matrix. Se abre con doble clic en **`iniciar_app.bat`** (no requiere activar el entorno) o desde la terminal:

```powershell
python gui_app.py
```

| Botón | Qué hace |
|---|---|
| **01 REGISTRAR ROSTRO** | Pide el nombre y captura hasta 100 muestras con la cámara. Debe haber **un solo rostro** en cuadro. |
| **02 ENTRENAR MODELO** | Entrena el reconocedor con todas las personas registradas. **Parpadea** cuando hay rostros nuevos sin entrenar. |
| **03 VALIDAR RECONOCIMIENTO** | Reconoce rostros en vivo: verde con el nombre, rojo si es *Desconocido*. |
| **DETENER** (o **Esc**) | Termina la captura o la validación en curso. |

Flujo típico: **01** (por cada persona) → **02** → **03**. Repita **02** cada vez que registre a alguien nuevo. El panel *Estado* muestra si el modelo está `LISTO`, `DESACTUALIZADO` o `SIN ENTRENAR`, y la consola registra cada paso.

Los pasos 2, 3 y 4 de abajo hacen lo mismo desde la línea de comandos.

## 2. Registrar personas (por consola)

```powershell
python capture_faces.py
```

1. Escriba el nombre de la persona y presione Enter.
2. Mire a la cámara y mueva un poco la cabeza (de lado a lado, arriba y abajo).
3. Se guardan hasta 100 fotos. Puede terminar antes con **Esc**.

Repita este paso para cada persona. Se recomienda registrar **al menos 2 personas**.

## 3. Entrenar el modelo

```powershell
python train_model.py
```

Debe mostrar cuántas personas y muestras se usaron. Repita este paso cada vez que registre a alguien nuevo.

## 4. Reconocer rostros

```powershell
python recognize_faces.py
```

| Lo que se ve en pantalla | Significado |
|---|---|
| Nombre en **verde** | Persona reconocida (confianza menor a 70) |
| **Desconocido** en **rojo** | Rostro no registrado o confianza insuficiente |
| Valor de **Confianza** | Distancia del modelo: *menor = más parecido* |

Presione **Esc** para salir.

## Problemas frecuentes

| Problema | Solución |
|---|---|
| `No module named 'tkinter'` al abrir la interfaz | Reinstale Python marcando la opción **tcl/tk and IDLE**. |
| `no se pudo abrir la cámara` | Cierre Teams, Zoom u otras apps que usen la cámara. Si hay varias cámaras, cambie `VideoCapture(0)` por `1` en el script. |
| `No module named 'numpy._core._multiarray_umath'` (menciona `cp312` y `cpython-314`) | Los paquetes del `.venv` se instalaron con otra versión de Python. Ejecute `.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-cache-dir -r requirements.txt`. Instale siempre con `python -m pip` con el entorno activo, no con un `pip` global. |
| `falta cv2.face` | Ejecute `pip uninstall opencv-python` y luego `pip install -r requirements.txt`. |
| `no existe el modelo entrenado` | Ejecute primero `python train_model.py`. |
| `no hay imágenes en data/` | Ejecute primero `python capture_faces.py`. |
| Reconoce mal | Capture más muestras con una iluminación parecida a la de uso. |
