# Reconocimiento facial con OpenCV (LBPH)

Práctica académica de Seminario de IA. El sistema registra personas desde la cámara web, entrena un reconocedor facial y, al volver a verlas, muestra su nombre sobre el rostro. Si la confianza no es suficiente, muestra **Desconocido**.

## Conceptos

- **Haar Cascade**: clasificador clásico que *detecta* dónde hay un rostro frontal en la imagen, buscando patrones de claro/oscuro (ojos, nariz, frente). Viene incluido en OpenCV (`cv2.data.haarcascades`).
- **LBPH** (Local Binary Patterns Histograms): algoritmo que describe la textura local de cada rostro mediante histogramas y compara el rostro nuevo con los entrenados. Devuelve una **distancia**: *menor = más parecido*.
- **Detectar vs. reconocer**: detectar responde "¿hay un rostro aquí?"; reconocer responde "¿de quién es?". Haar hace lo primero; LBPH lo segundo.

## Instalación (Windows + Visual Studio Code)

Abrir la carpeta del proyecto en VS Code (`Archivo > Abrir carpeta`) y, en una terminal PowerShell (`Ctrl + ñ`):

```powershell
cd reconocimiento_facial_opencv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloquea la activación (típico cuando el proyecto está en una unidad de red), ejecute
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y vuelva a activar, o use
`.\.venv\Scripts\python.exe` directamente. Detalles en la [Guía de uso](GUIA_DE_USO.md).
En VS Code, seleccione el intérprete con `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`.

> Debe instalarse `opencv-contrib-python`, **no** `opencv-python` (no incluye `cv2.face`).

## Interfaz gráfica

```powershell
python gui_app.py        # o doble clic en iniciar_app.bat
```

Ventana tema Matrix (verde/negro) con tres botones: **01 Registrar rostro** (pide el nombre y captura), **02 Entrenar modelo** y **03 Validar reconocimiento**. Reutiliza `capture_faces.py`, `train_model.py` y `recognize_faces.py`. Usa solo `tkinter` (incluido con Python), OpenCV y numpy. Detalle en la [Guía de uso](GUIA_DE_USO.md).

## Flujo de uso (por consola)

```powershell
python capture_faces.py     # 1. Registrar una persona (repetir por cada persona)
python train_model.py       # 2. Entrenar el modelo
python recognize_faces.py   # 3. Reconocer en tiempo real
```

1. **Captura**: escriba el nombre, mire a la cámara y mueva ligeramente la cabeza. Se guardan hasta 100 muestras en `data/<nombre>/`. Esc para salir.
2. **Entrenamiento**: lee todas las carpetas de `data/` y genera `models/lbph_model.yml` y `models/labels.json`. Vuelva a ejecutarlo cada vez que agregue personas.
3. **Reconocimiento**: nombre en verde si la distancia es menor a 70; "Desconocido" en rojo si es 70 o más. Se muestra el valor de confianza. Esc para salir.

El umbral (`THRESHOLD = 70`) se puede ajustar en `recognize_faces.py`.

## Estructura

```
reconocimiento_facial_opencv/
├── data/               # Imágenes de entrenamiento, una carpeta por persona
├── models/             # lbph_model.yml y labels.json (generados)
├── capture_faces.py    # Captura de rostros
├── train_model.py      # Entrenamiento LBPH
├── recognize_faces.py  # Reconocimiento en tiempo real
├── requirements.txt
├── README.md
└── .gitignore
```

Todas las rutas se calculan a partir de la ubicación de los scripts, por lo que funcionan desde cualquier carpeta del proyecto.

## Limitaciones

- **Iluminación**: LBPH es sensible a cambios fuertes de luz; capture y reconozca en condiciones similares.
- **Posición del rostro**: Haar solo detecta rostros frontales y poco inclinados.
- **Muestras**: pocas o borrosas muestras, o de una sola persona, dan resultados poco fiables (con 1 persona casi todo se asignará a ella). Se recomiendan 2 o más personas y variedad de gestos.
- Los nombres se guardan sin acentos ni símbolos (OpenCV no los dibuja bien).
- **Prototipo académico**: no es un sistema de seguridad ni de autenticación; puede confundir personas o ser engañado con una fotografía. Las imágenes son datos biométricos: obtenga consentimiento y no las suba al repositorio (ya están en `.gitignore`).
