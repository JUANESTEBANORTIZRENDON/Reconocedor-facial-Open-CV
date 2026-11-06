"""Captura rostros de una persona desde la cámara web.

Guarda hasta 100 imágenes en escala de grises (150x150) en data/<nombre>/.
Salir: tecla Esc.
"""
import re
import sys
import unicodedata
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent      # carpeta del proyecto
DATA_DIR = BASE_DIR / "data"
FACE_SIZE = (150, 150)                          # mismo tamaño en todos los scripts
MAX_SAMPLES = 100
FRAME_STEP = 3                                  # guarda 1 de cada 3 frames con rostro (más variedad)
ESC = 27


def clean_name(raw: str) -> str:
    """Convierte el nombre en un nombre de carpeta seguro (sin acentos ni símbolos)."""
    text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text


def main() -> int:
    name = clean_name(input("Nombre de la persona: ").strip())
    if not name:
        print("Error: el nombre no puede estar vacío (use letras o números).")
        return 1

    person_dir = DATA_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    # Continúa la numeración si la persona ya tenía muestras
    count = len(list(person_dir.glob("*.jpg")))
    if count >= MAX_SAMPLES:
        print(f"'{name}' ya tiene {count} muestras. Borre la carpeta data/{name} para recapturar.")
        return 0

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        print("Error: no se pudo cargar el Haar Cascade.")
        return 1

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: no se pudo abrir la cámara. Verifique que esté conectada y libre.")
        return 1

    print(f"Capturando para '{name}'. Mire a la cámara y mueva un poco la cabeza. Esc para salir.")
    frame_id = 0
    while count < MAX_SAMPLES:
        ok, frame = cam.read()
        if not ok:
            print("Error: no se pudo leer un frame de la cámara.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        # Solo se guarda cuando hay exactamente un rostro, para evitar muestras de otras personas
        if len(faces) == 1:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame_id += 1
            if frame_id % FRAME_STEP == 0:
                face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
                count += 1
                cv2.imwrite(str(person_dir / f"{count:03d}.jpg"), face)

        cv2.putText(frame, f"{name}: {count}/{MAX_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Captura de rostros (Esc para salir)", frame)
        if cv2.waitKey(1) & 0xFF == ESC:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Listo: {count} muestras en data/{name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
