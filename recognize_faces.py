"""Reconocimiento facial en tiempo real con el modelo LBPH entrenado.

Verde + nombre si la distancia < 70; rojo + "Desconocido" en caso contrario.
Salir: tecla Esc.
"""
import json
import sys
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "lbph_model.yml"
LABELS_PATH = BASE_DIR / "models" / "labels.json"
FACE_SIZE = (150, 150)          # igual que en captura y entrenamiento
THRESHOLD = 70                  # LBPH: menor distancia = mayor parecido
GREEN, RED = (0, 255, 0), (0, 0, 255)
ESC = 27


def main() -> int:
    if not hasattr(cv2, "face"):
        print("Error: falta cv2.face. Instale opencv-contrib-python (y desinstale opencv-python).")
        return 1
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        print("Error: no existe el modelo entrenado. Primero ejecute: python train_model.py")
        return 1

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        print("Error: no se pudo cargar el Haar Cascade.")
        return 1

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: no se pudo abrir la cámara. Verifique que esté conectada y libre.")
        return 1

    print("Reconociendo... presione Esc para salir.")
    while True:
        ok, frame = cam.read()
        if not ok:
            print("Error: no se pudo leer un frame de la cámara.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for x, y, w, h in faces:
            # Misma preparación que en el entrenamiento: recorte gris 150x150
            face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            label, distance = recognizer.predict(face)

            if distance < THRESHOLD and label in labels:
                text, color = labels[label].replace("_", " "), GREEN
            else:
                text, color = "Desconocido", RED

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Confianza: {distance:.1f}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Reconocimiento facial (Esc para salir)", frame)
        if cv2.waitKey(1) & 0xFF == ESC:
            break

    cam.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
