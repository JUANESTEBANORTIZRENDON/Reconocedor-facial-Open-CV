"""Entrena un reconocedor LBPH con las imágenes de data/.

Genera models/lbph_model.yml y models/labels.json (etiqueta -> nombre).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "lbph_model.yml"
LABELS_PATH = MODELS_DIR / "labels.json"
FACE_SIZE = (150, 150)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_dataset():
    """Recorre data/<persona>/ y devuelve imágenes, etiquetas y el mapeo etiqueta->nombre."""
    images, labels, mapping = [], [], {}
    people = sorted(p for p in DATA_DIR.iterdir() if p.is_dir()) if DATA_DIR.exists() else []
    for person in people:
        label = len(mapping)
        samples = 0
        for path in sorted(person.iterdir()):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Aviso: no se pudo leer {path.name}, se omite.")
                continue
            images.append(cv2.resize(img, FACE_SIZE))   # garantiza el mismo tamaño
            labels.append(label)
            samples += 1
        if samples:
            mapping[label] = person.name
            print(f"  - {person.name}: {samples} muestras")
        else:
            print(f"Aviso: la carpeta '{person.name}' no tiene imágenes válidas, se omite.")
    return images, labels, mapping


def main() -> int:
    if not hasattr(cv2, "face"):
        print("Error: falta cv2.face. Instale opencv-contrib-python (y desinstale opencv-python).")
        return 1

    print("Leyendo imágenes de data/ ...")
    images, labels, mapping = load_dataset()

    if not images:
        print("Error: no hay imágenes en data/. Ejecute primero capture_faces.py.")
        return 1
    if len(mapping) < 2:
        print("Aviso: solo hay 1 persona registrada; todo rostro tenderá a asignarse a ella. "
              "Se recomienda registrar al menos 2.")

    # Los labels se reasignaron de forma consecutiva (0..n-1) si se omitió alguna carpeta vacía
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels, dtype=np.int32))

    MODELS_DIR.mkdir(exist_ok=True)
    recognizer.write(str(MODEL_PATH))
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)

    print(f"\nEntrenamiento completo: {len(mapping)} persona(s), {len(images)} muestras.")
    print(f"Modelo:    models/{MODEL_PATH.name}")
    print(f"Etiquetas: models/{LABELS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
