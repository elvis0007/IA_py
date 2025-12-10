import os
import torch
import cv2
import numpy as np
from pathlib import Path

from biometrics_ml.faces.siamese_faces_model import SiameseNet, EmbeddingNet

# ============================================================
# CONFIGURACIÓN DE DIRECTORIOS
# ============================================================

DATASET_DIR = Path("dataset_faces")          
OUTPUT_DIR = Path("face_embeddings")        
OUTPUT_DIR.mkdir(exist_ok=True)

# Ruta correcta al modelo entrenado
MODEL_PATH = Path("models/siamese_faces.pth")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ No se encontró el modelo en: {MODEL_PATH}")

# Dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# CARGAR MODELO ENTRENADO
# ============================================================

model = SiameseNet(EmbeddingNet()).to(device)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
model.eval()

# Tamaño esperado
IMG_SIZE = (160, 160)

# ============================================================
# PREPROCESAMIENTO DE IMAGEN
# ============================================================

def preprocess(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"No se pudo leer imagen: {img_path}")

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

    return img

# ============================================================
# GENERAR EMBEDDINGS POR USUARIO
# ============================================================

def build_embeddings_for_person(person_dir):
    person_name = person_dir.name
    embeddings = []

    for img_path in person_dir.glob("*.jpg"):
        x = preprocess(img_path)

        with torch.no_grad():
            emb = model.embedding(x).cpu().numpy()[0]
            embeddings.append(emb)

    embeddings = np.array(embeddings)
    np.savez(OUTPUT_DIR / f"{person_name}.npz", embeddings=embeddings)

    print(f"✔ Embeddings generados: {person_name} ({len(embeddings)} imágenes)")


# ============================================================
# MAIN
# ============================================================

def main():
    print("🔵 Generando embeddings para cada sujeto...")

    for folder in DATASET_DIR.iterdir():
        if folder.is_dir():
            build_embeddings_for_person(folder)

    print("\n✅ Todos los embeddings listos en /face_embeddings/\n")


if __name__ == "__main__":
    main()
