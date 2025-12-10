import os
import torch
import numpy as np
import librosa
from pathlib import Path

from biometrics_ml.voices.siamese_voice_model import VoiceEmbeddingNet, SiameseVoiceNet

# ============================
# CONFIGURACIÓN
# ============================

DATASET_DIR = Path("dataset_voices")           # dataset voces
OUTPUT_DIR = Path("voice_embeddings")          # dónde guardar embeddings
MODEL_PATH = Path("models/siamese_voice.pth")  # modelo entrenado

OUTPUT_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# CARGAR MODELO ENTRENADO
# ============================

model = SiameseVoiceNet(VoiceEmbeddingNet()).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ============================
# FUNCIÓN PROCESAR AUDIO
# ============================

def load_audio(path):
    """
    Carga un audio WAV, lo recorta o rellena a 3 segundos
    y devuelve tensor adecuado para el encoder.
    """
    audio, sr = librosa.load(path, sr=16000)

    TARGET_LEN = 16000 * 3

    # Rellenar o recortar
    if len(audio) < TARGET_LEN:
        pad = TARGET_LEN - len(audio)
        audio = np.pad(audio, (0, pad))
    else:
        audio = audio[:TARGET_LEN]

    audio = audio.astype("float32")
    audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0).to(device)

    return audio


# ============================
# GENERAR EMBEDDINGS PARA UNA PERSONA
# ============================

def build_embeddings_for_person(person_dir: Path):
    person_name = person_dir.name
    print(f"🔵 Procesando {person_name} ...")

    embeddings = []

    for wav in person_dir.glob("*.wav"):
        x = load_audio(wav)

        with torch.no_grad():
            emb = model.embedding(x).cpu().numpy()[0]
            embeddings.append(emb)

    embeddings = np.array(embeddings)

    # Guardar archivo .npz
    out_path = OUTPUT_DIR / f"{person_name}.npz"
    np.savez(out_path, embeddings=embeddings)

    print(f"✔ Embeddings generados para {person_name} ({len(embeddings)} audios). Guardado en {out_path}")


# ============================
# MAIN
# ============================

def main():
    print("🎧 Generando embeddings de voz...")

    for person in DATASET_DIR.iterdir():
        if person.is_dir():
            build_embeddings_for_person(person)

    print("🎉 TODOS LOS EMBEDDINGS DE VOZ ESTÁN LISTOS")
    print(f"📁 Guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
