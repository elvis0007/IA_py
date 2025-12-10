import numpy as np
import torch
import librosa
from pathlib import Path

from siamese_voice_model import SiameseVoiceNet, VoiceEmbeddingNet

EMB_DIR = Path("voice_embeddings")
MODEL_PATH = "models/siamese_voice.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SiameseVoiceNet(VoiceEmbeddingNet()).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def load_audio(path):
    audio, sr = librosa.load(path, sr=16000)
    audio = audio[:16000 * 3]
    audio = librosa.util.pad_center(audio, 16000 * 3)
    audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    return audio


def verify_voice(test_audio, threshold=0.7):
    x = load_audio(test_audio)

    with torch.no_grad():
        new_emb = model.embedding(x).cpu().numpy()[0]

    best_user = None
    best_dist = 999

    for file in EMB_DIR.glob("*.npz"):
        emb = np.load(file)["embeddings"]
        d = np.linalg.norm(emb - new_emb, axis=1).min()

        if d < best_dist:
            best_dist = d
            best_user = file.stem

    print(f"Más parecido: {best_user} | Distancia: {best_dist:.4f}")

    if best_dist < threshold:
        print("✔ Voz coincide")
        return best_user
    else:
        print("❌ No coincide")
        return None
