import numpy as np
import torch
import cv2
import librosa
from pathlib import Path

from biometrics_ml.faces.siamese_faces_model import SiameseNet, EmbeddingNet
from biometrics_ml.voices.siamese_voice_model import SiameseVoiceNet, VoiceEmbeddingNet


# ======================================================
# CONFIGURACIÓN
# ======================================================

FACE_MODEL_PATH = Path("models/siamese_face.pth")
VOICE_MODEL_PATH = Path("models/siamese_voice.pth")

FACE_EMB_DIR = Path("face_embeddings")
VOICE_EMB_DIR = Path("voice_embeddings")

device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = (160, 160)
AUDIO_LEN = 16000 * 3  # 3 segundos


# ======================================================
# CARGAR MODELOS ENTRENADOS
# ======================================================

# Modelo de rostro
face_model = SiameseNet(EmbeddingNet()).to(device)
face_model.load_state_dict(torch.load(FACE_MODEL_PATH, map_location=device))
face_model.eval()

# Modelo de voz
voice_model = SiameseVoiceNet(VoiceEmbeddingNet()).to(device)
voice_model.load_state_dict(torch.load(VOICE_MODEL_PATH, map_location=device))
voice_model.eval()


# ======================================================
# UTILIDADES DE PROCESAMIENTO
# ======================================================

def preprocess_face(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
    return img


def preprocess_voice(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)

    if len(audio) < AUDIO_LEN:
        audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
    else:
        audio = audio[:AUDIO_LEN]

    audio = audio.astype("float32")
    audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0).to(device)
    return audio


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ======================================================
# INFERENCE: ROSTRO
# ======================================================

def identify_face(face_img_path, threshold=0.65):
    x = preprocess_face(face_img_path)

    with torch.no_grad():
        emb_query = face_model.embedding(x).cpu().numpy()[0]

    best_person = None
    best_score = -1

    for npz_file in FACE_EMB_DIR.glob("*.npz"):
        data = np.load(npz_file)
        emb_list = data["embeddings"]

        # comparar query con todos los embeddings de esa persona
        sims = [cosine_similarity(emb_query, e) for e in emb_list]
        max_sim = max(sims)

        if max_sim > best_score:
            best_score = max_sim
            best_person = npz_file.stem

    is_valid = best_score >= threshold

    return {
        "person": best_person,
        "score": float(best_score),
        "accepted": is_valid
    }


# ======================================================
# INFERENCE: VOZ
# ======================================================

def identify_voice(audio_path, threshold=0.60):
    x = preprocess_voice(audio_path)

    with torch.no_grad():
        emb_query = voice_model.embedding(x).cpu().numpy()[0]

    best_person = None
    best_score = -1

    for npz_file in VOICE_EMB_DIR.glob("*.npz"):
        data = np.load(npz_file)
        emb_list = data["embeddings"]

        sims = [cosine_similarity(emb_query, e) for e in emb_list]
        max_sim = max(sims)

        if max_sim > best_score:
            best_score = max_sim
            best_person = npz_file.stem

    is_valid = best_score >= threshold

    return {
        "person": best_person,
        "score": float(best_score),
        "accepted": is_valid
    }


# ======================================================
# MATCH DUAL: FACE + VOICE
# ======================================================

def identify_both(face_img, audio_wav):
    face_res = identify_face(face_img)
    voice_res = identify_voice(audio_wav)

    same_person = (
        face_res["person"] == voice_res["person"]
        and face_res["accepted"]
        and voice_res["accepted"]
    )

    return {
        "face": face_res,
        "voice": voice_res,
        "match": same_person
    }
