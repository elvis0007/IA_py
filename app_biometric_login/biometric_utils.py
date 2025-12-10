import base64
import io
import numpy as np
import cv2
import torch
import librosa
import soundfile as sf
from pathlib import Path

# Importar tus modelos reales
from biometrics_ml.faces.siamese_faces_model import SiameseNet, EmbeddingNet
from biometrics_ml.voices.siamese_voice_model import SiameseVoiceNet, VoiceEmbeddingNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Rutas a tus modelos entrenados
VOICE_MODEL_PATH = Path("models/siamese_voice.pth")
FACE_MODEL_PATH = Path("models/siamese_faces.pth")


# -------------------------------------------------------------
# CARGAR MODELO DE VOZ
# -------------------------------------------------------------
def load_voice_model():
    model = SiameseVoiceNet(VoiceEmbeddingNet()).to(DEVICE)
    model.load_state_dict(torch.load(VOICE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# -------------------------------------------------------------
# CARGAR MODELO DE ROSTRO
# -------------------------------------------------------------
def load_face_model():
    model = SiameseNet(EmbeddingNet()).to(DEVICE)
    model.load_state_dict(torch.load(FACE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


VOICE_MODEL = load_voice_model()
FACE_MODEL = load_face_model()


# -------------------------------------------------------------
# OBTENER EMBEDDING DE ROSTRO DESDE BASE64
# -------------------------------------------------------------
def get_face_embedding_from_base64(data_url: str) -> np.ndarray:
    if "," in data_url:
        _, b64data = data_url.split(",", 1)
    else:
        b64data = data_url

    # Decodificar imagen
    image_bytes = base64.b64decode(b64data)
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Preprocesamiento EXACTO COMO ENTRENASTE
    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, 1))  # (1,1,160,160)
    img_tensor = torch.tensor(img).to(DEVICE)

    # Obtener embedding real
    with torch.no_grad():
        emb = FACE_MODEL.embedding(img_tensor).cpu().numpy().flatten()

    return emb


# -------------------------------------------------------------
# OBTENER EMBEDDING DE VOZ DESDE ARCHIVO
# -------------------------------------------------------------

def get_voice_embedding_from_file(django_file):
    """
    Lee audio desde FormData (generalmente WebM/OGG) 
    y lo convierte en un vector.
    """
    from pydub import AudioSegment
    import numpy as np
    import torch

    # Convertir blob a AudioSegment
    audio_bytes = django_file.read()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

    # Convertir a 16kHz mono WAV interno
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Asegurar duración exacta de 3s
    target_ms = 3000
    if len(audio) < target_ms:
        silence = AudioSegment.silent(duration=target_ms - len(audio))
        audio = audio + silence
    else:
        audio = audio[:target_ms]

    # Convertir a array numpy float32
    samples = np.array(audio.get_array_of_samples()).astype("float32")
    samples = samples / np.iinfo(audio.array_type).max  # normalización

    # Convertir a tensor
    audio_tensor = torch.tensor(samples).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 🔥 Aquí debes llamar a tu modelo real (por ahora generamos uno aleatorio)
    embedding = np.random.rand(128).astype("float32")

    return embedding


# -------------------------------------------------------------
# DISTANCIAS
# -------------------------------------------------------------
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype("float32")
    b = b.astype("float32")
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a) + 1e-8
    norm_b = np.linalg.norm(b) + 1e-8
    return 1.0 - (dot / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def combine_distances(face_dist: float, voice_dist: float, alpha: float = 0.5) -> float:
    return alpha * face_dist + (1 - alpha) * voice_dist
