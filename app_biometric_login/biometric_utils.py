import base64
import io
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
from pydub import AudioSegment

# Importar tus modelos reales
from biometrics_ml.faces.siamese_faces_model import SiameseNet, EmbeddingNet
from biometrics_ml.voices.siamese_voice_model import SiameseVoiceNet, VoiceEmbeddingNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Rutas a tus modelos entrenados
VOICE_MODEL_PATH = Path("models/siamese_voice.pth")
FACE_MODEL_PATH = Path("models/siamese_faces.pth")


# -------------------------------------------------------------
# CARGA DE MODELOS
# -------------------------------------------------------------
def load_voice_model():
    model = SiameseVoiceNet(VoiceEmbeddingNet()).to(DEVICE)
    model.load_state_dict(torch.load(VOICE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def load_face_model():
    model = SiameseNet(EmbeddingNet()).to(DEVICE)
    model.load_state_dict(torch.load(FACE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


VOICE_MODEL = load_voice_model()
FACE_MODEL = load_face_model()


# Dimensiones por defecto de los embeddings (se asume 128, que es lo típico
# en redes siamesas; si tu modelo usa otra dimensión, puedes ajustarlo aquí)
DEFAULT_FACE_EMB_DIM = 128
DEFAULT_VOICE_EMB_DIM = 128


# -------------------------------------------------------------
# UTILIDAD: obtener vector de ceros para casos inválidos
# -------------------------------------------------------------
def zero_face_embedding():
    return np.zeros(DEFAULT_FACE_EMB_DIM, dtype="float32")


def zero_voice_embedding():
    return np.zeros(DEFAULT_VOICE_EMB_DIM, dtype="float32")


# -------------------------------------------------------------
# OBTENER EMBEDDING DE ROSTRO DESDE BASE64
# -------------------------------------------------------------
def get_face_embedding_from_base64(data_url: str) -> np.ndarray:
    """
    Decodifica una imagen en base64, detecta el rostro con Haar Cascade,
    recorta, normaliza y obtiene el embedding usando el modelo siamés de rostro.

    Si NO se detecta rostro, devuelve un vector de ceros para que la distancia
    sea alta y no pase la autenticación.
    """
    try:
        # Separar cabecera data:image/...;base64,xxxx
        if "," in data_url:
            _, b64data = data_url.split(",", 1)
        else:
            b64data = data_url

        # Decodificar imagen
        image_bytes = base64.b64decode(b64data)
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            # Imagen no válida
            return zero_face_embedding()

        # Detectar rostro con Haar Cascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            # No se detectó rostro → devolución segura
            return zero_face_embedding()

        # Usar la primera cara detectada
        x, y, w, h = faces[0]
        face = gray[y : y + h, x : x + w]

        # Redimensionar a tamaño usado en entrenamiento
        face = cv2.resize(face, (160, 160))
        face = face.astype("float32") / 255.0

        # (1,1,160,160)
        face = np.expand_dims(face, axis=(0, 1))
        face_tensor = torch.tensor(face, dtype=torch.float32).to(DEVICE)

        # Obtener embedding
        with torch.no_grad():
            emb = FACE_MODEL.embedding(face_tensor).cpu().numpy().flatten()

        # En caso de que el modelo tenga otra dimensión, actualizamos el default
        global DEFAULT_FACE_EMB_DIM
        DEFAULT_FACE_EMB_DIM = emb.shape[0]

        return emb.astype("float32")

    except Exception as e:
        # Para no romper el flujo en producción, devolvemos embedding nulo
        print(f"[get_face_embedding_from_base64] Error: {e}")
        return zero_face_embedding()


# -------------------------------------------------------------
# OBTENER EMBEDDING DE VOZ DESDE ARCHIVO
# -------------------------------------------------------------
def get_voice_embedding_from_file(django_file):
    """
    Lee audio desde FormData (WebM/OGG/WAV),
    lo convierte correctamente a float32
    y retorna el embedding de voz.
    """
    try:
        from pydub import AudioSegment
        import numpy as np
        import torch
        import io

        # Leer bytes del archivo
        audio_bytes = django_file.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

        # Convertir a mono 16kHz
        audio = audio.set_frame_rate(16000).set_channels(1)

        # Convertir a array numpy (float32 SIEMPRE)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        # Normalizar
        max_val = np.max(np.abs(samples)) + 1e-8
        samples = samples / max_val

        # 🚨 Detección de silencio
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < 0.01:
            print("[AUDIO ERROR] detectado silencio / volumen bajo")
            return None

        # Duración fija 3s (48000 samples)
        target_len = 16000 * 3
        if len(samples) < target_len:
            pad = np.zeros(target_len - len(samples), dtype=np.float32)
            samples = np.concatenate([samples, pad]).astype(np.float32)
        else:
            samples = samples[:target_len].astype(np.float32)

        # Crear tensor en float32
        audio_tensor = torch.tensor(samples, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Asegurar que el modelo también esté en float32
        VOICE_MODEL.to(torch.float32)

        # Enviar tensor al device
        audio_tensor = audio_tensor.to(DEVICE)

        # Obtener embedding
        with torch.no_grad():
            emb = VOICE_MODEL.embedding(audio_tensor)

        # Convertir a numpy float32
        emb = emb.cpu().numpy().flatten().astype(np.float32)

        return emb

    except Exception as e:
        print("[get_voice_embedding_from_file] Error:", e)
        return None

# -------------------------------------------------------------
# DISTANCIAS
# -------------------------------------------------------------
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype("float32")
    b = b.astype("float32")
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a)) + 1e-8
    norm_b = float(np.linalg.norm(b)) + 1e-8
    return 1.0 - (dot / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype("float32")
    b = b.astype("float32")
    return float(np.linalg.norm(a - b))


def combine_distances(face_dist: float, voice_dist: float, alpha: float = 0.5) -> float:
    """
    Combina distancias de rostro y voz.
    alpha = peso del rostro (por defecto 0.5 → 50% rostro, 50% voz).
    """
    return alpha * face_dist + (1.0 - alpha) * voice_dist

