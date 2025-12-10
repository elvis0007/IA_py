from django.views import View
from django.shortcuts import render
from django.http import JsonResponse
from django.views.generic import TemplateView
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import os
import subprocess
import base64
import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path(settings.BASE_DIR)

# Carpetas del dataset
RAW_FACE_DIR = BASE_DIR / "dataset_raw/faces"
RAW_VOICE_DIR = BASE_DIR / "dataset_raw/voices"

PRO_FACE_DIR = BASE_DIR / "dataset_faces"   # caras procesadas
PRO_FACE_DIR.mkdir(parents=True, exist_ok=True)

FACE_SIZE = (160, 160)

# Haarcascade
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def ensure_folder(path: Path):
    """Crea carpeta si no existe."""
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# CAPTURA DE ROSTROS (PROCESADO DESDE BASE64)
# ============================================================

@method_decorator(csrf_exempt, name='dispatch')
class CaptureFaceWeb(View):

    def post(self, request):
        username = request.POST.get("username")
        image_b64 = request.POST.get("image")

        if not username or not image_b64:
            return JsonResponse({"error": "missing data"}, status=400)

        user_folder = PRO_FACE_DIR / username
        ensure_folder(user_folder)

        # ---- Convertir base64 a imagen ----
        try:
            img_data = base64.b64decode(image_b64.split(",")[1])
        except:
            return JsonResponse({"error": "invalid_b64"}, status=400)

        np_data = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if frame is None:
            return JsonResponse({"error": "decode_failed"}, status=400)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Detección del rostro ----
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return JsonResponse({"status": "no_face"})

        # Seleccionar rostro más grande
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face = gray[y:y + h, x:x + w]

        # Redimensionar
        face_resized = cv2.resize(face, FACE_SIZE)

        # Guardar incrementalmente
        existing = list(user_folder.glob("face_*.jpg"))
        file_number = len(existing) + 1
        filename = user_folder / f"face_{file_number:03d}.jpg"

        cv2.imwrite(str(filename), face_resized)

        return JsonResponse({"status": "ok", "count": file_number})


# ============================================================
# CAPTURA DE VOZ (RECIBE WAV)
# ============================================================

@method_decorator(csrf_exempt, name='dispatch')
class CaptureVoiceView(View):

    def post(self, request):
        username = request.POST.get("username")
        audio_file = request.FILES.get("audio")

        if not username or not audio_file:
            return JsonResponse({"error": "missing data"}, status=400)

        user_dir = RAW_VOICE_DIR / username
        ensure_folder(user_dir)

        count = len(os.listdir(user_dir))
        file_path = user_dir / f"{username}_{count}.wav"

        with open(file_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        return JsonResponse({"status": "ok", "saved": str(file_path)})


# ============================================================
# ENTRENAMIENTO DE MODELOS
# ============================================================

@method_decorator(csrf_exempt, name='dispatch')
class TrainFaceModelView(View):
    def get(self, request):
        return render(request, "app_dataset_creator/train_models.html")

    def post(self, request):
        subprocess.run(["python", "app_dataset_creator/scripts/train_face_model.py"])
        return JsonResponse({"status": "ok"})


@method_decorator(csrf_exempt, name='dispatch')
class TrainVoiceModelView(View):
    def post(self, request):
        subprocess.run(["python", "app_dataset_creator/scripts/train_voice_model.py"])
        return JsonResponse({"status": "ok"})


# ============================================================
# VISTA GENERAL: HTML UNIFICADO (ROSTRO + VOZ)
# ============================================================

class CaptureAllView(TemplateView):
    template_name = "app_dataset_creator/capture_all.html"
