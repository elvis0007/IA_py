from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth import get_user_model
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required

from .models import BiometricProfile
from .biometric_utils import (
    get_face_embedding_from_base64,
    get_voice_embedding_from_file,
    cosine_distance,
    combine_distances,
)

import json
import numpy as np

User = get_user_model()

# ============================================================
#                       VISTAS HTML
# ============================================================

def login_page(request):
    return render(request, "app_biometric_login/login.html")

from django.contrib.auth import logout
from django.shortcuts import redirect

def logout_user(request):
    logout(request)
    return redirect("login")

def login_email(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        user = authenticate(request, username=email, password=password)
        if user:
            login(request, user)
            return redirect("welcome")

        return render(
            request,
            "app_biometric_login/login_email.html",
            {"error": "Correo o contraseña incorrectos"},
        )

    return render(request, "app_biometric_login/login_email.html")


def login_biometric_page(request):
    return render(request, "app_biometric_login/login_biometric.html")


def register_page(request):
    return render(request, "app_biometric_login/register.html")


@login_required
def welcome_page(request):
    return render(request, "app_biometric_login/welcome.html")


# ============================================================
#                  API: REGISTRO BIOMÉTRICO
# ============================================================

@csrf_exempt
def register_biometric(request):
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=400)

    name = request.POST.get("name")
    email = request.POST.get("email")
    password = request.POST.get("password")

    # Aquí CORREGIMOS — ahora sí recibe correctamente las 5 fotos
    face_images_json = request.POST.get("face_images")
    voice_audio = request.FILES.get("voice_audio")
    print(">>> POST DATA:", request.POST)
    print(">>> FILES RECIBIDOS:", request.FILES)
    print(">>> voice_audio:", voice_audio)
    print(">>> face_images_json:", face_images_json)

    # Validación
    if not all([name, email, password, face_images_json, voice_audio]):
        return JsonResponse({"status": "fail", "message": "Faltan datos"}, status=400)

    face_images = json.loads(face_images_json)

    if len(face_images) < 5:
        return JsonResponse({"status": "fail", "message": "Debes tomar 5 fotos"}, status=400)
    if not all([name, email, password, face_images_json, voice_audio]):
        return JsonResponse({"status": "fail", "message": "Faltan datos"}, status=400)

    # Convertir string JSON → lista Python
    face_images = json.loads(face_images_json)

    if len(face_images) < 5:
        return JsonResponse({"status": "fail", "message": "Debes tomar 5 fotos"}, status=400)

    # Validar usuario existente
    if User.objects.filter(username=email).exists():
        return JsonResponse({"status": "fail", "message": "El usuario ya existe"}, status=400)

    # Crear usuario
    user = User.objects.create_user(username=email, email=email, password=password)
    user.first_name = name
    user.save()

    # ==============================
    #     EMBEDDING FACIAL PROMEDIO
    # ==============================
    emb_list = []
    for img64 in face_images:
        emb = get_face_embedding_from_base64(img64)
        emb_list.append(emb)

    face_embedding = np.mean(emb_list, axis=0)

    # ==============================
    #     EMBEDDING DE VOZ
    # ==============================
    voice_embedding = get_voice_embedding_from_file(voice_audio)

    # Guardar en BD
    BiometricProfile.objects.create(
        user=user,
        face_embedding=face_embedding.tolist(),
        voice_embedding=voice_embedding.tolist(),
    )

    return JsonResponse({"status": "success"})


# ============================================================
#             API: LOGIN / VERIFICACIÓN BIOMÉTRICA
# ============================================================
@csrf_exempt
def verify_biometric(request):
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=400)

    face_image = request.POST.get("face_image")
    voice_audio = request.FILES.get("voice_audio")

    if not face_image or not voice_audio:
        return JsonResponse({"status": "fail", "message": "Faltan datos"}, status=400)

    # Procesar rostro
    face_emb_try = get_face_embedding_from_base64(face_image)

    # Procesar voz
    voice_emb_try = get_voice_embedding_from_file(voice_audio)
    if voice_emb_try is None:
        return JsonResponse({
            "status": "fail",
            "message": "No se detectó voz. Habla claramente."
        })

    profiles = BiometricProfile.objects.select_related("user").all()

    if not profiles:
        return JsonResponse({"status": "fail", "message": "No hay usuarios registrados"}, status=400)

    best_user = None
    best_score = 999
    THRESHOLD = 0.65  # ajustable según precisión real

    # Comparación contra todos los usuarios
    for profile in profiles:
        face_saved = np.array(profile.face_embedding, dtype="float32")
        voice_saved = np.array(profile.voice_embedding, dtype="float32")

        d_face = cosine_distance(face_emb_try, face_saved)
        d_voice = cosine_distance(voice_emb_try, voice_saved)

        score = combine_distances(d_face, d_voice, alpha=0.5)

        if score < best_score:
            best_score = score
            best_user = profile.user

    # Validación final
    if best_user and best_score < THRESHOLD:
        login(request, best_user)
        return JsonResponse({
            "status": "success",
            "user": best_user.first_name,
            "score": float(best_score)
        })

    return JsonResponse({
        "status": "fail",
        "message": "Datos biométricos no coinciden",
        "score": float(best_score)
    })
