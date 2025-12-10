from django.urls import path
from .views import (
    CaptureFaceWeb,
    CaptureVoiceView,
    CaptureAllView,
    TrainFaceModelView,
    TrainVoiceModelView
)

urlpatterns = [
    path("capture/face/", CaptureFaceWeb.as_view(), name="capture_face"),
    path("capture/voice/", CaptureVoiceView.as_view(), name="capture_voice"),
    path("capture/all/", CaptureAllView.as_view(), name="capture_all"),

    # Entrenamiento
    path("train/face/", TrainFaceModelView.as_view()),
    path("train/voice/", TrainVoiceModelView.as_view()),
]
