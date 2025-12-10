from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class BiometricProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="biometric_profile")
    # Guardamos los embeddings como JSON (lista de floats)
    face_embedding = models.JSONField(null=True, blank=True)
    voice_embedding = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"Biometría de {self.user.email if hasattr(self.user, 'email') else self.user.username}"
