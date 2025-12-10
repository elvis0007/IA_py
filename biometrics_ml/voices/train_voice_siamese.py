import os
import random
import torch
import torch.optim as optim
import numpy as np
import librosa
from pathlib import Path

from biometrics_ml.voices.siamese_voice_model import SiameseVoiceNet, contrastive_loss

# ================================================================
# CONFIGURACIÓN
# ================================================================

DATASET = Path("dataset_voices")
MODEL_PATH = Path("models/siamese_voice.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔥 Iniciando entrenamiento de VOZ...")
print("📂 Dataset esperado en:", DATASET.resolve())
print("💾 Guardado de modelo en:", MODEL_PATH.resolve())
print("⚙️ Device:", device)


# ================================================================
# CARGA DE AUDIO CON LOGS
# ================================================================
def load_audio(path):
    print(f"   🔍 Cargando audio: {path}")

    try:
        audio, sr = librosa.load(path, sr=16000)
    except Exception as e:
        print(f"   ❌ ERROR leyendo {path}: {e}")
        raise e

    TARGET_LEN = 16000 * 3  # 3 segundos

    if len(audio) < TARGET_LEN:
        pad_width = TARGET_LEN - len(audio)
        audio = np.pad(audio, (0, pad_width))
        print(f"   ➕ Padding aplicado (faltaban {pad_width} muestras)")

    elif len(audio) > TARGET_LEN:
        audio = audio[:TARGET_LEN]
        print(f"   ✂ Recorte aplicado (audio excedía los 3s)")

    audio = audio.astype("float32")

    # reshape → (batch, channels, time)
    tensor_audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0).to(device)

    print(f"   ✔ Audio cargado correctamente. Shape final: {tensor_audio.shape}")
    return tensor_audio


# ================================================================
# GENERAR PAR POSITIVO/NEGATIVO CON LOGS
# ================================================================
def generate_pair():
    persons = [p for p in DATASET.iterdir() if p.is_dir()]

    if len(persons) < 2:
        raise Exception("❌ ERROR: Se necesitan al menos 2 personas en dataset_voices/")

    print("\n🔄 Generando par...")

    # 50% positivos
    if random.random() < 0.5:
        p = random.choice(persons)
        print(f"   👤 Par positivo (misma persona): {p.name}")

        wavs = list(p.glob("*.wav"))
        if len(wavs) < 2:
            raise Exception(f"❌ ERROR: La carpeta {p} no tiene 2 audios")

        a1, a2 = random.sample(wavs, 2)

        return load_audio(a1), load_audio(a2), torch.tensor([1.0]).to(device)

    # 50% negativos
    p1, p2 = random.sample(persons, 2)
    print(f"   ⚔ Par negativo (personas diferentes): {p1.name} vs {p2.name}")

    a1 = random.choice(list(p1.glob("*.wav")))
    a2 = random.choice(list(p2.glob("*.wav")))

    return load_audio(a1), load_audio(a2), torch.tensor([0.0]).to(device)


# ================================================================
# ENTRENAMIENTO CON LOGS DETALLADOS
# ================================================================
def train():
    print("\n🚀 Preparando modelo Siamese Voice...")
    model = SiameseVoiceNet().to(device)
    optimz = optim.Adam(model.parameters(), lr=0.0002)

    print("✔ Modelo creado.")

    for epoch in range(20):
        print(f"\n==============================")
        print(f"   🟢 EPOCH {epoch+1}/20")
        print(f"==============================")

        loss_total = 0

        for i in range(60):
            print(f"\n--- 🧪 Iteración {i+1}/200 ---")

            x1, x2, y = generate_pair()
            print(f"   📌 Etiqueta del par: {y.item()}")

            z1, z2 = model(x1, x2)

            loss = contrastive_loss(z1, z2, y)
            print(f"   📉 Loss parcial: {loss.item():.4f}")

            optimz.zero_grad()
            loss.backward()
            optimz.step()

            loss_total += loss.item()

        avg = loss_total / 60
        print(f"\n📊 Epoch {epoch+1} completado. Loss promedio = {avg:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print("\n🎉 ENTRENAMIENTO COMPLETADO")
    print("✔ Modelo de voz guardado en:", MODEL_PATH)


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    train()
