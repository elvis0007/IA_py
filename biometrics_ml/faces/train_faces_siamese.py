from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset_faces_siamese import SiameseFacesDataset
from .siamese_faces_model import SiameseNet, EmbeddingNet, contrastive_loss


# rutas (ajusta si tu estructura es distinta)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset_faces"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "siamese_faces.pth"

BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 20


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    dataset = SiameseFacesDataset(DATASET_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SiameseNet(EmbeddingNet()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    MODEL_DIR.mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            z1, z2 = model(img1, img2)
            loss = contrastive_loss(z1, z2, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Modelo Siamese de rostro guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    train()
