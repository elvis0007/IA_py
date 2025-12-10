import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    """
    ConvNet simple que convierte una cara 160x160 en un embedding de tamaño 128
    """
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 160 -> 80

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 80 -> 40

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 40 -> 20
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 20 * 20, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # normalizamos para que los embeddings estén en la esfera unitaria
        x = F.normalize(x, p=2, dim=1)
        return x


class SiameseNet(nn.Module):
    def __init__(self, embedding_net: nn.Module | None = None):
        super().__init__()
        self.embedding = embedding_net or EmbeddingNet()

    def forward(self, x1, x2):
        z1 = self.embedding(x1)
        z2 = self.embedding(x2)
        return z1, z2


def contrastive_loss(z1, z2, label, margin: float = 1.0):
    """
    label = 1 -> misma persona
    label = 0 -> distinta
    """
    # distancia euclídea entre embeddings
    d = F.pairwise_distance(z1, z2)

    # pérdida para pares similares: queremos d pequeña
    loss_similar = label.squeeze() * d.pow(2)

    # pérdida para pares distintos: queremos d > margin
    loss_dissimilar = (1 - label.squeeze()) * F.relu(margin - d).pow(2)

    return (loss_similar + loss_dissimilar).mean()
