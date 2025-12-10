import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceEmbeddingNet(nn.Module):
    """
    Red para convertir un audio (waveform) en un embedding de tamaño 128.
    Combina Conv1D + LSTM para capturar información temporal.
    """
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

        # LSTM toma la secuencia después del conv
        self.lstm = nn.LSTM(
            input_size=128,  # canales después del conv
            hidden_size=128,
            batch_first=True
        )

        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        """
        x: tensor forma (batch, 1, time)
        """
        x = self.conv(x)          # (B, 128, T_new)
        x = x.transpose(1, 2)     # (B, T_new, 128)

        x, _ = self.lstm(x)       # salida de LSTM
        x = x[:, -1, :]           # último estado temporal

        x = self.fc(x)            # proyección final

        # Normalizar embedding (muy importante)
        x = F.normalize(x, p=2, dim=1)

        return x


class SiameseVoiceNet(nn.Module):
    """
    Red siamesa que toma dos audios y retorna sus embeddings.
    """
    def __init__(self, embedding_net=None):
        super().__init__()
        self.embedding = embedding_net or VoiceEmbeddingNet()

    def forward(self, x1, x2):
        z1 = self.embedding(x1)
        z2 = self.embedding(x2)
        return z1, z2


def contrastive_loss(z1, z2, label, margin=1.0):
    """
    label = 1 → misma persona
    label = 0 → personas diferentes
    """
    distances = F.pairwise_distance(z1, z2)

    # pérdida para pares similares
    loss_similar = label * distances.pow(2)

    # pérdida para pares distintos
    loss_dissimilar = (1 - label) * F.relu(margin - distances).pow(2)

    return (loss_similar + loss_dissimilar).mean()
