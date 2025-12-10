import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class SiameseFacesDataset(Dataset):
    """
    Crea pares (img1, img2, label)
    label = 1.0 -> misma persona
    label = 0.0 -> personas distintas
    """
    def __init__(self, root_dir: str | Path, image_size=(160, 160)):
        self.root_dir = Path(root_dir)

        # carpetas persona_01, persona_02, ...
        self.people = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])

        # dict: person_idx -> [lista de imágenes]
        self.person_to_images: dict[int, list[Path]] = {}
        self.samples: list[tuple[Path, int]] = []

        for idx, person_dir in enumerate(self.people):
            imgs = sorted(list(person_dir.glob("*.jpg"))) + \
                   sorted(list(person_dir.glob("*.png")))  # por si acaso

            # necesitamos al menos 2 imágenes por persona
            if len(imgs) < 2:
                continue

            self.person_to_images[idx] = imgs
            for img in imgs:
                self.samples.append((img, idx))

        if not self.samples:
            raise ValueError(f"No se encontraron imágenes en {self.root_dir}")

        self.person_indices = list(self.person_to_images.keys())

        self.transforms = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path):
        img = Image.open(path).convert("L")
        return self.transforms(img)

    def __getitem__(self, idx: int):
        img_path, person_idx = self.samples[idx]

        # 50% positivo (misma persona), 50% negativo (otra persona)
        if random.random() < 0.5:
            # positivo
            same_imgs = self.person_to_images[person_idx]
            img2_path = img_path
            while img2_path == img_path:
                img2_path = random.choice(same_imgs)
            label = 1.0
        else:
            # negativo
            other_person_idx = random.choice(
                [p for p in self.person_indices if p != person_idx]
            )
            img2_path = random.choice(self.person_to_images[other_person_idx])
            label = 0.0

        img1 = self._load_image(img_path)
        img2 = self._load_image(img2_path)

        return img1, img2, torch.tensor([label], dtype=torch.float32)
