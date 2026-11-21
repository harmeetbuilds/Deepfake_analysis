import io, random
from pathlib import Path

from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLASSES = ["real", "fake"]  # label 0 = real, 1 = fake


class RandomJPEG:
    """Simulate re-compression (social-media style) so the model is robust to it."""
    def __init__(self, p=0.3, q=(40, 95)):
        self.p, self.q = p, q

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(*self.q))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomBlur:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5))) if random.random() < self.p else img


def get_train_transform(size=224):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        RandomBlur(), RandomJPEG(),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])


def get_eval_transform(size=224):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


class FaceDataset(Dataset):
    """Expects root/{real,fake}/*.jpg  (e.g. data/processed/train)."""
    def __init__(self, root, transform):
        self.items = []
        for label, cls in enumerate(CLASSES):
            self.items += [(p, label) for p in sorted((Path(root) / cls).glob("*.*"))]
        if not self.items:
            raise RuntimeError(f"No images found in {root}")
        self.labels = [l for _, l in self.items]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), torch.tensor(y, dtype=torch.float32), str(p)
