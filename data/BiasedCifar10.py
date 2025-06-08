import torch

from torchvision import datasets, transforms
from torch.utils.data import Dataset

from . import BiasInjector


class BiasedCifar10(Dataset):
    COLORS = [
        [1.0, 0.0, 0.0],  # red
        [0.0, 1.0, 0.0],  # green
        [0.0, 0.0, 1.0],  # blue
        [1.0, 1.0, 0.0],  # yellow
        [1.0, 0.0, 1.0],  # magenta
        [0.0, 1.0, 1.0],  # cyan
        [1.0, 0.5, 0.0],  # orange
        [0.5, 0.0, 0.5],  # purple
        [0.6, 0.4, 0.2],  # brown
        [1.0, 0.75, 0.8],  # pink
    ]

    def __init__(self, root: str, p_y_a, p_a, train: bool, download: bool = False, seed=None):
        transform = transforms.ToTensor()
        base = datasets.CIFAR10(root, train=train, transform=transform, download=download)

        self.square_size = 5
        self.square_position = (0, 0)

        self.colors = [torch.tensor(c, dtype=torch.float32).view(3, 1, 1) for c in self.COLORS]
        self.biased_dataset = BiasInjector(base, 10, p_y_a, p_a, self._paint_square, seed)

    def _paint_square(self, imgs, j):
        single = imgs.dim() == 3

        if single:
            imgs = imgs.unsqueeze(0)

        color = self.colors[j].to(imgs.device).view(1, 3, 1, 1)
        r0, c0 = self.square_position
        s = self.square_size

        imgs[:, :, r0 : r0 + s, c0 : c0 + s] = color

        if single:
            imgs = imgs.squeeze(0)

        return imgs

    def __getitem__(self, idx):
        return self.biased_dataset[idx]

    def __len__(self):
        return len(self.biased_dataset)
