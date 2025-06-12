import torch
from torch import Tensor
from torchvision import datasets, transforms

from . import BiasedDataset


class BiasedCifar10(BiasedDataset):
    COLORS = Tensor(
        [
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
    )

    def __init__(
        self,
        root: str,
        p_y_a: Tensor | list[list[float]],
        train: bool = False,
        download: bool = False,
        seed: int | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        transform = transforms.ToTensor()
        base = datasets.CIFAR10(root, train=train, transform=transform, download=download)

        self.square_size = 5
        self.square_position = (1, 1)

        super().__init__(base, 10, p_y_a, seed, device)

    def bias_fn(self, img: Tensor, a: int) -> Tensor:
        color = self.COLORS[a].to(self._device).view(1, 3, 1, 1)
        r0, c0 = self.square_position
        s = self.square_size

        img[..., r0 : r0 + s, c0 : c0 + s] = color

        return img
