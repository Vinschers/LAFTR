import torch
from torch import Tensor
from torchvision import datasets, transforms

from . import BiasedDataset


class BiasedBinaryMNIST(BiasedDataset):
    COLORS = Tensor(
        [
            [1.0, 0.0, 0.0],  # red
            [0.0, 0.0, 1.0],  # blue
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
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda img: torch.cat([img, img, img], dim=0)),
            ]
        )
        base = datasets.MNIST(root, train=train, transform=transform, download=download)
        base.targets = (base.targets % 2)

        super().__init__(base, 2, p_y_a, seed, device)

    def bias_fn(self, img: Tensor, a: int) -> Tensor:
        color = self.COLORS[a].to(self._device).view(1, 3, 1, 1)
        
        background_mask = (img == 0).all(dim=1, keepdim=True).to(self._device)
        img = torch.where(background_mask, color, img)

        return img