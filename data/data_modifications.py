import torch
import random

from torchvision import datasets, transforms
from torch.utils.data import Dataset


class ColoredMNIST(Dataset):

    RED = (1.0, 0.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)

    def __init__(self, root, alpha_odd, alpha_even, train=True, download=True):
        self.raw_dataset = datasets.MNIST(
            root=root, train=train, transform=transforms.ToTensor(), download=download
        )
        self.alpha_odd = alpha_odd
        self.alpha_even = alpha_even
        self.data = self._process()

    def _to_rgb(self, image):
        """Converts a grayscale image (1, H, W) to RGB format (3, H, W)
        by duplicating the grayscale channel into all three color channels."""

        return torch.cat([image, image, image], dim=0)

    def _apply_background_color(self, image_rgb, color):
        """Replaces all black background pixels in an RGB image with a specified color."""

        background_mask = (image_rgb == 0).all(dim=0)

        for c in range(3):
            image_rgb[c][background_mask] = color[c]
        return image_rgb

    def _process(self):
        """
        Processes an MNIST dataset by:
        - Converting each image to RGB format
        - Assigning a binary label: 1 for odd digits, 0 for even
        - Modifying the background color of a subset of images:
            - alpha_odd% of odd-labeled images get a blue background
            - alpha_even% of even-labeled images get a red background
        """

        new_data = []

        odd_indices = [
            i for i, (_, label) in enumerate(self.raw_dataset) if label % 2 == 1
        ]
        even_indices = [
            i for i, (_, label) in enumerate(self.raw_dataset) if label % 2 == 0
        ]

        num_odd_to_color = int(self.alpha_odd * len(odd_indices))
        num_even_to_color = int(self.alpha_even * len(even_indices))

        selected_odd = set(random.sample(odd_indices, num_odd_to_color))
        selected_even = set(random.sample(even_indices, num_even_to_color))

        for i, (img, label) in enumerate(self.raw_dataset):
            rgb_img = self._to_rgb(img)
            binary_label = torch.tensor(1 if label % 2 == 1 else 0)

            if label % 2 == 1 and i in selected_odd:
                rgb_img = self._apply_background_color(rgb_img, color=self.RED)  # Red
            elif label % 2 == 0 and i in selected_even:
                rgb_img = self._apply_background_color(rgb_img, color=self.BLUE)  # Blue

            new_data.append((rgb_img, binary_label))

        return new_data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
