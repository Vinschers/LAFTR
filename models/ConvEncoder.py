from typing import Union
import torch
import torch.nn as nn

from ._BaseEncoder import _BaseEncoder


class ConvEncoder(_BaseEncoder):

    def _set_net(
        self,
        latent_dim: int,
        image_dim: Union[int, tuple[int, int]] = 28,
        in_channels: int = 3
    ):
        """
        Implements a convolutional encoder that maps image input to a latent representation.

        Args:
            latent_dim (int): Dimension of the output latent space.
            image_dim (int or tuple[int]): Input image size, either as an int (for square images)
                                           or a tuple (height, width).
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
        """

        if isinstance(image_dim, int):
            h, w = image_dim, image_dim
        elif isinstance(image_dim, tuple) and len(image_dim) == 2:
            h, w = image_dim
        else:
            raise ValueError(
                "image_dim must be an int or a tuple of two ints (height, width)"
            )

        # Define the convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=2, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Determine the flattened output size after conv layers using a dummy input
        dummy_input = torch.zeros(1, in_channels, h, w)

        with torch.no_grad():
            out = self.conv(dummy_input)
            flatten_dim = out.view(1, -1).shape[1]

        # Compose final encoder: conv -> flatten -> linear projection to latent_dim
        second_conv = nn.Sequential(
            self.conv,
            nn.Conv2d(2, latent_dim, kernel_size=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # Outputs a 1 x 1 x latent_dim
            nn.Flatten(),
        )
        self.net = second_conv