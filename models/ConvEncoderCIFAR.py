import torch
import torch.nn as nn
from ._BaseEncoder import _BaseEncoder
from typing import Sequence, Union

class ConvEncoderCIFAR(_BaseEncoder):
    
    def _set_net(
        self,
        latent_dim: int,
        image_dim: Union[int, tuple[int,int]] = 32,
        in_channels: int = 3,
    ):
        """
        Implements convolutional encoder for CIFAR10 images with 5 layers and batch norm.

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

        layers = []
        channels = [32, 64, 128, 256, latent_dim]
        prev_channels = in_channels

        for out_channel in channels:
            layers.append(nn.Conv2d(prev_channels, out_channel, kernel_size= 3, stride=2, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(latent_dim))
            prev_channels = out_channel

        layers.append(nn.Flatten())  # final shape (B, latent_dim)

        self.net = nn.Sequential(*layers)
