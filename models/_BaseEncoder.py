import torch.nn as nn
from abc import ABC, abstractmethod


class _BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for encoder modules that map input data to a latent representation.

    Args:
        latent_dim (int): Dimensionality of the latent space output.
        *args, **kwargs: Additional arguments to be passed to the subclass implementation.
    """

    def __init__(self, latent_dim: int = 16, *args, **kwargs):
        super().__init__()

        self._set_net(latent_dim=latent_dim, *args, **kwargs)

    @abstractmethod
    def _set_net(self, latent_dim, *args, **kwargs):
        """
        Abstract method for defining the encoder's internal neural network.
        Must be implemented by subclasses to initialize self.net.
        """
        pass

    def forward(self, x):
        return self.net(x)
