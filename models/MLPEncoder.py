import torch.nn as nn
from ._BaseEncoder import _BaseEncoder


class MLPEncoder(_BaseEncoder):

    def _set_net(
        self,
        latent_dim: int,
        in_dim: int = 28 * 28 * 3,
        output_neurons: tuple[int, ...] = (256, 128) # Hidden layer sizes
    ):
        """
        Implements a multi-layer perceptron (MLP) encoder that maps input data to a latent representation.
        The number of hidden layers is configurable based on the output neurons.

        Args:
            latent_dim (int): Dimension of the latent output representation.
            in_dim (int): Dimensionality of the input data (flattened).
            output_neurons (tuple[int]): Sizes of the hidden layers in the MLP.
        """

        dims = (in_dim,) + output_neurons

        layers = []
        layers.append(nn.Flatten())

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(output_neurons[-1], latent_dim))

        self.net = nn.Sequential(*layers)
