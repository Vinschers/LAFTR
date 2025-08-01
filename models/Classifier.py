import torch.nn as nn


class Classifier(nn.Module):
    """
    A simple neural network classifier for predicting task labels from latent representations.

    Args:
        latent_dim (int): Dimensionality of the input latent representation z.
        C (int): Number of output classes. Tipically set to 2 for binary logits.
    """

    def __init__(self, latent_dim: int = 16, C: int = 2):
        super().__init__()

        self.net = nn.Linear(latent_dim, C) # A single linear layer, no hidden layers or ReLU

    def forward(self, z):
        return self.net(z)
