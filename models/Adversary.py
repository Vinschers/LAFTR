import torch.nn as nn


class Adversary(nn.Module):
    """
    An adversarial neural network module for predicting sensitive attributes from latent representations.
    
    Args:
        latent_dim (int): Dimensionality of the input latent representation z.
        hidden_dim (int): Number of units in the hidden layer.
        K (int): Number of output logits. Typically set to 2 for binary classification.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64, K: int = 2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K)
        )

    def forward(self, z):
        return self.net(z).squeeze(-1) # Squeeze is used to handle shape (batch_size, 1) -> (batch_size)
