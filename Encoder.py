import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Flattens, pushes through 2 hidden layera
    """
    def __init__(self, latent_dim=16, hidden_dims=(256, 128)):
        super().__init__()
        in_dim = 28 * 28 * 3     
        h1, h2 = hidden_dims

        self.net = nn.Sequential(
            nn.Flatten(),              
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1,    h2), nn.ReLU(),
            nn.Linear(h2, latent_dim)    
        )

    def forward(self, x):
        return self.net(x)