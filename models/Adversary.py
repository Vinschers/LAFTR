import torch
import torch.nn as nn
import torch.nn.functional as F

class Adversary(nn.Module):
    def __init__(self, latent_dim, hidden_dims=(64, 32), use_y=True):
        super().__init__()
        self.use_y = use_y
        in_dim = latent_dim + (1 if use_y else 0)
        h1, h2 = hidden_dims

        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1,    h2), nn.ReLU(),
            nn.Linear(h2, 1)
        )

    def forward(self, z, y_pred=None):
        if self.use_y:
            z = torch.cat([z, y_pred.unsqueeze(-1)], dim=1)
        return torch.sigmoid(self.net(z).squeeze(-1))