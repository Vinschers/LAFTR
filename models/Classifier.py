import torch  
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, latent_dim, hidden_dims=(64, 32)):
        super().__init__()
        in_dim = latent_dim
        h1, h2 = hidden_dims

        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1,    h2), nn.ReLU(),
            nn.Linear(h2, 1)
        )

    def forward(self, z):
        return torch.sigmoid(self.net(z).squeeze(-1))