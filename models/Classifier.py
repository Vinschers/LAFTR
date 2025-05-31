import torch  
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.net(z).squeeze(-1) # output between (-inf, +inf)