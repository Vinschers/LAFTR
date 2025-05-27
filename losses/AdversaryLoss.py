import torch
from torch.nn import Module, functional as F


class AdversaryLossDP(Module):
    """
    Adversarial loss for Demographic Parity.

    Parameters
    ----------
        K: (scalar) the number of possible labels for the sensitive attribute A
    """

    def __init__(self, K=2):
        super().__init__()
        self.K = K

    def forward(self, A, adv_prediction):
        if adv_prediction.dim() == 1:
            p1 = adv_prediction
            p0 = 1 - p1
            probs = torch.stack([p0, p1], dim=1)
        else:
            probs = adv_prediction

        A_onehot = F.one_hot(A, num_classes=self.K).float()
        dists = torch.norm(probs - A_onehot, p=1, dim=1)

        loss = torch.tensor(0.0, device=probs.device)

        for a in range(self.K):
            mask = A == a

            if mask.any():
                loss = loss + dists[mask].mean()
            else:
                loss = loss + 1

        return loss / self.K - 1  # Negative of objective function
