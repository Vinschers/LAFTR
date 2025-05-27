import torch
from torch.nn import Module


def _dist_from_true(A, adv_prediction):
    # If adv_prediction is 1D or a column vector
    if adv_prediction.dim() == 1 or (adv_prediction.dim() == 2 and adv_prediction.size(1) == 1):
        p1 = adv_prediction.view(-1)
        p_true = torch.where(A.bool(), p1, 1 - p1)
    else:
        p_true = adv_prediction.gather(1, A.unsqueeze(1)).squeeze(1)

    return 1 - p_true


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
        dists = _dist_from_true(A, adv_prediction)
        loss = 0

        for a in range(self.K):
            mask = A == a

            if mask.any():
                loss += dists[mask].mean()

        return loss - 1  # Negative of objective function


class AdversaryLossEO(Module):
    """
    Adversarial loss for Equalized Odds.

    Parameters
    ----------
        K: (scalar) the number of possible labels for the sensitive attribute A
        C: (scalar) the number of possible classes for prediction
    """

    def __init__(self, K=2, C=2):
        super().__init__()
        self.K = K
        self.C = C

    def forward(self, A, Y_pred, adv_prediction):
        dists = _dist_from_true(A, adv_prediction)
        loss = 0

        for a in range(self.K):
            for c in range(self.C):
                mask = (A == a) & (Y_pred == c)

                if mask.any():
                    loss += dists[mask].mean()

        return loss - self.C  # Negative of objective function
