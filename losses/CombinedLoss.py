from torch.nn import Module


class CombinedLoss(Module):
    """
    Combined loss.

    Parameters
    ----------
        alpha: (scalar) the weight for the classification loss
        beta: (scalar) the weight for the reconstruction loss
        gamma: (scalar) the weight for the adversary loss
    """

    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, L_C, L_Dev, L_Adv):
        return self.alpha * L_C + self.beta * L_Dev + self.gamma * L_Adv
