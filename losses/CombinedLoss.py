from torch import Tensor
from torch.nn import Module


class CombinedLoss(Module):
    """
    Combined loss for adversarially‐trained models.

    This module implements the loss
        L = L_C − γ × L_adv
    where:
      - L_C is the primary (e.g. classification) loss,
      - L_adv is the adversarial loss, and
      - γ (gamma) balances the strength of the adversary term.

    Attributes
    ----------
    gamma : float
        Weighting factor for the adversarial loss term.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        """
        Parameters
        ----------
        gamma : float, optional
            Scale factor for the adversarial loss, by default 1.0.
        """
        super().__init__()
        self.gamma: float = gamma

    def forward(self, L_C: Tensor, L_adv: Tensor) -> Tensor:
        """
        Compute the combined loss.

        Parameters
        ----------
        L_C : Tensor
            The primary loss (e.g., classification loss). Should be a scalar tensor.
        L_adv : Tensor
            The adversarial loss. Should be a scalar tensor.

        Returns
        -------
        Tensor
            The scalar combined loss: L_C − gamma * L_adv.
        """
        return L_C - self.gamma * L_adv
