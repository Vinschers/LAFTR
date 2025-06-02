import torch
from torch import Tensor
from torch.nn import Module, functional as F


class AdversaryLoss(Module):
    """
    Adversarial loss for demographic parity.

    This module computes the demographic‐parity adversarial loss:
        loss = ∑_{i=1}^K (1/|D_i|) · ∑_{n: A_n = i} ‖softmax(adv_logits_n) − one_hot(A_n)‖₁ − 1
    where:
      - adv_logits are the adversary’s raw prediction logits of shape (N, K),
      - A is a tensor of sensitive attribute labels in {0, …, K−1},
      - D_i = {n ∣ A_n = i} is the subset of examples with attribute i,
      - and the final subtraction of 1 yields the negative objective value.
    """

    def forward(self, adv_logits: Tensor, A: Tensor) -> Tensor:
        """
        Compute the demographic‐parity adversarial loss.

        Parameters
        ----------
        adv_logits : Tensor
            Adversary raw logits before softmax, a tensor of shape (N, K).
        A : Tensor
            Sensitive attribute labels, an integer tensor of shape (N,).

        Returns
        -------
        Tensor
            Scalar tensor representing the demographic‐parity adversarial loss.
        """
        K = adv_logits.size(1)

        one_hot_A = F.one_hot(A, num_classes=K).float()
        counts_A = one_hot_A.sum(dim=0)  # | D_i | for i = 1, ..., K

        pred = F.softmax(adv_logits, dim=1)
        # pred = F.one_hot(torch.argmax(pred, dim=1), num_classes=K).float()

        errors = torch.norm(pred - one_hot_A, p=1, dim=1)

        loss = one_hot_A.T @ errors  # K x 1 vector where each element is the sum of errors for 1, ..., K
        loss[loss > 0] /= counts_A[loss > 0]  # Avoid division by 0

        return loss.sum()  # Negative of objective function without constant terms
