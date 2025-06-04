import torch
from torch import Tensor
from torch.nn import Module, functional as F


class AdversaryLoss(Module):
    """
    Adversarial loss for demographic parity and equalized odds.

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
        Compute the DP or EO adversarial loss.

        Parameters
        ----------
        adv_logits : Tensor
            Adversary raw logits before softmax, a tensor of shape (N, K).
        A : Tensor
            Sensitive attribute labels, an integer tensor of shape (N,).

        Returns
        -------
        Tensor
            Scalar tensor representing the adversarial loss.
        """
        K = adv_logits.size(1) # A = {0, 1} -> K=2

        one_hot_A = F.one_hot(A, num_classes=K).float() # [[0, 1], [1,0], [1, 0]] (N, K)
        counts_A = torch.bincount(A, minlength=K).float()  # | D_i | for i = 1, ..., K ([2, 1] (1, K)))

        # [[0.1, 0.9], [0.95, 0.05], [0.7, 0.3]] (N, K) -> h(z) (prediction)
        pred = F.softmax(adv_logits, dim=1)
        errors = torch.norm(pred - one_hot_A, p=1, dim=1) # L1-norm -> ||[[0.1, 0.9], [0.95, 0.05], [0.7, 0.3]] - [[0, 1], [1,0], [1, 0]]||

        loss = torch.zeros(K, device=adv_logits.device)

        loss = loss.index_add(0, A, errors)  # K x 1 vector where each element is the sum of errors for 1, ..., K
        loss[counts_A > 0] /= counts_A[counts_A > 0]  # Avoid division by 0

        return loss.sum() - 1  # Negative of objective function without constant terms
