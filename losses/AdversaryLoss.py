import torch
from torch import Tensor
from torch.nn import Module, functional as F


class AdversaryLossDP(Module):
    """
    Adversarial loss for demographic parity.

    This module computes the demographic‐parity adversarial loss:
        loss = ∑_{i=1}^K (1/|D_i|) · ∑_{n: A_n = i} ‖softmax(adv_logits_n) − one_hot(A_n)‖₁ − 1
    where:
      - A is a tensor of sensitive attribute labels in {0, …, K−1},
      - adv_logits are the adversary’s raw prediction logits of shape (N, K),
      - D_i = {n ∣ A_n = i} is the subset of examples with attribute i,
      - and the final subtraction of 1 yields the negative objective value.
    """

    def forward(self, A: Tensor, adv_logits: Tensor) -> Tensor:
        """
        Compute the demographic‐parity adversarial loss.

        Parameters
        ----------
        A : Tensor
            Sensitive attribute labels, an integer tensor of shape (N,).
        adv_logits : Tensor
            Adversary raw logits before softmax, a tensor of shape (N, K).

        Returns
        -------
        Tensor
            Scalar tensor representing the demographic‐parity adversarial loss.
        """
        K = adv_logits.size(1)

        probs = F.softmax(adv_logits, dim=1)
        one_hot_A = F.one_hot(A, num_classes=K).float()
        counts_A = one_hot_A.sum(dim=0)  # | D_i | for i = 1, ..., K

        errors = torch.norm(probs - one_hot_A, p=1, dim=1)

        loss = one_hot_A.T @ errors  # K x 1 vector where each element is the sum of errors for 1, ..., K
        loss[loss > 0] /= counts_A[loss > 0]  # Avoid division by 0

        return loss.sum() - 1  # Negative of objective function


class AdversaryLossEO(Module):
    """
    Adversarial loss for equalized odds.

    This module computes the equalized‐odds adversarial loss:
        loss = ∑_{i=1}^K ∑_{j=1}^C (1/|D_i^j|) · ∑_{n: A_n=i, Y_n=j} ‖softmax(adv_logits_n) − one_hot(A_n)‖₁ − C
    where:
      - A is a tensor of sensitive attribute labels in {0, …, K−1},
      - y_pred is the tensor of predicted class labels in {0, …, C−1},
      - adv_logits are the adversary’s raw logits of shape (N, K),
      - D_i^j = {n ∣ A_n = i and Y_n = j},
      - and C is the number of predicted classes.
    """

    def forward(self, A: Tensor, y_pred: Tensor, adv_logits: Tensor) -> Tensor:
        """
        Compute the equalized‐odds adversarial loss.

        Parameters
        ----------
        A : Tensor
            Sensitive attribute labels, an integer tensor of shape (N,).
        y_pred : Tensor
            Predicted class labels, an integer tensor of shape (N,).
        adv_logits : Tensor
            Adversary raw logits before softmax, a tensor of shape (N, K).

        Returns
        -------
        Tensor
            Scalar tensor representing the equalized‐odds adversarial loss.
        """
        K = adv_logits.size(1)
        C = int(y_pred.max().item()) + 1

        one_hot_A = F.one_hot(A, num_classes=K).float()
        one_hot_Y = F.one_hot(y_pred, num_classes=C).float()

        # One-hot 3D matrix. if element (i, j, k) = 1, then the i-th element from the batch has A = j and Y = k
        joint_AY = one_hot_A.unsqueeze(2) * one_hot_Y.unsqueeze(1)
        one_hot_AY = joint_AY.reshape(-1, K * C)  # Now, it is a 2D matrix that corresponds to the combinations of A and Y

        probs = F.softmax(adv_logits, dim=1)
        counts_AY = one_hot_AY.sum(dim=0)  # | D_i^j | for i = 1, ..., K and j = 1, ..., C

        errors = torch.norm(probs - one_hot_A, p=1, dim=1)  # Vector the size of the batch

        loss = one_hot_AY.T @ errors  # KC x 1 vector where each element is the sum of errors for combinations of A and Y
        loss[loss > 0] /= counts_AY[loss > 0]  # Avoid division by 0

        return loss.sum() - C
