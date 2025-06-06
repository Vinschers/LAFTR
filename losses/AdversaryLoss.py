import torch
from torch import Tensor
from torch.nn import Module, functional as F


class AdversaryLoss(Module):
    """
    Adversarial loss for demographic parity.

    This module computes the adversarial loss over sensitive‐attribute groups as:

        loss = ∑_{i=0}^{K-1} (1/|D_i|) · ∑_{n: A_n = i} (1 − p_{n,i})  − 1

    where:
      - adv_logits is a tensor of shape (N, K) containing the adversary’s raw prediction logits,
      - p_{n,i} = softmax(adv_logits_n)[i] is the predicted probability for class i on example n,
      - A is a tensor of sensitive‐attribute labels in {0, …, K−1},
      - D_i = {n ∣ A_n = i} is the set of examples whose sensitive attribute equals i,
      - and the final “−1” shifts the average‐per‐group error into a negative‐objective form.
    """

    def forward(self, adv_logits: Tensor, A: Tensor) -> Tensor:
        """
        Compute the demographic‐parity adversarial loss.

        For each example n, let p_{n,A_n} be the softmax probability assigned to its true
        sensitive‐attribute label A_n. We define the per‐example “error” as (1 − p_{n,A_n}).
        Then we average these errors within each sensitive‐attribute group i, sum across all
        groups, and subtract 1.

        Parameters
        ----------
        adv_logits : Tensor
            Adversary raw logits before softmax, a tensor of shape (N, K).
        A : Tensor
            Sensitive‐attribute labels, an integer tensor of shape (N,), with values in {0,…,K−1}.

        Returns
        -------
        Tensor
            A scalar tensor containing the adversarial loss.
        """
        K = adv_logits.size(1)  # A = {0, 1} -> K=2

        probs = F.softmax(adv_logits, dim=1)  # (N, K) -> h(z) (prediction)

        prob_true = probs.gather(dim=1, index=A.unsqueeze(1)).squeeze(1)
        errors = 1 - prob_true

        loss = torch.zeros(K, device=adv_logits.device)
        loss = loss.index_add(0, A, errors)  # K x 1 vector where each element is the sum of errors for 1, ..., K

        counts_A = torch.bincount(A, minlength=K).float()  # | D_i | for i = 1, ..., K
        loss[counts_A > 0] /= counts_A[counts_A > 0]  # Avoid division by 0

        return loss.sum() - 1  # Negative of objective function
