from typing import Sized, Any, cast
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.utils.data import Dataset


class BiasedDataset(Dataset, ABC):
    """
    Wrap any dataset to inject a spurious attribute A via Bayes' rule,
    pre-applying the bias so __getitem__ is just a tensor lookup.

    Parameters
    ----------
    base_dataset : torch.utils.data.Dataset
        Underlying dataset; __getitem__ must return (image, label) with
        label in {0,...,C-1}.
    C : int
        Number of classes.
    p_y_a : array-like, shape (C, K)
        P(Y=i | A=j), for i=0..C-1 and j=0..K-1. Each column must sum to 1.
    p_a : array-like, shape (K,)
        Prior P(A=j). Must sum to 1.
    bias_fn : Callable[[Tensor, int], Tensor]
        Called as bias_fn(imgs, a), returns the biased image(s).
    seed : int, optional
        Random seed for reproducible attribute assignment.
    device : torch.device, optional (default=torch.device("cpu"))
        Device on which to store all tensors (priors, images, labels, attributes).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        C: int,
        p_y_a: Tensor | list[list[float]],
        seed: int | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self._base = base_dataset
        self._C = C

        if device.type == "mps":
            self._device = torch.device("cpu")
        else:
            self._device = device

        self._extract_labels()
        self._prepare_priors(p_y_a)

        self._K = self.p_a.size(0)
        self.p_a_y = self._compute_bayes_theorem(self.p_y_a.T, self.p_a[:, None], self.p_y[None, :])

        if seed is not None:
            torch.manual_seed(seed)

        self._assign_attributes()
        self._load_and_stack_images()
        self._apply_bias_to_images()

        if device.type == "mps":
            self._device = device

            self.x = self.x.to(self._device)
            self.a = self.a.to(self._device)
            self.y = self.y.to(self._device)

        self.dataset = (self.x, self.a, self.y)

    @abstractmethod
    def bias_fn(self, img: Tensor, a: int) -> Tensor:
        return torch.zeros_like(img)

    def _prepare_priors(self, p_y_a: Tensor | list[list[float]]):
        """
        Validate and store prior P(A) and conditional P(Y|A).

        Parameters
        ----------
        p_y_a : array-like, shape (C, K)
        p_a   : array-like, shape (K,)
        """
        self.p_y_a = torch.as_tensor(p_y_a, dtype=torch.float32, device=self._device)  # (K, C)

        counts = torch.bincount(self.y, minlength=self._C).float()
        self.p_y = counts / self._n  # shape (C,)

        assert torch.all(self.p_y_a >= 0), "All probabilities in P(Y | A) must be non-negative."
        self.p_y_a /= self.p_y_a.sum(dim=0, keepdim=True)  # Normalize columns

        self.p_a = torch.linalg.lstsq(self.p_y_a, self.p_y).solution
        assert (self.p_a >= 0).all(), "P(Y) is not in the convex-hull of P(Y | A)."

        self.p_a /= self.p_a.sum()

    def _extract_labels(self):
        """
        Extract labels Y from base_dataset into a tensor self.y.

        Raises
        ------
        AssertionError if labels are out of range [0, C-1].
        """
        self._n = len(cast(Sized, self._base))
        if hasattr(self._base, "targets"):
            y = torch.as_tensor(cast(Any, self._base).targets, dtype=torch.long, device=self._device)
        elif hasattr(self._base, "labels"):
            y = torch.as_tensor(cast(Any, self._base).labels, dtype=torch.long, device=self._device)
        else:
            y = torch.tensor([self._base[i][1] for i in range(self._n)], dtype=torch.long, device=self._device)
        assert y.min() >= 0 and y.max() < self._C, "labels out of range"
        self.y = y

    def _compute_bayes_theorem(self, likelihood: Tensor | float, prior: Tensor | float, evidence: Tensor | float) -> Tensor:
        """
        Computes the posterior probability P(A|Y) using Bayes' theorem.

        Args:
            likelihood (Tensor or float): P(Y|A).
            prior (Tensor or float): P(A).
            evidence (Tensor or float): P(Y).

        Returns:
            Tensor or float: Posterior probability P(A|Y).
        """
        return torch.as_tensor((likelihood * prior) / evidence)

    def _assign_attributes(self):
        """
        Vectorized draw of A_i ~ P(A | Y=y_i) for each sample i.
        Stores result in self.a of shape (N,).
        """
        probs = self.p_a_y[:, self.y].T  # (N, K)
        self.a = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(1)

    def _load_and_stack_images(self):
        """
        Read all raw images into a tensor self.x of shape (N, C_img, H, W),
        moved to self.device.
        """
        imgs = [self._base[i][0] for i in range(self._n)]
        self.x = torch.stack(imgs, dim=0).to(self._device)

    def _apply_bias_to_images(self):
        """
        Pre-apply bias_fn to all images based on sampled attributes,
        storing the result back in self.x.
        """
        x_biased = self.x.clone()
        for j in range(self._K):
            idx_j = (self.a == j).nonzero(as_tuple=True)[0]
            if idx_j.numel() == 0:
                continue
            try:
                x_biased[idx_j] = self.bias_fn(self.x[idx_j], j)
            except Exception:
                for i in idx_j:
                    x_biased[i] = self.bias_fn(self.x[i], j)
        self.x = x_biased

    def __getitem__(self, idx: int | list[int] | Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Retrieve the biased sample.

        Parameters
        ----------
        idx : int
            Index of the desired sample.

        Returns
        -------
        tuple (x_biased, a, y)
            x_biased : Tensor on self.device
            a        : LongTensor on self.device
            y        : LongTensor on self.device
        """
        return self.x[idx], self.a[idx], self.y[idx]

    def __len__(self):
        """
        Total number of samples.

        Returns
        -------
        int
        """
        return self._n
