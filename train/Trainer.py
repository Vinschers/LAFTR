from typing import Sequence

import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm

from losses import CombinedLoss, AdversaryLoss
from predict import Predictor


class Trainer:
    """
    Train and evaluate a LAFTR-style model with optional demographic-parity or per-class adversaries.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader yielding training batches of (x, a_true, y_true).
    encoder : Module
        Neural network mapping x -> z (representation).
    classifier : Module
        Neural network mapping z -> y_pred.
    adversary : Module or ModuleList
        If a single Module, use a demographic-parity adversary applied to all examples;
        if a ModuleList of length C, use one adversary network per class label y_true.
    C : int, default=2
        Number of class labels (used when adversary is a ModuleList).
    K : int, default=2
        Number of sensitive-attribute categories (output dimension of each adversary).
    device : torch.device, default=torch.device("cpu")
        Device on which to run the encoder, classifier, and adversary networks.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        encoder: Module,
        classifier: Module,
        adversary: Module | ModuleList,
        C: int = 2,
        K: int = 2,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the Trainer with data loaders, model components, and configuration.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader yielding training batches of (x, a_true, y_true).
        encoder : Module
            Neural network mapping x -> z (representation).
        classifier : Module
            Neural network mapping z -> y_pred.
        adversary : Module or ModuleList
            If a single Module, use a demographic-parity adversary; if ModuleList, one adversary per class.
        C : int, default=2
            Number of class labels (used when adversary is a ModuleList).
        K : int, default=2
            Number of sensitive-attribute categories (output dimension of each adversary).
        device : torch.device, default=torch.device("cpu")
            Device on which to run the encoder, classifier, and adversary networks.
        """
        self._C = C
        self._K = K
        self._device = device

        self._dp = not isinstance(adversary, ModuleList)

        self._train_loader = train_loader

        self.encoder = encoder.to(self._device)
        self.classifier = classifier.to(self._device)
        self.adversary = adversary.to(self._device)

        self._pred_logits_adversary = Predictor(self.adversary, self._K, self._device)

    def train(
        self,
        criterion_classifier: Module,
        optimizer_enc_class: Optimizer,
        optimizer_adv: Optimizer | Sequence[Optimizer],
        gamma: float = 1e-4,
        epochs: int = 12,
        verbose: bool = False,
    ):
        """
        Train encoder, classifier, and adversary networks over multiple epochs.

        Parameters
        ----------
        criterion_classifier : Module
            Classification loss function computing loss given (y_pred, y_true).
        optimizer_enc_class : Optimizer
            Optimizer for encoder and classifier parameters.
        optimizer_adv : Optimizer or sequence of Optimizer
            If demographic parity (single adversary), a single Optimizer; otherwise,
            a sequence of Optimizers (one per adversary in ModuleList).
        gamma : float, default=1e-4
            Weight on the adversarial fairness penalty in the encoder+classifier loss.
        epochs : int, default=12
            Number of full passes through the training data.
        verbose : bool, default=False
            If True, print progress after each epoch.

        Returns
        -------
        tuple of list of float
            (losses_enc_class, losses_adv), each of length `epochs`:
            - losses_enc_class[e]: encoder+classifier loss averaged over epoch e.
            - losses_adv[e]: adversary loss (sum over classes or DP) averaged over epoch e.
        """
        self._criterion_classifier = criterion_classifier
        self._optimizer_enc_class = optimizer_enc_class
        self._check_optimizers_adv(optimizer_adv)

        self._criterion_enc_class = CombinedLoss(gamma)
        self._criterion_adv = AdversaryLoss()

        self._losses_enc_class = []
        self._losses_adv = []

        self._reset_models()

        for e in range(epochs):
            loss_enc_class, loss_adv = self._train_epoch(e, verbose)

            self._losses_enc_class.append(loss_enc_class)
            self._losses_adv.append(loss_adv)

        self._set_mode(
            train_encoder=False, train_classifier=False, train_adversary=False
        )
        return self._losses_enc_class, self._losses_adv

    def _train_epoch(self, epoch: int = 0, verbose: bool = False):
        """
        Run one training epoch: update encoder+classifier and adversary networks on all minibatches.

        Returns
        -------
        tuple of float
            (average encoder+classifier loss over this epoch, average adversary loss over this epoch).
        """
        loss_enc_class_total = 0
        loss_adv_total = 0
        n = 0

        loader = (
            tqdm(self._train_loader, unit="batch", leave=False)
            if verbose
            else self._train_loader
        )

        for x, a_true, y_true in loader:
            batch_size = x.size(0)

            x = x.to(self._device)
            a_true = a_true.to(self._device)
            y_true = y_true.to(self._device)

            loss_enc_class = self._train_enc_class(x, a_true, y_true)
            loss_adv = self._train_adversary(x, a_true, y_true)

            loss_enc_class_total += loss_enc_class * batch_size
            loss_adv_total += loss_adv * batch_size

            n += batch_size

        loss_enc_class = loss_enc_class_total / n
        loss_adv = loss_adv_total / n

        if verbose:
            print(
                f"Epoch {epoch + 1} (encoder+classifier loss: {loss_enc_class:.4f}, adversary loss: {loss_adv:.4f})"
            )

        return loss_enc_class, loss_adv

    def _train_enc_class(self, x: Tensor, a_true: Tensor, y_true: Tensor):
        """
        Perform one optimization step for encoder and classifier with adversary fixed.

        Parameters
        ----------
        x : Tensor
            Input features of shape [batch_size, ...].
        a_true : Tensor
            True sensitive-attribute labels of shape [batch_size].
        y_true : Tensor
            True class labels of shape [batch_size], in {0,...,C-1}.

        Returns
        -------
        float
            The scalar encoder+classifier loss (classification + γ·fairness) for this batch.
        """
        self._set_mode(train_encoder=True, train_classifier=True, train_adversary=False)

        z = self.encoder(x)
        y_pred = self.classifier(z)

        a_pred = self._pred_logits_adversary(z, y_true)
        loss_adv = self._criterion_adv(a_pred, a_true)

        loss_class = self._criterion_classifier(y_pred, y_true)
        loss_enc_class = self._criterion_enc_class(loss_class, loss_adv)

        self._optimizer_enc_class.zero_grad()
        loss_enc_class.backward()
        self._optimizer_enc_class.step()

        return loss_enc_class.item()

    def _train_adversary(self, x: Tensor, a_true: Tensor, y_true: Tensor):
        """
        Perform one optimization step for adversary networks with encoder and classifier fixed.

        Parameters
        ----------
        x : Tensor
            Input features of shape [batch_size, ...].
        a_true : Tensor
            True sensitive-attribute labels of shape [batch_size].
        y_true : Tensor
            True class labels of shape [batch_size], in {0,...,C-1}.

        Returns
        -------
        float
            The sum of adversary losses (mean over respective subsets) for this batch.
        """
        self._set_mode(
            train_encoder=False, train_classifier=False, train_adversary=True
        )

        with torch.no_grad():
            z = self.encoder(x)

        adv_loss = 0

        if self._dp:
            a_pred = self.adversary(z)
            adv_loss = self._criterion_adv(a_pred, a_true)

            self.optimizer_adv.zero_grad()  # type: ignore[operator]
            adv_loss.backward()
            self.optimizer_adv.step()  # type: ignore[operator]

            adv_loss = adv_loss.item()
        else:
            for y in range(self._C):
                mask_y = y_true == y

                if mask_y.any():
                    adv = self.adversary[y]  # type: ignore[operator]
                    optim_adv = self.optimizer_adv[y]  # type: ignore[operator]

                    z_y = z[mask_y]
                    a_y = a_true[mask_y]

                    a_pred = self._to_pred_matrix(adv(z_y))
                    loss_adv_y = self._criterion_adv(a_pred, a_y)

                    optim_adv.zero_grad()
                    loss_adv_y.backward()
                    optim_adv.step()

                    adv_loss += loss_adv_y.item()

        return adv_loss

    def _check_optimizers_adv(self, optimizer_adv: Optimizer | Sequence[Optimizer]):
        """
        Validate and store the adversary optimizer(s) based on whether DP or per-class adversaries are used.

        Parameters
        ----------
        optimizer_adv : Optimizer or sequence of Optimizer
            If demographic parity (single adversary), a single Optimizer; otherwise,
            a sequence of Optimizers (one per adversary in ModuleList).

        Raises
        ------
        AssertionError
            If `optimizer_adv` does not match the expected type/length given `self.dp`.
        """
        if self._dp:
            assert isinstance(
                optimizer_adv, Optimizer
            ), "When you pass a single adversary Module, `optimizer_adv` must be a single Optimizer, not a list."
        else:
            assert isinstance(optimizer_adv, (list, tuple)) and len(
                optimizer_adv
            ) == len(
                self.adversary  # type: ignore[operator]
            ), "When you pass multiple adversaries, `optimizer_adv` must be a list/tuple of optimizers of the same length."

        self.optimizer_adv = optimizer_adv

    def _set_mode(
        self, train_encoder: bool, train_classifier: bool, train_adversary: bool
    ):
        """
        Set training/evaluation mode for encoder, classifier, and adversary networks.

        Parameters
        ----------
        train_encoder : bool
            If True, call `.train()` on the encoder; otherwise, `.eval()`.
        train_classifier : bool
            If True, call `.train()` on the classifier; otherwise, `.eval()`.
        train_adversary : bool
            If True, call `.train()` on the adversary network(s); otherwise, `.eval()`.
        """
        if train_encoder:
            self.encoder.train()
        else:
            self.encoder.eval()

        if train_classifier:
            self.classifier.train()
        else:
            self.classifier.eval()

        if train_adversary:
            self.adversary.train()
        else:
            self.adversary.eval()

    def _reset_models(self):
        """
        Reset parameters of encoder, classifier, and adversary networks (if available).

        Loops through all modules in each network and calls `reset_parameters()` if the method exists.
        """
        for m in self.encoder.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()  # type: ignore[operator]

        for m in self.classifier.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()  # type: ignore[operator]

        for m in self.adversary.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()  # type: ignore[operator]

    def _to_pred_matrix(self, pred: Tensor):
        if pred.dim() == 2:
            return pred

        return torch.stack((1 - pred, pred), dim=1)
