from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module, ModuleList
from torch.optim import Optimizer

from models import Encoder, Classifier
from losses import CombinedLoss, AdversaryLoss


def _to_pred_matrix(pred: Tensor):
    if pred.dim() == 2:
        return pred

    return torch.stack((1 - pred, pred), dim=1)


def _train_enc_class(
    x: Tensor,
    a_true: Tensor,
    y_true: Tensor,
    encoder: Encoder,
    classifier: Classifier,
    adversaries: ModuleList,
    criterion_enc_class: Module,
    optimizer_enc_class: Optimizer,
    criterion_class: Module,
    criterion_adv: Module,
):
    """
    Perform one optimization step for the encoder and classifier with fixed adversaries.

    Parameters
    ----------
    x : Tensor
        Input features of shape [batch_size, ...].
    a_true : Tensor
        True sensitive-attribute labels of shape [batch_size].
    y_true : Tensor
        True prediction labels of shape [batch_size], in {0,...,C-1}.
    encoder : Encoder
        Neural network mapping x -> z (representation).
    classifier : Classifier
        Neural network mapping z -> y_pred.
    adversaries : ModuleList
        List of C adversary networks, one per value of y_true.
    criterion_enc_class : Module
        Combined loss function that accepts (loss_class, loss_adv) and returns a scalar.
    optimizer_enc_class : Optimizer
        Optimizer for encoder and classifier parameters.
    criterion_class : Module
        Classification loss function computing loss given (y_pred, y_true).
    criterion_adv : Module
        Adversarial loss function computing fairness penalty given (a_pred, a_true).

    Returns
    -------
    float
        The scalar encoder+classifier loss (classification + γ·fairness) for this batch.
    """
    C = len(adversaries)

    # Fix adversaries
    for adv in adversaries:
        adv.eval()

    encoder.train()
    classifier.train()

    optimizer_enc_class.zero_grad()

    z = encoder(x)
    y_pred = classifier(z)

    loss_adv = 0

    for y in range(C):
        mask_y = y_true == y

        if mask_y.any():
            z_y = z[mask_y]
            a_y = a_true[mask_y]
            adv = adversaries[y]

            a_pred = _to_pred_matrix(adv(z_y))

            loss_adv += criterion_adv(a_pred, a_y)

    loss_class = criterion_class(y_pred, y_true.float())

    loss_enc_class = criterion_enc_class(loss_class, loss_adv)
    loss_enc_class.backward()
    optimizer_enc_class.step()

    return loss_enc_class.item()


def _train_adversaries(
    x: Tensor,
    a_true: Tensor,
    y_true: Tensor,
    encoder: Encoder,
    classifier: Classifier,
    adversaries: ModuleList,
    optimizers_adv: Sequence[Optimizer],
    criterion_adv: Module,
):
    """
    Perform one optimization step for each adversary with encoder and classifier fixed.

    Parameters
    ----------
    x : Tensor
        Input features of shape [batch_size, ...].
    a_true : Tensor
        True sensitive-attribute labels of shape [batch_size].
    y_true : Tensor
        True prediction labels of shape [batch_size], in {0,...,C-1}.
    encoder : Encoder
        Neural network mapping x -> z. Its parameters are frozen during this step.
    classifier : Classifier
        Neural network mapping z -> y_pred. Its parameters are frozen during this step.
    adversaries : ModuleList
        List of C adversary networks, one per value of y_true.
    optimizers_adv : Sequence[Optimizer]
        List of optimizers corresponding to each adversary.
    criterion_adv : Module
        Adversarial loss function computing fairness penalty given (a_pred, a_true).

    Returns
    -------
    float
        The sum of each adversary’s loss (mean over its subset) for this batch.
    """
    C = len(adversaries)

    # Fix encoder and classifier
    encoder.eval()
    classifier.eval()

    for adv in adversaries:
        adv.train()

    with torch.no_grad():
        z = encoder(x)

    adv_loss = 0

    for y in range(C):
        mask_y = y_true == y

        if mask_y.any():
            adv = adversaries[y]
            optim_adv = optimizers_adv[y]

            optim_adv.zero_grad()
            z_y = z[mask_y]
            a_y = a_true[mask_y]

            a_pred = _to_pred_matrix(adv(z_y))

            loss_adv_y = criterion_adv(a_pred, a_y)
            loss_adv_y.backward()
            optim_adv.step()

            adv_loss += loss_adv_y.item()

    return adv_loss


def _train_epoch(
    encoder: Encoder,
    classifier: Classifier,
    adversaries: ModuleList,
    train_loader: DataLoader,
    criterion_enc_class: Module,
    optimizer_enc_class: Optimizer,
    criterion_class: Module,
    criterion_adv: Module,
    optimizers_adv: Sequence[Optimizer],
    device: torch.device = torch.device("cpu"),
):
    """
    Run one training epoch: update encoder+classifier and adversaries over all minibatches.

    Parameters
    ----------
    encoder : Encoder
        Neural network mapping x -> z.
    classifier : Classifier
        Neural network mapping z -> y_pred.
    adversaries : ModuleList
        List of C adversary networks, one per value of y_true.
    train_loader : DataLoader
        DataLoader yielding (x, a_true, y_true) batches.
    criterion_enc_class : Module
        Combined loss function for encoder+classifier.
    optimizer_enc_class : Optimizer
        Optimizer for encoder and classifier parameters.
    criterion_class : Module
        Classification loss function computing loss given (y_pred, y_true).
    criterion_adv : Module
        Adversarial loss function computing fairness penalty given (a_pred, a_true).
    optimizers_adv : Sequence[Optimizer]
        List of optimizers, one per adversary in `adversaries`.
    device : torch.device, default=torch.device("cpu")
        Device on which to perform computations.

    Returns
    -------
    tuple of float
        (average encoder+classifier loss over epoch, average adversary loss over epoch).
    """
    loss_enc_class_total = 0
    loss_adv_total = 0
    n = 0

    for x, a_true, y_true in train_loader:
        x = x.to(device)
        a_true = a_true.to(device)
        y_true = y_true.to(device)
        batch_size = x.size(0)

        loss_enc_class = _train_enc_class(
            x, a_true, y_true, encoder, classifier, adversaries, criterion_enc_class, optimizer_enc_class, criterion_class, criterion_adv
        )
        loss_adv = _train_adversaries(x, a_true, y_true, encoder, classifier, adversaries, optimizers_adv, criterion_adv)

        loss_enc_class_total += loss_enc_class * batch_size
        loss_adv_total += loss_adv * batch_size
        n += batch_size

    return loss_enc_class_total / n, loss_adv_total / n


def train_laftr(
    encoder: Encoder,
    classifier: Classifier,
    adversaries: ModuleList,
    criterion_class: Module,
    optimizer_enc_class: Optimizer,
    optimizers_adv: Sequence[Optimizer],
    train_loader: DataLoader,
    gamma: float = 1.0,
    epochs: int = 12,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
):
    """
    Train a LAFTR model using adversarial DP/EO criteria over multiple epochs.

    Parameters
    ----------
    encoder : Encoder
        Neural network mapping x -> z (representation).
    classifier : Classifier
        Neural network mapping z -> y_pred.
    adversaries : ModuleList
        List of C adversary networks, one per value of y_true.
    criterion_class : Module
        Classification loss function computing loss given (y_pred, y_true).
    optimizer_enc_class : Optimizer
        Optimizer for encoder and classifier parameters.
    optimizers_adv : Sequence[Optimizer]
        Sequence of optimizers corresponding to each adversary in `adversaries`.
    train_loader : DataLoader
        DataLoader yielding batches of (x, a_true, y_true).
    gamma : float, default=1.0
        Weight on the adversarial fairness penalty in the encoder+classifier loss.
    epochs : int, default=12
        Number of full passes through `train_loader`.
    device : torch.device, default=torch.device("cpu")
        Device on which to perform computations.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    tuple of list of float
        (losses_enc_class, losses_adv), each a list of length `epochs` containing:
        - losses_enc_class[e]: encoder+classifier loss (classification + γ·fairness) averaged over epoch e.
        - losses_adv[e]: sum of adversaries’ losses averaged over epoch e.
    """
    criterion_enc_class = CombinedLoss(gamma)
    criterion_adv = AdversaryLoss()

    losses_enc_class = []
    losses_adv = []

    for e in range(epochs):
        loss_enc_class, loss_adv = _train_epoch(
            encoder, classifier, adversaries, train_loader, criterion_enc_class, optimizer_enc_class, criterion_class, criterion_adv, optimizers_adv, device
        )

        losses_enc_class.append(loss_enc_class)
        losses_adv.append(loss_adv)

        if verbose:
            print(f"Epoch {e} (encoder+classifier loss: {loss_enc_class:.4f}, adversary loss: {loss_adv:.4f})")

    return losses_enc_class, losses_adv
