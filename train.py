import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer, Adam

from models import Encoder, Classifier, Adversary
from losses import CombinedLoss, AdversaryLossDP


def _train_epoch_dp(
    encoder: Encoder,
    classifier: Classifier,
    adversary: Adversary,
    train_loader: DataLoader,
    criterion_enc_class: Module,
    optimizer_enc_class: Optimizer,
    criterion_adv: Module,
    optimizer_adv: Optimizer,
    device: torch.device = torch.device("cpu"),
):
    encoder.train()
    classifier.train()
    adversary.train()

    loss_enc_class = 0
    loss_adv = 0
    n = 0

    for x, a_true, y_true in train_loader:
        x = x.to(device)
        a_true = a_true.to(device)
        y_true = y_true.to(device)
        batch_size = x.size(0)

        optimizer_enc_class.zero_grad()
        optimizer_adv.zero_grad()

        # Train encoder and classifier with fixed adversary
        z = encoder(x)
        y_pred = classifier(z)

        loss = criterion_enc_class(y_pred, y_true)
        loss.backward()
        optimizer_enc_class.step()

        loss_enc_class += loss.item() * batch_size

        # Train adversary with fixed encoder and classifier
        a_pred = adversary(z.detach())

        loss = criterion_adv(a_pred, a_true)
        loss.backward()
        optimizer_adv.step()

        loss_adv += loss.item() * batch_size

        n += batch_size

    return loss_enc_class / n, loss_adv / n


def train_dp(
    encoder: Encoder,
    classifier: Classifier,
    adversary: Adversary,
    train_loader: DataLoader,
    gamma: float = 1.0,
    epochs: int = 12,
    device: torch.device = torch.device("cpu"),
):
    criterion_enc_class = CombinedLoss(gamma)
    criterion_adv = AdversaryLossDP()

    optimizer_enc_class = Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-3)
    optimizer_adv = Adam(adversary.parameters(), lr=1e-3)

    losses_enc_class = []
    losses_adv = []

    for _ in range(epochs):
        loss_enc_class, loss_adv = _train_epoch_dp(
            encoder, classifier, adversary, train_loader, criterion_enc_class, optimizer_enc_class, criterion_adv, optimizer_adv, device
        )

        losses_enc_class.append(loss_enc_class)
        losses_adv.append(loss_adv)

    return losses_enc_class, losses_adv
