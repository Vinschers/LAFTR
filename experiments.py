import os

from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import ModuleList, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam

from data import BiasedCifar10, BiasedBinaryMNIST

from predict import Predictor
from data import BiasedCifar10, BiasedBinaryMNIST
from models import MLPEncoder, ConvEncoder, ConvEncoderCIFAR, Classifier, Adversary
from train import Trainer


def run_experiments(C: int, K: int, bias=0.0, encoder_type="MLP", dataset_name="MNIST", gammas=[], device: torch.device = torch.device("cpu")):

    # Creates Biased Data
    train_set, test_set_same_bias, test_set_no_bias, test_set_modified_bias = (
        create_datasets(C, K, bias=bias, dataset_name=dataset_name, device=device)
    )

    # set training for DP
    latent_dim = 24
    batch_size = 1 << 12
    learning_rate = 1e-3
    epochs = 10

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    encoder = set_encoder(dataset_name, encoder_type, latent_dim).to(device)
    classifier = Classifier(latent_dim, C=C).to(device)
    adversary_dp = Adversary(latent_dim, K=K).to(device)
    advs = [Adversary(latent_dim, K=K).to(device) for _ in range(C)]
    adversary_eo = ModuleList(advs)

    trainer_dp = Trainer(train_loader, encoder, classifier, adversary_dp, C, K, device)
    trainer_eo = Trainer(train_loader, encoder, classifier, adversary_eo, C, K, device)

    criterion_class = CrossEntropyLoss()
    optimizer_enc_class = Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate, weight_decay=5e-4)
    optimizer_adv_dp = Adam(adversary_dp.parameters(), lr=learning_rate, weight_decay=5e-4)
    optimizer_adv_eo = [Adam(adv.parameters(), lr=learning_rate, weight_decay=5e-4) for adv in advs]

    num_gammas = len(gammas)
    results_matrix = np.zeros((2, 3, num_gammas, 2))

    for idx, gamma in enumerate(gammas):
        # DP
        _, _ = trainer_dp.train(criterion_class, optimizer_enc_class, optimizer_adv_dp, gamma, epochs, verbose=False)
        result_dp = run_scenarios(adversary_dp, classifier, encoder, K, C, test_set_same_bias, test_set_no_bias, test_set_modified_bias)
        results_matrix[0, :, idx, :] = result_dp[:, 0, :]
        print(f"Trained LAFTR on DP : {idx+1}/{len(gammas)}.")

        # EO
        _, _ = trainer_eo.train(criterion_class, optimizer_enc_class, optimizer_adv_eo, gamma, epochs, verbose=False)
        result_eo = run_scenarios(adversary_eo, classifier, encoder, K, C, test_set_same_bias, test_set_no_bias, test_set_modified_bias)
        results_matrix[1, :, idx, :] = result_eo[:, 0, :]
        print(f"Trained LAFTR on EO : {idx+1}/{len(gammas)}.")

    # plot_results(results_matrix, bias, gammas)

    return results_matrix


def run_scenarios(adversary, classifier, encoder, K, C, test_set_same_bias, test_set_no_bias, test_set_modified_bias):

    y_true_sb, y_pred_sb, a_true_sb, a_pred_sb = run_tests(adversary, classifier, encoder, K, C, test_set_same_bias)
    y_true_nb, y_pred_nb, a_true_nb, a_pred_nb = run_tests(adversary, classifier, encoder, K, C, test_set_no_bias)
    y_true_mb, y_pred_mb, a_true_mb, a_pred_mb = run_tests(adversary, classifier, encoder, K, C, test_set_modified_bias)

    classifier_acc_sb = compute_accuracy(y_true_sb, y_pred_sb)
    adv_acc_sb = compute_accuracy(a_true_sb, a_pred_sb)

    classifier_acc_nb = compute_accuracy(y_true_nb, y_pred_nb)
    adv_acc_nb = compute_accuracy(a_true_nb, a_pred_nb)

    classifier_acc_mb = compute_accuracy(y_true_mb, y_pred_mb)
    adv_acc_mb = compute_accuracy(a_true_mb, a_pred_mb)

    matrix = np.array([[[classifier_acc_sb, adv_acc_sb]], [[classifier_acc_nb, adv_acc_nb]], [[classifier_acc_mb, adv_acc_mb]]])

    return matrix


def create_datasets(
    C: int,
    K: int,
    bias: int = 0.0,
    dataset_name="MNIST",
    device: torch.device = torch.device("cpu"),
):
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    data_dir = os.path.join(project_root, "data/datasets")

    p_y_a = generate_p_y_a(bias, C, K, device)
    p_y_a_no_bias = generate_p_y_a(0, C, K, device)
    p_y_a_modified = torch.roll(p_y_a, shifts=1, dims=1)

    if dataset_name.upper() == "MNIST":
        train_set = BiasedBinaryMNIST(data_dir, p_y_a, train=True, device=device)
        test_set_same_bias = BiasedBinaryMNIST(data_dir, p_y_a, train=False, device=device)
        test_set_no_bias = BiasedBinaryMNIST(data_dir, p_y_a_no_bias, train=False, device=device)
        test_set_modified_bias = BiasedBinaryMNIST(data_dir, p_y_a_modified, train=False, device=device)

    elif dataset_name.upper() == "CIFAR10":
        train_set = BiasedCifar10(data_dir, p_y_a, train=True, device=device)
        test_set_same_bias = BiasedCifar10(data_dir, p_y_a, train=False, device=device)
        test_set_no_bias = BiasedCifar10(data_dir, p_y_a_no_bias, train=False, device=device)
        test_set_modified_bias = BiasedCifar10(data_dir, p_y_a_modified, train=False, device=device)

    else:
        print("Invalid dataset_name at create_dataset.")

    return train_set, test_set_same_bias, test_set_no_bias, test_set_modified_bias


def run_tests(adversary, classifier, encoder, K, C, test_set):
    device = next(encoder.parameters()).device

    predict_adversary = Predictor(model=adversary, N=K, device=device)
    predict_classifier = Predictor(model=classifier, N=C)

    x, a_true, y_true = test_set[:]

    x = x.to(device)
    y_true = y_true.to(device)

    z = encoder(x)

    a_pred = predict_adversary.predict_class(z, y_true)
    y_pred = predict_classifier.predict_class(z)

    return y_true, y_pred, a_true, a_pred


def plot_results(results_matrix, list_gammas, K, baseline_acc):
    """
    Simple 2×2 grid: Encoders vs Adversaries × DP vs EO.
    Drops γ=0 (log scale can’t show zero), plots plain log‐x, a baseline,
    and one legend per subplot.
    """
    # — drop zero gamma (so log‐scale works) —
    gammas = np.array(list_gammas)
    mask = gammas > 0
    gammas = gammas[mask]
    data = results_matrix[:, :, mask, :]  # shape still (2,3,len(gammas),2)

    fairness = ["DP", "EO"]
    tests = ["Bias", "Bias = 0", "Inversed Bias"]
    models = ["Encoders", "Adversaries"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Loop over columns (DP/EO) and rows (Encoder/Adv)
    for col in range(2):
        for row in range(2):
            ax = axes[row, col]
            for t_idx, label in enumerate(tests):
                y = data[col, t_idx, :, row]  # row=0→encoder,1→adv
                ax.plot(gammas, y, label=label)
            if row == 0:
                # only on encoder row
                ax.axhline(baseline_acc, ls="--", color="gray", label="Baseline")
            else:
                ax.axhline(1 / K, ls="--", color="gray", label="Baseline")
            ax.set_xscale("log")
            ax.set_title(f"{models[row]} – {fairness[col]}")
            ax.set_xlabel("Gamma")
            ax.set_ylabel("Accuracy")
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    plt.show()


def set_encoder(dataset_name, encoder_type, latent_dim):
    if dataset_name.upper() == "MNIST":
        if encoder_type.upper() == "MLP":
            encoder = MLPEncoder(latent_dim)
        elif encoder_type.upper() == "CONV":
            encoder = ConvEncoder(latent_dim)
        else:
            print("Invalid encoder_type at set_encoder.")
            return
    elif dataset_name.upper() == "CIFAR10":
        encoder = ConvEncoderCIFAR(latent_dim)
    else:
        print("Invalid dataset_name at set_encoder.")
        return

    return encoder


def generate_p_y_a(bias, C, K, device):
    if bias >= 1.0:
        raise ValueError("bias has to be in the interval [0, 1)")

    p_y_a = torch.zeros(C, K)

    dominant_classes = torch.randperm(C)[:K]

    for j in range(K):

        uniform = torch.full((C,), 1.0 / C, device=device)

        one_hot = torch.zeros(C, device=device)
        one_hot[dominant_classes[j].item()] = 1.0

        p_y_a[:, j] = (1 - bias) * uniform + bias * one_hot

    return p_y_a


def compute_accuracy(true, pred):
    """
    Computes accuracy given true and predicted labels.

    Parameters:
    - true: array-like of true labels
    - pred: array-like of predicted labels

    Returns:
    - accuracy: float between 0 and 1
    """
    true = true.cpu().numpy()
    pred = pred.cpu().numpy()
    correct = np.sum(true == pred)
    accuracy = correct / len(true)

    return accuracy
