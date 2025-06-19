import numpy as np
import matplotlib.pyplot as plt


# def run_experiments(dataset = 'mnist', bias, list_gammas, mode)
    

def plot_results(results_matrix, bias, list_gammas):
    """
    Plots classifier and adversary accuracy curves from a results matrix.

    Parameters:
    - results_matrix: numpy array of shape (2, 3, len(list_gammas), 2)
    - bias: the bias value used in the experiments (for title labeling)
    - list_gammas: list or array of gamma values used in experiments
    """
    fairness_modes = ['DP', 'EO']
    test_types = [f'Bias = {bias}', 'Bias = 0', 'Inversed Bias']

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)

    for i in range(3):  # test types
        for j in range(2):  # fairness modes
            ax = axes[i, j]
            classifier_acc = results_matrix[j, i, :, 0]
            adversary_acc = results_matrix[j, i, :, 1]
            ax.plot(list_gammas, classifier_acc, label='Classifier', color='darkblue')
            ax.plot(list_gammas, adversary_acc, label='Adversary', color='darkred')
            ax.set_title(f"{test_types[i]} - {fairness_modes[j]}")
            ax.set_xticks(list_gammas)
            if i == 2:
                ax.set_xlabel("Gamma")
            if j == 0:
                ax.set_ylabel("Accuracy")
            ax.legend()

    fig.suptitle(f"Experiment Results (Bias = {bias})", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()