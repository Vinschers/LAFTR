import torch
from torch import Tensor
from torch.nn import Module, ModuleList, functional as F

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Predictor:
    def __init__(
        self,
        model: Module | ModuleList,
        N: int = 2,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self._multi_model = isinstance(
            self.model, ModuleList
        )  # check if multi-model setup
        self._N = N  # number of classes the model is predicting
        self._device = device

    def __call__(self, z: Tensor, y_true: Tensor | None = None) -> Tensor:
        """
        Predict logits using a single model or the y-th model if multi-adversary.
        """

        batch_size = z.size(0)
        logits = torch.zeros(batch_size, self._N, device=self._device)

        if self._multi_model:

            if y_true is None:
                raise ValueError("y_true is required for multi-adversary setup.")

            for y in range(len(self.model)):
                mask_y = y_true == y

                if mask_y.any():
                    idx = mask_y.nonzero(as_tuple=True)[0]

                    z_y = z[mask_y]

                    logits[idx] = self.model[y](z_y)

        else:
            logits = self.model(z)

        return self._to_pred_matrix(logits)

    def predict_class(self, z: Tensor, y_true: Tensor | None = None):
        """Return one-hot predictions (0/1) from logits"""

        logits = self.__call__(z, y_true)

        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)  # returns the index of the max prob

        return preds

    def print_classification_report(self, true_labels: Tensor, pred_labels: Tensor):
        true_labels = true_labels.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()

        print(classification_report(true_labels, pred_labels, zero_division=0))

    def plot_confusion_matrix(
        self,
        true_labels: Tensor,
        pred_labels: Tensor,
        normalize=False,
        labels=None,
        labels_names=None,
        figsize=(5, 4),
        ylabel="True Label",
        xlabel="Predicted Label",
        save_path=None,
    ):
        true_labels = true_labels.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()

        cm = confusion_matrix(true_labels, pred_labels, labels=labels)
        classes = (
            labels_names
            or labels
            or np.unique(np.concatenate((true_labels, pred_labels)))
        )

        if normalize:
            with np.errstate(divide="ignore", invalid="ignore"):
                cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
                cm = np.nan_to_num(cm)  #  replace NaNs with 0
            fmt = ".2f"
            vmin, vmax = 0.0, 1.0

        else:
            fmt = "d"
            vmin, vmax = None, None

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            linewidths=0.5,
            linecolor="gray",
            vmin=vmin,
            vmax=vmax,
        )

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()

        return cm

    def _to_pred_matrix(self, pred: Tensor):
        if pred.dim() == 2:
            return pred

        return torch.stack((1 - pred, pred), dim=1)
