import torch
import random

from torchvision import datasets, transforms
from torch.utils.data import Dataset


def _compute_bayes_theorem(likelihood, prior, evidence):
    """
    Computes the posterior probability P(A|Y) using Bayes' theorem.

    Args:
        likelihood (float): P(Y|A) — probability of Y given A.
        prior (float): P(A) — prior probability of A.
        evidence (float): P(Y) — marginal probability of Y.

    Returns:
        float: Posterior probability P(A|Y).
    """

    return (likelihood * prior) / evidence


class BinaryColoredMNIST(Dataset):
    """
    Custom dataset derived from MNIST with colored backgrounds based on digit parity (even/odd).
    Assigns a sensitive attribute (A) based on background color: red (0) or blue (1).
    """

    COLOR_RED = torch.tensor([1.0, 0.0, 0.0])
    COLOR_BLUE = torch.tensor([0.0, 0.0, 1.0])

    def __init__(
        self, root, p_odd_red, p_even_red, prob_a=0.5, train=True, download=True
    ):
        """
        Initializes the BinaryColoredMNIST dataset.

        Args:
            root (str): Root directory for dataset.
            p_odd_red (float): P(Y=odd | A=red) — likelihood of an odd digit given red background.
            p_even_red (float): P(Y=even | A=red) — likelihood of an even digit given red background.
            prob_a (float): Prior probability P(A=red).
            train (bool): If True, use training data; otherwise test data.
            download (bool): Whether to download the dataset if not found locally.
        """

        self.raw_dataset = datasets.MNIST(
            root=root, train=train, transform=transforms.ToTensor(), download=download
        )
        self.p_odd_red = p_odd_red
        self.p_even_red = p_even_red
        self.prob_a = prob_a
        self.data = self._process()

    def _to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        """
        Converts a grayscale image (1, H, W) to RGB (3, H, W) format.

        Args:
            image (torch.Tensor): Grayscale image tensor.

        Returns:
            torch.Tensor: RGB image tensor.
        """

        return torch.cat([image, image, image], dim=0)

    def _apply_background_color(
        self, image_rgb: torch.Tensor, color: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the given color to the background (black pixels) of an RGB image.

        Args:
            image_rgb (torch.Tensor): An RGB image tensor.
            color (torch.Tensor): A 3-element tensor representing an RGB color.

        Returns:
            torch.Tensor: Image tensor with background pixels replaced by the color.
        """

        background_mask = (image_rgb == 0).all(dim=0)

        for c in range(3):
            image_rgb[c][background_mask] = color[c]
        return image_rgb

    def _parity_label(self):
        """
        Converts original digit labels into binary parity labels:
        - 0 for even digits
        - 1 for odd digits

        Returns:
            List[Tuple[Tensor, int]]: List of (image, parity_label) tuples.
        """

        return [(img, label % 2) for img, label in self.raw_dataset]

    def _process(self):
        """
        Processes the MNIST dataset by:
        - Assigning binary parity labels to digits (even=0, odd=1).
        - Computing P(A=red | Y=even) and P(A=red | Y=odd) using Bayes' theorem.
        - Selecting samples to color red based on these probabilities.
        - Coloring remaining samples blue.
        - Assigning a sensitive attribute A (0 for red, 1 for blue).

        Returns:
            List[Tuple[Tensor, int, int]]: List of (RGB image, sensitive attribute A, label Y).
        """

        new_data = []

        labeled_data = self._parity_label()

        # Maps the indices for each parity (odd or even)
        even_indices = [i for i, (_, label) in enumerate(labeled_data) if label == 0]
        odd_indices = [i for i, (_, label) in enumerate(labeled_data) if label == 1]

        # Computes the probability of each parity on the MNIST data P(Y)
        prob_even = len(even_indices) / len(labeled_data)
        prob_odd = 1 - prob_even

        # Bayes theorem to recover the P(A|Y) to generate the colored data
        posterior_even_red = _compute_bayes_theorem(
            likelihood=self.p_even_red, prior=self.prob_a, evidence=prob_even
        )
        posterior_odd_red = _compute_bayes_theorem(
            likelihood=self.p_odd_red, prior=self.prob_a, evidence=prob_odd
        )

        # Selects the samples of each parity that will turn RED
        selected_even_to_red = set(
            random.sample(even_indices, int(posterior_even_red * len(even_indices)))
        )
        selected_odd_to_red = set(
            random.sample(odd_indices, int(posterior_odd_red * len(odd_indices)))
        )

        # Logic to color the digits and create the new data as a truple [img, A, Y]
        # sensitive attribute = {red: 0, blue: 1}
        for i, (img, label) in enumerate(labeled_data):
            rgb_img = self._to_rgb(img)

            is_red = (
                i in selected_odd_to_red if label == 1 else i in selected_even_to_red
            )
            color = self.COLOR_RED if is_red else self.COLOR_BLUE
            sensitive = 0 if is_red else 1
            rgb_img = self._apply_background_color(rgb_img, color)

            new_data.append((rgb_img, torch.tensor(sensitive), torch.tensor(label)))

        return new_data

    def __getitem__(self, idx):
        """
        Retrieves the dataset sample at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Tensor, int, int]: (RGB image, sensitive attribute A, label Y)
        """

        return self.data[idx]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Dataset length.
        """

        return len(self.data)
