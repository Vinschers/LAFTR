import torch
from torch import Tensor
from torchvision import datasets, transforms
from torch.utils.data import Dataset


def _compute_bayes_theorem(likelihood: Tensor | float, prior: Tensor | float, evidence: Tensor | float) -> Tensor | float:
    """
    Computes the posterior probability P(A|Y) using Bayes' theorem.

    Args:
        likelihood: P(Y|A) — probability of Y given A.
        prior: P(A) — prior probability of A.
        evidence: P(Y) — marginal probability of Y.

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

    def __init__(self, root, p_even_red, p_even_blue, prob_a=0.5, train=True, download=True, device: torch.device = torch.device("cpu")):
        """
        Initialize the MNIST dataset with colored backgrounds according to specified probabilities.

        Parameters
        ----------
        root : str
            Path to the dataset root directory.
        train : bool
            If True, load the training split; otherwise, load the test split.
        p_even_red : float
            Probability that an even digit is colored red.
        p_even_blue : float
            Probability that an even digit is colored blue.
        prob_a : float
            Prior probability of the sensitive attribute A.
        download : bool
            Whether to download the dataset if not found locally.
        device : torch.device
            Device on which to store tensors.
        """

        self.p_even_red = p_even_red
        self.p_even_blue = p_even_blue
        self.p_a = prob_a

        self.device = device

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda img: torch.cat([img, img, img], dim=0)),
            ]
        )

        self.raw_dataset = datasets.MNIST(root=root, train=train, transform=transform, download=download)

        self.x = torch.stack([self.raw_dataset[i][0] for i in range(len(self.raw_dataset))], dim=0).to(self.device)
        self.a = torch.zeros(self.x.size(0), dtype=torch.long, device=self.device)
        self.y = (self.raw_dataset.targets % 2).to(self.device)

        self.n = self.raw_dataset.data.size(0)
        self.m = self.raw_dataset.data.size(1)

        self._find_probs()
        self._transform_data()

        self.dataset = self.x, self.a, self.y

    def _find_probs(self):
        """
        Compute class-conditional and marginal probabilities for coloring.

        This method calculates:
        - p_odd_red and p_odd_blue as complements of p_even_red and p_even_blue.
        - p_even and p_odd as the empirical probabilities of even/odd classes.
        - p_red_even, p_red_odd, p_blue_even, and p_blue_odd using Bayes' theorem,
          based on p_even_red, p_even_blue, prior p_a, and class marginals.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Stores computed probabilities in instance attributes:
            p_odd_red, p_odd_blue, p_even, p_odd, p_red_even, p_red_odd, p_blue_even, p_blue_odd.
        """
        self.p_odd_red = 1 - self.p_even_red
        self.p_odd_blue = 1 - self.p_even_blue

        self.p_even = (self.y == 0).sum().item() / self.n
        self.p_odd = 1 - self.p_even

        self.p_red_even = _compute_bayes_theorem(self.p_even_red, self.p_a, self.p_even)
        self.p_red_odd = _compute_bayes_theorem(self.p_odd_red, self.p_a, self.p_odd)
        self.p_blue_even = _compute_bayes_theorem(self.p_even_blue, self.p_a, self.p_even)
        self.p_blue_odd = _compute_bayes_theorem(self.p_odd_blue, self.p_a, self.p_odd)

    def _transform_data(self):
        """
        Assign background colors to each image according to computed probabilities.

        This method:
        - Identifies indices for even and odd labels.
        - Determines how many even digits should be red versus blue, and similarly for odd digits.
        - Randomly selects which samples of each class receive a red background versus a blue background.
        - Applies the corresponding background color using `_apply_background_color`.
        - Sets the sensitive attribute `a` to 1 for blue-background samples.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Modifies `self.x` in-place by coloring images and updates `self.a` accordingly.
        """
        idx_even = (self.y == 0).nonzero().squeeze()
        idx_odd = (self.y == 1).nonzero().squeeze()

        n_even = idx_even.size(0)
        n_odd = idx_odd.size(0)

        n_red_even = int(self.p_red_even * n_even)
        n_red_odd = int(self.p_red_odd * n_odd)

        perm_even = torch.randperm(n_even)
        idx_red_even = idx_even[perm_even[:n_red_even]]
        idx_blue_even = idx_even[perm_even[n_red_even:]]

        perm_odd = torch.randperm(n_odd)
        idx_red_odd = idx_odd[perm_odd[:n_red_odd]]
        idx_blue_odd = idx_odd[perm_odd[n_red_odd:]]

        idx_red = torch.cat([idx_red_even, idx_red_odd], dim=0)
        idx_blue = torch.cat([idx_blue_even, idx_blue_odd], dim=0)

        self.x[idx_red] = self._apply_background_color(self.x[idx_red], self.COLOR_RED)
        self.x[idx_blue] = self._apply_background_color(self.x[idx_blue], self.COLOR_BLUE)

        self.a[idx_blue] = 1

    def _apply_background_color(self, img: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
        """
        Applies the given color to the background (black pixels) of an RGB image.

        Parameters
        ----------
        img : torch.Tensor
            An RGB image tensor.
        color : torch.Tensor
            A 3-element tensor representing an RGB color.

        Returns
        -------
        torch.Tensor
            Image tensor with background pixels replaced by the color.
        """
        single_img = img.dim() == 3

        color = color.view(1, 3, 1, 1)
        if single_img:
            img = img.unsqueeze(0)

        background_mask = (img == 0).all(dim=1, keepdim=True).to(self.device)

        img = torch.where(background_mask, color.to(self.device), img)

        if single_img:
            img = img.squeeze(0)

        return img

    def __getitem__(self, idx):
        """
        Retrieves the dataset sample at the given index.

        Parameters
        ----------
        idx : int or list
            Index of the sample.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            RGB image, sensitive attribute A, and label Y.
        """

        return self.x[idx], self.a[idx], self.y[idx]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
        int
            Dataset length.
        """

        return self.n
