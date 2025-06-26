# LAFTR

A fairness-aware representation learning project using modified MNIST datasets.

## Project Structure

```
LAFTR/
├── data/                          # Data modules and resources
│   ├── BiasedBinaryMNIST.py       # Binary background for MNIST
│   ├── BiasedCifar10.py           # CIFAR10 dataset with color bias
│   ├── BiasedDataset.py           # Base dataset class for biasing
│   ├── BinaryColoredMNIST.py      # Colored MNIST (DP/EO setups)
│   └── __init__.py                # Module init
├── losses/                        # Custom loss functions (DP, EO)
│   ├── AdversaryLoss.py
│   ├── CombinedLoss.py
│   └── __init__.py
├── models/                        # Neural model architectures
│   ├── Adversary.py               # Adversary network
│   ├── Classifier.py              # Simple classifier
│   ├── ConvEncoderCIFAR.py        # Conv encoder for CIFAR10
│   ├── ConvEncoderMNIST.py        # Conv encoder for MNIST
│   ├── MLPEncoder.py              # MLP encoder
│   ├── _BaseEncoder.py            # Abstract base encoder
│   └── __init__.py
├── predict/                       # Prediction utilities
│   ├── Predictor.py               # Inference and reporting wrapper
│   └── __init__.py
├── train/                         # Training loops
│   ├── Trainer.py                 # Unified training class
│   └── __init__.py
├── notebooks/                     # Main experiments
│   ├── LAFTR.ipynb
│   ├── cifar10.ipynb
│   └── training_original.ipynb
├── papers/                        # Reference papers
│   ├── Learning Adversarially Fair and Transferable Representations.pdf
│   └── On Fairness and Calibration.pdf
├── LICENSE
├── .gitignore
├── experiments.py                 # Entrypoint for experiments
├── structure.txt                  # ASCII project layout
├── class_diagram.sh               # Script for generating UML
├── classes_LAFTR.png              # Class diagram image
├── packages_LAFTR.png             # Package diagram
└── README.md
```
