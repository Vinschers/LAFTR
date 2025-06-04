# LAFTR

A fairness-aware representation learning project using modified MNIST datasets.

## Project Structure

```
LAFTR/
├── data/                         # Data-related modules and resources
│   ├── mnist_data/               # Downloaded MNIST dataset
│   └── BinaryColoredMNIST.py     # Custom dataset transformations (BinaryColoredMNIST)
│
├── losses/                       # Custom loss functions
│
├── models/                       # Model architectures and utilities
│
├── papers/                       # Papers related to this project
│
├── LAFTR.ipynb                   # Main Jupyter notebook for training/testing
├── .gitignore                    # Git ignore rules
└── LICENSE                       # License for the project
└── train.py                      # Training functions for the DP and EO models
```
