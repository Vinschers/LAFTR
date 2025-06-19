import os

import torch

from data import BiasedCifar10, BiasedBinaryMNIST

def run_experiments():
    pass

def create_datasets(C: int, K: int, bias: int=0., dataset_name='MNIST', device: torch.device = torch.device("cpu")):
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    data_dir = os.path.join(project_root, "data/datasets")

    p_y_a = generate_p_y_a(bias, C, K, device)
    p_y_a_no_bias = generate_p_y_a(0, C, K, device)
    p_y_a_modified = torch.roll(p_y_a, shifts=1, dims=1)

    if dataset_name.upper() == 'MNIST':
        train_set = BiasedBinaryMNIST(data_dir, p_y_a, train=True, device=device)
        test_set_same_bias = BiasedBinaryMNIST(data_dir, p_y_a, train=False, device=device)
        test_set_no_bias = BiasedBinaryMNIST(data_dir, p_y_a_no_bias, train=False, device=device)
        test_set_modified_bias = BiasedBinaryMNIST(data_dir, p_y_a_modified, train=False, device=device)

    elif dataset_name.upper() == 'CIFAR10':
        train_set = BiasedCifar10(data_dir, p_y_a, train=True, device=device)
        test_set_same_bias = BiasedCifar10(data_dir, p_y_a, train=False, device=device)
        test_set_no_bias = BiasedCifar10(data_dir, p_y_a_no_bias, train=False, device=device)
        test_set_modified_bias = BiasedCifar10(data_dir, p_y_a_modified, train=False, device=device)

    else: 
        print("Invalid dataset_name.")

    return train_set, test_set_same_bias, test_set_no_bias, test_set_modified_bias


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