import os
import torch
from torchvision import datasets, transforms

def download_mnist(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.ToTensor()
    
    train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    torch.save(train_data, os.path.join(data_dir, "mnist_train.pt"))
    torch.save(test_data, os.path.join(data_dir, "mnist_test.pt"))

if __name__ == "__main__":
    download_mnist()