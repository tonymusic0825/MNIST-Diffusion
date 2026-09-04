import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1407)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))
])

def download_MNIST(root="./data", download=True):

    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        transform=transform,
        download=download
    )

    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        transform=transform,
        download=download
    )

    return train_dataset, test_dataset

def create_dataloader(shuffle=True, pin_memory=True, num_workers=0, test=True, batch_size=32):

    train_dataset, test_dataset = download_MNIST()

    if test:
        print(train_dataset[0][0])
        plt.imshow(train_dataset[0][0].squeeze())
        plt.show()

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=True,
    )

    return train_dataloader, test_dataloader

