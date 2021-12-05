"""Dataset setting and data loader for FashionMNIST."""


import torch
from torchvision import datasets, transforms

import params

def get_fashionmnist(train):
    """Get FashionMNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    # dataset and data loader
    fashionmnist_dataset = datasets.FashionMNIST(root=params.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    fashionmnist_data_loader = torch.utils.data.DataLoader(
        dataset=fashionmnist_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return fashionmnist_data_loader