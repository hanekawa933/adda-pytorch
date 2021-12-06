"""Dataset setting and data loader for EMNIST."""


import torch
from torchvision import datasets, transforms

import params

def get_emnist(train):
    """Get EMNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=params.dataset_mean,
                                        std=params.dataset_std)])

    # dataset and data loader
    emnist_dataset = datasets.EMNIST(root=params.data_root,split='mnist',
                                train=train,
                                transform=pre_process,
                                download=True)

    emnist_data_loader = torch.utils.data.DataLoader(
        dataset=emnist_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return emnist_data_loader