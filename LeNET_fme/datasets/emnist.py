"""Dataset setting and data loader for EMNIST."""


import torch
from torchvision import datasets, transforms

def get_emnist(train):
    """Get EMNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=dataset_mean,
                                          std=dataset_std)])

    # dataset and data loader
    emnist_dataset = datasets.EMNIST(root=data_root,split='mnist',
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    emnist_data_loader = torch.utils.data.DataLoader(
        dataset=emnist_dataset,
        batch_size=batch_size,
        shuffle=True)

    return emnist_data_loader