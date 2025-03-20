# Custom Class to Hold the Euro Satellite dataset

import torch
from torch.utils import data
from torch.utils.data import Dataset



# Extend Pytorch dataset class for applying custom transformations - crop, flip and normalize
class EuroSAT(data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    # returns inputs and label set for the given index in the dataset
    def __getitem__(self, index):
        x = self.dataset[index][0]

        if self.transform:
            x = self.transform(x)

        y = self.dataset[index][1]
        return x, y

    def __len__(self):
        return len(self.dataset)