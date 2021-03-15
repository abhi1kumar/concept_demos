"""
    Dataset loaders.
"""
import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, x_train, y_train):
        # Read the csv file separated by whitespace
        self.x_train   = x_train
        self.y_train   = y_train
        self.data_len  = len(self.x_train)

    def __getitem__(self, index):
        # Get image name from the pandas df
        x = self.x_train[index]
        y = self.y_train[index]

        # Return image and the label
        return x.unsqueeze(0), y.unsqueeze(0)

    def __len__(self):
        return self.data_len
