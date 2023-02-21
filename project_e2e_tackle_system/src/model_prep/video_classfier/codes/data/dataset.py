from os import path as osp
from typing import List, Optional

import numpy as np
from torch.utils.data import Dataset

class RugbyVideoDataset(Dataset):

    def __init__(self, data, transform:Optional[List]=None) -> None:
        """
        Args:
            data
            label
        Returns: 
            None
        """

        self.data = data[:, 0]
        self.label = data[:, 1].astype(float).astype(int)
        self.data_idx = np.array([
            osp.basename(data_path) for data_path in self.data
        ])

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]
        data_idx = self.data_idx[index]

        if self.transform:
            sample = {"data": data, "label": label}
            sample = self.transform(sample)
            data, label = sample["data"], sample["label"]
        
        return data, label, data_idx

if __name__ == "__main__":
    pass