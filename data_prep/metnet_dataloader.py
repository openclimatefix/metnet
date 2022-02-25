import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .prepare_data_MetNet import load_data


class MetNetDataset(Dataset):
    def __init__(self, X, Y):
        """
        Input:
        file_name: path to hdf5-file.
        N: time-window in 5minut increments. N=1000 means 5000 minutes of data
        lead_times: Lead times
        Output:
        X: numpy array shape (None, time, channels, width, height),
        Y: numpy array shape (None, lead_times, channels, width, height)
        """

        self.x = X
        self.y = Y

        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        # allows indexing dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # Will allow len(data)
        return self.n_samples


if __name__ == "__main__":
    dataset = MetNetDataset("combination_all_pn157.h5", N=1000)
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    eps = 2
    tot_samps = len(dataset)
    n_iterations = math.ceil(tot_samps / 4)
    print(tot_samps, n_iterations)
    for epoch in range(eps):
        for i, (inputs, labels) in enumerate(dataloader):
            if (i + 1) % 5 == 0:
                print(f"epoch {epoch}/{eps}, step {i+1}/{n_iterations}, input {inputs.shape}")

    print("DONE NOW")
    input()
    dataiter = iter(dataloader)

    data = dataiter.next()
    features, labels = data
    print(features.shape, labels.shape)
