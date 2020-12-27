import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyDataset(Dataset):
    def __init__(self, shuffle_seed=0):
        x_0 = np.load(f"exp_0/fMRI_data_0.pkl.npy")
        # x_0 = np.zeros(shape=[10,2])
        y_0 = np.zeros(shape=(len(x_0)))
        x_1 = np.load(f"exp_0/fMRI_data_100.pkl.npy")
        # x_1 = np.ones(shape=[10,2])
        x = np.concatenate((x_0, x_1))
        y = np.zeros(shape=(len(x)))
        y[len(x_0):] = 1
        
        np.random.seed(shuffle_seed)
        p = np.random.permutation(len(x))
        self.x, self.y = x[p], y[p]
        self.y = self.y.astype(np.int64)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = { "x": self.x[idx], "y": self.y[idx] }
        return sample

ds0 = MyDataset(shuffle_seed=0)

ds = DataLoader(ds0, shuffle=True)
for x in ds:
    print(x)
    break