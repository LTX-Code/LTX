import torch
from torch.utils.data import Dataset
import h5py
import os


class ImagenetResults(Dataset):
    def __init__(self, path):
        super(ImagenetResults, self).__init__()

        self.path = os.path.join(path, "results.hdf5")
        self.data = None

        with h5py.File(self.path, "r") as f:
            self.data_length = len(f["/image"])

    def __len__(self):
        return self.data_length

    def __getitem__(self, item):
        if self.data is None:
            self.data = h5py.File(self.path, 'r')

        image = torch.tensor(self.data["image"][item])
        vis = torch.tensor(self.data["vis"][item])
        target = torch.tensor(self.data["target"][item]).long()

        return image, vis, target
