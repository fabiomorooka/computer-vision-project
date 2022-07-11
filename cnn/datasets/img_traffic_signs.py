# encoding: utf-8
import torch
from torch.utils.data import Dataset

# Prepare data to use with pytorch
class ImageTrafficSignsDataset(Dataset):
    def __init__(self, x, y, transform=None):
      self._x = torch.tensor(x)
      self._y = torch.tensor(y).squeeze()
      self._transform = transform  

    def __len__(self):
      return self._x.shape[0]

    def __getitem__(self, idx):
      item = self._x[idx]
      label = self._y[idx]
      if self._transform is not None:
        item = self._transform(item)
      return item.double(), label.double()
