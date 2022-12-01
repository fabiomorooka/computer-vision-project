# encoding: utf-8
import torch
from torch.utils.data import Dataset

from PIL import Image

# Prepare data to use with pytorch
class ImageTrafficSignsDataset(Dataset):
    def __init__(self, x, y, transform=None):
      self._x = x
      self._y = y
      self._transform = transform  

    def __len__(self):
      return len(self._x)

    def __getitem__(self, idx):
      item = Image.fromarray(self._x[idx], mode="RGB")

      label = int(self._y[idx])
      if self._transform is not None:
        item = self._transform(item)
      return item, label
