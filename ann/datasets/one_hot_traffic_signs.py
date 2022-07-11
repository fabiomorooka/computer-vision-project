# encoding: utf-8
import torch
import torch.nn.functional as F

from datasets.traffic_signs import TrafficSignsDataset

# Prepare data to use with pytorch
class OneHotTrafficSignsDataset(TrafficSignsDataset):
    def __init__(self, x, y, transform=None, one_hot=False):
      self.one_hot = one_hot

      if self.one_hot:
        self._y_enc = torch.tensor(y).reshape(-1)
        self._y_enc= F.one_hot(self._y_enc.long(), num_classes=43)
      
      super(OneHotTrafficSignsDataset, self).__init__(x=x, y=y, transform=transform)

    def __len__(self):
      return self._x.shape[0]

    def __getitem__(self, idx):
      item = self._x[idx]
      if self.one_hot:
        label = self._y_enc[idx]
      else:
        label = self._y[idx]
      if self._transform is not None:
        item = self._transform(item)
      item = item.flatten()
      return item, label
