# encoding: utf-8
from numpy import size
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, output_size, p_dropout = 0.05):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.output_size = output_size
        self.dropout = nn.Dropout(p_dropout)
    
        self.layer = nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=32, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=32),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=32, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=32),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=32, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=32),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=32, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=32),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2),

          nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=64),
          nn.ReLU(),
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=64),
          nn.ReLU(),
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=64),
          nn.ReLU(),
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=64),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2),

          nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=128),
          nn.ReLU(),
          nn.Conv2d(in_channels=128, out_channels=128, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=128),
          nn.ReLU(),
          nn.Conv2d(in_channels=128, out_channels=128, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=128),
          nn.ReLU(),
          nn.Conv2d(in_channels=128, out_channels=128, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=128),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2),

          nn.Conv2d(in_channels=128, out_channels=128, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=128),
          nn.ReLU(),
          nn.Conv2d(in_channels=128, out_channels=128, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=128),
          nn.ReLU(),
          nn.Conv2d(in_channels=128, out_channels=128, kernel_size = 3, stride = 1, padding = 1),
          nn.BatchNorm2d(num_features=128),
          nn.ReLU(),
          nn.AvgPool2d(kernel_size=4),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=self.output_size, bias=True, dtype=torch.double)
        )

        self.output = nn.Linear(256, self.output_size, bias=True, dtype=torch.double)

    def forward(self, x):
        hidden = self.layer(x)
        #hidden = self.dropout(hidden)
        output = hidden.view(-1, 128)
        output = self.fc(output)
        return output
