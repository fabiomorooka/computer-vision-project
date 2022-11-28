# encoding: utf-8
from numpy import size
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, p_dropout = 0.05):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = nn.Dropout(p_dropout)
            
        self.layer = nn.Sequential(
          nn.Conv2d(self.input_size, 32, kernel_size = 3, padding = 1),
          nn.ReLU(),
          nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
          nn.ReLU(),
          nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
          nn.ReLU(),
          nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
          nn.Dropout(0.25),

          nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
          nn.ReLU(),
          nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
          nn.ReLU(),
          nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
          nn.ReLU(),
          nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
          nn.Dropout(0.25),
        )

        self.flatten = nn.Sequential(
          nn.AdaptiveMaxPool2d(2), 
          nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_size, bias=True, dtype=torch.double)
        )

        self.output = nn.Linear(256, self.output_size, bias=True, dtype=torch.double)

    def forward(self, x):
        hidden = self.layer(x)
        #hidden = self.dropout(hidden)
        hidden = self.flatten(hidden)
        output = self.fc(hidden)
        return F.softmax(output, dim = 1)
