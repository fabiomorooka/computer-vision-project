# encoding: utf-8
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
          nn.Conv2d(self.input_size, 64, kernel_size = 3, padding = 1),
          nn.ReLU(),
          nn.Conv2d(64,128, kernel_size = 3, stride = 1, padding = 1),
          nn.ReLU(),
          nn.MaxPool2d(2,2),

          nn.Flatten(),
          nn.Linear(2048, 1024),
          nn.ReLU(),
          nn.Linear(1024, 512),
          nn.ReLU(),
          nn.Linear(512, 256)          
        )

        self.relu = nn.ReLU()
        self.output = nn.Linear(256, self.output_size, bias=True, dtype=torch.double)

    def forward(self, x):
        hidden = self.layer(x)
        hidden = self.dropout(hidden)
        hidden = torch.flatten(hidden, 1)
        relu = self.relu(hidden)
        output = self.output(relu)
        return F.softmax(output, dim = 1)
