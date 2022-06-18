# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, weight_init, output_size, p_dropout = 0.00):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size  = hidden_size
        self.hidden_layers = hidden_layers
        self.weight_init = weight_init
        self.dropout = nn.Dropout(p_dropout)

        layers = []
        for i in range(self.hidden_layers):
          if i < 1: 
            layers.append(torch.nn.Linear(self.input_size, self.hidden_size, bias=True, dtype=torch.float64))
            layers.append(torch.nn.ReLU())
          else : 
            layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True, dtype=torch.float64))
            layers.append(torch.nn.ReLU())
            
        self.layer = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(self.hidden_size, self.output_size, bias=True, dtype=torch.float64)

    def forward(self, x):
        hidden = self.layer(x)
        hidden = self.dropout(hidden)
        relu = self.relu(hidden)
        output = self.output(relu)
        return F.softmax(output, dim = 1)
