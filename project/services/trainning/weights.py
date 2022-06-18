# encoding: utf-8
import random

import numpy as np
import torch
import torch.nn as nn


def reset_seeds():
    '''This function is used to reset all random seeds of the project
        It is used before initializing the weights from the neural networks 
        to assure they have the same weights and bias. 

    Input
    -----
    None

    Output
    ------
    None

    '''

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

def initialize_weights(m, weight_init = 'xavier_uniform'):
  '''This function initialize the neural network weights

  Input
  -----
  m :
    The model to be insert the weights
  weight_init :
    The type of initialization (xavier uniform by default)

  Output
  ------
  None
  '''

  if isinstance(m, nn.Linear):
      if weight_init == 'xavier_uniform':
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)

      elif weight_init == 'uniform':
        nn.init.uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

      else: 
        nn.init.zeros_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
