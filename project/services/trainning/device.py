# encoding: utf-8
import torch

def get_device():
    # Utilização de GPU como no notebook de referência

    if torch.cuda.is_available(): 
        dev = "cuda:0"
        print(f'GPU available: {torch.cuda.get_device_name(dev)}')
    else: 
        dev = "cpu" 
    dev = 'cpu'
    print(f'The device used is {dev}')
    device = torch.device(dev)
    
    return device