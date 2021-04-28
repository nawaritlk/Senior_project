import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import time

import model
import function
import transformer
import os

def set_optimizer(img,optimizer,num_iteration,lbfgs_num_correction,learning_rate):
    if optimizer == 'lbfgs':
        print("Using optimizer with L-BFGS")
        optim_state = {
            'max_iter' : num_iteration,
            'tolerance_change' : -1,
            'tolerance_grad' : -1
        }
        if lbfgs_num_correction != 100:
            optim_state['history_size'] = lbfgs_num_correction
        optimizer = optim.LBFGS([img],**optim_state)
        loopVal = 1
    elif optimizer == 'adam':
        print("Using optimizer with Adam")
        optimizer = optim.Adam([img],lr=learning_rate)
        loopVal = num_iteration-1
    return optimizer, loopVal



