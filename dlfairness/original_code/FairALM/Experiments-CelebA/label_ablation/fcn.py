import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

class MultilayerPerceptron(torch.nn.Module):

    def __init__(self, num_features):
        super(MultilayerPerceptron, self).__init__()
        
        ### 1st hidden layer
        self.linear_1 = torch.nn.Linear(num_features, 64)
        
        ### 2nd hidden layer
        self.linear_2 = torch.nn.Linear(64, 1)
        
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear_1(out)
        out = F.relu(out)
        out = self.linear_2(out)

        probas = torch.cat(((out < 0.), (out >= 0.)),  1)
        
        return out, probas

class MyHingeLoss(torch.nn.Module):

    def __init__(self):
        super(MyHingeLoss, self).__init__()
    
    def forward(self, output, target):
        target_new = target.clone()
        target_new[target < 1.] = -1.

        hinge_loss = 1 - torch.mul(torch.squeeze(output), target_new.float())
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss
    
def fcn(num_features):
    model = MultilayerPerceptron(num_features)
    return model
