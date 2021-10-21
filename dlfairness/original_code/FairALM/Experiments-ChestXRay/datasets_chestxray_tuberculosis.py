# credits: https://github.com/udion/XTBTorch/blob/master/src/Xray_trainloop.ipynb
# credits: https://www.kaggle.com/andrewmvd/workshop-tuberculosis-classification

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os, sys, random
import numpy as np
import PIL
from PIL import Image


# Hyper-params
BATCH_SIZE = 128

def get_lbl_from_name(fname):
    lbl = int(fname.split('.png')[0][-1])
    return lbl

class XrayDset(Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor()):
        self.root_dir = root_dir
        self.fnames = os.listdir(root_dir)
        self.labels = [get_lbl_from_name(f) for f in self.fnames]
        self.transform = transform
        
    def __getitem__(self, index):
        I0 = Image.open(self.root_dir+self.fnames[index])
        I = self.transform(I0)

        if 'CHN' in self.fnames[index]:
            protected = 0
        elif 'MCU' in self.fnames[index]:
            protected = 1
        else:
            print('ERROR!! Site not found')
            exit(1)
        
        return I, self.labels[index], protected
    
    def __len__(self):
        return len(self.fnames)

def get_loaders(processed_data = True):

    if processed_data:
        data_root = './tuberculosis-data-processed/'
    else:
        data_root = None # To raise an error

    custom_transform = transforms.Compose([transforms.ToTensor(),
                                           lambda x : (x-x.min())/(x.max()-x.min())])

    train_dataset = XrayDset(data_root + 'train/',
                             transform = custom_transform)

    test_dataset = XrayDset(data_root + 'test/',
                            transform = custom_transform)

    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle = True)
    
    test_loader = DataLoader(dataset = test_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle = False)

    return train_loader, test_loader, None

    
if __name__ == '__main__':
    get_loaders()
