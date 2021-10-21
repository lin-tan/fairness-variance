import os

import numpy as np
import pandas as pd

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from PIL import Image

# Hyper-params
BATCH_SIZE = 256
data_root = './data/' # Please specify the location of CelebA dataset.
label_attr = 'Attractive' # The response label is the noun `attractive` in the CelebA dataset.
# Please find a remark on the choice of the labels in Section 6 of https://arxiv.org/pdf/2004.01355.pdf
protected_attr = 'Male'

class CelebaDataset(Dataset):
    
    def __init__(self, csv_path, img_dir, transform=None, label_attr = 'Attractive', protected_attr = 'Male'):
        
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df[label_attr].values
        self.p = df[protected_attr].values
        self.transform = transform
        
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)
            
        label = self.y[index]
        protected = self.p[index]
        return img, label, protected
        
    def __len__(self):
        return self.y.shape[0]


def prepare_dataset(label_attr = 'Attractive', protected_attr = 'Male'):

    print('Using label attribute:', label_attr)

    df1 = pd.read_csv(data_root + 'list_attr_celeba.txt', sep="\s+", skiprows=1, usecols=[label_attr, protected_attr])
    # Make 0 (female) & 1 (male) labels instead of -1 & 1
    df1.loc[df1[label_attr] == -1, label_attr] = 0
    df1.loc[df1[protected_attr] == -1, protected_attr] = 0

    df2 = pd.read_csv(data_root + 'list_eval_partition.txt', sep="\s+", skiprows=0, header=None)
    df2.columns = ['Filename', 'Partition']
    df2 = df2.set_index('Filename')

    df3 = df1.merge(df2, left_index=True, right_index=True)

    df3.to_csv(data_root + 'celeba-attractive-partitions.csv')
    df4 = pd.read_csv(data_root + 'celeba-attractive-partitions.csv', index_col=0)

    df4.loc[df4['Partition'] == 0].to_csv(data_root + 'celeba-attractive-train.csv')
    df4.loc[df4['Partition'] == 1].to_csv(data_root + 'celeba-attractive-valid.csv')
    df4.loc[df4['Partition'] == 2].to_csv(data_root + 'celeba-attractive-test.csv')
    
def get_loaders():

    prepare_dataset(label_attr, protected_attr)
    
    # Note that transforms.ToTensor()
    # already divides pixels by 255. internally

    custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                           transforms.Resize((128, 128)),
                                           transforms.ToTensor()])

    train_dataset = CelebaDataset(csv_path=data_root + 'celeba-attractive-train.csv',
                                  img_dir=data_root + 'img_align_celeba/',
                                  transform=custom_transform,
                                  label_attr=label_attr,
                                  protected_attr = protected_attr)

    valid_dataset = CelebaDataset(csv_path=data_root + 'celeba-attractive-valid.csv',
                                  img_dir=data_root + 'img_align_celeba/',
                                  transform=custom_transform,
                                  label_attr=label_attr,
                                  protected_attr = protected_attr)

    test_dataset = CelebaDataset(csv_path=data_root + 'celeba-attractive-test.csv',
                                 img_dir=data_root + 'img_align_celeba/',
                                 transform=custom_transform,
                                 label_attr=label_attr,
                                 protected_attr = protected_attr)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=4)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4)

    return train_loader, valid_loader, test_loader

