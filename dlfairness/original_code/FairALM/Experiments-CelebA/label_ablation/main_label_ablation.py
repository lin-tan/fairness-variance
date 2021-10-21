import os
import time

import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

from datasets_celebA import get_loaders
from fcn import fcn, MyHingeLoss
from utils import *
import pdb

def run_all(config):

    # Architecture
    NUM_FEATURES = config['IMAGE_SIZE']*config['IMAGE_SIZE']*3 # Use 128*128 for resnet18 and vgg
    NUM_CLASSES = 2
    GRAYSCALE = False

    # Check these hyper manually
    if config['LR'] == 0.001:
        config['OPTIMIZER_'] = 'Adam'
    else:
        config['OPTIMIZER_'] = 'SGD'
    config['MODEL_'] = 'fcn'
    config['SHUFFLE_'] = True

    config['file_name'] = '/tmp/' + config['RESPONSE'] + '_' \
                          + config['PROTECTED'] + '_' \
                          + config['ALGORITHM'] + '_' \
                          + config['OPTIMIZER_'] + '_' \
                          + str(config['ETA_INIT'])
    def save_everything(epoch, net, optimizer, train_acc, val_acc):
        # Save checkpoint.
        state = {
            'epoch': epoch,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc
        }
        if not os.path.isdir(config['file_name']):
            os.mkdir(config['file_name'])
            torch.save(state, config['file_name'] + '/ckpt_' + str(epoch) + '.t7')
                            

    train_loader, valid_loader, test_loader = get_loaders(IMAGE_SIZE = config['IMAGE_SIZE'],
                                                          BATCH_SIZE = config['BATCH_SIZE'],
                                                          label_attr = config['RESPONSE'],
                                                          protected_attr = config['PROTECTED'])


    model = fcn(NUM_FEATURES)
    model = model.cuda()
    criterion = MyHingeLoss()

    if config['LR'] == 0.001:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config['LR'])

    lam0 = config['LAM0_PRIOR'] 
    lam1 = config['LAM1_PRIOR']
    lam2 = config['LAM2_PRIOR'] 
    eta = config['ETA_INIT']

    print(config)
    
    start_time = time.time()

    try:
        for epoch in range(config['NUM_EPOCHS']):
            eta = eta * config['ETA_BETA']

            model.train()
            for batch_idx, (features, targets, protected) in enumerate(train_loader):
                if config['DEBUG'] and batch_idx > 1:
                    break
		
                features = features.cuda()
                targets = targets.cuda()
                protected = protected.cuda()
            
                if config['ALGORITHM'] == 'FAIR_ALM':
                    for _ in range(config['NUM_INNER']):
                        ### FORWARD AND BACK PROP
                        logits, _ = model(features)
                        loss_all = criterion(logits, targets)
                        loss_t0_s0 = loss_all[(targets == 0) & (protected==0)].mean()
                        loss_t0_s1 = loss_all[(targets == 0) & (protected==1)].mean()
                        loss_t1_s0 = loss_all[(targets == 1) & (protected==0)].mean()
                        loss_t1_s1 = loss_all[(targets == 1) & (protected==1)].mean()
                        train_loss = loss_all.mean()
                        penalty_loss = loss_t1_s0 - loss_t1_s1

                        # Primal Update
                        optimizer.zero_grad()
                        
                        loss = loss_all.mean() \
                               + (eta/4 + (lam0 - lam1)/2) * loss_t1_s0 \
                               + (eta/4 + (lam1 - lam0)/2) * loss_t1_s1 
                
                        loss.backward()
                        optimizer.step()
            
                    # Dual Update
                    lam0 = 0.5*(lam0 - lam1) + 0.5 * eta * (loss_t1_s0.item()  - loss_t1_s1.item())
                    lam1 = 0.5*(lam1 - lam0) + 0.5 * eta * (loss_t1_s1.item()  - loss_t1_s0.item())
            
                else:
                    ### FORWARD AND BACK PROP
                    logits, _ = model(features)
                    loss_all = criterion(logits, targets)
                    loss_t0_s0 = loss_all[(targets == 0) & (protected==0)].mean()
                    loss_t0_s1 = loss_all[(targets == 0) & (protected==1)].mean()
                    loss_t1_s0 = loss_all[(targets == 1) & (protected==0)].mean()
                    loss_t1_s1 = loss_all[(targets == 1) & (protected==1)].mean()
                    train_loss = loss_all.mean()
                    penalty_loss = loss_t0_s0 - loss_t0_s1
                
                    optimizer.zero_grad()
                    loss = loss_all.mean()
                    loss.backward()
                    optimizer.step()
        
                ### LOGGING
                if not batch_idx % 50:
                    print ('Epoch: %03d/%03d | Batch %04d/%04d | train_loss: %.4f | penalty_loss: %.4f'%(epoch+1, config['NUM_EPOCHS'], batch_idx,len(train_loader), train_loss, penalty_loss))
                    print('eta: %.3f | lam0: %.3f | lam1: %.3f | lam2: %.3f' % (eta, lam0, lam1, lam2))

            model.eval()
            with torch.set_grad_enabled(False): # save memory during inference
                train_stats = compute_accuracy(config, model, train_loader)
                print_stats(config, epoch, train_stats, stat_type='Train')

                valid_stats = compute_accuracy(config, model, valid_loader)
                print_stats(config, epoch, valid_stats, stat_type='Valid')

            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

            if epoch == 4 and config['SAVE_CKPT']:
                save_everything(epoch, model, optimizer, train_stats, valid_stats)
    except:
        pass


if __name__ == '__main__':
    ### SETTINGS
    config = {}
    config['ALGORITHM'] = 'FAIR_ALM'  # 'FAIR_ALM' or 'L2_PENALTY' or 'NO_CONSTRAINTS'
    config['CONSTRAINT'] = 'DEO' # DEO, DDP, PPV 
    config['LAM0_PRIOR'] = 0.
    config['LAM1_PRIOR'] = 0.
    config['LAM2_PRIOR'] = 0.
    config['ETA_INIT'] = 0.01
    config['ETA_BETA'] = 1.01
    config['SAVE_CKPT'] = True
    config['DEBUG'] = False 
    config['RESPONSE'] = '5_o_Clock_Shadow'
    config['PROTECTED'] = 'Young'
    
    # Hyperparameters
    config['LR'] = 0.01
    config['NUM_EPOCHS'] = 5
    config['NUM_INNER'] = 1
    config['BATCH_SIZE'] = 1024 
    config['IMAGE_SIZE'] = 28


    list_response = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                     'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                     'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                     'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                     'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                     'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                     'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                     'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                     'Wearing_Necktie']

    for response in list_response:
        config['RESPONSE'] = response
        run_all(config)
