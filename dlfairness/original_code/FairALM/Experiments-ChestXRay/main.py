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

from datasets_chestxray_tuberculosis import get_loaders
from resnet import resnet18
from utils import *
import pdb

import argparse
import random
from pathlib import Path
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='NO_CONSTRAINTS') # config['ALGORITHM']
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

random.seed(a=args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

### SETTINGS
config = {}
#config['ALGORITHM'] = 'FAIR_ALM'  # 'FAIR_ALM' or 'L2_PENALTY' or 'NO_CONSTRAINTS'
config['ALGORITHM'] = args.algorithm
config['CONSTRAINT'] = 'DEO' # DEO, DDP, PPV 
config['LAM0_PRIOR'] = 0.
config['LAM1_PRIOR'] = 0.
config['LAM2_PRIOR'] = 0.
config['ETA_INIT'] = 50.0
config['ETA_BETA'] = 1.01
config['SAVE_CKPT'] = True
config['DEBUG'] = False
#config['file_name'] = '/tmp/' # Specify the location to save the checkpoints
config['file_name'] = args.checkpoint_path
config['PROTECTED'] = 'site'

# Hyperparameters
config['LR'] = 0.01
#config['NUM_EPOCHS'] = 500
config['NUM_EPOCHS'] = args.num_epochs
config['NUM_INNER'] = 1

# Architecture
NUM_FEATURES = 128*128
NUM_CLASSES = 2
GRAYSCALE = True

# Check these hyper manually
config['OPTIMIZER_'] = 'SGD'
config['MODEL_'] = 'resnet18'
config['SHUFFLE_'] = True

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
                            

train_loader, valid_loader, test_loader = get_loaders()

model = resnet18(NUM_CLASSES, GRAYSCALE)

#### DATA PARALLEL START ####
if torch.cuda.device_count() > 1:
	print("Using", torch.cuda.device_count(), "GPUs")
	model = nn.DataParallel(model)
#### DATA PARALLEL END ####

model = model.cuda()
criterion = torch.nn.CrossEntropyLoss(reduce=False)
optimizer = torch.optim.SGD(model.parameters(), lr=config['LR'])

lam0 = config['LAM0_PRIOR'] 
lam1 = config['LAM1_PRIOR']
lam2 = config['LAM2_PRIOR'] 
eta = config['ETA_INIT']

print(config)

start_time = time.time()

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
                logits, probas = model(features)
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
        elif config['ALGORITHM'] == 'L2_PENALTY':
            ### FORWARD AND BACK PROP
            logits, probas = model(features)
            loss_all = criterion(logits, targets)
            loss_t0_s0 = loss_all[(targets == 0) & (protected==0)].mean()
            loss_t0_s1 = loss_all[(targets == 0) & (protected==1)].mean()
            loss_t1_s0 = loss_all[(targets == 1) & (protected==0)].mean()
            loss_t1_s1 = loss_all[(targets == 1) & (protected==1)].mean()
            train_loss = loss_all.mean()
            penalty_loss = loss_t1_s0 - loss_t1_s1

            loss = loss_all.mean() + 0.5 * eta * (loss_t1_s0 - loss_t1_s1)**2
            loss.backward()
            optimizer.step()
        else:
            # Unconstrained algorithm
            
            ### FORWARD AND BACK PROP
            logits, probas = model(features)
            loss_all = criterion(logits, targets)
            loss_t0_s0 = loss_all[(targets == 0) & (protected==0)].mean()
            loss_t0_s1 = loss_all[(targets == 0) & (protected==1)].mean()
            loss_t1_s0 = loss_all[(targets == 1) & (protected==0)].mean()
            loss_t1_s1 = loss_all[(targets == 1) & (protected==1)].mean()
            train_loss = loss_all.mean()
            penalty_loss = loss_t1_s0 - loss_t1_s1

            optimizer.zero_grad()
            loss = loss_all.mean()
            loss.backward()
            optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | train_loss: %.4f | penalty_loss: %.4f'%(epoch+1, config['NUM_EPOCHS'], batch_idx, 
                     len(train_loader), train_loss, penalty_loss))
            print('eta: %.3f | lam0: %.3f | lam1: %.3f | lam2: %.3f' % (eta, lam0, lam1, lam2))

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        train_stats = compute_accuracy(config, model, train_loader)
        print_stats(config, epoch, train_stats, stat_type='Train')

        valid_stats = compute_accuracy(config, model, valid_loader)
        print_stats(config, epoch, valid_stats, stat_type='Valid')

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    if epoch % 20 == 0 and config['SAVE_CKPT']:
        save_everything(epoch, model, optimizer, train_stats, valid_stats)

        '''
        with torch.set_grad_enabled(False): # save memory during inference
            test_stats = compute_accuracy(config, model, test_loader)
            print_stats(config, epoch, test_stats, stat_type='Test')
        '''
        
'''        
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
with torch.set_grad_enabled(False): # save memory during inference
    test_stats = compute_accuracy(config, model, test_loader)
    print_stats(config, epoch, test_stats, stat_type='Test')
'''

print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

# Save raw predictions
with torch.set_grad_enabled(False): # save memory during inference
    test_stats, features, targets, protected, pred = compute_accuracy(config, model, valid_loader, dump_raw = True)
    print_stats(config, epoch, test_stats, stat_type='Test')

    p = Path(args.checkpoint_path, 'test_stats.yaml')
    with open(str(p), 'w') as f:
        yaml.safe_dump(map_0d_tensor_to_float(test_stats), f)
    torch.save(features, str(Path(args.checkpoint_path, 'features.pth')))
    torch.save(targets, str(Path(args.checkpoint_path, 'targets.pth')))
    torch.save(protected, str(Path(args.checkpoint_path, 'protected.pth')))
    torch.save(pred, str(Path(args.checkpoint_path, 'pred.pth')))

# Save log
p = Path('./logs')
p.mkdir(exist_ok=True)
test_stats = map_0d_tensor_to_float(test_stats)
with open(str(Path(p, 'test_result.txt')), 'w') as f:
    f.write('Accuracy: ' + str(test_stats['acc']))
with open(str(Path(p, 'bias_result.txt')), 'w') as f:
    f.write('Demographic Parity: ' + str(test_stats['ddp']))
