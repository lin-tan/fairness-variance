import argparse
import pandas as pd
import json
import pickle
import numpy as np
from pathlib import Path
from scipy.special import softmax
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--raw_result_dir', type=str)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config_json = json.load(f)

for config in config_json:
    class_bias_result = []
    for no_try in range(16):
        exp_result_path = Path(
            args.raw_result_dir,
            "{0}_{1}_{2}_{3}/{4}".format(config['network'],
                                        config['training_type'],
                                        config['dataset'],
                                        config['random_seed'],
                                        str(no_try)))

        if not ('adv-conv' in config['training_type']):
            if config['training_type'] == 'no_gender':
                log_dir = 'origin_0'
            else:
                log_dir = config['training_type'].replace('-','_')

            ckpt_path = Path(exp_result_path, 'models', log_dir, 'checkpoint.pth.tar')
        else:
            if config['dataset'] == 'coco':
                if config['training_type'] == 'adv-conv4':
                    log_dir = 'conv4_300_2.0_0.2_0_linear'
                elif config['training_type'] == 'adv-conv5':
                    log_dir = 'conv5_300_2.0_0.2_0_linear'
            elif config['dataset'] == 'imSitu':
                if config['training_type'] == 'adv-conv4':
                    log_dir = 'conv4_300_1.0_0.2_linear_0'
                elif config['training_type'] == 'adv-conv5':
                    log_dir = 'conv5_300_1.0_0.2_linear_0'
            
            ckpt_path = Path(exp_result_path, 'adv/models', log_dir, 'checkpoint.pth.tar')
        
        if config['dataset'] == 'coco':
            tech = 'C-'
        elif config['dataset'] == 'imSitu':
            tech = 'I-'
        
        if config['training_type'] == 'no_gender':
            tech += 'Base'
        elif config['training_type'].startswith('ratio-'):
            tech += 'R' + config['training_type'][-1]
        elif config['training_type'].startswith('adv-conv'):
            tech += 'A' + config['training_type'][-1]       
        
        copy_path = Path(args.output_dir, tech, 'run_' + str(no_try).zfill(2) + '.pth')
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ckpt_path, copy_path)
