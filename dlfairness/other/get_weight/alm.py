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
        if (config['dataset'] != 'CelebA') or (not config['training_type'] in ['no-constraints', 'l2-penalty', 'fair-alm']):
            continue

        exp_result_path = Path(
            args.raw_result_dir,
            "{0}_{1}_{2}_{3}/{4}".format(config['network'],
                                        config['training_type'],
                                        config['dataset'],
                                        config['random_seed'],
                                        str(no_try)))
        p = Path(exp_result_path, 'checkpoint')
        ckpt_path = Path(p, 'ckpt_80.t7')

        if config['training_type'] == 'no-constraints':
            tech = 'A-Base'
        elif config['training_type'] == 'l2-penalty':
            tech = 'A-L2'
        elif config['training_type'] == 'fair-alm':
            tech = 'A-ALM'
        
        copy_path = Path(args.output_dir, tech, 'run_' + str(no_try).zfill(2) + '.pth')
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ckpt_path, copy_path)