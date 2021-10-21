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
        ckpt_path = Path(
            exp_result_path,
            "record/{0}_{1}/e1/ckpt.pth".format(config['dataset'],
                                                             config['training_type'].replace('-', '_')))


        if config['training_type'] == 'baseline':
            tech = 'S-Base'
            tech_count = 1
        elif config['training_type'] == 'sampling':
            tech = 'S-RS'
            tech_count = 1
        elif config['training_type'] == 'uniconf-adv':
            tech = 'S-UC'
            tech_count = 1
        elif config['training_type'] == 'gradproj-adv':
            tech = 'S-GR'
            tech_count = 1
        elif config['training_type'] == 'domain-discriminative':
            tech = 'S-DD'
            tech_count = 4
        elif config['training_type'] == 'domain-independent':
            tech = 'S-DI'
            tech_count = 2

        if tech_count == 1:
            copy_path = Path(args.output_dir, tech, 'run_' + str(no_try).zfill(2) + '.pth')
            copy_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ckpt_path, copy_path)
        else:
            for tc in range(1, tech_count + 1):
                copy_path = Path(args.output_dir, tech + str(tc), 'run_' + str(no_try).zfill(2) + '.pth')
                copy_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ckpt_path, copy_path)

