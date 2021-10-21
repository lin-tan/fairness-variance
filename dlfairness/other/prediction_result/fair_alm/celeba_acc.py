import argparse
import pandas as pd
import json
import pickle
import numpy as np
from pathlib import Path
from scipy.special import softmax
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--raw_result_dir', type=str)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

def load_acc_txt(p):
    result = {}
    with open(str(p), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if ',' in line:
                line = line.split(',')[0].strip()
            key = line.split(':')[0].strip()
            value = float(line.split(':')[1].strip())
            result[key] = value
    return result   

with open(args.config, 'r') as f:
    config_json = json.load(f)

#with open('./celeba_alm_acc.yaml', 'r') as f:
#    full_result = yaml.load(f)

full_result = {} # {setting: []}
for config in config_json:
    for no_try in range(16):
        if config['dataset'] == 'ChestXRay':
            continue
#        if config['training_type'] != 'l2-penalty':
#            continue
        exp_result_path = Path(
            args.raw_result_dir,
            "{0}_{1}_{2}_{3}/{4}".format(config['network'],
                                         config['training_type'],
                                         config['dataset'],
                                         config['random_seed'],
                                         str(no_try)))
        log_path = Path(exp_result_path, config['main_result_rel_path'])
        acc_dict = load_acc_txt(log_path)

        key = config['training_type'] + '/'
        full_result.setdefault(key, [])
        full_result[key].append(acc_dict['Accuracy'])

with open('./celeba_alm_acc.yaml', 'w') as f:
    yaml.dump(full_result, f)
