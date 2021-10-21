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

full_raw_result = {} # {setting: []}
for config in config_json:
    for no_try in range(16):
        exp_result_path = Path(
            args.raw_result_dir,
            "{0}_{1}_{2}_{3}/{4}".format(config['network'],
                                         config['training_type'],
                                         config['dataset'],
                                         config['random_seed'],
                                         str(no_try)))
        log_path = Path(exp_result_path,
            "record/{0}_{1}/e1/test_result.txt".format(config['dataset'],
                                                             config['training_type'].replace('-', '_')))

        acc_dict = load_acc_txt(log_path)
        if config['training_type'] == 'domain-discriminative':
            # Sum-of-prob-no-prior-shift/
            acc = (acc_dict['Test on color images accuracy sum prob without prior shift'] + acc_dict['Test on gray images accuracy sum prob without prior shift']) / 2
            full_raw_result.setdefault('domain-discriminative/Sum-of-prob-no-prior-shift/', [])
            full_raw_result['domain-discriminative/Sum-of-prob-no-prior-shift/'].append(acc)

            # domain-discriminative/Max-of-prob-prior-shift/
            acc = (acc_dict['Test on color images accuracy max prob with prior shift'] + acc_dict['Test on gray images accuracy max prob with prior shift']) / 2
            full_raw_result.setdefault('domain-discriminative/Max-of-prob-prior-shift/', [])
            full_raw_result['domain-discriminative/Max-of-prob-prior-shift/'].append(acc)

            # domain-discriminative/Sum-of-prob-prior-shift/
            acc = (acc_dict['Test on color images accuracy sum prob with prior shift'] + acc_dict['Test on gray images accuracy sum prob with prior shift']) / 2
            full_raw_result.setdefault('domain-discriminative/Sum-of-prob-prior-shift/', [])
            full_raw_result['domain-discriminative/Sum-of-prob-prior-shift/'].append(acc)

            # domain-discriminative/rba/
            acc = (acc_dict['RBA Gray Accuracy'] + acc_dict['RBA Color Accuracy']) / 2 * 100
            full_raw_result.setdefault('domain-discriminative/rba/', [])
            full_raw_result['domain-discriminative/rba/'].append(acc)

        elif config['training_type'] == 'domain-independent':
            # domain-independent/Conditional/
            acc = (acc_dict['Test on color images accuracy conditional'] + acc_dict['Test on gray images accuracy conditional']) / 2
            full_raw_result.setdefault('domain-independent/Conditional/', [])
            full_raw_result['domain-independent/Conditional/'].append(acc)

            # domain-independent/Sum/
            acc = (acc_dict['Test on color images accuracy sum out'] + acc_dict['Test on gray images accuracy sum out']) / 2
            full_raw_result.setdefault('domain-independent/Sum/', [])
            full_raw_result['domain-independent/Sum/'].append(acc)

        else:
            acc = (acc_dict['Test on color images accuracy'] + acc_dict['Test on gray images accuracy']) / 2
            key = config['training_type'] + '/'
            full_raw_result.setdefault(key, [])
            full_raw_result[key].append(acc)

with open('./cifar-10s.yaml', 'w') as f:
    yaml.dump(full_raw_result, f)
