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

coco_f1 = {} # {setting: []}
coco_mAP = {}
imSitu_f1 = {}
imSitu_mAP = {}
for config in config_json:
    for no_try in range(16):
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
        if config['dataset'] == 'coco':
            key = 'threshold_0.5/' + key
            coco_f1.setdefault(key, [])
            coco_f1[key].append(acc_dict['F1 Score'])
            coco_mAP.setdefault(key, [])
            coco_mAP[key].append(acc_dict['mAP'])
        elif config['dataset'] == 'imSitu':
            imSitu_f1.setdefault(key, [])
            imSitu_f1[key].append(acc_dict['F1 Score'])
            imSitu_mAP.setdefault(key, [])
            imSitu_mAP[key].append(acc_dict['mAP'])

with open('./coco_f1.yaml', 'w') as f:
    yaml.dump(coco_f1, f)
with open('./coco_mAP.yaml', 'w') as f:
    yaml.dump(coco_mAP, f)
with open('./imSitu_f1.yaml', 'w') as f:
    yaml.dump(imSitu_f1, f)
with open('./imSitu_mAP.yaml', 'w') as f:
    yaml.dump(imSitu_mAP, f)
