import argparse
from os import lseek
import pandas as pd
import json
import pickle
import numpy as np
from pathlib import Path
from scipy.special import softmax
from scipy.special import expit as sigmoid
from sklearn.metrics import f1_score
import torch
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--raw_result_dir', type=str)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config_json = json.load(f)

for config in config_json:
    if (config['dataset'] == 'CelebA') and (config['training_type'] == 'l2-penalty'):
        for no_try in range(config['no_tries']):
            exp_result_path = Path(
                args.raw_result_dir,
                "{0}_{1}_{2}_{3}/{4}".format(config['network'],
                                            config['training_type'],
                                            config['dataset'],
                                            config['random_seed'],
                                            str(no_try)))
            
            p = Path(exp_result_path, 'checkpoint')
            #features_path = Path(p, 'features.pth')
            target_path = Path(p, 'targets.pth')
            protected_path = Path(p, 'protected.pth')
            pred_path = Path(p, 'pred.pth')

            #features = torch.load(str(features_path)).numpy()
            targets = torch.load(str(target_path)).numpy()
            protected = torch.load(str(protected_path)).numpy()
            pred = torch.load(str(pred_path)).numpy()

            df = pd.DataFrame({
                'idx': list(range(pred.shape[0])),
                'ground_truth': list(targets),
                'prediction_result': list(pred),
                'protected_label': list(protected)
            })
            df.set_index('idx', inplace=True)

            output_path = Path(args.output_dir, config['training_type'])
            output_path.mkdir(exist_ok=True, parents=True)
            csv_path = Path(output_path, 'try_{0:02d}.csv'.format(no_try))
            df.to_csv(str(csv_path))
