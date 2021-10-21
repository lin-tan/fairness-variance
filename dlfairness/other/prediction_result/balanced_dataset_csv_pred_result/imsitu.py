import argparse
from os import lseek
import pandas as pd
import json
import pickle
import numpy as np
from pathlib import Path
from scipy.special import softmax
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
    if config['dataset'] == 'imSitu':
        for no_try in range(config['no_tries']):
            exp_result_path = Path(
                args.raw_result_dir,
                "{0}_{1}_{2}_{3}/{4}".format(config['network'],
                                            config['training_type'],
                                            config['dataset'],
                                            config['random_seed'],
                                            str(no_try)))

            if config['training_type'] == 'no_gender':
                log_dir = 'origin_0'
            else:
                log_dir = config['training_type'].replace('-','_')

            feature_path = Path(exp_result_path, 'models', log_dir, 'image_features')
            image_id_path = Path(feature_path, 'test_image_ids.pth')
            targets_path = Path(feature_path, 'test_targets.pth')
            genders_path = Path(feature_path, 'test_genders.pth')
            potentials_path = Path(feature_path, 'test_potentials.pth')
            
            image_id = torch.load(str(image_id_path)).numpy()
            targets = torch.load(str(targets_path)).numpy()
            genders = torch.load(str(genders_path)).numpy()
            potentials = torch.load(str(potentials_path)).numpy()

            pred = np.argmax(softmax(potentials, axis=1), axis=1)
            gt = np.argmax(targets, axis=1)

            image_id_list = list(image_id.squeeze())
            ground_truth_list = list(gt)
            pred_list = list(pred)
            gender_list = list(np.argmax(genders, axis=1))        

            df = pd.DataFrame({
                'idx': image_id_list,
                'ground_truth': ground_truth_list,
                'prediction_result': pred_list,
                'protected_label': gender_list
            })
            df.set_index('idx', inplace=True)

            output_path = Path(args.output_dir, config['training_type'])
            output_path.mkdir(exist_ok=True, parents=True)
            csv_path = Path(output_path, 'try_{0:02d}.csv'.format(no_try))

            df.to_csv(str(csv_path))
