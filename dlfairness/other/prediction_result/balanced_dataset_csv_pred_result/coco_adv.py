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
parser.add_argument('--threshold', type=float, default=0.5)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config_json = json.load(f)

for config in config_json:
    if (config['dataset'] == 'coco') and ('adv-conv' in config['training_type']):
        for no_try in range(config['no_tries']):
            exp_result_path = Path(
                args.raw_result_dir,
                "{0}_{1}_{2}_{3}/{4}".format(config['network'],
                                            config['training_type'],
                                            config['dataset'],
                                            config['random_seed'],
                                            str(no_try)))

            if config['training_type'] == 'adv-conv4':
                log_dir = 'conv4_300_2.0_0.2_0_linear'
            elif config['training_type'] == 'adv-conv5':
                log_dir = 'conv5_300_2.0_0.2_0_linear'

            feature_path = Path(exp_result_path, 'adv/models', log_dir, 'image_features')
            image_id_path = Path(feature_path, 'test_image_ids.pth')
            targets_path = Path(feature_path, 'test_targets.pth')
            genders_path = Path(feature_path, 'test_genders.pth')
            potentials_path = Path(feature_path, 'test_potentials.pth')
            
            image_id = torch.load(str(image_id_path)).numpy()
            targets = torch.load(str(targets_path)).numpy().astype(int)
            genders = torch.load(str(genders_path)).numpy()
            potentials = torch.load(str(potentials_path)).numpy()

            pred = (sigmoid(potentials) >= args.threshold).astype(int)

            image_id_list = list(image_id.squeeze())
            ground_truth_list = [list(e) for e in list(targets)]
            pred_list = [list(e) for e in list(pred)]
            gender_list = list(np.argmax(genders, axis=1))        

            df = pd.DataFrame({
                'idx': image_id_list,
                'ground_truth': ground_truth_list,
                'prediction_result': pred_list,
                'protected_label': gender_list
            })
            df.set_index('idx', inplace=True)

            output_path = Path(args.output_dir, 'threshold_' + str(args.threshold), config['training_type'])
            output_path.mkdir(exist_ok=True, parents=True)
            csv_path = Path(output_path, 'try_{0:02d}.csv'.format(no_try))
            json_path = Path(output_path, 'try_{0:02d}.json'.format(no_try))

            df.to_csv(str(csv_path))
            df.to_json(str(json_path))

            #if no_try == 0:
            #    print(args.threshold, config['training_type'], 'F1:', f1_score(ground_truth_list, pred_list, average='macro'))

            '''
            print(f1_score(ground_truth_list, pred_list, average='macro'))
            count = 0
            for idx, gt, pred in zip(image_id_list, ground_truth_list, pred_list):
                #print(idx, f1_score(gt, pred, average = 'macro'))
                
                t = (np.array(gt) != np.array(pred)).astype(int).sum()
                if t != 0:
                    print(idx, t)
                    count += 1
                
            sys.exit(0)
            '''    