import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--raw_result_dir', type=str)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config_json = json.load(f)

for config in config_json:
    bias_result_dict = {}
    for no_try in range(config['no_tries']):
        exp_result_path = Path(
            args.raw_result_dir,
            "{0}_{1}_{2}_{3}/{4}".format(config['network'],
                                         config['training_type'],
                                         config['dataset'],
                                         config['random_seed'],
                                         str(no_try)))
        bias_result_path = Path(
            exp_result_path,
            "record/{0}_{1}/e1/bias_result.txt".format(config['dataset'],
                                                             config['training_type'].replace('-', '_')))
        
        bias_f = open(str(bias_result_path), 'r')
        for line in bias_f:
            metric = line.split(':')[0]
            value = float(line.split(':')[1].strip()) * 200
            if metric in bias_result_dict:
                bias_result_dict[metric].append(value)
            else:
                bias_result_dict[metric] = [value]
    
    print(bias_result_dict)
    
    for metric, values in bias_result_dict.items():
        fig = plt.figure()
        ax = fig.subplots(1, 1)
        ax.boxplot(values)
        fig.suptitle(metric)
        fig_fn = "{0}_{1}_{2}.png".format(config['network'], config['training_type'], metric).replace(' ', '-').replace('/', '')

        fig.savefig(fig_fn)
