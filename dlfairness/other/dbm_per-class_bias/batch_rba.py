import json
import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--raw_result_dir', type=str)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config_json = json.load(f)

for config in config_json:
    if config['training_type'] == 'domain-discriminative':
        for no_try in range(config['no_tries']):
            parent_folder = str(
                Path(
                    args.raw_result_dir,
                    "{0}_{1}_{2}_{3}/{4}".format(config['network'],
                                                 config['training_type'],
                                                 config['dataset'],
                                                 config['random_seed'],
                                                 str(no_try))))
            record_name = "{0}_{1}".format(config['dataset'], config['training_type'].replace('-', '_'))

            cmd = ['python', 'rba.py', '--record_name', record_name, '--parent_folder', parent_folder]

            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            o, e = p.communicate()
            print(o)
            #print(e)
