import csv

from pathlib import Path

path_list = ['../result_fairness', '../result_fairness_2', '../result_balanced_dataset_coco', '../result_fairalm', '../result_fairness_DBM_seed', '../result_nifr']
fn = 'train_done.csv'

time_sec = 0
for path in path_list:
    p = Path(path, fn)
    with open(str(p), newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_sec += float(row['time'])

print(time_sec / 3600.0)


