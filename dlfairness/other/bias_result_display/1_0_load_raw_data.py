import yaml
import json
import csv
import sys

from pathlib import Path
from collections import OrderedDict

with open('./bias_metric.yaml', 'r') as f:
    bias_metric_list = yaml.full_load(f)

with open('./configs.yaml', 'r') as f:
    configs = yaml.full_load(f)

for paper, config in configs.items():
    baseline, mitigation = config['settings']
    class_count = config['class']
    all_settings = baseline + mitigation

    overall_stat = OrderedDict()  # {(setting, bias_metric): {stat: value}}
    per_class_stat = OrderedDict()  # {(setting, bias_metric): [{stat: value}]}
    overall_raw = OrderedDict()  # {(setting, bias_metric): [value (each run)]}
    per_class_raw = OrderedDict()  # {(setting, bias_metric): [[value (each run)]]}

    for setting in all_settings:
        for metric_name in bias_metric_list:
            stat_csv = Path('./raw_data', paper, metric_name + '.csv')
            raw_json = Path('./raw_data', paper, metric_name + '_raw.json')

            # Get the stat value
            start_flag = False
            class_counter = 0
            with open(str(stat_csv), 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if 'rel_maxdiff(%)' in row:
                        continue  # Skip header
                    cur_setting, class_tag, max_diff, max_v, min_v, std_dev, mean_v, rel_maxdiff = row

                    if setting not in cur_setting:
                        if not start_flag:
                            continue  # Skip, not the setting we need
                        else:
                            break

                    if not start_flag:  # The first class in setting
                        start_flag = True
                    else:
                        class_counter += 1

                    if (class_counter < class_count) and ('class' in class_tag):
                        per_class_stat.setdefault((setting, metric_name), [])
                        per_class_stat[(setting, metric_name)].append({
                            'max_diff': max_diff,
                            'max': max_v,
                            'min': min_v,
                            'std_dev': std_dev,
                            'mean': mean_v,
                            'rel_maxdiff': rel_maxdiff
                        })
                    elif (class_counter == class_count) and ('overall' in class_tag):
                        overall_stat[(setting, metric_name)] = {
                            'max_diff': max_diff,
                            'max': max_v,
                            'min': min_v,
                            'std_dev': std_dev,
                            'mean': mean_v,
                            'rel_maxdiff': rel_maxdiff
                        }

                assert len(per_class_stat[(setting, metric_name)]) == class_count
            
            # Get the raw value
            with open(str(raw_json), 'r') as f:
                raw_values = json.load(f)
            for one_run in raw_values[setting]:
                overall_value = one_run[-1]
                per_class_value = one_run[:-1]
                
                overall_raw.setdefault((setting, metric_name), [])
                overall_raw[(setting, metric_name)].append(overall_value)
                per_class_raw.setdefault((setting, metric_name), [])
                per_class_raw[(setting, metric_name)].append(per_class_value)
    
    dump_parent = Path('./processed_data', paper)
    dump_parent.mkdir(exist_ok=True, parents=True)
    with open(str(Path(dump_parent, 'overall_stat.yaml')), 'w') as f:
        yaml.dump(overall_stat, f)
    with open(str(Path(dump_parent, 'per_class_stat.yaml')), 'w') as f:
        yaml.dump(per_class_stat, f)
    with open(str(Path(dump_parent, 'overall_raw.yaml')), 'w') as f:
        yaml.dump(overall_raw, f)
    with open(str(Path(dump_parent, 'per_class_raw.yaml')), 'w') as f:
        yaml.dump(per_class_raw, f)
