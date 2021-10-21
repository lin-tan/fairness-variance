import yaml
import scipy.stats
import csv

from pathlib import Path
from functools import partial
from collections import OrderedDict

from yaml import loader
import math

import decimal_utils
from abbrs import prefix_abbr, setting_abbr, metric_abbr, setting_order
from utils import write_tex_table_line, stat_significance, stat_threshold_formatter, stat_formatter, stat_threshold_formatter_percentage, stat_formatter_percentage, ndi_mapper, ndi_mapper_csv


def main():
    with open('./configs.yaml', 'r') as f:
        configs = yaml.full_load(f)

    configs_t = OrderedDict()
    for k in setting_order:
        configs_t[k] = configs[k]
    configs = configs_t

    with open('./bias_metric.yaml', 'r') as f:
        bias_metric_list = yaml.full_load(f)

    
    csv_file = {} # {(mitigation, metric): []}
    for paper, config in configs.items():
        baseline, mitigation = config['settings']

        p = Path('./processed_data', paper, 'overall_raw.yaml')
        with open(str(p), 'r') as f:
            overall_raw = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): [value (each run)]}
        p = Path('./processed_data', paper, 'overall_stat.yaml')
        with open(str(p), 'r') as f:
            overall_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}

        p = Path('./processed_data', paper, 'per_class_stat.yaml')
        with open(str(p), 'r') as f:
            per_class_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}
        p = Path('./processed_data', paper, 'per_class_raw.yaml')
        with open(str(p), 'r') as f:
            per_class_raw = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): [[value (each run)]]}

        # Overall for 16 runs
        p = Path('./result_submit/overall')
        p.mkdir(exist_ok=True, parents=True)
        for setting in baseline + mitigation:
            exp_key = setting_abbr[paper][setting]

            dump = {}
            for bias_metric in bias_metric_list:
                metric_key = ndi_mapper_csv(metric_abbr[bias_metric])
                dump[metric_key] = overall_raw[(setting, bias_metric)]
            with open(str(Path(p, exp_key + '.yaml')), 'w') as f:
                yaml.safe_dump(dump, f, sort_keys=False)

        # Per-class for 16 runs
        p = Path('./result_submit/per_class')
        p.mkdir(exist_ok=True, parents=True)
        for setting in baseline + mitigation:
            exp_key = setting_abbr[paper][setting]
            pp = Path(p, exp_key)
            pp.mkdir(exist_ok=True, parents=True)

            dump = {}
            for idx in range(16):
                for bias_metric in bias_metric_list:
                    metric_key = ndi_mapper_csv(metric_abbr[bias_metric])
                    dump[metric_key] = per_class_raw[(setting, bias_metric)][idx]
                with open(str(Path(pp, 'run_' + str(idx).zfill(2) + '.yaml')), 'w') as f:
                    yaml.safe_dump(dump, f, sort_keys=False)



        stat_p = Path('./stats/base_v_mitigation')
        for bias_metric in bias_metric_list:
            with open(str(Path(stat_p, paper, metric_abbr[bias_metric] + '.yaml')), 'r') as f:
                result = yaml.load(f, Loader=yaml.Loader)  # {setting: {mean, norm_var, mean_p, norm_var_p}}

            baseline_bias = stat_formatter(result[baseline[0]]['mean'], 6)
            baseline_var = stat_formatter(result[baseline[0]]['norm_var'], 6)
            
            for _mitigation in mitigation:
                mitigation_bias = stat_formatter(result[_mitigation]['mean'], 6)
                mitigation_var = stat_formatter(result[_mitigation]['norm_var'], 6)
                bias_p = stat_formatter(result[_mitigation]['mean_p'], 5)
                sig = stat_significance(result[baseline[0]]['mean'], result[_mitigation]['mean'], result[_mitigation]['mean_p'])
                bias_p += ' (' + sig + ')'
                
                if sig != 'NS':
                    bias_d = stat_formatter(abs(result[_mitigation]['mean_d']), 2)
                else:
                    bias_d = 'N/A'
                
                var_p = stat_formatter(result[_mitigation]['norm_var_p'], 5)
                sig = stat_significance(result[baseline[0]]['norm_var'], result[_mitigation]['norm_var'], result[_mitigation]['norm_var_p'])
                var_p += ' (' + sig + ')'

                csv_file[(setting_abbr[paper][_mitigation], ndi_mapper_csv(metric_abbr[bias_metric]))] = [baseline_bias, mitigation_bias, bias_p, bias_d, baseline_var, mitigation_var, var_p]
            
    # Statisitcal tests
    with open('./result_submit/stat_tests.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'Mitigation',
            'Metric',
            'Baseline Bias',
            'Mitigation Bias',
            'p-value (significance)',
            'Cohen\'s d',
            'Baseline Variance (N)',
            'Mitigation Variance (N)',
            'p-value (significance)'
        ]
        writer.writerow(header)
        for paper, config in configs.items():
            _, mitigation = config['settings']
            for _mitigation in mitigation:
                exp_key = setting_abbr[paper][_mitigation]
                first_line = True
                for bias_metric in bias_metric_list:
                    metric_key = ndi_mapper_csv(metric_abbr[bias_metric])

                    if first_line:
                        row = [exp_key]
                        first_line = False
                    else:
                        row = ['']
                    
                    row.append(metric_key)
                    row += csv_file[(exp_key, metric_key)]

                    writer.writerow(row)

if __name__ == '__main__':
    main()
