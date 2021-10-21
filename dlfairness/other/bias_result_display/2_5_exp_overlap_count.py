import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statistics
import scipy.stats

from matplotlib_venn import venn3, venn3_unweighted
from typing import *
from pathlib import Path
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
from collections import OrderedDict

from yaml.loader import Loader

from abbrs import prefix_abbr, setting_abbr, metric_abbr, setting_order
from utils import stat_significance

with open('./configs.yaml', 'r') as f:
    configs = yaml.load(f, Loader=yaml.Loader)

configs_t = OrderedDict()
for k in setting_order:
    configs_t[k] = configs[k]
configs = configs_t

with open('./bias_metric.yaml', 'r') as f:
    bias_metric_list = yaml.load(f, Loader=yaml.Loader)

overlap_count = 0
overlap_bad_count = 0

for paper, config in configs.items():
    baseline, mitigation = config['settings']
    baseline = baseline[0]

    with open(str(Path('./processed_data', paper, 'overall_stat.yaml')), 'r') as f:
        overall_stat = yaml.load(f, Loader=yaml.Loader)
    
    for bias_metric in bias_metric_list:
        baseline_min = float(overall_stat[(baseline, bias_metric)]['min'])
        baseline_max = float(overall_stat[(baseline, bias_metric)]['max'])

        metric_key = metric_abbr[bias_metric]
        with open(str(Path('./stats/base_v_mitigation', paper, metric_key + '.yaml')), 'r') as f:
            stat_result = yaml.load(f, Loader=yaml.Loader)
        
        for _mitigation in mitigation:
            exp_key = setting_abbr[paper][_mitigation]
            sig = stat_significance(stat_result[baseline]['mean'], stat_result[_mitigation]['mean'], stat_result[_mitigation]['mean_p'])

            mitigation_min = float(overall_stat[(_mitigation, bias_metric)]['min'])
            mitigation_max = float(overall_stat[(_mitigation, bias_metric)]['max'])

            if (baseline_min < mitigation_min < baseline_max) or (mitigation_min < baseline_min < mitigation_max):
                overlap_count += 1
                print(exp_key, metric_key, sig)
                if sig == '+':
                    overlap_bad_count += 1
                    #print(exp_key, metric_key)    
print("Overlap:", overlap_count)
print("Overlap bad:", overlap_bad_count)


