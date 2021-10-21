# Baseline vs. others
# One model, one metric per graph

import matplotlib.pyplot as plt
import yaml

from pathlib import Path 

from abbrs import prefix_abbr, setting_abbr, metric_abbr, metric_display

needed_paper_list = ['EffStrategies', 'FairALM-CelebA', 'NIFR']
needed_setting_list = ['S-Base', 'S-GR', 'A-Base', 'A-ALM', 'A-L2', 'N-Base', 'N-Flow']


with open('./configs.yaml', 'r') as f:
    configs = yaml.full_load(f)

with open('./bias_metric.yaml', 'r') as f:
    bias_metric_list = yaml.full_load(f)

for paper, config in configs.items():
    if paper not in needed_paper_list:
        continue

    baseline, mitigation = config['settings']
    all_settings = baseline + mitigation

    p = Path('./processed_data', paper)
    with open(str(Path(p, 'overall_stat.yaml')), 'r') as f:
        overall_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}
    
    with open(str(Path(p, 'overall_raw.yaml')), 'r') as f:
        overall_raw = yaml.load(f, Loader=yaml.Loader) # {(setting, bias_metric): [value (each run)]}
    
    p = Path('./figures', paper)
    p.mkdir(exist_ok=True, parents=True)
    for bias_metric in bias_metric_list:
        print('Start:', paper, bias_metric)

        '''
        fig = plt.figure()
        ax = fig.subplots(1, 1)
        values = []
        labels = []
        for setting in all_settings:
            values.append(float(overall_stat[(setting, bias_metric)]['rel_maxdiff']))
            labels.append(setting)
        ax.bar(labels, values)
        '''

        
        fig = plt.figure(figsize=(7, 2))
        ax = fig.subplots(1, 1)
        values = []
        labels = []
        for idx, setting in enumerate(all_settings):
            if setting_abbr[paper][setting] not in needed_setting_list:
                continue

            values.append(overall_raw[(setting, bias_metric)])
            labels.append(setting_abbr[paper][setting])
        
        ax.boxplot(values, vert=False, labels=labels, widths=0.5, showfliers=True)
        ax.invert_yaxis()
        #ax.set_title(metric_display[bias_metric])
        

        #fig.suptitle(metric_display[bias_metric])
        #fig.tight_layout(pad=0.5)
        fig_fn = str(Path(p, metric_abbr[bias_metric] + '.png'))
        fig.savefig(fig_fn, bbox_inches='tight', dpi=600)
        
        plt.close(fig)