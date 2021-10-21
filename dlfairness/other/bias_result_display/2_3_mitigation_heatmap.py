import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from typing import *
from pathlib import Path
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
from collections import OrderedDict

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


def convert_to_nparray(
        d: Dict[Tuple[str, str], Union[str, float]],
        stat=True,
        second_d: Dict[Tuple[str, str], str] = None) -> Tuple[np.ndarray, List[str], List[str], np.ndarray]:
    exp_count = len(set([e[0] for e in d.keys()]))
    metric_count = len(set([e[1] for e in d.keys()]))
    print('Shape:', exp_count, metric_count)

    exp_name = []
    metric_name = []
    arr = np.empty([exp_count, metric_count])
    label = np.empty([exp_count, metric_count], dtype=object)
    exp_idx = -1
    #huge_minus_ct = 0
    for paper, config in configs.items():
        _, mitigation = config['settings']
        for _mitigation in mitigation:
            exp_idx += 1
            exp_key = setting_abbr[paper][_mitigation]
            exp_name.append(exp_key)
            opt_target = {'C': 'DP', 'I': 'DP', 'S': 'BA', 'A': 'EOFP', 'N': 'DP'}
            for metric_idx, bias_metric in enumerate(bias_metric_list):
                metric_key = metric_abbr[bias_metric]
                if len(metric_name) < metric_count:
                    if metric_key == 'DI':
                        metric_name.append('$\\overline{\\rm DI}$')
                    else:
                        metric_name.append(metric_key)
                raw_value = d[(exp_key, metric_key)]
                if stat:
                    stat_map = {'+': 1, '-': -1, 'NS': 0}
                    mapped_value = stat_map[raw_value]
                else:
                    if raw_value >= 2.0:
                        mapped_value = 5
                        #huge_minus_ct += 1
                    elif 1.2 <= raw_value < 2.0:
                        mapped_value = 4
                    elif 0.8 <= raw_value < 1.2:
                        mapped_value = 3
                    elif 0.5 <= raw_value < 0.8:
                        mapped_value = 2
                    elif 0.2 <= raw_value < 0.5:
                        mapped_value = 1
                    elif -0.2 < raw_value < 0.2:
                        mapped_value = 0
                    elif -0.5 < raw_value <= -0.2:
                        mapped_value = -1
                    elif -0.8 < raw_value <= -0.5:
                        mapped_value = -2
                    elif -1.2 < raw_value <= -0.8:
                        mapped_value = -3
                    elif -2.0 < raw_value <= -1.2:
                        mapped_value = -4
                    elif raw_value <= -2.0:
                        mapped_value = -5

                arr[exp_idx][metric_idx] = mapped_value

                if second_d is None:
                    if metric_key == opt_target[exp_key[0]]:
                        label[exp_idx][metric_idx] = '*'
                    else:
                        label[exp_idx][metric_idx] = ''
                    if exp_key[0] in ['C', 'I']:
                        if metric_key == 'BA':
                            label[exp_idx][metric_idx] = 'o'
                else:
                    if (second_d[(exp_key, metric_key)] != '-') and (d[(exp_key, metric_key)] != 'NS'):
                        label[exp_idx][metric_idx] = 'X'
                    else:
                        label[exp_idx][metric_idx] = ''

    #print(huge_minus_ct)
    return arr, exp_name, metric_name, label


def main():
    mean_bias_stat: Dict[Tuple[str, str], str] = {}  # {(Exp, Metric): Stat}
    mean_bias_d: Dict[Tuple[str, str], float] = {}  # {(Exp, Metric): Cohen's d}
    variance_stat: Dict[Tuple[str, str], str] = {}

    for paper, config in configs.items():
        baseline, mitigation = config['settings']
        baseline = baseline[0]
        for bias_metric in bias_metric_list:
            metric_key = metric_abbr[bias_metric]
            for _mitigation in mitigation:
                exp_key = setting_abbr[paper][_mitigation]

                with open(Path('./stats/base_v_mitigation', paper, metric_key + '.yaml'), 'r') as f:
                    stat_result: Dict[str, Dict[str, float]] = yaml.load(f, Loader=yaml.Loader)

                mean_bias_stat[(exp_key, metric_key)] = stat_significance(stat_result[baseline]['mean'],
                                                                          stat_result[_mitigation]['mean'],
                                                                          stat_result[_mitigation]['mean_p'])
                if mean_bias_stat[(exp_key, metric_key)] == 'NS':
                    mean_bias_d[(exp_key, metric_key)] = 0
                else:
                    mean_bias_d[(exp_key, metric_key)] = stat_result[_mitigation]['mean_d']
                variance_stat[(exp_key, metric_key)] = stat_significance(stat_result[baseline]['norm_var'],
                                                                         stat_result[_mitigation]['norm_var'],
                                                                         stat_result[_mitigation]['norm_var_p'])

    p = Path('./figures')
    p.mkdir(exist_ok=True)

    def plot(fn, d, stat=True, annotation=True, second_d=None):
        arr, exp_key, metric_key, label = convert_to_nparray(d, stat=stat, second_d=second_d)
        arr = np.transpose(arr)
        label = np.transpose(label)
        fig = plt.figure(figsize=(9, 1.8))
        ax = fig.subplots(1, 1)

        if stat:
            #color = [(0, '#00e600'), (0.5, 'lightgray'), (1, '#e60000')] # Green ,Grey, Red
            color = [(0, '#85c0f9'), (0.5, 'lightgray'), (1, '#f5793a')] # Green ,Grey, Red
            cmap = LinearSegmentedColormap.from_list('custom', color, N=3)
            #myColors = ((0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
            #cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
            #cmap = sns.mpl_palette('RdBu', 3)
            vmin = -1
            vmax = 1
        else:
            # If Mitigation < Bias, we mark them as a "-", but Cohen's d will get a positive number
            # So, -5 here is red
            color = [(0, '#e84d00'), (0.1, '#f5793a'), (0.2, '#f29c6f'), (0.3, '#f0b799'), (0.4, '#f2d4c4'),
                     (0.5, 'lightgray'), (0.6, '#a6cff7'), (0.7, '#78b7f5'), (0.8, '#419af0'), (0.9, '#0081ff'),
                     (1, '#0061bf')] # Red -> Green (+ -> -)
            cmap = LinearSegmentedColormap.from_list('custom', color, N=11)
            vmin = None
            vmax = None
        if annotation:
            annot = label
            fmt = ''
        else:
            annot = None
            fmt = None
        sns.heatmap(arr,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    ax=ax,
                    annot=annot,
                    fmt=fmt,
                    xticklabels=exp_key,
                    yticklabels=metric_key,
                    linewidths=1.0,
                    linecolor='white',
                    cbar=False)
        ax.yaxis.set_tick_params(rotation=0)
        #ax.xaxis.set_tick_params(rotation=30)

        if stat:
            minus_patch = mpatches.Patch(color='#85c0f9', label='-')
            plus_patch = mpatches.Patch(color='#f5793a', label='+')
            ns_patch = mpatches.Patch(color='lightgray', label='NS')
            fig.legend(handles=[minus_patch, ns_patch, plus_patch],
                       loc='upper center',
                       ncol=3,
                       bbox_to_anchor=(0.5, 1.05),
                       frameon=False)
        else:
            '''
            phuge_patch = mpatches.Patch(color='#ad0000', label='Huge +')
            pvl_patch = mpatches.Patch(color='#ff0000', label='Very Large +')
            plarge_patch = mpatches.Patch(color='#ff8f8f', label='Large +')
            pmed_patch = mpatches.Patch(color='#ffc2c2', label='Medium +')
            psmall_patch = mpatches.Patch(color='#ffe6e6', label='Small +')
            ns_patch = mpatches.Patch(color='lightgray', label='NS')
            msmall_patch = mpatches.Patch(color='#e6ffe6', label='Small -')
            mmed_patch = mpatches.Patch(color='#c2ffc2', label='Medium -')
            mlarge_patch = mpatches.Patch(color='#8fff8f', label='Large -')
            mvl_patch = mpatches.Patch(color='#00ff00', label='Very Large -')
            mhuge_patch = mpatches.Patch(color='#00ad00', label='Huge -')
            '''           
            phuge_patch = mpatches.Patch(color='#e84d00', label='Huge +')
            pvl_patch = mpatches.Patch(color='#f5793a', label='Very Large +')
            plarge_patch = mpatches.Patch(color='#f29c6f', label='Large +')
            pmed_patch = mpatches.Patch(color='#f0b799', label='Medium +')
            psmall_patch = mpatches.Patch(color='#f2d4c4', label='Small +')
            ns_patch = mpatches.Patch(color='lightgray', label='NS')
            msmall_patch = mpatches.Patch(color='#a6cff7', label='Small -')
            mmed_patch = mpatches.Patch(color='#78b7f5', label='Medium -')
            mlarge_patch = mpatches.Patch(color='#419af0', label='Large -')
            mvl_patch = mpatches.Patch(color='#0081ff', label='Very Large -')
            mhuge_patch = mpatches.Patch(color='#0061bf', label='Huge -')
            fig.legend(handles=[
                mhuge_patch,
                mvl_patch,
                mlarge_patch,
                mmed_patch,
                msmall_patch,
                ns_patch,
                psmall_patch,
                pmed_patch,
                plarge_patch,
                pvl_patch,
                phuge_patch
            ],
                       bbox_to_anchor=(0.5, 1.17),
                       frameon=False,
                       loc='upper center',
                       handletextpad=0.5,
                       ncol=6)

        #fig.tight_layout(pad=0.5)
        fig.savefig(str(Path(p, fn)), bbox_inches='tight', dpi=600)
        plt.close(fig)

    # Draw mean bias with stat result (+/-/NS)
    plot('mean_bias_stat.png', mean_bias_stat, stat=True, annotation=True)
    # Draw mean bias with d value
    plot('mean_bias_d.png', mean_bias_d, stat=False, annotation=True)
    # Draw variance stat result
    plot('variance_stat.png', variance_stat, stat=True, annotation=True, second_d=mean_bias_stat)


if __name__ == '__main__':
    main()