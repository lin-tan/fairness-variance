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

    acc_stat: Dict[str, str] = {} # {Exp: stat}
    acc_var_stat: Dict[str, str] = {} # {Exp: stat}

    for paper, config in configs.items():
        #print("Start:", paper)
        baseline, mitigation = config['settings']
        baseline = baseline[0]
        perf_file = Path('./model_perf', config['performance_file'])
        with open(str(perf_file), 'r') as f:
            perf = yaml.load(f, Loader=yaml.Loader)

        for bias_metric in bias_metric_list:
            baseline_perf = perf[baseline]
            baseline_perf_mean = statistics.mean(baseline_perf)
            #baseline_perf_var = statistics.variance([e / baseline_perf_mean for e in baseline_perf])
            baseline_perf_var = statistics.variance(baseline_perf)
            with open(str(Path('./stats/base_v_mitigation', paper, metric_abbr[bias_metric] + '.yaml')), 'r') as f:
                stat_result = yaml.load(f, Loader=yaml.Loader)  # {setting: {mean, norm_var, mean_p, norm_var_p}}

            for _mitigation in mitigation:
                mitigation_perf = perf[_mitigation]
                mitigation_perf_mean = statistics.mean(mitigation_perf)
                mitigation_perf_var = statistics.variance([e / mitigation_perf_mean for e in mitigation_perf])
                #mitigation_perf_var = statistics.variance(mitigation_perf)

                _, perf_p = scipy.stats.mannwhitneyu(baseline_perf, mitigation_perf, alternative=('less' if baseline_perf_mean < mitigation_perf_mean else 'greater'))
                _, perf_var_p = scipy.stats.levene([e / baseline_perf_mean for e in baseline_perf], [e / mitigation_perf_mean for e in mitigation_perf])
                #_, perf_var_p = scipy.stats.levene(baseline_perf, mitigation_perf)
                
                exp_key = setting_abbr[paper][_mitigation]
                acc_stat[exp_key] = stat_significance(baseline_perf_mean, mitigation_perf_mean, perf_p)
                acc_var_stat[exp_key] = stat_significance(baseline_perf_var, mitigation_perf_var, perf_var_p)

    raw_result: Dict[Tuple[str, str], Tuple[str, str, str, str]] = {} # {(Exp, Metric): (Mean_Bias, Bias_Var, Acc, Acc_Var)}
    for paper, config in configs.items():
        baseline, mitigation = config['settings']
        for bias_metric in bias_metric_list:
            metric_key = metric_abbr[bias_metric]
            for _mitigation in mitigation:
                exp_key = setting_abbr[paper][_mitigation]
                raw_result[(exp_key, metric_key)] = (mean_bias_stat[exp_key, metric_key], variance_stat[exp_key, metric_key], acc_stat[exp_key], acc_var_stat[exp_key])
    
    bias_dec = set()
    cost_bias_var = set()
    cost_acc = set()
    cost_acc_var = set()
    for (exp_key, metric_key), (mean_bias, bias_var, acc, acc_var) in raw_result.items():
        if mean_bias == '-':
            bias_dec.add((exp_key, metric_key))
            if bias_var == '+':
                cost_bias_var.add((exp_key, metric_key))
            if acc == '-':
                cost_acc.add((exp_key, metric_key))
            if acc_var == '+':
                cost_acc_var.add((exp_key, metric_key))
    print("Bias Decrease:", len(bias_dec))
    print("Cost Bias Var:", len(cost_bias_var))
    print("Cost Acc:", len(cost_acc))
    print("Cost Acc Var:", len(cost_acc_var))
    print("Cost Bias Var + Acc:", len(cost_bias_var.intersection(cost_acc)))
    print("Cost Bias Var + Acc Var:", len(cost_bias_var.intersection(cost_acc_var)))
    print("Cost Acc + Acc Var:", len(cost_acc.intersection(cost_acc_var)))
    print("Cost Bias Var + Acc + Acc Var:", len(cost_bias_var.intersection(cost_acc, cost_acc_var)))
    print("Cost Any:", len(cost_bias_var.union(cost_acc, cost_acc_var)))

    #print("Cost Bias Var + Acc + Acc Var:", cost_bias_var.intersection(cost_acc, cost_acc_var))
    #print("No cost:", sorted(bias_dec.difference(cost_bias_var.union(cost_acc, cost_acc_var))))

    fig = plt.figure(figsize=(3, 3))
    ax = fig.subplots(1, 1)
    out = venn3_unweighted([cost_acc, cost_acc_var, cost_bias_var], set_labels=('Accuracy', 'Accuracy Variance', 'Bias Variance'), ax=ax)        
    for text in out.set_labels:
        text.set_fontsize(14)
    for x in range(len(out.subset_labels)):
        if out.subset_labels[x] is not None:
            out.subset_labels[x].set_fontsize(16)
    fig.savefig('./figures/cost_venn.png', bbox_inches='tight', dpi=600)
    plt.close(fig)

    def draw(stat, map, fn, title, green_label='-', red_label='+', annot=None, fmt=None):
        arr = np.empty([1, len(stat)])
        exp_name = []
        idx = -1
        for paper, config in configs.items():
            baseline, mitigation = config['settings']
            for setting in mitigation:
                idx += 1
                exp_key = setting_abbr[paper][setting]
                exp_name.append(exp_key)

                arr[0][idx] = map[stat[exp_key]]
        
        color = [(0, '#85c0f9'), (0.5, 'lightgray'), (1, '#f5793a')]
        cmap = LinearSegmentedColormap.from_list('custom', color, N=3)
        vmin = -1
        vmax = 1
        fig = plt.figure(figsize=(9, 0.2))
        ax = fig.subplots(1, 1)
        sns.heatmap(arr,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    ax=ax,
                    xticklabels=exp_name,
                    yticklabels=False,
                    linewidths=1.0,
                    linecolor='white',
                    annot=annot,
                    fmt=fmt,
                    cbar=False)
        ax.yaxis.set_tick_params(rotation=0)
        minus_patch = mpatches.Patch(color='#85c0f9', label=green_label)
        plus_patch = mpatches.Patch(color='#f5793a', label=red_label)
        ns_patch = mpatches.Patch(color='lightgray', label='NS')
        fig.legend(handles=[minus_patch, ns_patch, plus_patch],
                    loc='upper center',
                    ncol=3,
                    bbox_to_anchor=(0.5, 2.25),
                    frameon=False)
        p = Path('./figures')
        p.mkdir(exist_ok=True)
        ax.set_title(title, pad=16, fontsize=11)
        fig.savefig(str(Path(p, fn)), bbox_inches='tight', dpi=600)
        plt.close(fig)
    
    stat_map = {'+': -1, '-': 1, 'NS': 0}
    label = np.full([1, len(acc_stat)], '')
    label[0][1] = 'X'
    label[0][2] = 'X'
    label[0][3] = 'X'
    draw(acc_stat, stat_map, 'acc_heatmap.png', 'Model Accuracy', green_label='+', red_label='-', annot=label, fmt='')
    stat_map = {'+': 1, '-': -1, 'NS': 0}
    label = np.full([1, len(acc_var_stat)], '')
    label[0][1] = 'X'
    label[0][2] = 'X'
    draw(acc_var_stat, stat_map, 'acc_var_heatmap.png', 'Variance on Model Accuracy', green_label='-', red_label='+', annot=label, fmt='')


    #print(raw_result)


if __name__ == '__main__':
    main()
