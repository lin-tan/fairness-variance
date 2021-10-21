import yaml
import scipy.stats
import statistics

from pathlib import Path
from functools import partial

from yaml import loader

import decimal_utils
from abbrs import prefix_abbr, setting_abbr, metric_abbr
from utils import write_tex_table_line, stat_significance, stat_threshold_formatter, stat_formatter, cohen_d


def main():

    with open('./configs.yaml', 'r') as f:
        configs = yaml.full_load(f)

    with open('./bias_metric.yaml', 'r') as f:
        bias_metric_list = yaml.full_load(f)

    for paper, config in configs.items():
        print("Start:", paper)
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

            p = Path('./table', 'mitigation_on_model_perf', paper)
            p.mkdir(exist_ok=True, parents=True)
            with open(str(Path(p, metric_abbr[bias_metric] + '.tex')), 'w') as tex_f:
                header = [
                    '\\textbf{Exp}',
                    '\\textbf{Acc}',
                    '\\textbf{Cohen\'s d}',
                    '\\textbf{Acc\_Var}'
                ]
                write_tex_table_line(tex_f, header)
                tex_f.write('\\midrule \n')

                for _mitigation in mitigation:
                    mitigation_perf = perf[_mitigation]
                    mitigation_perf_mean = statistics.mean(mitigation_perf)
                    mitigation_perf_var = statistics.variance([e / mitigation_perf_mean for e in mitigation_perf])
                    #mitigation_perf_var = statistics.variance(mitigation_perf)

                    _, perf_p = scipy.stats.mannwhitneyu(baseline_perf, mitigation_perf, alternative=('less' if baseline_perf_mean < mitigation_perf_mean else 'greater'))
                    _, perf_var_p = scipy.stats.levene([e / baseline_perf_mean for e in baseline_perf], [e / mitigation_perf_mean for e in mitigation_perf])
                    #_, perf_var_p = scipy.stats.levene(baseline_perf, mitigation_perf)
                    perf_d = cohen_d(mitigation_perf, baseline_perf)
                    
                    line = [
                        setting_abbr[paper][_mitigation] + '\t',
                        stat_significance(baseline_perf_mean, mitigation_perf_mean, perf_p),
                        stat_formatter(perf_d, 2),
                        stat_significance(baseline_perf_var, mitigation_perf_var, perf_var_p)
                    ]
                    write_tex_table_line(tex_f, line)


if __name__ == '__main__':
    main()