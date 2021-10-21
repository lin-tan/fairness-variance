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


def get_largest_rel_maxdiff(class_stat_list):  #[{stat: value}]
    max_rel_diff = -1000
    max_idx = -1
    max_stat = {}
    for idx, stat in enumerate(class_stat_list):
        rel_maxdiff = float(stat['rel_maxdiff'])
        if (abs(rel_maxdiff) > max_rel_diff):
            max_rel_diff = abs(rel_maxdiff)
            max_idx = idx
            max_stat = stat
    return max_stat, max_idx


def get_largest_maxdiff(class_stat_list):  #[{stat: value}]
    max_max_diff = -1000
    max_idx = -1
    max_stat = {}
    for idx, stat in enumerate(class_stat_list):
        max_diff = float(stat['max_diff'])
        if (abs(max_diff) > max_max_diff):
            max_max_diff = abs(max_diff)
            max_idx = idx
            max_stat = stat
    return max_stat, max_idx


def main():

    with open('./configs.yaml', 'r') as f:
        configs = yaml.full_load(f)

    configs_t = OrderedDict()
    for k in setting_order:
        configs_t[k] = configs[k]
    configs = configs_t

    with open('./bias_metric.yaml', 'r') as f:
        bias_metric_list = yaml.full_load(f)

    output_p = Path('./table')
    output_p.mkdir(exist_ok=True, parents=True)

    # Baseline for all metrics
    '''
    print("Start: baseline for all metrics")
    with open(str(Path(output_p, 'over_baseline.tex')), 'w') as tex_f:
        for paper, config in sorted(configs.items()):
            baseline, _ = config['settings']
            baseline = baseline[0]

            p = Path('./processed_data', paper, 'overall_stat.yaml')
            with open(str(p), 'r') as f:
                overall_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}

            tex_f.write('\\midrule')
            tex_f.write('\n')

            first_line = True
            for bias_metric in bias_metric_list:
                output_line = []
                if first_line:
                    f_string = '\\multirow{{{0}}}'.format(len(bias_metric_list))
                    f_string += '{*}'
                    f_string += '{{\\rotatebox{{90}}{{{0}-Baseline}}}}'.format(prefix_abbr[paper])
                    output_line.append(f_string)

                    first_line = False
                else:
                    output_line.append('')
                output_line.append(metric_abbr[bias_metric])
                cur_stat = overall_stat[(baseline, bias_metric)]
                output_line += [cur_stat['mean'], cur_stat['max_diff'], cur_stat['std_dev'], cur_stat['rel_maxdiff']]

                write_tex_table_line(tex_f, output_line)
    '''

    # All experiments, all metrics, one table each, Overall
    print("Start: One table for all metrics overall")
    parent_p = Path('./table/overall_stat_for_all_exp')
    parent_p.mkdir(exist_ok=True, parents=True)
    for paper, config in configs.items():
        baseline, mitigation = config['settings']
        all_settings = baseline + mitigation

        p = Path('./processed_data', paper, 'overall_stat.yaml')
        with open(str(p), 'r') as f:
            overall_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}

        for setting in all_settings:
            tex_p = Path(parent_p, paper)
            tex_p.mkdir(exist_ok=True, parents=True)
            with open(str(Path(tex_p, setting_abbr[paper][setting] + '.tex')), 'w') as tex_f:
                header = [
                    '\\textbf{Exp}',
                    '\\textbf{Metric}',
                    '\\textbf{Mean}',
                    '\\textbf{Max}',
                    '\\textbf{Min}',
                    '\\textbf{MaxDiff}',
                    '\\textbf{STDEV}'
                ]
                write_tex_table_line(tex_f, header)
                tex_f.write('\\midrule \n')
                first_line = True
                for bias_metric in bias_metric_list:
                    stat = overall_stat[(setting, bias_metric)]

                    if first_line:
                        f_string = '\\multirow{{{0}}}'.format(len(bias_metric_list))
                        f_string += '{*}'
                        f_string += '{{\\rotatebox{{90}}{{{0}}}}}'.format(setting_abbr[paper][setting])
                        first_line = False
                    else:
                        f_string = ''

                    line = [
                        f_string,
                        ndi_mapper(metric_abbr[bias_metric]),
                        stat_threshold_formatter_percentage(stat['mean'], 0.1, 1),
                        stat_threshold_formatter_percentage(stat['max'], 0.1, 1),
                        stat_threshold_formatter_percentage(stat['min'], 0.1, 1),
                        stat_threshold_formatter_percentage(stat['max_diff'], thres=0.1, decimal_points=1),
                        stat_threshold_formatter_percentage(stat['std_dev'], 0.1, 1)
                    ]

                    write_tex_table_line(tex_f, line)

    # One Table for all the experiments
    print("Start: One big table for all metrics overall + per-class")
    tex_p = Path('./table/supp_material.tex')
    with open(str(tex_p), 'w') as tex_f:
        tex_f.write('\\toprule \n')
        line = [
            '\\multirow{2}{*}{\\textbf{\\Tech}}',
            '\\multirow{2}{*}{\\textbf{Metric}}',
            '\\multicolumn{5}{c}{\\textbf{Overall (\%)}}',
            '\\multicolumn{5}{c}{\\textbf{Per-class (\%)}}'
        ]
        write_tex_table_line(tex_f, line)
        tex_f.write('\\cmidrule(lr){3-7} \\cmidrule(lr){8-12} \n')
        header = ['\\textbf{Mean}', '\\textbf{Max}', '\\textbf{Min}', '\\textbf{MaxDiff}', '\\textbf{STDEV}']
        line = ['', ''] + header + header
        write_tex_table_line(tex_f, line)
        tex_f.write('\\midrule \n')

        for paper, config in configs.items():
            baseline, mitigation = config['settings']
            all_settings = baseline + mitigation

            p = Path('./processed_data', paper, 'overall_stat.yaml')
            with open(str(p), 'r') as f:
                overall_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}
            p = Path('./processed_data', paper, 'per_class_stat.yaml')
            with open(str(p), 'r') as f:
                per_class_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}

            for setting in all_settings:
                first_line = True
                for bias_metric in bias_metric_list:
                    stat = overall_stat[(setting, bias_metric)]
                    cur_stat, cur_class_idx = get_largest_maxdiff(per_class_stat[(setting, bias_metric)])

                    if first_line:
                        f_string = '\\multirow{{{0}}}'.format(len(bias_metric_list))
                        f_string += '{*}'
                        f_string += '{{\\rotatebox{{90}}{{{0}}}}}'.format(setting_abbr[paper][setting])
                        first_line = False
                    else:
                        f_string = ''

                    line = [
                        f_string,
                        ndi_mapper(metric_abbr[bias_metric]),
                        stat_threshold_formatter_percentage(stat['mean'], 0.1, 1),
                        stat_threshold_formatter_percentage(stat['max'], 0.1, 1),
                        stat_threshold_formatter_percentage(stat['min'], 0.1, 1),
                        stat_threshold_formatter_percentage(stat['max_diff'], thres=0.1, decimal_points=1),
                        stat_threshold_formatter_percentage(stat['std_dev'], 0.1, 1),
                        stat_threshold_formatter_percentage(cur_stat['mean'], 0.1, 1),
                        stat_threshold_formatter_percentage(cur_stat['max'], 0.1, 1),
                        stat_threshold_formatter_percentage(cur_stat['min'], 0.1, 1),
                        stat_threshold_formatter_percentage(cur_stat['max_diff'], thres=0.1, decimal_points=1),
                        stat_threshold_formatter_percentage(cur_stat['std_dev'], 0.1, 1)
                    ]

                    write_tex_table_line(tex_f, line)

                tex_f.write('\\midrule \n')

    # One Table for all the experiments
    print("Start: One big table for all metrics overall + per-class CSV")
    csv_p = Path('./table/supp_material.csv')
    with open(str(csv_p), 'w', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(['', '', 'Overall (%)', '', '', '', '', "Per-class (%)", '', '', '', ''])
        writer.writerow(
            ['Technique', 'Metric', 'Mean', 'Max', 'Min', 'MaxDiff', 'STDEV', 'Mean', 'Max', 'Min', 'MaxDiff', 'STDEV'])

        for paper, config in configs.items():
            baseline, mitigation = config['settings']
            all_settings = baseline + mitigation

            p = Path('./processed_data', paper, 'overall_stat.yaml')
            with open(str(p), 'r') as f:
                overall_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}
            p = Path('./processed_data', paper, 'per_class_stat.yaml')
            with open(str(p), 'r') as f:
                per_class_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}

            for setting in all_settings:
                first_line = True
                for bias_metric in bias_metric_list:
                    stat = overall_stat[(setting, bias_metric)]
                    cur_stat, cur_class_idx = get_largest_maxdiff(per_class_stat[(setting, bias_metric)])

                    if first_line:
                        f_string = setting_abbr[paper][setting]
                        first_line = False
                    else:
                        f_string = ''

                    line = [
                        f_string,
                        ndi_mapper_csv(metric_abbr[bias_metric]),
                        stat_threshold_formatter_percentage(stat['mean'], 0.1, 1),
                        stat_threshold_formatter_percentage(stat['max'], 0.1, 1),
                        stat_threshold_formatter_percentage(stat['min'], 0.1, 1),
                        stat_threshold_formatter_percentage(stat['max_diff'], thres=0.1, decimal_points=1),
                        stat_threshold_formatter_percentage(stat['std_dev'], 0.1, 1),
                        stat_threshold_formatter_percentage(cur_stat['mean'], 0.1, 1),
                        stat_threshold_formatter_percentage(cur_stat['max'], 0.1, 1),
                        stat_threshold_formatter_percentage(cur_stat['min'], 0.1, 1),
                        stat_threshold_formatter_percentage(cur_stat['max_diff'], thres=0.1, decimal_points=1),
                        stat_threshold_formatter_percentage(cur_stat['std_dev'], 0.1, 1)
                    ]

                    writer.writerow(line)

    # All experiments, all metrics, one table each, Per-class
    print("Start: One table for all metrics per-class")
    parent_p = Path('./table/per-class_stat_for_all_exp')
    parent_p.mkdir(exist_ok=True, parents=True)
    for paper, config in configs.items():
        baseline, mitigation = config['settings']
        all_settings = baseline + mitigation

        p = Path('./processed_data', paper, 'per_class_stat.yaml')
        with open(str(p), 'r') as f:
            per_class_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}

        for setting in all_settings:
            tex_p = Path(parent_p, paper)
            tex_p.mkdir(exist_ok=True, parents=True)
            with open(str(Path(tex_p, setting_abbr[paper][setting] + '.tex')), 'w') as tex_f:
                header = [
                    '\\textbf{Exp}',
                    '\\textbf{Metric}',
                    '\\textbf{Mean}',
                    '\\textbf{Max}',
                    '\\textbf{Min}',
                    '\\textbf{MaxDiff}',
                    '\\textbf{STDEV}'
                ]
                write_tex_table_line(tex_f, header)
                tex_f.write('\\midrule \n')
                first_line = True
                for bias_metric in bias_metric_list:
                    # stat = overall_stat[(setting, bias_metric)]
                    cur_stat, cur_class_idx = get_largest_maxdiff(per_class_stat[(setting, bias_metric)])

                    if first_line:
                        f_string = '\\multirow{{{0}}}'.format(len(bias_metric_list))
                        f_string += '{*}'
                        f_string += '{{\\rotatebox{{90}}{{{0}}}}}'.format(setting_abbr[paper][setting])
                        first_line = False
                    else:
                        f_string = ''

                    line = [
                        f_string,
                        ndi_mapper(metric_abbr[bias_metric]),
                        stat_threshold_formatter_percentage(cur_stat['mean'], 0.1, 1),
                        stat_threshold_formatter_percentage(cur_stat['max'], 0.1, 1),
                        stat_threshold_formatter_percentage(cur_stat['min'], 0.1, 1),
                        stat_threshold_formatter_percentage(cur_stat['max_diff'], thres=0.1, decimal_points=1),
                        stat_threshold_formatter_percentage(cur_stat['std_dev'], 0.1, 1)
                    ]

                    write_tex_table_line(tex_f, line)

    # Per-class for all metrics
    '''
    print('Start: per-class baseline for all mtrics')
    with open(str(Path(output_p, 'class_baseline.tex')), 'w') as tex_f:
        for paper, config in sorted(configs.items()):
            baseline, _ = config['settings']
            baseline = baseline[0]

            p = Path('./processed_data', paper, 'per_class_stat.yaml')
            with open(str(p), 'r') as f:
                per_class_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): [{stat: value}]}

            tex_f.write('\\midrule')
            tex_f.write('\n')

            first_line = True
            for bias_metric in bias_metric_list:
                output_line = []
                if first_line:
                    f_string = '\\multirow{{{0}}}'.format(len(bias_metric_list))
                    f_string += '{*}'
                    f_string += '{{\\rotatebox{{90}}{{{0}-Baseline}}}}'.format(prefix_abbr[paper])
                    output_line.append(f_string)

                    first_line = False
                else:
                    output_line.append('')
                output_line.append(metric_abbr[bias_metric])
                cur_stat, cur_class_idx = get_largest_rel_maxdiff(per_class_stat[(baseline, bias_metric)])
                output_line += [cur_stat['mean'], cur_stat['max_diff'], cur_stat['std_dev'], cur_stat['rel_maxdiff']]

                write_tex_table_line(tex_f, output_line)
    '''

    # Overall + Per-class baseline
    print('Start: Overall + Per-class')
    with open(str(Path(output_p, 'overall+class_baseline.tex')), 'w') as tex_f:
        for paper, config in sorted(configs.items()):
            baseline, _ = config['settings']
            baseline = baseline[0]

            p = Path('./processed_data', paper, 'per_class_stat.yaml')
            with open(str(p), 'r') as f:
                per_class_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): [{stat: value}]}
            p = Path('./processed_data', paper, 'overall_stat.yaml')
            with open(str(p), 'r') as f:
                overall_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}

            tex_f.write('\\midrule')
            tex_f.write('\n')

            first_line = True
            for bias_metric in bias_metric_list:
                output_line = []
                if first_line:
                    f_string = '\\multirow{{{0}}}'.format(len(bias_metric_list))
                    f_string += '{*}'
                    f_string += '{{\\rotatebox{{90}}{{{0}-Base}}}}'.format(prefix_abbr[paper])
                    output_line.append(f_string)

                    first_line = False
                else:
                    output_line.append('')
                output_line.append(metric_abbr[bias_metric])
                c_overall_stat = overall_stat[(baseline, bias_metric)]
                c_class_stat, cur_class_idx = get_largest_maxdiff(per_class_stat[(baseline, bias_metric)])
                output_line += [
                    stat_threshold_formatter_percentage(c_overall_stat['max_diff'], thres=0.1, decimal_points=1),
                    stat_threshold_formatter_percentage(c_overall_stat['std_dev'], thres=0.1, decimal_points=1),
                    #stat_formatter(c_overall_stat['rel_maxdiff']),
                    stat_threshold_formatter_percentage(c_class_stat['max_diff'], thres=0.1, decimal_points=1),
                    stat_threshold_formatter_percentage(c_class_stat['std_dev'], thres=0.1, decimal_points=1)
                    #stat_formatter(c_class_stat['rel_maxdiff'])
                ]

                write_tex_table_line(tex_f, output_line)

    # Overall + Per-class EO-TP
    for bias_metric in bias_metric_list:
        metric_key = metric_abbr[bias_metric]
        print('Start: Overall + Per-class ' + metric_key)
        with open(str(Path(output_p, 'overall+class_all_exp_' + metric_key + '.tex')), 'w') as tex_f:
            for paper, config in configs.items():
                baseline, mitigation = config['settings']

                p = Path('./processed_data', paper, 'per_class_stat.yaml')
                with open(str(p), 'r') as f:
                    per_class_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): [{stat: value}]}
                p = Path('./processed_data', paper, 'overall_stat.yaml')
                with open(str(p), 'r') as f:
                    overall_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}

                tex_f.write('\\midrule')
                tex_f.write('\n')

                for setting in baseline + mitigation:
                    #bias_metric = 'equality_of_odds_true_positive'
                    output_line = [setting_abbr[paper][setting]]

                    c_overall_stat = overall_stat[(setting, bias_metric)]
                    c_class_stat, cur_class_idx = get_largest_maxdiff(per_class_stat[(setting, bias_metric)])
                    #c_class_stat, cur_class_idx = get_largest_rel_maxdiff(per_class_stat[(setting, bias_metric)])
                    output_line += [
                        stat_threshold_formatter_percentage(c_overall_stat['max_diff'], thres=0.1, decimal_points=1),
                        stat_threshold_formatter_percentage(c_overall_stat['std_dev'], thres=0.1, decimal_points=1),
                        #stat_formatter(c_overall_stat['rel_maxdiff']),
                        stat_threshold_formatter_percentage(c_class_stat['max_diff'], thres=0.1, decimal_points=1),
                        stat_threshold_formatter_percentage(c_class_stat['std_dev'], thres=0.1, decimal_points=1)
                        #stat_formatter(c_class_stat['rel_maxdiff'])
                    ]

                    write_tex_table_line(tex_f, output_line)

    # Overall + Per-class all
    print("Start: Overall + Per-class for all")
    all_p = Path('./table/overall+per-class')
    all_p.mkdir(exist_ok=True, parents=True)
    for paper, config in configs.items():
        baseline, mitigation = config['settings']

        p = Path('./processed_data', paper, 'per_class_stat.yaml')
        with open(str(p), 'r') as f:
            per_class_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): [{stat: value}]}
        p = Path('./processed_data', paper, 'overall_stat.yaml')
        with open(str(p), 'r') as f:
            overall_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}

        paper_p = Path(all_p, paper)
        paper_p.mkdir(exist_ok=True, parents=True)
        for setting in baseline + mitigation:
            with open(str(Path(paper_p, setting_abbr[paper][setting] + '.tex')), 'w') as tex_f:
                first_line = True
                for bias_metric in bias_metric_list:
                    output_line = []
                    if first_line:
                        f_string = '\\multirow{{{0}}}'.format(len(bias_metric_list))
                        f_string += '{*}'
                        f_string += '{{\\rotatebox{{90}}{{{0}}}}}'.format(setting_abbr[paper][setting])
                        output_line.append(f_string)

                        first_line = False
                    else:
                        output_line.append('')
                    output_line.append(metric_abbr[bias_metric])
                    c_overall_stat = overall_stat[(setting, bias_metric)]
                    c_class_stat, cur_class_idx = get_largest_maxdiff(per_class_stat[(setting, bias_metric)])
                    output_line += [
                        stat_threshold_formatter_percentage(c_overall_stat['max_diff'], thres=0.1, decimal_points=1),
                        stat_threshold_formatter_percentage(c_overall_stat['std_dev'], thres=0.1, decimal_points=1),
                        #stat_formatter(c_overall_stat['rel_maxdiff']),
                        stat_threshold_formatter_percentage(c_class_stat['max_diff'], thres=0.1, decimal_points=1),
                        stat_threshold_formatter_percentage(c_class_stat['std_dev'], thres=0.1, decimal_points=1)
                        #stat_formatter(c_class_stat['rel_maxdiff'])
                    ]

                    write_tex_table_line(tex_f, output_line)

    # Correlation between metrics
    print("Start: Correlation")
    corr_p = Path(output_p, 'correlation')
    corr_p.mkdir(exist_ok=True, parents=True)
    for paper, config in configs.items():
        baseline, mitigation = config['settings']
        all_settings = baseline + mitigation

        p = Path('./processed_data', paper, 'overall_raw.yaml')
        with open(str(p), 'r') as f:
            overall_raw = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): [value (each run)]}
        p = Path('./processed_data', paper, 'overall_stat.yaml')
        with open(str(p), 'r') as f:
            overall_stat = yaml.load(f, Loader=yaml.Loader)  # {(setting, bias_metric): {stat: value}}

        cur_p = Path(corr_p, paper)
        cur_p.mkdir(exist_ok=True, parents=True)
        for setting in all_settings:
            metric_correlation = []  # [[]]
            for m1 in bias_metric_list:
                metric_correlation.append([])
                for m2 in bias_metric_list:
                    s1 = overall_raw[(setting, m1)]
                    s2 = overall_raw[(setting, m2)]
                    pearsonr, _ = scipy.stats.pearsonr(s1, s2)
                    metric_correlation[-1].append(stat_formatter(pearsonr))

            # Table for pearsonR value
            fn = (setting.rstrip('/') + '.tex').replace('/', '|')
            with open(str(Path(cur_p, fn)), 'w') as tex_f:
                tex_f.write('\\begin{tabular}{l|rrrrrrr|r}\n')
                tex_f.write('\\toprule\n')
                header = ['']
                for m in bias_metric_list:
                    header.append(metric_abbr[m])
                header.append('Diff(\\%)')
                write_tex_table_line(tex_f, header)
                tex_f.write('\\midrule\n')
                for m, r_values in zip(bias_metric_list, metric_correlation):
                    line = [metric_abbr[m]] + r_values + [stat_formatter(overall_stat[(setting, m)]['rel_maxdiff'])]
                    write_tex_table_line(tex_f, line)

                tex_f.write('\\bottomrule\n')
                tex_f.write('\\end{tabular}\n\n')

                # Table for rel_diff value
                '''
                tex_f.write('\\begin{tabular}{l|l}\n')
                tex_f.write('\\toprule\n')
                header = ['', 'Diff(\\%)']
                write_tex_table_line(tex_f, header)
                tex_f.write('\\midrule\n')
                for m in bias_metric_list:
                    row = [metric_abbr[m], overall_stat[(setting, m)]['rel_maxdiff']]
                    write_tex_table_line(tex_f, row)
                tex_f.write('\\bottomrule\n')
                tex_f.write('\\end{tabular}\n\n')
                '''

    # Bias: Baseline vs Mitigation [Metric fixed]
    print("Start: Baseline vs. mitigation")
    stat_p = Path('./stats/base_v_mitigation')
    rf = partial(decimal_utils.round_significant_format, digit=decimal_utils.GLOBAL_ROUND_DIGIT, scientific=False)
    for paper, config in configs.items():
        baseline, mitigation = config['settings']
        for bias_metric in bias_metric_list:
            with open(str(Path(stat_p, paper, metric_abbr[bias_metric] + '.yaml')), 'r') as f:
                result = yaml.load(f, Loader=yaml.Loader)  # {bias_metric: {mean, norm_var, mean_p, norm_var_p}}
            p = Path(output_p, 'bias_v_mitigation', paper)
            p.mkdir(parents=True, exist_ok=True)
            with open(str(Path(p, metric_abbr[bias_metric] + '.tex')), 'w') as tex_f:
                row = [
                    setting_abbr[paper][baseline[0]],
                    stat_formatter(result[baseline[0]]['mean'], 4),
                    '/',
                    stat_formatter(math.sqrt(result[baseline[0]]['norm_var']), 4),
                    '/'
                ]
                write_tex_table_line(tex_f, row)
                for _mitigation in mitigation:
                    row = [setting_abbr[paper][_mitigation]]
                    row += [
                        stat_formatter(result[_mitigation]['mean'], 4),
                        stat_significance(result[baseline[0]]['mean'],
                                          result[_mitigation]['mean'],
                                          result[_mitigation]['mean_p']),
                        stat_formatter(math.sqrt(result[_mitigation]['norm_var']), 4),
                        stat_significance(result[baseline[0]]['norm_var'],
                                          result[_mitigation]['norm_var'],
                                          result[_mitigation]['norm_var_p']),
                    ]
                    write_tex_table_line(tex_f, row)

    # Bias: Comparison between metrics [Mitigation fixed]
    print("Start: Comparison between bias metrics")
    stat_p = Path('./stats/bias_metric_comparison')
    rf = partial(decimal_utils.round_significant_format, digit=decimal_utils.GLOBAL_ROUND_DIGIT, scientific=False)
    for paper, config in configs.items():
        baseline, mitigation = config['settings']
        baseline = baseline[0]
        for _mitigation in mitigation:
            with open(str(Path(stat_p, paper, setting_abbr[paper][_mitigation] + '.yaml')), 'r') as f:
                result = yaml.load(f, Loader=yaml.Loader)
            p = Path(output_p, 'bias_metric_comparison', paper)
            p.mkdir(exist_ok=True, parents=True)
            with open(str(Path(p, setting_abbr[paper][_mitigation] + '.tex')), 'w') as tex_f:
                tex_f.write('\\toprule\n')
                write_tex_table_line(
                    tex_f,
                    ['', '\\multicolumn{3}{c}{\\textbf{Mean Bias}}', '\\multicolumn{3}{c}{\\textbf{Normalized Stdev}}'])
                tex_f.write('\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n')
                write_tex_table_line(tex_f,
                                     [
                                         '\\textbf{Metric}',
                                         '\\multicolumn{1}{c}{\\textbf{' + setting_abbr[paper][baseline] + '}}',
                                         '\\multicolumn{1}{c}{\\textbf{' + setting_abbr[paper][_mitigation] + '}}',
                                         '\\textbf{S}',
                                         '\\multicolumn{1}{c}{\\textbf{' + setting_abbr[paper][baseline] + '}}',
                                         '\\multicolumn{1}{c}{\\textbf{' + setting_abbr[paper][_mitigation] + '}}',
                                         '\\textbf{S}'
                                     ])
                tex_f.write('\\midrule\n')
                for bias_metric in bias_metric_list:
                    cur_stat = result[metric_abbr[bias_metric]]
                    row = [
                        metric_abbr[bias_metric],
                        stat_formatter(cur_stat['baseline_mean'], 4),
                        stat_formatter(cur_stat['mitigation_mean'], 4),
                        stat_significance(cur_stat['baseline_mean'], cur_stat['mitigation_mean'], cur_stat['mean_p']),
                        stat_formatter(math.sqrt(cur_stat['baseline_norm_var']), 4),
                        stat_formatter(math.sqrt(cur_stat['mitigation_norm_var']), 4),
                        stat_significance(cur_stat['baseline_norm_var'],
                                          cur_stat['mitigation_norm_var'],
                                          cur_stat['norm_var_p']),
                    ]
                    write_tex_table_line(tex_f, row)

    # FairALM vs Baseline
    paper = 'FairALM-CelebA'
    baseline = 'no-constraints/'
    _mitigation = 'fair-alm/'
    bias_metric = 'equality_of_odds_true_positive'
    with open(str(Path('./processed_data', paper, 'overall_stat.yaml')), 'r') as f:
        overall_stat = yaml.load(f, Loader=yaml.Loader)
    with open(str(Path(stat_p, paper, setting_abbr[paper][_mitigation] + '.yaml')), 'r') as f:
        result = yaml.load(f, Loader=yaml.Loader)
    cur_stat = result[metric_abbr[bias_metric]]

    p = Path('./table')
    p.mkdir(exist_ok=True)
    with open(str(Path(p, 'alm_result.tex')), 'w') as tex_f:
        tex_f.write('\\toprule \n')
        write_tex_table_line(tex_f, ['', '\\textbf{A-Base}', '\\textbf{A-ALM}', '\\textbf{Stat}'])
        tex_f.write('\\midrule \n')
        row = [
            'Mean',
            stat_formatter(cur_stat['baseline_mean'], 4),
            stat_formatter(cur_stat['mitigation_mean'], 4),
            stat_significance(cur_stat['baseline_mean'], cur_stat['mitigation_mean'], cur_stat['mean_p'])
        ]
        write_tex_table_line(tex_f, row)
        row = [
            'MaxDiff',
            stat_formatter(overall_stat[(baseline, bias_metric)]['max_diff'], 4),
            stat_formatter(overall_stat[(_mitigation, bias_metric)]['max_diff'], 4),
            '/'
        ]
        write_tex_table_line(tex_f, row)
        row = [
            'Diff(\%)',
            stat_formatter(overall_stat[(baseline, bias_metric)]['rel_maxdiff'], 4),
            stat_formatter(overall_stat[(_mitigation, bias_metric)]['rel_maxdiff'], 4),
            '/'
        ]
        write_tex_table_line(tex_f, row)
        row = [
            'STDEV(N)',
            stat_formatter(math.sqrt(cur_stat['baseline_norm_var']), 4),
            stat_formatter(math.sqrt(cur_stat['mitigation_norm_var']), 4),
            stat_significance(cur_stat['baseline_norm_var'], cur_stat['mitigation_norm_var'], cur_stat['norm_var_p'])
        ]
        write_tex_table_line(tex_f, row)
        tex_f.write('\\bottomrule \n')

    # S-GR vs Baseline
    paper = 'EffStrategies'
    baseline = 'baseline/'
    _mitigation = 'gradproj-adv/'
    bias_metric = 'bias_amplification'
    with open(str(Path('./processed_data', paper, 'overall_stat.yaml')), 'r') as f:
        overall_stat = yaml.load(f, Loader=yaml.Loader)
    with open(str(Path(stat_p, paper, setting_abbr[paper][_mitigation] + '.yaml')), 'r') as f:
        result = yaml.load(f, Loader=yaml.Loader)
    cur_stat = result[metric_abbr[bias_metric]]

    p = Path('./table')
    p.mkdir(exist_ok=True)
    with open(str(Path(p, 's-gr_result.tex')), 'w') as tex_f:
        tex_f.write('\\toprule \n')
        write_tex_table_line(tex_f, ['', '\\textbf{S-Base}', '\\textbf{S-GR}', '\\textbf{Stat}'])
        tex_f.write('\\midrule \n')
        row = [
            'Avg',
            stat_formatter_percentage(cur_stat['baseline_mean'], 1),
            stat_formatter_percentage(cur_stat['mitigation_mean'], 1),
            stat_significance(cur_stat['baseline_mean'], cur_stat['mitigation_mean'], cur_stat['mean_p'])
        ]
        write_tex_table_line(tex_f, row)
        row = [
            'RSD',
            stat_formatter_percentage(math.sqrt(cur_stat['baseline_norm_var']), 1),
            stat_formatter_percentage(math.sqrt(cur_stat['mitigation_norm_var']), 1),
            stat_significance(cur_stat['baseline_norm_var'], cur_stat['mitigation_norm_var'], cur_stat['norm_var_p'])
        ]
        write_tex_table_line(tex_f, row)
        row = [
            'MaxDiff (\\%)',
            stat_threshold_formatter_percentage(overall_stat[(baseline, bias_metric)]['max_diff']),
            stat_threshold_formatter_percentage(overall_stat[(_mitigation, bias_metric)]['max_diff']),
            '/'
        ]
        write_tex_table_line(tex_f, row)
        row = [
            'STDEV (\\%)',
            stat_threshold_formatter_percentage(overall_stat[(baseline, bias_metric)]['std_dev']),
            stat_threshold_formatter_percentage(overall_stat[(_mitigation, bias_metric)]['std_dev']),
            '/'
        ]
        write_tex_table_line(tex_f, row)
        #write_tex_table_line(tex_f, row)
        #row = [
        #    'Diff(\%)',
        #    stat_formatter(overall_stat[(baseline, bias_metric)]['rel_maxdiff'], 4),
        #    stat_formatter(overall_stat[(_mitigation, bias_metric)]['rel_maxdiff'], 4),
        #    '/'
        #]
        #write_tex_table_line(tex_f, row)

        tex_f.write('\\bottomrule \n')


if __name__ == '__main__':
    main()