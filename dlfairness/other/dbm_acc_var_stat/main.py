import argparse
import sys
import scipy.stats
from pathlib import Path
import json
import yaml
import itertools
import csv
import os
import statistics
from functools import partial
import math
from collections import OrderedDict
from decimal import Decimal

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import decimal_utils

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--raw_result_dir', type=str)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()


def read_result_file(file_name):
    file_name = str(file_name)
    values_dict = dict()

    file = open(file_name, 'r')
    count = 0

    while True:
        count += 1

        line = file.readline()

        if not line:
            break

        comma_splited = line.strip().split(',')
        if len(comma_splited) > 1:
            first_name = ''
            for i in range(len(comma_splited)):
                cs = comma_splited[i]

                splited = cs.strip().split(':')
                if len(splited) == 1:
                    splited = cs.strip().split(';')

                if i == 0:
                    values_dict[splited[0].strip()] = float(splited[1])
                    first_name = splited[0].strip()
                else:
                    values_dict[first_name + ' ' + splited[0].strip()] = float(splited[1])
        else:
            splited = line.strip().split(':')
            values_dict[splited[0].strip()] = float(splited[1])

    file.close()

    return values_dict


def add_result(raw_result, exp_name, gray_acc, color_acc, bias):
    if exp_name not in raw_result:
        tdict = {'gray_acc': [], "color_acc": [], 'bias': []}
        raw_result[exp_name] = tdict
    raw_result[exp_name]['gray_acc'].append(gray_acc)
    raw_result[exp_name]['color_acc'].append(color_acc)
    raw_result[exp_name]['bias'].append(bias)

    return raw_result


def get_relmaxdiff(maxdiff, mean):
    rd = partial(decimal_utils.round_significant_digit, digit=decimal_utils.GLOBAL_ROUND_DIGIT)
    if math.isclose(mean, 0):
        rel_maxdiff = 0
    else:
        rel_maxdiff = rd(maxdiff) / rd(mean)

    return rel_maxdiff

def cohen_d(mean_1, mean_2, stdev_1, stdev_2):
    s_pooled = Decimal(math.sqrt((stdev_1 ** 2 + stdev_2 ** 2) / 2))
    d = (mean_1 - mean_2) / s_pooled

    return d


def main():
    with open(args.config, 'r') as f:
        config_json = json.load(f)

    # Read result data
    raw_result = {}  # {Exp_Name: {Gray_acc: [], Color_acc: [], Bias: []}}
    for config in config_json:
        for no_try in range(config['no_tries']):
            #print(no_try)
            exp_result_path = Path(
                args.raw_result_dir,
                "{0}_{1}_{2}_{3}/{4}".format(config['network'],
                                             config['training_type'],
                                             config['dataset'],
                                             config['random_seed'],
                                             str(no_try)))

            result_path = Path(exp_result_path, 'record', 'cifar-s_' + config['training_type'].replace('-', '_'), 'e1')
            acc_result_path = Path(result_path, 'test_result.txt')
            bias_result_path = Path(result_path, 'bias_result.txt')

            acc_result = read_result_file(acc_result_path)
            bias_result = read_result_file(bias_result_path)

            if config['training_type'] == 'baseline':
                exp_name = 'baseline'
                gray_acc = acc_result['Test on gray images accuracy']
                color_acc = acc_result['Test on color images accuracy']
                bias = bias_result['Mean bias']
                raw_result = add_result(raw_result, exp_name, gray_acc, color_acc, bias)
            elif config['training_type'] == 'sampling':
                exp_name = 'oversampling'
                gray_acc = acc_result['Test on gray images accuracy']
                color_acc = acc_result['Test on color images accuracy']
                bias = bias_result['Mean bias']
                raw_result = add_result(raw_result, exp_name, gray_acc, color_acc, bias)
            elif config['training_type'] == 'uniconf-adv':  # Adv uniform
                exp_name = 'adv_uniform'
                gray_acc = acc_result['Test on gray images accuracy']
                color_acc = acc_result['Test on color images accuracy']
                bias = bias_result['Mean bias']
                raw_result = add_result(raw_result, exp_name, gray_acc, color_acc, bias)
            elif config['training_type'] == 'gradproj-adv':
                exp_name = 'adv_gradproj'
                gray_acc = acc_result['Test on gray images accuracy']
                color_acc = acc_result['Test on color images accuracy']
                bias = bias_result['Mean bias']
                raw_result = add_result(raw_result, exp_name, gray_acc, color_acc, bias)
            elif config['training_type'] == 'domain-discriminative':
                exp_name = 'domain_discr-1_sum'
                gray_acc = acc_result['Test on gray images accuracy sum prob without prior shift']
                color_acc = acc_result['Test on color images accuracy sum prob without prior shift']
                bias = bias_result['Sum prob w/o prior shift (result 1)']
                raw_result = add_result(raw_result, exp_name, gray_acc, color_acc, bias)

                exp_name = 'domain_discr-2_max_shift'
                gray_acc = acc_result['Test on gray images accuracy max prob with prior shift']
                color_acc = acc_result['Test on color images accuracy max prob with prior shift']
                bias = bias_result['Max prob w/ prior shift (result 2)']
                raw_result = add_result(raw_result, exp_name, gray_acc, color_acc, bias)

                exp_name = 'domain_discr-3_sum_shift'
                gray_acc = acc_result['Test on gray images accuracy sum prob with prior shift']
                color_acc = acc_result['Test on color images accuracy sum prob with prior shift']
                bias = bias_result['Sum prob w/ prior shift (result 3)']
                raw_result = add_result(raw_result, exp_name, gray_acc, color_acc, bias)

                exp_name = 'domain_discr-4_rba'
                gray_acc = acc_result['RBA Gray Accuracy'] * 100.0  # RBA script doesn't write the accuracy in percentage
                color_acc = acc_result['RBA Color Accuracy'] * 100.0
                bias = bias_result['RBA bias']
                raw_result = add_result(raw_result, exp_name, gray_acc, color_acc, bias)
            elif config['training_type'] == 'domain-independent':
                exp_name = 'domain_indep-1_cond'
                gray_acc = acc_result['Test on gray images accuracy conditional']
                color_acc = acc_result['Test on color images accuracy conditional']
                bias = bias_result['Bias conditional (result 1)']
                raw_result = add_result(raw_result, exp_name, gray_acc, color_acc, bias)

                exp_name = 'domain_indep-2_sum'
                gray_acc = acc_result['Test on gray images accuracy sum out']
                color_acc = acc_result['Test on color images accuracy sum out']
                bias = bias_result['Bias sum (result 2)']
                raw_result = add_result(raw_result, exp_name, gray_acc, color_acc, bias)

    raw_result_yaml = Path(args.output_dir, 'all_acc_bias_raw.yaml')
    with open(str(raw_result_yaml), 'w') as f:
        yaml.safe_dump(raw_result, f)

    # U-Test and Levene's test for pairs
    stat_result = OrderedDict()
    # {(exp1, exp2): [[gray_mean_1, gray_mean_2, gray_u_p, color_mean_1, color_mean_2, color_u_p, bias_mean_1, bias_mean_2, bias_u_p],
    #                 [gray_normvar_1, gray_normvar_2, gray_l_p, color_normvar_1, color_normvar_2, color_l_p, bias_normvar_1, bias_normvar_2, bias_l_p],
    #                 [gray_maxdiff_1, gray_maxdiff_2, , color_maxdiff_1, color_maxdiff_2, , bias_maxdiff_1, bias_maxdiff_2],
    #                 [gray_rel_maxdiff_1, gray_rel_maxdiff_2, , color_rel_maxdiff_1, color_rel_maxdiff_2, , bias_rel_maxdiff_1, bias_rel_maxdiff_2]]}
    for exp1, exp2 in itertools.product(sorted(list(raw_result.keys())), sorted(list(raw_result.keys()))):
        if exp1 == exp2:
            continue
        if exp1 != 'baseline':
            continue

        gray_acc_1 = raw_result[exp1]['gray_acc']
        gray_acc_2 = raw_result[exp2]['gray_acc']
        color_acc_1 = raw_result[exp1]['color_acc']
        color_acc_2 = raw_result[exp2]['color_acc']
        bias_1 = raw_result[exp1]['bias']
        bias_2 = raw_result[exp2]['bias']


        rl = partial(decimal_utils.round_list, digit=decimal_utils.GLOBAL_ROUND_DIGIT)
        gray_mean_1 = statistics.mean(rl(gray_acc_1))
        gray_mean_2 = statistics.mean(rl(gray_acc_2))
        color_mean_1 = statistics.mean(rl(color_acc_1))
        color_mean_2 = statistics.mean(rl(color_acc_2))
        bias_mean_1 = statistics.mean(rl(bias_1))
        bias_mean_2 = statistics.mean(rl(bias_2))

        _, gray_u_p = scipy.stats.mannwhitneyu(gray_acc_1, gray_acc_2, alternative=('less' if gray_mean_1 < gray_mean_2 else 'greater'))  # <0.05: have different acc
        _, color_u_p = scipy.stats.mannwhitneyu(color_acc_1, color_acc_2, alternative=('less' if color_mean_1 < color_mean_2 else 'greater'))  # <0.05: have different acc
        _, bias_u_p = scipy.stats.mannwhitneyu(bias_1, bias_2, alternative=('less' if bias_mean_1 < bias_mean_2 else 'greater'))  # <0.05: have different bias
        #_, gray_l_p = scipy.stats.levene(gray_acc_1, gray_acc_2) # <0.05: have different var
        #_, color_l_p = scipy.stats.levene(color_acc_1, color_acc_2)
        #_, bias_l_p = scipy.stats.levene(bias_1, bias_2)
        _, gray_l_p = scipy.stats.levene([e / float(gray_mean_1) for e in gray_acc_1], [e / float(gray_mean_2) for e in gray_acc_2])
        _, color_l_p = scipy.stats.levene([e / float(color_mean_1) for e in color_acc_1], [e / float(color_mean_2) for e in color_acc_2])
        _, bias_l_p = scipy.stats.levene([e / float(bias_mean_1) for e in bias_1], [e / float(bias_mean_2) for e in bias_2])

        gray_normvar_1 = statistics.variance(rl([e / float(gray_mean_1) for e in gray_acc_1]))
        gray_normvar_2 = statistics.variance(rl([e / float(gray_mean_2) for e in gray_acc_2]))
        color_normvar_1 = statistics.variance(rl([e / float(color_mean_1) for e in color_acc_1]))
        color_normvar_2 = statistics.variance(rl([e / float(color_mean_2) for e in color_acc_2]))
        bias_normvar_1 = statistics.variance(rl([e / float(bias_mean_1) for e in bias_1]))
        bias_normvar_2 = statistics.variance(rl([e / float(bias_mean_2) for e in bias_2]))

        gray_stdev_1 = statistics.stdev(rl(gray_acc_1))
        gray_stdev_2 = statistics.stdev(rl(gray_acc_2))
        color_stdev_1 = statistics.stdev(rl(color_acc_1))
        color_stdev_2 = statistics.stdev(rl(color_acc_2))
        bias_stdev_1 = statistics.stdev(rl(bias_1))
        bias_stdev_2 = statistics.stdev(rl(bias_2))

        gray_d = cohen_d(gray_mean_1, gray_mean_2, gray_stdev_1, gray_stdev_2)
        color_d = cohen_d(color_mean_1, color_mean_2, color_stdev_1, color_stdev_2)
        bias_d = cohen_d(bias_mean_1, bias_mean_2, bias_stdev_1, bias_stdev_2)

        gray_maxdiff_1 = max(rl(gray_acc_1)) - min(rl(gray_acc_1))
        gray_maxdiff_2 = max(rl(gray_acc_2)) - min(rl(gray_acc_2))
        color_maxdiff_1 = max(rl(color_acc_1)) - min(rl(color_acc_1))
        color_maxdiff_2 = max(rl(color_acc_2)) - min(rl(color_acc_2))
        bias_maxdiff_1 = max(rl(bias_1)) - min(rl(bias_1))
        bias_maxdiff_2 = max(rl(bias_2)) - min(rl(bias_2))

        gray_rel_maxdiff_1 = get_relmaxdiff(gray_maxdiff_1, gray_mean_1)
        gray_rel_maxdiff_2 = get_relmaxdiff(gray_maxdiff_2, gray_mean_2)
        color_rel_maxdiff_1 = get_relmaxdiff(color_maxdiff_1, color_mean_1)
        color_rel_maxdiff_2 = get_relmaxdiff(color_maxdiff_2, color_mean_2)
        bias_rel_maxdiff_1 = get_relmaxdiff(bias_maxdiff_1, bias_mean_1)
        bias_rel_maxdiff_2 = get_relmaxdiff(bias_maxdiff_2, bias_mean_2)

        tlist = [
            [
            gray_mean_1,
            gray_mean_2,
            gray_u_p,
            gray_d,
            color_mean_1,
            color_mean_2,
            color_u_p,
            color_d,
            bias_mean_1,
            bias_mean_2,
            bias_u_p,
            bias_d
            ],
            [
                gray_normvar_1,
                gray_normvar_2,
                gray_l_p,
                '',
                color_normvar_1,
                color_normvar_2,
                color_l_p,
                '',
                bias_normvar_1,
                bias_normvar_2,
                bias_l_p,
                ''
            ],
            [
                gray_maxdiff_1,
                gray_maxdiff_2,
                '',
                '',
                color_maxdiff_1,
                color_maxdiff_2,
                '',
                '',
                bias_maxdiff_1,
                bias_maxdiff_2,
                '',
                ''
            ],
            [
                gray_rel_maxdiff_1,
                gray_rel_maxdiff_2,
                '',
                '',
                color_rel_maxdiff_1,
                color_rel_maxdiff_2,
                '',
                '',
                bias_rel_maxdiff_1,
                bias_rel_maxdiff_2,
                '',
                ''
            ]
        ] # yapf: disable

        stat_result[(exp1, exp2)] = tlist

    csv_file = Path(args.output_dir, 'acc_bias_pairwise_variance_analysis.csv')
    with open(str(csv_file), 'w', newline='') as f:
        w = csv.writer(f)
        for key, value in stat_result.items():
            exp1, exp2 = key
            header = [
                exp1 + '||' + exp2,
                'gray_1',
                'gray_2',
                'gray_p',
                'gray_d',
                'color_1',
                'color_2',
                'color_p',
                'color_d',
                'bias_1',
                'bias_2',
                'bias_p',
                'bias_d'
            ]
            w.writerow(header)
            rf = partial(decimal_utils.round_significant_format, digit=decimal_utils.GLOBAL_ROUND_DIGIT)
            remarks = ['Mean', 'Norm_Var', 'Maxdiff', 'Rel_maxdiff']
            for i in range(4):
                w.writerow([remarks[i]] + list(map(lambda e: rf(e) if not isinstance(e, str) else e, value[i])))

            w.writerow([])


if __name__ == "__main__":
    main()