import argparse
import pandas as pd
import numpy as np
import statistics as st
import json
from pathlib import Path
import math
from functools import partial
import decimal


import decimal_utils


def read_result_file(file_name):
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


def add_values_to_dict(v_dict, main_v_dict):
    for key in list(v_dict.keys()):
        if key in list(main_v_dict.keys()):
            main_v_dict[key].append(v_dict[key])
        else:
            main_v_dict[key] = [v_dict[key]]


def main():
    # read the parameter argument parsing
    parser = argparse.ArgumentParser(description='Analyze the variance of multiple runs')
    parser.add_argument('result_dir', help="the path of the result folder")
    parser.add_argument('json_filename', help="the json filename")
    args = parser.parse_args()

    result_dir = args.result_dir
    json_filename = args.json_filename

    with open(json_filename) as config_file:
        configs = json.load(config_file)

    for config in configs:

        no_tries = config['no_tries']

        analysis_out = open(
            '%s/%s_%s_%s_%s_variance_analysis.csv' %
            (result_dir, config['network'], config['training_type'], config['dataset'], config['random_seed']),
            'w')

        main_values_dict = dict()

        for iTry in range(no_tries):
            if 'main_result_rel_path' in config:
                result_file = '{0}/{1}_{2}_{3}_{4}/{5}/'.format(result_dir,
                                                               config['network'],
                                                               config['training_type'],
                                                               config['dataset'],
                                                               config['random_seed'],
                                                               iTry) + config['main_result_rel_path']
            else:
                result_file = '%s/%s_%s_%s_%s/%d/record/cifar-s_%s/e1/test_result.txt' \
                            % (result_dir, config['network'], config['training_type'], config['dataset'], config['random_seed'], iTry, config['training_type'].replace('-', '_'))

            values_dict = read_result_file(result_file)

            add_values_to_dict(values_dict, main_values_dict)

            if 'bias_result_rel_path' in config:
                bias_result_file = '{0}/{1}_{2}_{3}_{4}/{5}/'.format(result_dir,
                                                                    config['network'],
                                                                    config['training_type'],
                                                                    config['dataset'],
                                                                    config['random_seed'],
                                                                    iTry) + config['bias_result_rel_path']
            else:
                bias_result_file = '%s/%s_%s_%s_%s/%d/record/cifar-s_%s/e1/bias_result.txt' \
                            % (result_dir, config['network'], config['training_type'], config['dataset'], config['random_seed'], iTry, config['training_type'].replace('-', '_'))
            bias_result_file = Path(bias_result_file)
            if bias_result_file.is_file():
                bias_dict = read_result_file(str(bias_result_file))
                add_values_to_dict(bias_dict, main_values_dict)

        analysis_out.write('Metric,max_diff,max,min,std_dev,mean,rel_maxdiff\n')
        for metric in list(main_values_dict.keys()):
            values = main_values_dict[metric]

            rd = partial(decimal_utils.round_significant_digit, digit=decimal_utils.GLOBAL_ROUND_DIGIT)
            rf = partial(decimal_utils.round_significant_format, digit=decimal_utils.GLOBAL_ROUND_DIGIT)
            rl = partial(decimal_utils.round_list, digit=decimal_utils.GLOBAL_ROUND_DIGIT)
            
            with decimal.localcontext() as ctx:
                ctx.traps[decimal.InvalidOperation] = 0 # Handle NaN issue

                values = rl(values)
                max_v = max(values)
                min_v = min(values)
                max_diff = max_v - min_v
                std_dev = st.stdev(values)
                mean = st.mean(values)
                if math.isclose(mean, 0):
                    rel_maxdiff = 0
                else:
                    rel_maxdiff = rd(max_diff) / rd(mean)

            
            result_list = [metric, rf(max_diff), rf(max_v), rf(min_v), rf(std_dev), rf(mean), rf(rel_maxdiff)]
            analysis_out.write(','.join(result_list))
            analysis_out.write('\n')

            '''
            max_v = max(values)
            min_v = min(values)
            max_diff = max_v - min_v
            std_dev = st.stdev(values)
            mean = st.mean(values)
            if math.isclose(mean, 0):
                rel_maxdiff = 0
            else: 
                rel_maxdiff = round(max_diff, 4) / round(mean, 4)

            analysis_out.write('%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (metric, max_diff, max_v, min_v, std_dev, mean, rel_maxdiff))

            '''

        analysis_out.close()


if __name__ == "__main__":
    main()
