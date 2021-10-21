import json
import pickle
import numpy as np
import argparse
from pathlib import Path
from scipy.special import softmax
import csv
import statistics
import sys
import os
from functools import partial
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import decimal_utils

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--raw_result_dir', type=str)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()


def predict(color_score, gray_score, training_type, eval_mode=1):
    if training_type == 'domain-discriminative':
        score = np.concatenate([gray_score, color_score], axis=0)
        if eval_mode == 1:  # Sum without prior shift (result 1)
            probs = softmax(score, axis=1)
            predicted_classes = np.argmax(probs[:, :10] + probs[:, 10:], axis=1)
        elif eval_mode == 2:  # Max prob with prior shift (result 2)
            prior_shift_weight = [1 / 5 if i % 2 == 0 else 1 / 95
                                  for i in range(10)] + [1 / 95 if i % 2 == 0 else 1 / 5 for i in range(10)]
            probs = softmax(score, axis=1) * prior_shift_weight
            predicted_classes = np.argmax(np.stack((probs[:, :10], probs[:, 10:])).max(axis=0), axis=1)
        elif eval_mode == 3:  # Sum prob with prior shift
            prior_shift_weight = [1 / 5 if i % 2 == 0 else 1 / 95
                                  for i in range(10)] + [1 / 95 if i % 2 == 0 else 1 / 5 for i in range(10)]
            probs = softmax(score, axis=1) * prior_shift_weight
            predicted_classes = np.argmax(probs[:, :10] + probs[:, 10:], axis=1)
    elif training_type == 'domain-independent':
        if eval_mode == 1:  # Conditional (result 1)
            outputs = np.concatenate([gray_score[:, 10:], color_score[:, :10]], axis=0)
            predicted_classes = np.argmax(outputs, axis=1)
        elif eval_mode == 2:  # Sum (result 2)
            outputs = np.concatenate([gray_score, color_score], axis=0)
            outputs = outputs[:, :10] + outputs[:, 10:]
            predicted_classes = np.argmax(outputs, axis=1)
    else:
        score = np.concatenate([gray_score, color_score], axis=0)
        predicted_classes = np.argmax(score, axis=1)

    return predicted_classes


def get_bias(predicted_classes, test_labels):
    domain_zeros = np.zeros([
        10000,
    ], dtype=np.int32)
    domain_ones = np.ones([
        10000,
    ], dtype=np.int32)
    domain_targets = np.concatenate([domain_zeros, domain_ones], axis=0)

    class_targets = np.array(test_labels + test_labels)
    class_count = 10
    test_set_size = class_targets.shape[0]

    count_per_class = np.zeros((class_count, 2), dtype=np.float64)
    for i in range(test_set_size):
        cur_predict = int(predicted_classes[i])
        count_per_class[cur_predict][int(domain_targets[i])] += 1
    bias = np.amax(count_per_class, axis=1) / np.sum(count_per_class, axis=1)
    total_bias = np.abs(bias - 0.5)
    mean_class_bias = np.mean(total_bias)

    ret = {}
    for idx in range(class_count):
        key = 'class_' + str(idx) + '_bias'
        ret[key] = total_bias[idx]
    ret['mean_bias'] = mean_class_bias

    return ret


with open(args.config, 'r') as f:
    config_json = json.load(f)

for config in config_json:
    class_bias_result = []
    for no_try in range(config['no_tries']):
        exp_result_path = Path(
            args.raw_result_dir,
            "{0}_{1}_{2}_{3}/{4}".format(config['network'],
                                         config['training_type'],
                                         config['dataset'],
                                         config['random_seed'],
                                         str(no_try)))
        color_result_path = Path(
            exp_result_path,
            "record/{0}_{1}/e1/test_color_result.pkl".format(config['dataset'],
                                                             config['training_type'].replace('-', '_')))
        gray_result_path = Path(
            exp_result_path,
            "record/{0}_{1}/e1/test_gray_result.pkl".format(config['dataset'],
                                                            config['training_type'].replace('-', '_')))
        test_label_path = Path(exp_result_path, 'data/cifar_test_labels')

        with open(str(color_result_path), 'rb') as f:
            color_result = pickle.load(f)
        with open(str(gray_result_path), 'rb') as f:
            gray_result = pickle.load(f)
        with open(str(test_label_path), 'rb') as f:
            test_labels = pickle.load(f)

        if 'outputs' in color_result:
            outputs_key = 'outputs'
        else:
            outputs_key = 'class_outputs'
        color_score = color_result[outputs_key]
        gray_score = gray_result[outputs_key]

        # Get Bias results
        def eval_loop(modes):
            for eval_mode in range(1, modes + 1):
                if eval_mode <= 3:
                    predicted_classes = predict(color_score, gray_score, config['training_type'], eval_mode)
                    bias_result = get_bias(predicted_classes, test_labels)
                else:  # Load RBA bias_result
                    rba_bias_dict_path = Path(
                        exp_result_path,
                        "record/{0}_{1}/e1/rba_bias_dict.pkl".format(config['dataset'],
                                                                     config['training_type'].replace('-', '_')))
                    with open(str(rba_bias_dict_path), 'rb') as f:
                        bias_result = pickle.load(f)

                for key, value in bias_result.items():
                    if key not in class_bias_result[eval_mode - 1]:
                        class_bias_result[eval_mode - 1][key] = []
                    class_bias_result[eval_mode - 1][key].append(value)

        if config['training_type'] == 'domain-discriminative':
            if len(class_bias_result) == 0:
                class_bias_result = [{}, {}, {}, {}]
            eval_loop(4)
        elif config['training_type'] == 'domain-independent':
            if len(class_bias_result) == 0:
                class_bias_result = [{}, {}]
            eval_loop(2)
        else:
            if len(class_bias_result) == 0:
                class_bias_result = [{}]
            eval_loop(1)

    # Output
    def write_raw_dict(result_dict, writer):
        headers = sorted(list(result_dict.keys()))
        writer.writerow(['Trial'] + headers)
        for no_try in range(config['no_tries']):
            row = [no_try]
            for key in headers:
                row.append(str(result_dict[key][no_try]))
            writer.writerow(row)

    def write_stat(result_dict, writer):
        bias_keys = sorted(list(result_dict.keys()))
        writer.writerow(['Bias_name', 'max', 'min', 'mean', 'max_diff', 'stdev', 'rel_maxdiff'])
        for key in bias_keys:
            rd = partial(decimal_utils.round_significant_digit, digit=decimal_utils.GLOBAL_ROUND_DIGIT)
            rf = partial(decimal_utils.round_significant_format, digit=decimal_utils.GLOBAL_ROUND_DIGIT)
            rl = partial(decimal_utils.round_list, digit=decimal_utils.GLOBAL_ROUND_DIGIT)

            value = result_dict[key]
            value = rl(value)
            
            max_v = max(value)
            min_v = min(value)
            max_diff = max_v - min_v
            std_dev = statistics.stdev(value)
            mean = statistics.mean(value)
            if math.isclose(mean, 0):
                rel_maxdiff = 0
            else:
                rel_maxdiff = rd(max_diff) / rd(mean)


            writer.writerow([
                key,
                rf(max_v),
                rf(min_v),
                rf(mean),
                rf(max_diff),
                rf(std_dev),
                rf(rel_maxdiff)
            ])

    def write_one_result(raw_cw, analysis_cw, caption, result_dict):
        raw_cw.writerow([caption])
        raw_cw.writerow([])
        analysis_cw.writerow([caption])
        analysis_cw.writerow([])

        write_raw_dict(result_dict, raw_cw)
        raw_cw.writerow([])
        write_stat(result_dict, analysis_cw)
        analysis_cw.writerow([])

    raw_output_path = Path(
        args.output_dir,
        "{0}_{1}_{2}_{3}_perclass_bias_raw.csv".format(config['network'],
                                                       config['training_type'],
                                                       config['dataset'],
                                                       config['random_seed']))
    f = open(str(raw_output_path), 'w', newline='')
    cw = csv.writer(f)
    analysis_output_path = Path(
        args.output_dir,
        "{0}_{1}_{2}_{3}_perclass_variance_analysis.csv".format(config['network'],
                                                                config['training_type'],
                                                                config['dataset'],
                                                                config['random_seed']))
    f2 = open(str(analysis_output_path), 'w', newline='')
    cw2 = csv.writer(f2)

    if config['training_type'] == 'domain-discriminative':
        write_one_result(cw, cw2, 'Sum prob w/o prior shift (result 1)', class_bias_result[0])
        write_one_result(cw, cw2, 'Max prob w/ prior shift (result 2)', class_bias_result[1])
        write_one_result(cw, cw2, 'Sum prob w/ prior shift (result 3)', class_bias_result[2])
        write_one_result(cw, cw2, 'RBA (result 4)', class_bias_result[3])
    elif config['training_type'] == 'domain-independent':
        write_one_result(cw, cw2, 'Bias conditional (result 1)', class_bias_result[0])
        write_one_result(cw, cw2, 'Bias sum (result 2)', class_bias_result[1])
    else:
        write_one_result(cw, cw2, 'Mean bias', class_bias_result[0])
    f.close()
    f2.close()
