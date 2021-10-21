import argparse
import pandas as pd
import json
import pickle
import numpy as np
from pathlib import Path
from scipy.special import softmax

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
        test_labels = test_labels + test_labels

        if 'outputs' in color_result:
            outputs_key = 'outputs'
        else:
            outputs_key = 'class_outputs'
        color_score = color_result[outputs_key]
        gray_score = gray_result[outputs_key]

        # Get Bias results
        def eval_loop(modes):
            ret = []
            for eval_mode in range(1, modes + 1):
                if eval_mode <= 3:
                    predicted_classes = predict(color_score, gray_score, config['training_type'], eval_mode)
                else:  # Load RBA predicted result
                    test_rba_result_path = Path(
                        exp_result_path,
                        "record/{0}_{1}/e1/test_rba_result.pkl".format(config['dataset'],
                                                                       config['training_type'].replace('-', '_')))
                    with open(str(test_rba_result_path), 'rb') as f:
                        predicted_classes = pickle.load(f)
                ret.append(predicted_classes)

            return ret

        if config['training_type'] == 'domain-discriminative':
            predicted_classes_list = eval_loop(4)
        elif config['training_type'] == 'domain-independent':
            predicted_classes_list = eval_loop(2)
        else:
            predicted_classes_list = eval_loop(1)

        def write_pd(fn, predicted_classes):
            domain_zeros = np.zeros([
                10000,
            ], dtype=np.int32)
            domain_ones = np.ones([
                10000,
            ], dtype=np.int32)
            domain_targets = np.concatenate([domain_zeros, domain_ones], axis=0)
            # 0 for grayscale; 1 for color
            df = pd.DataFrame({
                'idx': list(range(20000)),
                'ground_truth': test_labels,
                'prediction_result': predicted_classes,
                'protected_label': domain_targets
            })
            df.set_index('idx', inplace=True)

            with open(fn, 'w') as f:
                df.to_csv(f)

        if config['training_type'] == 'domain-discriminative':
            sub_category = ['Sum-of-prob-no-prior-shift', 'Max-of-prob-prior-shift', 'Sum-of-prob-prior-shift', 'rba']
            for i in range(4):
                parent_path = Path(args.output_dir, '{0}/{1}'.format(config['training_type'], sub_category[i]))
                parent_path.mkdir(exist_ok=True, parents=True)
                csv_fn = Path(parent_path, 'try_{0:02d}.csv'.format(no_try))
                write_pd(str(csv_fn), predicted_classes_list[i])
        elif config['training_type'] == 'domain-independent':
            sub_category = ['Conditional', 'Sum']
            for i in range(2):
                parent_path = Path(args.output_dir, '{0}/{1}'.format(config['training_type'], sub_category[i]))
                parent_path.mkdir(exist_ok=True, parents=True)
                csv_fn = Path(parent_path, 'try_{0:02d}.csv'.format(no_try))
                write_pd(str(csv_fn), predicted_classes_list[i])
        else:
            parent_path = Path(args.output_dir, '{0}'.format(config['training_type']))
            parent_path.mkdir(exist_ok=True, parents=True)
            csv_fn = Path(parent_path, 'try_{0:02d}.csv'.format(no_try))
            write_pd(str(csv_fn), predicted_classes_list[0])
