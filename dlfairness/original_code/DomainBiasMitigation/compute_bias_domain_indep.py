import json
import pickle
import numpy as np
import argparse
from pathlib import Path
from scipy.special import softmax

GRAYSCALE = 0
COLOR = 1

parser = argparse.ArgumentParser()
parser.add_argument('--parent_folder', type=str)
parser.add_argument('--record_name', type=str)
parser.add_argument('--output_key', type=str, default='outputs')
args = parser.parse_args()

def bias(predicted_classes):
    global test_labels
    domain_zeros = np.zeros([10000,], dtype=np.int32)
    domain_ones = np.ones([10000,], dtype=np.int32)
    domain_targets = np.concatenate([domain_zeros, domain_ones], axis=0)

    class_targets = np.array(test_labels + test_labels)
    target_domain_ratios = 0.5 * np.ones((10,))
    domain_labels = ['Gray', 'Color']
    class_count = 10
    test_set_size = class_targets.shape[0]

    count_per_class = np.zeros((class_count, 2), dtype=np.float64)
    for i in range(test_set_size):
        predicted_class = int(predicted_classes[i])
        count_per_class[predicted_class][int(domain_targets[i])] += 1
    bias = np.amax(count_per_class, axis=1) / np.sum(count_per_class, axis=1)
    total_bias = np.abs(bias - 0.5)
    mean_class_bias = np.mean(total_bias)

    return mean_class_bias

def get_bias(gray_score, color_score):
    bias_ret = []

    # Conditional (result 1)
    outputs = np.concatenate([gray_score[:, 10:], color_score[:, :10]], axis=0)
    predicted_classes = np.argmax(outputs, axis=1)
    mean_bias = bias(predicted_classes)
    print("Bias conditional (result 1):", mean_bias)
    bias_ret.append(mean_bias)

    # Sum (result 2)
    outputs = np.concatenate([gray_score, color_score], axis=0)
    outputs = outputs[:, :10] + outputs[:, 10:]
    predicted_classes = np.argmax(outputs, axis=1)
    mean_bias = bias(predicted_classes)
    print("Bias sum (result 2):", mean_bias)    
    bias_ret.append(mean_bias)

    return bias_ret

color_result_path = Path(args.parent_folder, 'record', args.record_name, 'e1/test_color_result.pkl')
gray_result_path = Path(args.parent_folder, 'record', args.record_name, 'e1/test_gray_result.pkl')
test_label_path = Path(args.parent_folder, 'data/cifar_test_labels')

with open(color_result_path, 'rb') as f:
    color_result = pickle.load(f)
with open(gray_result_path, 'rb') as f:
    gray_result = pickle.load(f)
with open(test_label_path, 'rb') as f:
    test_labels = pickle.load(f)

#print(color_result['outputs'].shape)

bias_list = get_bias(gray_result[args.output_key], color_result[args.output_key])
output_file = Path(args.parent_folder, 'record', args.record_name, 'e1/bias_result.txt')
with open(output_file, 'w') as f:
    f.write("Bias conditional (result 1): {:4f}".format(bias_list[0]))
    f.write('\n')
    f.write("Bias sum (result 2): {:4f}".format(bias_list[1]))
