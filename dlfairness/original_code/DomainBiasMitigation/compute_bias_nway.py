import json
import pickle
import numpy as np
import argparse
from pathlib import Path

GRAYSCALE = 0
COLOR = 1

parser = argparse.ArgumentParser()
parser.add_argument('--parent_folder', type=str)
parser.add_argument('--record_name', type=str)
parser.add_argument('--output_key', type=str, default='outputs')
args = parser.parse_args()

def compute_bias(score): # Score need to have shape [, 2]
    global test_labels
    domain_zeros = np.zeros([10000,], dtype=np.int32)
    domain_ones = np.ones([10000,], dtype=np.int32)
    domain_targets = np.concatenate([domain_zeros, domain_ones], axis=0)

    class_targets = np.array(test_labels + test_labels)
    target_domain_ratios = 0.5 * np.ones((10,))
    domain_labels = ['Gray', 'Color']
    class_count = score.shape[1]
    test_set_size = class_targets.shape[0]

    # cifar_inference()
    #score = np.exp(score)
    predicted_classes = np.argmax(score, axis=1)   
    
    # cifar_count_domain_incidence_from_gt()
    count_per_class = np.zeros((class_count, 2), dtype=np.float64)
    for i in range(test_set_size):
        predicted_class = int(predicted_classes[i])
        count_per_class[predicted_class][int(domain_targets[i])] += 1
    bias = np.amax(count_per_class, axis=1) / np.sum(count_per_class, axis=1)
    total_bias = np.abs(bias - 0.5)
    mean_class_bias = np.mean(total_bias)

    return mean_class_bias

def preprocess_score(gray_result, color_result): # np.exp applied to the score

    score = np.concatenate([gray_result[args.output_key], color_result[args.output_key]], axis=0).astype(np.float128)
    score = np.exp(score)

    return score
        
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

score = preprocess_score(gray_result, color_result)
mean_bias = compute_bias(score)
print(mean_bias)

output_file = Path(args.parent_folder, 'record', args.record_name, 'e1/bias_result.txt')
with open(output_file, 'w') as f:
    f.write("Mean bias: {:4f}".format(mean_bias))
