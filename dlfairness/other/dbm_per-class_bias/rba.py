import json
import pickle
import numpy as np
import argparse
from pathlib import Path
import sys

GRAYSCALE = 0
COLOR = 1

parser = argparse.ArgumentParser()
parser.add_argument('--parent_folder', type=str)
parser.add_argument('--record_name', type=str)
parser.add_argument('--output_key', type=str, default='outputs')
args = parser.parse_args()

class optimize_potentials_given_known_domain():
    """Given a set of network outputs on a test set, updates potentials to reduce bias.
    
    Args:
      input_potentials: A float64 numpy array with shape (test_set_size, class_count).
        Contains the network outputs on each test example, for a single class prediction
        with known domain.
      gt_domain: An int32 numpy array with shape (test_set_size,). The ground truth
        domain. Used to do the optimization.
      gt_class: An int32 numpy array with shape (test_set_size,). The ground truth
        class label. Used only to compute accuracy.
      training_set_frequencies: A float32 numpy array with shape (test_set_size, class_count).
        The relative frequencies of the training set classes given each example's known domain.
        
    Returns:
      output_potentials: A float64 numpy array with shape (test_set_size, class_count).
        Contains the optimized network potentials that are the result of the optimization.
      output_predictions: An int32 numpy array with shape (test_set_size,). Contains
        the final network predictions (i.e. just an argmax over the potentials).
    """
    
    def __init__(self, input_potentials, gt_class, gt_domain, training_set_frequencies, lr,
                       margin, apply_prior_shift, inputs_are_activations, method_name,
                       target_domain_ratios, domain_labels, verbosity=2, total_epochs=100):
        
        self.test_set_size = gt_class.shape[0]
        self.class_count = input_potentials.shape[1]
        self.input_potentials = input_potentials
        self.gt_class = gt_class
        self.gt_domain = gt_domain
        self.training_set_frequencies = training_set_frequencies
        self.lr = lr
        self.margin = margin
        self.apply_prior_shift = apply_prior_shift
        self.inputs_are_activations = inputs_are_activations
        self.method_name = method_name
        self.target_domain_ratios = target_domain_ratios
        self.domain_labels = domain_labels
        self.verbosity = verbosity
        self.total_epochs = total_epochs
    
    def cifar_compute_accuracy(self, potentials, eval_dataset='balanced'):
        """Computes the accuracy from a set of potentials."""
        total_correct_class_count = 0
        predicted_class = self.cifar_inference(potentials)
        is_correct = np.equal(predicted_class, self.gt_class).astype(np.float64)
        if eval_dataset == 'gray':
            return np.mean(is_correct[self.gt_domain == GRAYSCALE])
        elif eval_dataset == 'color':
            return np.mean(is_correct[self.gt_domain == COLOR])
        elif eval_dataset == 'balanced':
            return np.mean(is_correct)
        elif eval_dataset == 'per_class_per_domain':
            class_count = int(np.max(self.gt_class)) + 1
            class_accs = np.zeros((class_count,))
            for i in range(class_count):
                i_grayscale_examples = (self.gt_class == i) & (self.gt_domain == GRAYSCALE)
                if np.any(i_grayscale_examples):
                    grayscale_acc = np.mean(is_correct[i_grayscale_examples])
                else:
                    grayscale_acc = 1.0
                i_color_examples = (self.gt_class == i) & (self.gt_domain == COLOR)
                if np.any(i_color_examples):
                    color_acc = np.mean(is_correct[i_color_examples])
                else:
                    color_acc = 1.0
                class_accs[i] = (grayscale_acc + color_acc) / 2.0
            return np.mean(class_accs)
        assert False
        
    def cifar_compute_bias(self, potentials):
        predicted_class = self.cifar_inference(potentials)
        count_per_class = self.cifar_count_domain_incidence_from_gt(predicted_class)
        bias_towards_grayscale = count_per_class[:, GRAYSCALE] / np.maximum(np.sum(count_per_class, axis=1), 1.0)
        total_bias = np.abs(bias_towards_grayscale - 0.5)
        mean_class_bias = np.mean(total_bias)
        return mean_class_bias

    def cifar_compute_class_bias(self, potentials):
        predicted_class = self.cifar_inference(potentials)
        count_per_class = self.cifar_count_domain_incidence_from_gt(predicted_class)
        bias_towards_grayscale = count_per_class[:, GRAYSCALE] / np.maximum(np.sum(count_per_class, axis=1), 1.0)
        total_bias = np.abs(bias_towards_grayscale - 0.5)
        mean_class_bias = np.mean(total_bias)

        ret = {}
        for idx in range(10):
            key = 'class_' + str(idx) + '_bias'
            ret[key] = total_bias[idx]
        ret['mean_bias'] = mean_class_bias
        return ret
    
    def cifar_count_domain_incidence_from_gt(self, predicted_classes):
        """Computes a dictionary mapping class_idx:[grayscale_count, color_count]
        
        This function uses the ground truth domain label to do the count.
        
        Args:
          predicted_classes: An int32 numpy array with shape (test_set_size,). The
            class labels.
        """
        count_per_class = np.zeros((self.class_count, 2), dtype=np.float64)
        for i in range(self.test_set_size):
            predicted_class = int(predicted_classes[i])
            count_per_class[predicted_class][int(self.gt_domain[i])] += 1
        return count_per_class
    
    def cifar_inference(self, potentials):
        """Computes the current class prediction from a set of potentials.
        
        Args:
          potentials: A float64 numpy array of shape (test_set_size, class_count)."""
        if self.inputs_are_activations:
            potentials = np.exp(potentials)
        if self.apply_prior_shift:
            probabilities = np.divide(potentials, self.training_set_frequencies)
            return np.argmax(probabilities, axis=1)
        predicted_classes = np.argmax(potentials, axis=1)
        return predicted_classes
    
    def cifar_generate_constraints(self, margin):
        constraints = np.zeros((self.class_count, 2, 2))
        constraints[:, 0, 0] = self.target_domain_ratios - 1 - self.margin
        constraints[:, 0, 1] = self.target_domain_ratios - self.margin
        constraints[:, 1, 0] = 1 - (self.margin + self.target_domain_ratios)
        constraints[:, 1, 1] = -(self.margin + self.target_domain_ratios)
        return constraints
    
    def optimize(self, input_potentials):
        if self.verbosity >= 1:
            initial_accuracy = self.cifar_compute_accuracy(input_potentials, 'balanced')
            initial_bias = self.cifar_compute_bias(input_potentials)
            name_in = ('Settings "%s", before opt.' % self.method_name).ljust(85)
            print('%s Acc. %f%%. Bias %f' % (name_in, 100*initial_accuracy, initial_bias))
            initial_gray_accuracy = self.cifar_compute_accuracy(input_potentials, 'gray')
            initial_color_accuracy = self.cifar_compute_accuracy(input_potentials, 'color')
            initial_mcd_accuracy = self.cifar_compute_accuracy(input_potentials, 'per_class_per_domain')
            print('%s Acc: %f%%. %s Acc: %f%%. Mean Per-C Per-D Acc: %f%%' % (
                self.domain_labels[0], 100*(initial_gray_accuracy), self.domain_labels[1],
                100*(initial_color_accuracy), 100*initial_mcd_accuracy))
        if self.verbosity >= 2:
            initial_accuracy = self.cifar_compute_accuracy(input_potentials, 'balanced')
            print('Pre optimization accuracy: %f%%' % (100.0*initial_accuracy))
        lambdas = np.zeros((self.class_count, 2), dtype=np.float64)
        current_potentials = input_potentials.copy()
        constraints = self.cifar_generate_constraints(self.margin)
        initial_predictions = self.cifar_inference(input_potentials)
        for epoch in range(self.total_epochs):
            violated_constraint_count = 0
            error = np.zeros((self.class_count, 2), dtype=np.float64)

            class_prediction = self.cifar_inference(current_potentials)
            count_per_class = self.cifar_count_domain_incidence_from_gt(class_prediction)
            count_per_class = np.reshape(count_per_class, [self.class_count, 1, 2])
            constraint_delta = np.sum(constraints * count_per_class, axis=2)
            lambdas += self.lr * constraint_delta
            error += constraint_delta
            count_per_class = np.reshape(count_per_class, [self.class_count, 2])

            lambdas = np.maximum(lambdas, 0)
            violated_constraint_count = np.count_nonzero(error > 0)
            current_potentials = input_potentials.copy()

            for i in range(len(current_potentials)):
                domain_idx = self.gt_domain[i]
                class_idx = class_prediction[i]
                constraint_idx = class_idx
                current_potentials[i][class_idx] -= lambdas[constraint_idx][0] * constraints[constraint_idx][0][domain_idx]
                current_potentials[i][class_idx] -= lambdas[constraint_idx][1] * constraints[constraint_idx][1][domain_idx]

            if (epoch % 10 == 0 or epoch == self.total_epochs-1) and self.verbosity >= 2:
                print('Finished %i-th Epoch.' % epoch)
                ratios = count_per_class[:, GRAYSCALE] / np.maximum((count_per_class[:, GRAYSCALE] + count_per_class[:, COLOR]), 1.0)
                bias = np.abs(ratios - 0.5)
                mean_bias = np.mean(bias)
                print('\tMean Bias: %0.4f' % mean_bias)
                constraint_count = len(constraints)
                print('\tConstraint Satisfaction: %i/%i' % (constraint_count-violated_constraint_count, constraint_count))
                current_accuracy = self.cifar_compute_accuracy(current_potentials, 'balanced')
                total_flipped_predictions = np.count_nonzero(self.cifar_inference(current_potentials) != initial_predictions)
                print('\tTotal Flipped Predictions: %i' % total_flipped_predictions)
                print('\tCurrent Accuracy: %0.2f%%' % (100.0*current_accuracy))

            if violated_constraint_count == 0:
                break
        if self.verbosity >= 1:
            final_accuracy = self.cifar_compute_accuracy(current_potentials, 'balanced')
            final_bias = self.cifar_compute_bias(current_potentials)
            name_in = ('Settings "%s", after opt' % self.method_name).ljust(85)
            print('%s Acc. %f%%. Bias %f' % (name_in, 100*(final_accuracy), (final_bias)))

            '''
            rba_cifar_inference = self.cifar_inference(current_potentials)
            output_file = Path(args.parent_folder, 'record', args.record_name, 'e1/test_rba_result.pkl')
            with open(str(output_file), 'wb') as f:
                pickle.dump(rba_cifar_inference, f)
            sys.exit(0)
            '''
            
            '''
            class_bias_dict = self.cifar_compute_class_bias(current_potentials)
            print(class_bias_dict)
            output_file = Path(args.parent_folder, 'record', args.record_name, 'e1/rba_bias_dict.pkl')
            with open(str(output_file), 'wb') as f:
                pickle.dump(class_bias_dict, f)
            sys.exit(0)
            '''
            
            final_gray_accuracy = self.cifar_compute_accuracy(current_potentials, 'gray')
            final_color_accuracy = self.cifar_compute_accuracy(current_potentials, 'color')
            final_mcd_accuracy = self.cifar_compute_accuracy(current_potentials, 'per_class_per_domain')
            #'''
            output_file = Path(args.parent_folder, 'record', args.record_name, 'e1/test_result.txt')
            with open(str(output_file), 'a') as f:
                f.write('RBA Gray Accuracy: ' + str(final_gray_accuracy))
                f.write('\n')
                f.write('RBA Color Accuracy: ' + str(final_color_accuracy))
            sys.exit(0)
            #'''
            print('%s Acc: %f%%. %s Acc: %f%%. Mean Per-C Per-D Acc: %f%%' % (
                self.domain_labels[0], 100*(final_gray_accuracy), self.domain_labels[1],
                100*(final_color_accuracy), 100*final_mcd_accuracy))
            
        print('Accuracy change: {}, bias change: {}'.format(final_accuracy-initial_accuracy, final_bias-initial_bias))
        return current_potentials, self.cifar_inference(current_potentials)
    
def run(hparams, data):
    optimize_probabilities = hparams['optimize_probabilities']
    reduction_method = hparams['reduction']
    apply_prior_shift = hparams['prior_shift']
    train_time_frequencies = data['train_time_frequencies']
    domain_targets = data['domain_targets']
    twon_activations = data['twon_activations']
    target_domain_ratios = data['target_domain_ratios']
    domain_labels = data['domain_labels']
    expected_class_count = len(train_time_frequencies)//2

    selected_outputs = []
    selected_train_probabilities = []
    # Cut down to relevant N values by subsituting in known domain:
    for i in range(domain_targets.shape[0]):
        cur_domain = domain_targets[i]
        if cur_domain == GRAYSCALE:   
            selected_train_probabilities.append(train_time_frequencies[expected_class_count:])
        elif cur_domain == COLOR:   
            selected_train_probabilities.append(train_time_frequencies[:expected_class_count])
        else:
            assert False
    train_probabilities = np.stack(selected_train_probabilities, axis=0)

    twon_activations = np.exp(twon_activations)
    if apply_prior_shift:
        twon_activations /= train_time_frequencies
        # We already applied prior shift:
        apply_prior_shift = False
        
    if reduction_method == 'sum':
        outputs = twon_activations[:, :expected_class_count] + twon_activations[:, expected_class_count:]

    if reduction_method == 'condition':
        for i in range(twon_activations.shape[0]):
            cur_domain = domain_targets[i]
            if cur_domain == GRAYSCALE:
                selected_outputs.append(twon_activations[i, expected_class_count:])
            elif cur_domain == COLOR:
                selected_outputs.append(twon_activations[i, :expected_class_count])
            else:
                assert False
        outputs = np.stack(selected_outputs, axis=0)
        
    margin = 0.05
    lr = hparams['lr']
    input_potentials = outputs
    
    optimization_str = 'optimize on probabilities' if hparams['optimize_probabilities'] else 'optimize on outputs'
    if hparams['optimize_probabilities'] and hparams['reduction'] == 'sum':
        reduction_str = 'sum probabilities'
    elif hparams['optimize_probabilities'] and hparams['reduction'] == 'condition':
        reduction_str = 'condition on d0'
    elif not hparams['optimize_probabilities'] and hparams['reduction'] =='sum':
        reduction_str = 'sum outputs'
    elif not hparams['optimize_probabilities'] and hparams['reduction'] == 'condition':
        reduction_str = 'condition on d0'
    else:
        assert False
    prior_shift_str = 'prior shift' if hparams['prior_shift'] else 'no prior shift'
    method_str = '%s, %s, %s' % (reduction_str, optimization_str, prior_shift_str)
            
    optimizer = optimize_potentials_given_known_domain(input_potentials=input_potentials,
                                                       gt_domain=domain_targets,
                                                       gt_class=class_targets,
                                                       training_set_frequencies=train_probabilities,
                                                       lr=lr,
                                                       margin=margin,
                                                       apply_prior_shift=apply_prior_shift,
                                                       inputs_are_activations=(not optimize_probabilities),
                                                       method_name=method_str,
                                                       target_domain_ratios=target_domain_ratios,
                                                       domain_labels=domain_labels,
                                                       verbosity=hparams['verbosity'],
                                                       total_epochs=hparams['total_epochs'])
    
    output_potentials, output_predictions = optimizer.optimize(input_potentials)
    return output_potentials, output_predictions

# Change this to the corresponding result path
#color_result_path = '/workspace/fairness/result_fairness/ResNet-18_baseline_cifar-s_fixed/0/record/cifar-s_baseline/e1/test_color_result.pkl'
#gray_result_path = '/workspace/fairness/result_fairness/ResNet-18_baseline_cifar-s_fixed/0/record/cifar-s_baseline/e1/test_gray_result.pkl'
#color_result_path = '/workspace/fairness/result_fairness/ResNet-18_domain-discriminative_cifar-s_fixed/3/record/cifar-s_domain_discriminative/e1/test_color_result.pkl'
#gray_result_path = '/workspace/fairness/result_fairness/ResNet-18_domain-discriminative_cifar-s_fixed/3/record/cifar-s_domain_discriminative/e1/test_gray_result.pkl'
color_result_path = Path(args.parent_folder, 'record', args.record_name, 'e1/test_color_result.pkl')
gray_result_path = Path(args.parent_folder, 'record', args.record_name, 'e1/test_gray_result.pkl')
with open(str(color_result_path), 'rb') as f:
    color_result = pickle.load(f)
with open(str(gray_result_path), 'rb') as f:
    gray_result = pickle.load(f)

test_label_path = Path(args.parent_folder, 'data/cifar_test_labels')
with open(str(test_label_path), 'rb') as f:
#with open('/workspace/fairness/result_fairness/ResNet-18_baseline_cifar-s_fixed/0/data/cifar_test_labels', 'rb') as f:
    test_labels = pickle.load(f)



train_time_frequencies = np.array([0.005, 0.095, 0.005, 0.095, 0.005, 
                                   0.095, 0.005, 0.095, 0.005, 0.095,
                                   0.095, 0.005, 0.095, 0.005, 0.095,
                                   0.005, 0.095, 0.005, 0.095, 0.005], dtype=np.float64) 

domain_zeros = np.zeros([10000,], dtype=np.int32)
domain_ones = np.ones([10000,], dtype=np.int32)
domain_targets = np.concatenate([domain_zeros, domain_ones], axis=0)

twon_gray_outs = color_result['outputs']
twon_color_outs = gray_result['outputs']

print(twon_gray_outs.shape)
print(twon_color_outs.shape)

twon_activations = np.concatenate([twon_gray_outs, twon_color_outs], axis=0)

class_targets = np.array(test_labels + test_labels)
target_domain_ratios = 0.5 * np.ones((10,))
domain_labels = ['Gray', 'Color']

data = {'train_time_frequencies': train_time_frequencies, 'domain_targets': domain_targets,
        'twon_activations': twon_activations, 'target_domain_ratios': target_domain_ratios,
        'domain_labels': domain_labels}

# sum without prior shift
hparams = {'optimize_probabilities': True, 'reduction': 'sum', 'prior_shift': False, 'lr': 3,
           'total_epochs': 100, 'verbosity': 1}

_,_ = run(hparams, data)

# sum with prior shift
hparams = {'optimize_probabilities': True, 'reduction': 'sum', 'prior_shift': True, 'lr': 3,
           'total_epochs': 100, 'verbosity': 1}

_,_ = run(hparams, data)

# condition without prior shift
hparams = {'optimize_probabilities': True, 'reduction': 'condition', 'prior_shift': False, 'lr': 3,
           'total_epochs': 100, 'verbosity': 1}

_,_ = run(hparams, data)

# condition with prior shift
hparams = {'optimize_probabilities': True, 'reduction': 'condition', 'prior_shift': True, 'lr': 3,
           'total_epochs': 100, 'verbosity': 1}

_,_ = run(hparams, data)
