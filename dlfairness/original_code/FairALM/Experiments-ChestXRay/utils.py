import torch
import pdb

def get_res_dict():
    res = {'acc': None,
           'ddp': None,
           'ppv': None,
           'fpr': None,
           'fnr': None,
           'tn_s0': None,
           'tn_s1': None,
           'fp_s0': None,
           'fp_s1': None,
           'fn_s0': None,
           'fn_s1': None,
           'tp_s0': None,
           'tp_s1': None
    }
    return res

def compute_accuracy(config, model, data_loader, dump_raw=False):
    correct_pred, num_examples = 0, 0

    num_protected0, num_protected1 = 0, 0
    num_correct_pred_protected0, num_correct_pred_protected1 = 0, 0

    num_pred1_protected0, num_pred1_protected1 = 0, 0
    num_targets0_protected0, num_targets0_protected1 = 0, 0
    num_targets1_protected0, num_targets1_protected1 = 0, 0
    
    num_pred0_targets0_protected0, num_pred0_targets0_protected1 = 0, 0
    num_pred1_targets0_protected0, num_pred1_targets0_protected1 = 0, 0
    num_pred0_targets1_protected0, num_pred0_targets1_protected1 = 0, 0
    num_pred1_targets1_protected0, num_pred1_targets1_protected1 = 0, 0

    if dump_raw:
        features_dump = []
        targets_dump = []
        protected_dump = []
        pred_dump = []

    for i, (features, targets, protected) in enumerate(data_loader):
        if config['DEBUG'] and i > 1:
            break
        features = features.cuda()
        targets = targets.cuda()
        protected = protected.cuda()
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += 1.*targets.size(0) 
        correct_pred += (predicted_labels == targets).float().sum() # CHANGED

        if dump_raw:
            features_dump.append(features.cpu())
            targets_dump.append(targets.cpu())
            protected_dump.append(protected.cpu())
            pred_dump.append(predicted_labels.detach().cpu())

        # For DDP metric
        num_protected0 += (protected == 0).float().sum()
        num_protected1 += (protected == 1).float().sum()
        num_correct_pred_protected0 += ((predicted_labels == targets) & (protected == 0)).float().sum()
        num_correct_pred_protected1 += ((predicted_labels == targets) & (protected == 1)).float().sum()

        # For PPV metric
        num_pred1_protected0 += ((predicted_labels == 1) & (protected == 0)).float().sum()
        num_pred1_protected1 += ((predicted_labels == 1) & (protected == 1)).float().sum()
        
        # For FPR metric
        num_targets0_protected0 += ((targets == 0) & (protected == 0)).float().sum()
        num_targets0_protected1 += ((targets == 0) & (protected == 1)).float().sum()
        
        # For FNR metric
        num_targets1_protected0 += ((targets == 1) & (protected == 0)).float().sum()
        num_targets1_protected1 += ((targets == 1) & (protected == 1)).float().sum()

        num_pred0_targets0_protected0 += ((predicted_labels == 0) & (targets == 0) & (protected == 0)).float().sum()
        num_pred0_targets0_protected1 += ((predicted_labels == 0) & (targets == 0) & (protected == 1)).float().sum()
        num_pred1_targets0_protected0 += ((predicted_labels == 1) & (targets == 0) & (protected == 0)).float().sum()
        num_pred1_targets0_protected1 += ((predicted_labels == 1) & (targets == 0) & (protected == 1)).float().sum()
        num_pred0_targets1_protected0 += ((predicted_labels == 0) & (targets == 1) & (protected == 0)).float().sum()
        num_pred0_targets1_protected1 += ((predicted_labels == 0) & (targets == 1) & (protected == 1)).float().sum()
        num_pred1_targets1_protected0 += ((predicted_labels == 1) & (targets == 1) & (protected == 0)).float().sum()
        num_pred1_targets1_protected1 += ((predicted_labels == 1) & (targets == 1) & (protected == 1)).float().sum()

    # res imported from utils
    res = get_res_dict()

    res['acc'] = correct_pred.float()/num_examples

    res['ddp'] = abs(num_correct_pred_protected0 / (num_protected0 + 1e-6) - num_correct_pred_protected1 / (num_protected1 + 1e-6)) 

    res['ppv'] = abs(num_pred1_targets1_protected0 / (num_pred1_protected0 + 1e-6) - num_pred1_targets1_protected1 / (num_pred1_protected1 + 1e-6)) 

    res['fpr'] = abs(num_pred0_targets0_protected0 / (num_targets0_protected0 + 1e-6) - num_pred0_targets0_protected1 / (num_targets0_protected1 + 1e-6)) 

    res['fnr'] = abs(num_pred1_targets1_protected0 / (num_targets1_protected0 + 1e-6) - num_pred1_targets1_protected1 / (num_targets1_protected1 + 1e-6)) 

    res['tn_s0'] = num_pred0_targets0_protected0
    res['tn_s1'] = num_pred0_targets0_protected1
    res['fp_s0'] = num_pred1_targets0_protected0
    res['fp_s1'] = num_pred1_targets0_protected1
    res['fn_s0'] = num_pred0_targets1_protected0
    res['fn_s1'] = num_pred0_targets1_protected1
    res['tp_s0'] = num_pred1_targets1_protected0
    res['tp_s1'] = num_pred1_targets1_protected1
    
    if dump_raw:
        features_dump = torch.cat(features_dump, 0)
        targets_dump = torch.cat(targets_dump, 0)
        protected_dump = torch.cat(protected_dump, 0)
        pred_dump = torch.cat(pred_dump, 0)
        return res, features_dump, targets_dump, protected_dump, pred_dump
    else:
        return res
    
def print_stats(config, epoch, stats, stat_type='Train'):
    if stat_type == 'Train':
        print('Epoch: %03d/%03d | Train Acc: %.3f%% | Train Ddp: %.3f%% | Train Ppv: %.3f%% | Train Fpr: %.3f%% | Train Fnr: %.3f%% ' % (
            epoch+1, config['NUM_EPOCHS'], 
            stats['acc'],
            stats['ddp'],
            stats['ppv'],
            stats['fpr'],
            stats['fnr']))
        print('                 | Train TP0: %d | Train FP0: %d | Train FN0: %d | Train TP0: %d' % (
            stats['tn_s0'],
            stats['fp_s0'],
            stats['fn_s0'],
            stats['tp_s0']))
        print('                 | Train TP1: %d | Train FP1: %d | Train FN1: %d | Train TP1: %d' % (
            stats['tn_s1'],
            stats['fp_s1'],
            stats['fn_s1'],
            stats['tp_s1']))
        '''
        print('                 | Train Acc: %.3f%% |                   | Train Ppv: %.3f%% | Train Fpr: %.3f%% | Train Fnr: %.3f%% ' % (
            100*(stats['tp_s0'] + stats['tp_s1'] + stats['tn_s0'] + stats['tn_s1'])/(stats['tp_s0'] + stats['tp_s1'] + stats['tn_s0'] + stats['tn_s1'] + stats['fp_s0'] + stats['fp_s1'] + stats['fn_s0'] + stats['fn_s1']),
            100*abs(stats['tp_s0']/(stats['tp_s0'] + stats['fp_s0']) - stats['tp_s1']/(stats['tp_s1'] + stats['fp_s1'])),
            100*abs(stats['fp_s0']/(stats['fp_s0'] + stats['tn_s0']) - stats['fp_s1']/(stats['fp_s1'] + stats['tn_s1'])),
            100*abs(stats['fn_s0']/(stats['fn_s0'] + stats['tp_s0']) - stats['fn_s1']/(stats['fn_s1'] + stats['tp_s1']))
        ))
        '''
    elif stat_type == 'Valid':
        print('Epoch: %03d/%03d | Valid Acc: %.3f%% | Valid Ddp: %.3f%% | Valid Ppv: %.3f%% | Valid Fpr: %.3f%% | Valid Fnr: %.3f%% ' % (
            epoch+1, config['NUM_EPOCHS'], 
            stats['acc'],
            stats['ddp'],
            stats['ppv'],
            stats['fpr'],
            stats['fnr']))
        print('                 | Valid TP0: %d | Valid FP0: %d | Valid FN0: %d | Valid TP0: %d' % (
            stats['tn_s0'],
            stats['fp_s0'],
            stats['fn_s0'],
            stats['tp_s0']))
        print('                 | Valid TP1: %d | Valid FP1: %d | Valid FN1: %d | Valid TP1: %d' % (
            stats['tn_s1'],
            stats['fp_s1'],
            stats['fn_s1'],
            stats['tp_s1']))
        '''
        print('                 | Valid Acc: %.3f%% |                   | Valid Ppv: %.3f%% | Valid Fpr: %.3f%% | Valid Fnr: %.3f%% ' % (
            100*(stats['tp_s0'] + stats['tp_s1'] + stats['tn_s0'] + stats['tn_s1'])/(stats['tp_s0'] + stats['tp_s1'] + stats['tn_s0'] + stats['tn_s1'] + stats['fp_s0'] + stats['fp_s1'] + stats['fn_s0'] + stats['fn_s1']),
            100*abs(stats['tp_s0']/(stats['tp_s0'] + stats['fp_s0']) - stats['tp_s1']/(stats['tp_s1'] + stats['fp_s1'])),
            100*abs(stats['fp_s0']/(stats['fp_s0'] + stats['tn_s0']) - stats['fp_s1']/(stats['fp_s1'] + stats['tn_s1'])),
            100*abs(stats['fn_s0']/(stats['fn_s0'] + stats['tp_s0']) - stats['fn_s1']/(stats['fn_s1'] + stats['tp_s1']))
        ))
        '''
    elif stat_type == 'Test':
        print('Epoch: %03d/%03d | Test Acc: %.3f%% | Test Ddp: %.3f%% | Test Ppv: %.3f%% | Test Fpr: %.3f%% | Test Fnr: %.3f%% ' % (
            epoch+1, config['NUM_EPOCHS'], 
            stats['acc'],
            stats['ddp'],
            stats['ppv'],
            stats['fpr'],
            stats['fnr']))
        print('                 | Test TP0: %d | Test FP0: %d | Test FN0: %d | Test TP0: %d' % (
            stats['tn_s0'],
            stats['fp_s0'],
            stats['fn_s0'],
            stats['tp_s0']))
        print('                 | Test TP1: %d | Test FP1: %d | Test FN1: %d | Test TP1: %d' % (
            stats['tn_s1'],
            stats['fp_s1'],
            stats['fn_s1'],
            stats['tp_s1']))
        '''
        print('                 | Test Acc: %.3f%% |                   | Test Ppv: %.3f%% | Test Fpr: %.3f%% | Test Fnr: %.3f%% ' % (
            100*(stats['tp_s0'] + stats['tp_s1'] + stats['tn_s0'] + stats['tn_s1'])/(stats['tp_s0'] + stats['tp_s1'] + stats['tn_s0'] + stats['tn_s1'] + stats['fp_s0'] + stats['fp_s1'] + stats['fn_s0'] + stats['fn_s1']),
            100*abs(stats['tp_s0']/(stats['tp_s0'] + stats['fp_s0']) - stats['tp_s1']/(stats['tp_s1'] + stats['fp_s1'])),
            100*abs(stats['fp_s0']/(stats['fp_s0'] + stats['tn_s0']) - stats['fp_s1']/(stats['fp_s1'] + stats['tn_s1'])),
            100*abs(stats['fn_s0']/(stats['fn_s0'] + stats['tp_s0']) - stats['fn_s1']/(stats['fn_s1'] + stats['tp_s1']))
        ))
        '''
    else:
        raise NameError('stat_type not defined!')
        
def map_0d_tensor_to_float(stat):
    ret = {}
    for key, value in stat.items():
        ret[key] = value.item()
    return ret