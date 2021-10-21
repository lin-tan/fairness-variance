'''
Script to plot the accuracy and the fairness measures for different algorithms
from the log files
'''

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import os
print(os.getcwd())
import numpy as np

plt.style.use('ggplot')

def create_acc_lists(filepath):
    train_acc = []
    train_ddp = []
    train_deo = []
    valid_acc = []
    valid_ddp = []
    valid_deo = []
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            #if 'Epoch: 040/100' in line:
            #    break
            if 'Train Acc' in line:
                line = line.strip()
                linesegs = line.split(' | ')
                train_acc.append(float(linesegs[1].split(': ')[1].strip('%')))
                train_ddp.append(float(linesegs[2].split(': ')[1].strip('%')))
                train_deo.append(float(linesegs[3].split(': ')[1].strip('%')))
            elif 'Valid Acc' in line:
                line = line.strip()
                linesegs = line.split(' | ')
                valid_acc.append(float(linesegs[1].split(': ')[1].strip('%')))
                valid_ddp.append(float(linesegs[2].split(': ')[1].strip('%')))
                valid_deo.append(float(linesegs[3].split(': ')[1].strip('%')))

                
            line = fp.readline()
            cnt += 1
    return train_acc, train_ddp, train_deo, valid_acc, valid_ddp, valid_deo

def color(R, G, B):
    return (float(R)/255, float(G)/255, float(B)/255)

def BLUE():
    return color(0, 77, 128)

def RED():
    return color(181, 23, 0)

def make_plot_helper(arr, legends, xlabel, ylabel, outname):
    epoch_list = np.arange(1, arr.shape[1] + 1)
    fig, axs = plt.subplots(1, 1, figsize=(5,4), sharey=False)
    fig.patch.set_visible(False)
    axs.set_facecolor(color(240, 240, 240))
    axs.tick_params(axis='x', colors='black')
    axs.tick_params(axis='y', colors='black')
    axs.xaxis.label.set_color('black')
    axs.yaxis.label.set_color('black')

    axs.set_ylim([0, arr.max() + 15])
    #plt.gca().set_color_cycle(['red', 'blue', 'green', 'yellow'])
    colors=[RED(), BLUE()]
    for value, legend, c in zip(arr, legends, colors):
        plt.plot(epoch_list, value, label=legend, color=c)
    axs.set_xlabel(xlabel, fontweight='bold')
    axs.set_ylabel(ylabel, fontweight='bold')
    title = ylabel.replace("%", "").upper()
    #plt.title(title, fontweight='bold')#, x=0.7, y=0.1)
    
    leg = axs.legend(loc='upper right', frameon=False)
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    fig.tight_layout()
    outname.replace('$', '_')
    fig.savefig(outname, bbox_inches='tight')
    print('Plotted ' + outname)

def make_plot(list1, list2, legend1, legend2, plot_type, suffix=None):
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    legend = [legend1, legend2]
    arr = np.array([arr1, arr2])
    xlabel = 'Epochs'
    if plot_type == 'acc':
        arr = 100 - arr
    ylabel = 'Error %' if plot_type == 'acc' else 'DEO'
    legend1 = '_'.join(legend1.split(' '))
    legend2 = '_'.join(legend2.split(' '))
    #pdb.set_trace()
    if 'penalty' in legend2:
        legend2 = 'l2_penalty'
    if 'penalty' in legend1:
        legend1 = 'l2_penalty'    
    outname = '_'.join([legend1, legend2, plot_type])
    if suffix is not None:
        outname += '_' + suffix
    make_plot_helper(arr, legend, xlabel, ylabel, outname)


def gen_main_plots():
    # Used in the main paper for generating plots
    file_name = 'no_1p_lr0p01.txt'
    _, _, _, no_acc, _, no_deo = create_acc_lists(file_name)

    file_name = 'with_1p_fairalm_eta60_inner5_lr0p01.txt'
    _, _, _, fair_acc, _, fair_deo = create_acc_lists(file_name)

    file_name = 'with_1e_L2_PENALTY_eta0p01_lr0p01.txt'
    _, _, _, l2_acc, _, l2_deo = create_acc_lists(file_name)
    
    
    
    MEDIUM_SIZE = 12

    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    
    make_plot(no_acc, fair_acc, 'Unconstrained', 'FairALM', 'acc')
    make_plot(no_deo, fair_deo, 'Unconstrained', 'FairALM', 'deo')
    
    make_plot(no_acc, l2_acc, 'Unconstrained', '$\ell_2$ penalty', 'acc')
    make_plot(no_deo, l2_deo, 'Unconstrained', '$\ell_2$ penalty', 'deo')

   
def gen_fair_alm_plots(no_filename, fair_alm_filename, suffix):
    _, _, _, no_acc, _, no_deo = create_acc_lists(no_filename)
    _, _, _, fair_acc, _, fair_deo = create_acc_lists(fair_alm_filename)
    make_plot(no_acc, fair_acc, 'Unconstrained', 'FairALM', 'acc', suffix)
    make_plot(no_deo, fair_deo, 'Unconstrained', 'FairALM', 'deo', suffix)
   
   
def gen_l2_plots(no_filename, l2_filename, suffix):
    _, _, _, no_acc, _, no_deo = create_acc_lists(no_filename)
    _, _, _, l2_acc, _, l2_deo = create_acc_lists(l2_filename)
    make_plot(no_acc, l2_acc, 'Unconstrained', "$\ell_2$ penalty", 'acc', suffix)
    make_plot(no_deo, l2_deo, 'Unconstrained', "$\ell_2$ penalty", 'deo', suffix)

    
def gen_l2_fair_alm_plots(l2_filename, fair_alm_filename, suffix):
    _, _, _, l2_acc, _, l2_deo = create_acc_lists(l2_filename)
    _, _, _, fair_acc, _, fair_deo = create_acc_lists(fair_alm_filename)
    make_plot(l2_acc, fair_acc, "$\ell_2$ penalty", 'FairALM', 'acc', suffix)
    make_plot(l2_deo, fair_deo, "$\ell_2$ penalty", 'FairALM', 'deo', suffix)


def gen_all_plots():
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    
    file_name = 'no_1p_lr0p01.txt'
    fair_alm_filenames = {'eta60': 'FAIR_ALM_eta60_inner5_lr0p01.txt',
                          'eta40': 'FAIR_ALM_eta40_inner5_lr0p01.txt',
                          'eta45': 'FAIR_ALM_eta45_lr0p01.txt',
                          'eta50': 'FAIR_ALM_eta50_lr0p01.txt',
                          'eta80': 'FAIR_ALM_eta80_inner5_lr0p01.txt',
                          'eta20': 'FAIR_ALM_eta20_inner5_lr0p01.txt'}
    l2_filenames = {'eta0p01': 'L2_PENALTY_eta0p01_lr0p01.txt',
                    'eta0p001': 'L2_PENALTY_eta0p001_lr0p01.txt',
                    'eta0p1': 'L2_PENALTY_eta0p1_lr0p01.txt',
                    'eta1': 'L2_PENALTY_eta1_lr0p01.txt'}

    for eta, name in fair_alm_filenames.items():
        gen_fair_alm_plots(file_name, name, eta)

    for eta, name in l2_filenames.items():
        gen_l2_plots(file_name, name, eta)

    for l2_eta, l2_name in l2_filenames.items():
        for alm_eta, alm_name in fair_alm_filenames.items():
            gen_l2_fair_alm_plots(l2_name, alm_name, l2_eta+'_'+alm_eta)

if __name__ == "__main__":
    gen_all_plots()



