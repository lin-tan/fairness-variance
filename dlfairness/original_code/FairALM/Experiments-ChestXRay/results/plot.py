import pdb
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import os
print(os.getcwd())
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import rc

plt.style.use('ggplot')

def color(R, G, B):
    return (float(R)/255, float(G)/255, float(B)/255)

def BLUE():
    return color(0, 77, 128)

def RED():
    return color(181, 23, 0)

def create_acc_list(filepath):
    val_acc = []
    val_deo = []
    cnt = 0
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            if 'Valid Acc' in line:
                lineseg = line.strip().split(' | ')
                val_acc.append(float(lineseg[1].split(': ')[1].strip('%')))
                val_deo.append(float(lineseg[5].split(': ')[1].strip('%')))
            line = fp.readline()
            cnt += 1
    return val_acc, val_deo

def make_plot_helper(arr, legends, xlabel, ylabel, outname):
    fig, axs = plt.subplots(1, 1, figsize=(3.1,3), sharey=False)
    fig.patch.set_visible(False)
    axs.set_facecolor(color(240, 240, 240))
    axs.tick_params(axis='x', colors='black')
    axs.set_xticklabels(labels=legends, fontweight='bold')
    
    axs.tick_params(axis='y', colors='black')
    axs.xaxis.label.set_color('black')
    axs.yaxis.label.set_color('black')
    if ylabel is 'DEO':
        mean = np.array(arr).mean()
    bp = axs.boxplot(arr, patch_artist=True)

    ## change outline color, fill color and linewidth of the boxes
    box1 = bp['boxes'][0]
    box2 = bp['boxes'][1]
    box1.set(facecolor=RED())
    box2.set(facecolor=BLUE())
    #for box in bp['boxes']:
    #    # change outline color
    #    box.set( color='black', linewidth=1)
    #    # change fill color
    #    box.set( facecolor = BLUE() )

    ## change color and linewidth of the whiskers
    #for whisker in bp['whiskers']:
    #    whisker.set(color='black', linewidth=2)

    ## change color and linewidth of the caps
    #for cap in bp['caps']:
    #    cap.set(color='black', linewidth=2)
        
    ## change color and linewidth of the medians
    median1 = bp['medians'][0]
    median2 = bp['medians'][1]
    #clr = color(181, 154, 173)
    #clr = color(177, 181, 151)

    clr = color(255, 160, 110)
    median1.set(color=clr, linewidth=3)

    median2.set(color=BLUE(), linewidth=4)     
        
    ## change the style of fliers and their fill
    #for flier in bp['fliers']:
    #    flier.set(marker='o', color='#e7298a', alpha=0.5)
    
    axs.set_ylabel(ylabel, fontweight='bold')
    if ylabel is 'DEO':
        title = 'Box plot for DEO'
    else:
        title = 'Box plot for Error%'
    plt.title(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(outname, bbox_inches='tight')
    print('Plotted ' + outname)


def make_plot(list1, list2, legend1, legend2, plot_type):
    start = 34
    list1 = np.array(list1[start:])
    list2 = np.array(list2[start:])
    std = 0.3
    if plot_type == 'acc':
        std = 0.3
    else:
        std = 0.8
        
    noise = np.random.normal(0,std,list2.shape[0])
    #list2 = list2 + noise
    ylabel = ''
    if plot_type is 'acc':
        list1 = 100 - list1
        list2 = 100 - list2
        ylabel = 'Error %'
    else:
        ylabel = 'DEO'
    arr = [list1, list2]
    legend = [legend1, legend2]
    outname = '_'.join([legend1, legend2, plot_type])
    make_plot_helper(arr, legend, 'xlabel', ylabel, outname)

def autolabel(axs, rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        axs.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                 '%d' % int(height),
                 ha='center', va='bottom', fontweight='bold')
    
def make_bar_plot():
    fig, axs = plt.subplots(1, 1, figsize=(2.7,3), sharey=False)
    fig.patch.set_visible(False)
    axs.set_facecolor(color(240, 240, 240))
    axs.tick_params(axis='x', colors='black')
    axs.set_ylim([0, 55])
    #axs.tick_params(axis='x', labelbottom=False)
    axs.set_xticklabels(labels=['Unconstrained', 'FairALM'], fontweight='bold')
    
    axs.tick_params(axis='y', colors='black')
    axs.xaxis.label.set_color('black')
    axs.yaxis.label.set_color('black')

    colors = [RED(), BLUE()]
    rect = axs.bar(['Unconstrained', 'FairALM'], height=[45, 27], width=0.5, color=colors)
    axs.set_ylabel('Epochs', fontweight='bold')
    title = 'Epochs to converge'
    plt.title(title, fontweight='bold')
    autolabel(axs, rect)
    #pdb.set_trace()
    plt.setp(axs.get_yticklabels()[0], visible=False)
    fig.tight_layout()
    outname = 'convergence_plot.png'
    fig.savefig(outname, bbox_inches='tight')
    print('Plotted ' + outname)
    

def gen_all_plots():
    no_cons_fname = 'no_lr0p01_batch128.txt'
    fair_alm_fname = 'with_lr0p01_batch128_eta25.txt'
    no_acc, no_deo = create_acc_list(no_cons_fname)
    alm_acc, alm_deo = create_acc_list(fair_alm_fname)

    MEDIUM_SIZE = 11
    LARGE_SIZE=17

    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    
    make_plot(no_acc, alm_acc, 'Unconstrained', 'FairALM', 'acc')
    make_plot(no_deo, alm_deo, 'Unconstrained', 'FairALM', 'deo')
    make_bar_plot()

if __name__ == "__main__":
    gen_all_plots()

