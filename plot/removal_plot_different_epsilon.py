import argparse
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['mathtext.fontset'] = 'cm'
from matplotlib import pyplot as plt
# plt.rcParams["figure.figsize"] = (3.5, 3)
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy import stats


parser = argparse.ArgumentParser(description='Synthetics experiments visualization')
    
# for synthetic functions
parser.add_argument('--metric', type=str, default='drge', help='metric to plot')
parser.add_argument('--lb', type=float, default=None, help='')
parser.add_argument('--ub', type=float, default=None, help='')
parser.add_argument('--step', type=int, default=30, help='')
parser.add_argument('--selected', type=int, default=1, help='')
args = parser.parse_args()
selected_fontsize = 17
metric_dict = {"drge": 0, "acc": 1, "loss": 2}
# names = ['Housing', 'Credit Card', 'Diabetes', 'MNIST', 'CIFAR-10']
metric_idx = metric_dict[args.metric]


# names = ['$\epsilon$=0.1', '$\epsilon$=1', '$\epsilon$=5', '$\epsilon$=10', '$\epsilon$=30']

# path_names = ["Kernel_Regression/housing-numdp-3000-seed-1-length_scale-2.0-epsilon-0.1-mu-0.01-evaldp-1000-allbaseline.npz",
#               "Kernel_Regression/housing-numdp-3000-seed-1-length_scale-2.0-epsilon-1.0-mu-0.01-evaldp-1000-allbaseline.npz",
#               "Kernel_Regression/housing-numdp-3000-seed-1-length_scale-2.0-epsilon-5-mu-0.01-evaldp-1000-allbaselinev1.npz",
#               "Kernel_Regression/housing-numdp-3000-seed-1-length_scale-2.0-epsilon-10.0-mu-0.01-evaldp-1000-allbaseline.npz",
#               "Kernel_Regression/housing-numdp-3000-seed-1-length_scale-2.0-epsilon-30.0-mu-0.01-evaldp-1000-allbaseline.npz"]

names = ['$\epsilon$=0.01', '$\epsilon$=0.1', '$\epsilon$=1', '$\epsilon$=10', '$\epsilon$=100']

path_names = ["Kernel_Regression/mnist-numdp-1000-seed-1-length_scale-50.0-epsilon-0.01-mu-0.0-evaldp-900-allbaseline.npz",
              "Kernel_Regression/mnist-numdp-1000-seed-1-length_scale-50.0-epsilon-0.1-mu-0.0-evaldp-900-allbaseline.npz",
              "Kernel_Regression/mnist-numdp-1000-seed-1-length_scale-50.0-epsilon-1.0-mu-0.0-evaldp-900-allbaseline.npz",
              "Kernel_Regression/mnist-numdp-1000-seed-1-length_scale-50.0-epsilon-10.0-mu-0.0-evaldp-900-allbaselinev1.npz",
              "Kernel_Regression/mnist-numdp-1000-seed-1-length_scale-50.0-epsilon-100.0-mu-0.0-evaldp-900-allbaseline.npz"]

if args.selected:
    names = ['$\epsilon$=1', '$\epsilon$=10', '$\epsilon$=100']
    path_names = ["Kernel_Regression/mnist-numdp-1000-seed-1-length_scale-50.0-epsilon-1.0-mu-0.0-evaldp-900-allbaseline.npz",
              "Kernel_Regression/mnist-numdp-1000-seed-1-length_scale-50.0-epsilon-10.0-mu-0.0-evaldp-900-allbaselinev1.npz",
              "Kernel_Regression/mnist-numdp-1000-seed-1-length_scale-50.0-epsilon-100.0-mu-0.0-evaldp-900-allbaseline.npz"]

figs, axes = plt.subplots(nrows=1, ncols=len(names), figsize=(10*len(names), 3))
for j, name in enumerate(names):
    regression = False if name in ['MNIST', 'CIFAR-10'] else True

    result_1 = np.load(path_names[j], allow_pickle=True)

    deviation_scores_m = result_1['deviation_scores_m']
    loo_scores_m = result_1['loo_scores_m']
    marginal_drge_m = result_1['marginal_drge_m']
    
    len_score = len(deviation_scores_m[0])
    show_idx = np.arange(0, len_score, args.step)
    drge_deviation_score_m = result_1['drge_deviation_score_m'][:,show_idx,metric_idx]
    drge_loo_score_m = result_1['drge_loo_score_m'][:,show_idx,metric_idx]
    drge_if_score_m = result_1['drge_if_score_m'][:,show_idx,metric_idx]
    drge_trackin_score_m = result_1['drge_trackin_score_m'][:,show_idx,metric_idx]
    if not regression:
        drge_lava_score_m = result_1['drge_lava_score_m'][:,show_idx,metric_idx]
    drge_davinz_score_m = result_1['drge_davinz_score_m'][:,show_idx,metric_idx]
    drge_random_score_m = result_1['drge_random_score_m'][:,show_idx,metric_idx]

    drge_deviation_score_l2h_m = result_1['drge_deviation_score_l2h_m'][:,show_idx,metric_idx]
    drge_loo_score_l2h_m = result_1['drge_loo_score_l2h_m'][:,show_idx,metric_idx]
    drge_if_score_l2h_m = result_1['drge_if_score_l2h_m'][:,show_idx,metric_idx]
    drge_trackin_score_l2h_m = result_1['drge_trackin_score_l2h_m'][:,show_idx,metric_idx]
    if not regression:
        drge_lava_score_l2h_m = result_1['drge_lava_score_l2h_m'][:,show_idx,metric_idx]
    drge_davinz_score_l2h_m = result_1['drge_davinz_score_l2h_m'][:,show_idx,metric_idx]
    drge_random_score_l2h_m = result_1['drge_random_score_l2h_m'][:,show_idx,metric_idx]

    drge_deviation_score_m = np.maximum.accumulate(drge_deviation_score_m,axis=1).astype(np.float32)
    drge_loo_score_m = np.maximum.accumulate(drge_loo_score_m,axis=1).astype(np.float32)
    drge_if_score_m = np.maximum.accumulate(drge_if_score_m,axis=1).astype(np.float32)
    drge_trackin_score_m = np.maximum.accumulate(drge_trackin_score_m,axis=1).astype(np.float32)
    if not regression:
        drge_lava_score_m = np.maximum.accumulate(drge_lava_score_m,axis=1).astype(np.float32)
    drge_davinz_score_m = np.maximum.accumulate(drge_davinz_score_m,axis=1).astype(np.float32)
    drge_random_score_m = np.maximum.accumulate(drge_random_score_m,axis=1).astype(np.float32)

    drge_deviation_score_l2h_m = np.maximum.accumulate(drge_deviation_score_l2h_m,axis=1).astype(np.float32)
    drge_loo_score_l2h_m = np.maximum.accumulate(drge_loo_score_l2h_m,axis=1).astype(np.float32)
    drge_if_score_l2h_m = np.maximum.accumulate(drge_if_score_l2h_m,axis=1).astype(np.float32)
    drge_trackin_score_l2h_m = np.maximum.accumulate(drge_trackin_score_l2h_m,axis=1).astype(np.float32)
    if not regression:
        drge_lava_score_l2h_m = np.maximum.accumulate(drge_lava_score_l2h_m,axis=1).astype(np.float32)
    drge_davinz_score_l2h_m = np.maximum.accumulate(drge_davinz_score_l2h_m,axis=1).astype(np.float32)
    drge_random_score_l2h_m = np.maximum.accumulate(drge_random_score_l2h_m,axis=1).astype(np.float32)

    drge_deviation_score = np.mean(drge_deviation_score_m, axis=0)
    drge_deviation_score_se = stats.sem(drge_deviation_score_m, axis=0)
    drge_loo_score = np.mean(drge_loo_score_m, axis=0)
    drge_loo_score_se = stats.sem(drge_loo_score_m, axis=0)
    drge_deviation_score_l2h = np.mean(drge_deviation_score_l2h_m, axis=0)
    drge_deviation_score_l2h_se = stats.sem(drge_deviation_score_l2h_m, axis=0)
    drge_loo_score_l2h = np.mean(drge_loo_score_l2h_m, axis=0)
    drge_loo_score_l2h_se = stats.sem(drge_loo_score_l2h_m, axis=0)

    drge_if_score = np.mean(drge_if_score_m, axis=0)
    drge_if_score_se = stats.sem(drge_if_score_m, axis=0)
    drge_trackin_score = np.mean(drge_trackin_score_m, axis=0)
    drge_trackin_score_se = stats.sem(drge_trackin_score_m, axis=0)
    if not regression:
        drge_lava_score = np.mean(drge_lava_score_m, axis=0)
        drge_lava_score_se = stats.sem(drge_lava_score_m, axis=0)
    else:
        drge_lava_score = drge_lava_score_se = 0
    drge_davinz_score = np.mean(drge_davinz_score_m, axis=0)
    drge_davinz_score_se = stats.sem(drge_davinz_score_m, axis=0)
    drge_random_score = np.mean(drge_random_score_m, axis=0)
    drge_random_score_se = stats.sem(drge_random_score_m, axis=0)

    drge_if_score_l2h = np.mean(drge_if_score_l2h_m, axis=0)
    drge_if_score_l2h_se = stats.sem(drge_if_score_l2h_m, axis=0)
    drge_trackin_score_l2h = np.mean(drge_trackin_score_l2h_m, axis=0)
    drge_trackin_score_l2h_se = stats.sem(drge_trackin_score_l2h_m, axis=0)
    if not regression:
        drge_lava_score_l2h = np.mean(drge_lava_score_l2h_m, axis=0)
        drge_lava_score_l2h_se = stats.sem(drge_lava_score_l2h_m, axis=0)
    else:
        drge_lava_score_l2h = drge_lava_score_l2h_se = 0
    drge_davinz_score_l2h = np.mean(drge_davinz_score_l2h_m, axis=0)
    drge_davinz_score_l2h_se = stats.sem(drge_davinz_score_l2h_m, axis=0)
    drge_random_score_l2h = np.mean(drge_random_score_l2h_m, axis=0)
    drge_random_score_l2h_se = stats.sem(drge_random_score_l2h_m, axis=0)
    
    length_score = len(drge_deviation_score)
    magnitude_se = 0.5
    baseline_names = ["Deviation", "LOO", "Influence", "TracIn", "LAVA", "DAVINZ", "Random"]
    means = [drge_deviation_score, drge_loo_score, drge_if_score, drge_trackin_score, drge_lava_score, drge_davinz_score, drge_random_score]
    ses = [drge_deviation_score_se, drge_loo_score_se, drge_if_score_se, drge_trackin_score_se, drge_lava_score_se, drge_davinz_score_se, drge_random_score_se]

    colors = ['C10', 'C2', 'C4', 'C6', 'C1', 'C3', 'C9', 'C6', 'C7', 'C8', 'C5']
    #, 'C9', 'C6', 'C7', 'C8', 'C5']
    markers = [r'$\heartsuit$', r'$\boxdot$', r'$\bigcirc$', r'$\bigtriangleup$',  r'$\diamondsuit$', r'$\heartsuit$', 'x']
    markers = markers[::-1]
    every_marker = int(length_score/8)
    #, r'$\boxdot$', r'$\bigcirc$', r'$\bigtriangleup$', r'$\diamondsuit$', r'$\heartsuit$', '2', '3', '4']
    # for i in range(len(means)):
    for i in range(len(baseline_names)):
        if regression and baseline_names[i] == "LAVA":
            continue
        # plt.plot(drge_deviation_score, label="Deviation")
        # plt.fill_between(range(length_score),drge_deviation_score - magnitude_se * drge_deviation_score_se, drge_deviation_score +  magnitude_se *drge_deviation_score_se, alpha=0.2)
        axes[j].plot(show_idx, means[i], marker=markers[i % len(markers)], color=colors[i % len(colors)], linestyle='--', label=baseline_names[i], markevery=every_marker)
        axes[j].fill_between(show_idx, means[i] - magnitude_se * ses[i], means[i] + magnitude_se * ses[i], alpha=0.2, color=colors[i % len(colors)])

        axes[j].set_title(names[j])
        
        if j == 0:
            if args.metric == 'loss':
                if args.selected:
                    axes[j].set_ylabel(r'Worst-case loss', fontsize=selected_fontsize)
                else:
                    axes[j].set_ylabel(r'Worst-case loss')
            else:
                if args.selected:
                    axes[j].set_ylabel(r'DRGE', fontsize=selected_fontsize)
                else:
                    axes[j].set_ylabel(r'DRGE')

    # axes[j].yaxis.set_major_formatter(FormatStrFormatter('%.f'))
    axes[j].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axes[j].set_ylim([args.lb, args.ub])
    
figs.tight_layout(rect=[0, 0.07, 1, 0.99], h_pad=0, w_pad=11)
# figs.tight_layout()
figs.set_figheight(2.5)
if args.selected:
    figs.text(0.5, 0.04, '$\#$ data points removed from high to low', ha='center', fontsize=selected_fontsize)
else:
    figs.text(0.5, 0.04, '$\#$ data points removed from high to low', ha='center')
figs.set_figwidth(2*len(names))

# plt.legend(loc="lower right", prop={'size': 8})
if args.selected:
    plt.savefig(f'figs/vary-epsilon-removing-{args.metric}-selected.pdf', bbox_inches="tight")

else:
    plt.savefig(f'figs/vary-epsilon-removing-{args.metric}.pdf', bbox_inches="tight")

figs, axes = plt.subplots(nrows=1, ncols=len(names), figsize=(10*len(names), 3))
for j, name in enumerate(names):
    regression = False if name in ['MNIST', 'CIFAR-10'] else True

    result_1 = np.load(path_names[j], allow_pickle=True)

    deviation_scores_m = result_1['deviation_scores_m']
    loo_scores_m = result_1['loo_scores_m']
    marginal_drge_m = result_1['marginal_drge_m']
    
    len_score = len(deviation_scores_m[0])
    show_idx = np.arange(0, len_score, args.step)
    drge_deviation_score_m = result_1['drge_deviation_score_m'][:,show_idx,metric_idx]
    drge_loo_score_m = result_1['drge_loo_score_m'][:,show_idx,metric_idx]
    drge_if_score_m = result_1['drge_if_score_m'][:,show_idx,metric_idx]
    drge_trackin_score_m = result_1['drge_trackin_score_m'][:,show_idx,metric_idx]
    if not regression:
        drge_lava_score_m = result_1['drge_lava_score_m'][:,show_idx,metric_idx]
    drge_davinz_score_m = result_1['drge_davinz_score_m'][:,show_idx,metric_idx]
    drge_random_score_m = result_1['drge_random_score_m'][:,show_idx,metric_idx]

    drge_deviation_score_l2h_m = result_1['drge_deviation_score_l2h_m'][:,show_idx,metric_idx]
    drge_loo_score_l2h_m = result_1['drge_loo_score_l2h_m'][:,show_idx,metric_idx]
    drge_if_score_l2h_m = result_1['drge_if_score_l2h_m'][:,show_idx,metric_idx]
    drge_trackin_score_l2h_m = result_1['drge_trackin_score_l2h_m'][:,show_idx,metric_idx]
    if not regression:
        drge_lava_score_l2h_m = result_1['drge_lava_score_l2h_m'][:,show_idx,metric_idx]
    drge_davinz_score_l2h_m = result_1['drge_davinz_score_l2h_m'][:,show_idx,metric_idx]
    drge_random_score_l2h_m = result_1['drge_random_score_l2h_m'][:,show_idx,metric_idx]

    drge_deviation_score_m = np.maximum.accumulate(drge_deviation_score_m,axis=1).astype(np.float32)
    drge_loo_score_m = np.maximum.accumulate(drge_loo_score_m,axis=1).astype(np.float32)
    drge_if_score_m = np.maximum.accumulate(drge_if_score_m,axis=1).astype(np.float32)
    drge_trackin_score_m = np.maximum.accumulate(drge_trackin_score_m,axis=1).astype(np.float32)
    if not regression:
        drge_lava_score_m = np.maximum.accumulate(drge_lava_score_m,axis=1).astype(np.float32)
    drge_davinz_score_m = np.maximum.accumulate(drge_davinz_score_m,axis=1).astype(np.float32)
    drge_random_score_m = np.maximum.accumulate(drge_random_score_m,axis=1).astype(np.float32)

    drge_deviation_score_l2h_m = np.maximum.accumulate(drge_deviation_score_l2h_m,axis=1).astype(np.float32)
    drge_loo_score_l2h_m = np.maximum.accumulate(drge_loo_score_l2h_m,axis=1).astype(np.float32)
    drge_if_score_l2h_m = np.maximum.accumulate(drge_if_score_l2h_m,axis=1).astype(np.float32)
    drge_trackin_score_l2h_m = np.maximum.accumulate(drge_trackin_score_l2h_m,axis=1).astype(np.float32)
    if not regression:
        drge_lava_score_l2h_m = np.maximum.accumulate(drge_lava_score_l2h_m,axis=1).astype(np.float32)
    drge_davinz_score_l2h_m = np.maximum.accumulate(drge_davinz_score_l2h_m,axis=1).astype(np.float32)
    drge_random_score_l2h_m = np.maximum.accumulate(drge_random_score_l2h_m,axis=1).astype(np.float32)

    drge_deviation_score = np.mean(drge_deviation_score_m, axis=0)
    drge_deviation_score_se = stats.sem(drge_deviation_score_m, axis=0)
    drge_loo_score = np.mean(drge_loo_score_m, axis=0)
    drge_loo_score_se = stats.sem(drge_loo_score_m, axis=0)
    drge_deviation_score_l2h = np.mean(drge_deviation_score_l2h_m, axis=0)
    drge_deviation_score_l2h_se = stats.sem(drge_deviation_score_l2h_m, axis=0)
    drge_loo_score_l2h = np.mean(drge_loo_score_l2h_m, axis=0)
    drge_loo_score_l2h_se = stats.sem(drge_loo_score_l2h_m, axis=0)

    drge_if_score = np.mean(drge_if_score_m, axis=0)
    drge_if_score_se = stats.sem(drge_if_score_m, axis=0)
    drge_trackin_score = np.mean(drge_trackin_score_m, axis=0)
    drge_trackin_score_se = stats.sem(drge_trackin_score_m, axis=0)
    if not regression:
        drge_lava_score = np.mean(drge_lava_score_m, axis=0)
        drge_lava_score_se = stats.sem(drge_lava_score_m, axis=0)
    else:
        drge_lava_score = drge_lava_score_se = 0
    drge_davinz_score = np.mean(drge_davinz_score_m, axis=0)
    drge_davinz_score_se = stats.sem(drge_davinz_score_m, axis=0)
    drge_random_score = np.mean(drge_random_score_m, axis=0)
    drge_random_score_se = stats.sem(drge_random_score_m, axis=0)

    drge_if_score_l2h = np.mean(drge_if_score_l2h_m, axis=0)
    drge_if_score_l2h_se = stats.sem(drge_if_score_l2h_m, axis=0)
    drge_trackin_score_l2h = np.mean(drge_trackin_score_l2h_m, axis=0)
    drge_trackin_score_l2h_se = stats.sem(drge_trackin_score_l2h_m, axis=0)
    if not regression:
        drge_lava_score_l2h = np.mean(drge_lava_score_l2h_m, axis=0)
        drge_lava_score_l2h_se = stats.sem(drge_lava_score_l2h_m, axis=0)
    else:
        drge_lava_score_l2h = drge_lava_score_l2h_se = 0
    drge_davinz_score_l2h = np.mean(drge_davinz_score_l2h_m, axis=0)
    drge_davinz_score_l2h_se = stats.sem(drge_davinz_score_l2h_m, axis=0)
    drge_random_score_l2h = np.mean(drge_random_score_l2h_m, axis=0)
    drge_random_score_l2h_se = stats.sem(drge_random_score_l2h_m, axis=0)
    
    length_score = len(drge_deviation_score)
    magnitude_se = 0.5
    baseline_names = ["Deviation", "LOO", "Influence", "TracIn", "LAVA", "DAVINZ", "Random"]
    means = [drge_deviation_score_l2h, drge_loo_score_l2h, drge_if_score_l2h, drge_trackin_score_l2h, drge_lava_score_l2h, drge_davinz_score_l2h, drge_random_score_l2h]
    ses = [drge_deviation_score_l2h_se, drge_loo_score_l2h_se, drge_if_score_l2h_se, drge_trackin_score_l2h_se, drge_lava_score_l2h_se, drge_davinz_score_l2h_se, drge_random_score_l2h_se]

    colors = ['C10', 'C2', 'C4', 'C6', 'C1', 'C3', 'C9', 'C6', 'C7', 'C8', 'C5']
    #, 'C9', 'C6', 'C7', 'C8', 'C5']
    markers = [r'$\heartsuit$', r'$\boxdot$', r'$\bigcirc$', r'$\bigtriangleup$',  r'$\diamondsuit$', r'$\heartsuit$', 'x']
    markers = markers[::-1]
    every_marker = int(length_score/8)
    #, r'$\boxdot$', r'$\bigcirc$', r'$\bigtriangleup$', r'$\diamondsuit$', r'$\heartsuit$', '2', '3', '4']
    # for i in range(len(means)):
    for i in range(len(baseline_names)):
        if regression and baseline_names[i] == "LAVA":
            continue
        # plt.plot(drge_deviation_score, label="Deviation")
        # plt.fill_between(range(length_score),drge_deviation_score - magnitude_se * drge_deviation_score_se, drge_deviation_score +  magnitude_se *drge_deviation_score_se, alpha=0.2)
        axes[j].plot(show_idx, means[i], marker=markers[i % len(markers)], color=colors[i % len(colors)], linestyle='--', label=baseline_names[i], markevery=every_marker)
        axes[j].fill_between(show_idx, means[i] - magnitude_se * ses[i], means[i] + magnitude_se * ses[i], alpha=0.2, color=colors[i % len(colors)])

        axes[j].set_title(names[j])
        
        if j == 0:
            if args.metric == 'loss':
                if args.selected:
                    axes[j].set_ylabel(r'Worst-case loss', fontsize=selected_fontsize)
                else:
                    axes[j].set_ylabel(r'Worst-case loss')
            else:
                if args.selected:
                    axes[j].set_ylabel(r'DRGE', fontsize=selected_fontsize)
                else:
                    axes[j].set_ylabel(r'DRGE')
        
        # if j == len(names) // 2:
        #     axes[j].set_xlabel(r'$\#$ data points removed')

    # axes[j].yaxis.set_major_formatter(FormatStrFormatter('%.f'))
    axes[j].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axes[j].set_ylim([args.lb, args.ub])
    
figs.tight_layout(rect=[0, 0.07, 1, 0.99], h_pad=0, w_pad=10)
if args.selected:
    figs.text(0.5, 0.04, '$\#$ data points removed from low to high', ha='center', fontsize=selected_fontsize)
else:
    figs.text(0.5, 0.04, '$\#$ data points removed from low to high', ha='center')
# figs.tight_layout()
figs.set_figheight(2.5)

figs.set_figwidth(2*len(names))

# plt.legend(loc="lower right", prop={'size': 8})
if args.selected:
    plt.savefig(f'figs/vary-epsilon-removing_l2h-{args.metric}-selected.pdf', bbox_inches="tight")
else:
    plt.savefig(f'figs/vary-epsilon-removing_l2h-{args.metric}.pdf', bbox_inches="tight")