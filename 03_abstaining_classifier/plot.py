import torch
import argparse
import numpy as np
import copy
import os

from data import load_cifar10
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Carmon2019Unlabeled')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--n_ex', type=int, default=10000, help='number of examples to evaluate on')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size for evaluation')
    parser.add_argument('--data_dir', type=str, default='./data', help='where to store downloaded datasets')
    parser.add_argument('--model_dir', type=str, default='./models', help='where to store downloaded models')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for computations')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    device = torch.device(args.device)

    if not os.path.exists('./figs'):
        os.makedirs('./figs')

    x_test, y_test = load_cifar10(args.n_ex, args.data_dir)
    x_test, y_test = x_test.to(device), y_test.to(device)

    y_soft_labels = np.load('../../cifar-10h/data/cifar10h-probs.npy')[:args.n_ex]
    probs_fstar = np.array([ y_soft_labels[i, y_test[i]] for i in range(len(y_test)) ])

    y_soft_labels_copy = copy.deepcopy(y_soft_labels)
    for i in range(len(y_test)):
        y_soft_labels_copy[i, y_test[i]] = 0
    probs_remain_top1 = np.max(y_soft_labels_copy, axis=1)
    lu_test = 1 - (probs_fstar - probs_remain_top1)


    normal_flags = np.load('./attack_log/std_'+args.norm+'_'+args.model_name+'.npy')
    robust_flags = np.load('./attack_log/rob_'+args.norm+'_'+args.model_name+'.npy')

    print(normal_flags)
    print(robust_flags)

## sort the images by label uncertainty, and plot accuracies vs abstaining ratio
    sorted_inds = np.argsort(lu_test)           ## lu from smallest to largest (0->2)
    ratio_arr = np.arange(0.00, 1.00, 0.001)
    print(ratio_arr)

    acc_arr = []
    rob_acc_arr = []
    for abs_ratio in ratio_arr:
        print('===== abstaining ratio: {:.2%}'.format(abs_ratio))
        n_selected = int(np.floor( (1 - abs_ratio) * len(lu_test) ))
        inds_certain = sorted_inds[:n_selected]

        acc_certain = np.sum(normal_flags[inds_certain])/ len(inds_certain)
        rob_acc_certain = np.sum(robust_flags[inds_certain]) / len(inds_certain)
        print('Clean acc: {:.2%}, Robust acc: {:.2%}'.format(acc_certain, rob_acc_certain))

        acc_arr.append(acc_certain)
        rob_acc_arr.append(rob_acc_certain)

    print('')

    plt.figure(figsize=(9, 6))
    plt.plot(ratio_arr, acc_arr, linewidth=3.0, color='mediumblue')
    plt.plot(ratio_arr, rob_acc_arr, linewidth=3.0, color='orangered')
    plt.legend(['clean acc', 'robust acc'], fontsize=16, loc=4)

    plt.xticks(np.arange(0.00, 0.51, step=0.10), fontsize=16)
    plt.yticks(np.arange(0.55, 1.01, step=0.05), fontsize=16)
    plt.xlim([0.0, 0.5])
    plt.ylim([0.55, 1.0])
    plt.xlabel("ratio of abstained high uncertainty inputs", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.grid(alpha=0.5, linestyle='--')


    plt.annotate('lu'+r'$\geq$'+'{:.2f}'.format(np.quantile(lu_test, 0.98)), (0.02, 0.55), 
            color='black', fontsize=12, xycoords='data', xytext=(0.05, 0.59), 
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            horizontalalignment='center', verticalalignment='top',)
    plt.annotate('lu'+r'$\geq$'+'{:.2f}'.format(np.quantile(lu_test, 0.90)), (0.10, 0.55), 
            color='black', fontsize=12, xycoords='data', xytext=(0.15, 0.59), 
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            horizontalalignment='center', verticalalignment='top',)
    plt.annotate('lu'+r'$\geq$'+'{:.2f}'.format(np.quantile(lu_test, 0.80)), (0.20, 0.55), 
            color='black', fontsize=12, xycoords='data', xytext=(0.25, 0.59), 
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            horizontalalignment='center', verticalalignment='top',)

    plt.savefig('./figs/abstain_ratio_'+args.norm+'_'+args.model_name+'.png')