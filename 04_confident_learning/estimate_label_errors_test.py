# cleanlab code for computing the 5 confident learning methods.
# psx is the n x m matrix of cross-validated predicted probabilities
# s is the array of noisy labels

import numpy as np
import os

import cleanlab
# from cleanlab import baseline_methods
from cleanlab.latent_estimation import compute_confident_joint

import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import argparse
import time
import copy
import csv

def get_probs(loader, model, args):
    # Switch to evaluate mode.
    model.eval()
    n_total = len(loader.dataset.imgs) / float(loader.batch_size)
    outputs = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(loader):
            print("\rComplete: {:.1%}".format(i / n_total), end="")
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            outputs.append(model(input))

    # Prepare outputs as a single matrix
    probs = np.concatenate([
        torch.nn.functional.softmax(z, dim=1).cpu().numpy()
        for z in outputs
    ])

    return probs

num_classes = 10
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Estimate Label Errors')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('--cifar10h_data', metavar='DIR', default='../00_data',
                    help='path to dataset')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel.'
                         'Use 128 for Co-Teaching.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--fig_dir', type=str, default='./figs',
                    help='path to save figures')
parser.add_argument('--img_dir', type=str, default='./images',
                    help='path to save images')
parser.add_argument('--model_dir', type=str, default='.',
                    help='path to saved pretrained models')
parser.add_argument('--save_dir', type=str, default='./saved',
                    help='path to save results')
parser.add_argument('--prune', default='conf_joint', type=str, 
                    choices=['conf_joint','confusion','pbc','pbnr','c+nr'],
                    help='type of pruning method')
parser.add_argument('--lu_thres', type=float, default=0.5,
                    help='lu threshold for precision-recall curve')
parser.add_argument('--is_vis', action='store_true', default=True,
                    help='whether visualize the error images')
args = parser.parse_args()


args.fig_dir += '/'+args.arch
args.img_dir += '/'+args.arch

if not os.path.exists(args.fig_dir):
    os.makedirs(args.fig_dir)

if not os.path.exists(args.img_dir):
    os.makedirs(args.img_dir)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

print("===== Construct Confident Joint for Testing Data")
# Prepare testing labels from CIFAR-10 (CL)
labels_cl = [label for img, label in
            datasets.ImageFolder(os.path.join(args.data, "test/")).imgs]
n_exp = len(labels_cl)

# Record the image index in CIFAR-10 testing dataset
cifar_cl_inds = []
for img, label in datasets.ImageFolder(os.path.join(args.data, "test/")).imgs:
    index = os.path.basename(img).rpartition('_')[0]
    cifar_cl_inds.append(int(index))

# Load CIFAR-10H human soft labels & compute label uncertainty
cifar10 = datasets.CIFAR10(args.cifar10h_data, download=True, 
                            train=False, transform=transforms.ToTensor())    
data_loader = torch.utils.data.DataLoader(cifar10, batch_size=n_exp)
_, labels = iter(data_loader).next()

soft_labels = np.load(args.cifar10h_data+'/cifar10h-probs.npy')
probs_fstar = np.array([ soft_labels[i, labels[i]] for i in range(n_exp) ])

soft_labels_copy = copy.deepcopy(soft_labels)
for i in range(n_exp):
    soft_labels_copy[i, labels[i]] = 0
probs_remain_top1 = np.max(soft_labels_copy, axis=1)
label_uncertainty = 1 - (probs_fstar - probs_remain_top1)

# Reorder label uncertainty based on the cifar_cl_inds
lu = label_uncertainty[cifar_cl_inds]
y_probs = torch.tensor(soft_labels[cifar_cl_inds])      # human label distribution

# Sanity check for label matching
labels_sorted = labels[cifar_cl_inds]
if labels_sorted.tolist() == labels_cl:
    print('Indices are matched!')
else:
    raise ValueError('There exists mismatch between two datasets!')

# Prepare testing data loader
testdir = os.path.join(args.data, 'test')
test_dataset = datasets.ImageFolder(
    testdir,
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]),
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True,
)

# Load the pretrained CIFAR-10 model
print("=> creating model '{}'".format(args.arch))
model = models.__dict__[args.arch](num_classes=num_classes)
if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
else:
    model = torch.nn.DataParallel(model).cuda()

checkpoint = torch.load(args.model_dir+"/model_"+args.arch+"_best.pth.tar")
model.load_state_dict(checkpoint['state_dict'])

psx = get_probs(test_loader, model, args)   # compute model predicted probs

# estimate label uncertainty using model predicted probs
est_probs_fstar = np.array([ psx[i][labels_cl[i]] for i in range(n_exp) ])
psx_copy = copy.deepcopy(psx)
for i in range(n_exp):
    psx_copy[i][labels_cl[i]] = 0
est_probs_remain_top1 = np.max(psx_copy, axis=1)
est_lu = 1 - (est_probs_fstar - est_probs_remain_top1)

# Compute overall accuracy
print('')
print('Computing Accuracy.', flush=True)
acc = sum(np.array(labels_cl) == np.argmax(psx, axis=1)) / float(n_exp)
print('Accuracy: {:.25}'.format(acc))

# Method: C_{\tilde{y}, y^*}
if args.prune == 'conf_joint':
    print('===== Prune by Confident Joint =====')
    label_error_mask = np.zeros(n_exp, dtype=bool)
    label_error_indices = compute_confident_joint(
        labels_cl, psx, return_indices_of_off_diagonals=True
    )[1]
    for idx in label_error_indices:
        label_error_mask[idx] = True
    baseline_conf_joint_only = label_error_mask

    err_inds = np.array([i for i, x in enumerate(baseline_conf_joint_only) if x])
    err_inds = err_inds[np.argsort(-est_lu[err_inds])]
    non_err_inds = [i for i in range(n_exp) if i not in err_inds]

    print('# of estimated errors:', len(err_inds))
    print('Average label uncertainty (err):', np.mean(lu[err_inds]))
    print('Average label uncertainty (non err):', np.mean(lu[non_err_inds]))

# Method: CL: PBNR
elif args.prune == 'pbnr':
    print('===== Prune by Noise Rate =====')
    err_inds = cleanlab.pruning.get_noise_indices(
                labels_cl, psx, 
                prune_method='prune_by_noise_rate', 
                frac_noise=1.0, 
                sorted_index_method='normalized_margin')
    non_err_inds = [i for i in range(n_exp) if i not in err_inds]

    print('# of estimated errors:', len(err_inds))
    print('Average label uncertainty (err):', np.mean(lu[err_inds]))
    print('Average label uncertainty (non-err):', np.mean(lu[non_err_inds]))

# save the error indices (from confident learning cifar10 testing dataset)
np.save(args.save_dir+'/est_err_inds_'+args.arch+'_'+args.prune+'_test.npy', err_inds)

# Histogram of human label uncertainty & estimated label uncertainty
plt.figure()
hist = plt.hist(lu, bins=50)
# plt.title('Distribution of Label Uncertainty')
plt.xlabel("Human Label Uncertainty")
plt.ylabel("Frequency")
plt.savefig('./figs/hum_lu_histogram.png')

plt.figure()
hist = plt.hist(est_lu, bins=50)
# plt.title('Distribution of Label Uncertainty')
plt.xlabel("Estimated Label Uncertainty")
plt.ylabel("Frequency")
plt.savefig(args.fig_dir+'/est_lu_histogram.png')


# precision-recall curve
print('label uncertainty threshold:', args.lu_thres)
inds_est_lu_sorted = np.argsort(-est_lu)

q_arr = np.arange(0.01, 1.0, 0.01)
P_arr = []
R_arr = []
F1_arr = []

for q in q_arr:  
    idx_pivot = int(np.floor(q * n_exp))
    inds_selected = inds_est_lu_sorted[range(idx_pivot)]
    inds_unselected = inds_est_lu_sorted[np.arange(idx_pivot, n_exp)]

    tp = np.sum(lu[inds_selected] >= args.lu_thres)
    fp = np.sum(lu[inds_selected] < args.lu_thres)
    fn = np.sum(lu[inds_unselected] >= args.lu_thres)
    tn = np.sum(lu[inds_unselected] > args.lu_thres)

    P = tp / (tp+fp)
    R = tp / (tp+fn)
    F1 = 2*P*R / (P+R)
    
    P_arr.append(P)
    R_arr.append(R)
    F1_arr.append(F1)

    # print('===== '+str(q)+' =====')
    # print(tp, fp, fn, tn)
    # print(P, R, F1)

plt.figure()
plt.plot(R_arr, P_arr, linewidth=2.0)
plt.xticks(np.arange(0.00, 1.01, step=0.20), fontsize=12)
plt.yticks(np.arange(0.00, 0.26, step=0.05),fontsize=12)


# plt.title('Precision Recall Curves (lu_thres = '+str(args.lu_thres)+')')
plt.xlabel("Recall Rate", fontsize=14)
plt.ylabel("Precision Rate", fontsize=14)
plt.savefig(args.fig_dir+'/precision_recall_'+args.prune+'_thres_'+str(args.lu_thres)+'.png')

# Histogram of label uncertainty of estimated error vs non-error images
# plt.figure()
# weights = np.ones_like(lu[err_inds]) / len(lu[err_inds])
# plt.hist(lu[err_inds], bins=50, weights=weights)
# # plt.title('Label Uncertainty Distribution of Error Images')
# plt.xlabel("Human Label Uncertainty")
# plt.ylabel("Density")
# plt.savefig(args.fig_dir+'/err_hist_'+args.prune+'.png')

# plt.figure()
# weights = np.ones_like(lu[non_err_inds]) / len(lu[non_err_inds])
# plt.hist(lu[non_err_inds], bins=50, weights=weights)
# # plt.title('Label Uncertainty Distribution of Non-Error Images')
# plt.xlabel("Human Label Uncertainty")
# plt.ylabel("Density")
# plt.savefig(args.fig_dir+'/non_err_hist_'+args.prune+'.png')

plt.figure()
bins = np.linspace(0.0, 2.0, 50)
# weights_err = np.ones_like(lu[err_inds]) / len(lu[err_inds])
# plt.hist(lu[err_inds], bins=bins, alpha=0.5, 
#          color='red', weights=weights_err, label='err')
# weights_non_err = np.ones_like(lu[non_err_inds]) / len(lu[non_err_inds])
# plt.hist(lu[non_err_inds], bins=bins, alpha=0.5, 
#          color='green', weights=weights_non_err, label='non-err')
weights_err = np.ones_like(lu[err_inds]) / len(lu[err_inds])
weights_non_err = np.ones_like(lu[non_err_inds]) / len(lu[non_err_inds])
plt.hist([lu[err_inds], lu[non_err_inds]], bins=bins, color=['darkorange', 'forestgreen'],
         weights=[weights_err, weights_non_err], label=['err', 'non-err'])
plt.legend(loc='upper right', fontsize=14)
# plt.title('Label Uncertainty Distribution')
plt.xlabel("Human Label Uncertainty", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(args.fig_dir+'/err_vs_non_err_hist_'+args.prune+'.png')

# Plot the top CIFAR-10 test images (sorted by estimated label uncertainty)
if args.is_vis:
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    test_dataset = datasets.ImageFolder(testdir, transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=n_exp, 
                                            shuffle=False, pin_memory=True)
    data_iter = iter(testloader)
    images_cl, _ = data_iter.next()

    n_rows = 4
    n_cols = 4
    n_exps = n_rows * n_cols // 2   # number of examples per batch
    n_batches = 20

    inds_est_lu_sorted = np.argsort(-est_lu)[:n_batches*n_exps]
    # imgs_est_lu_sorted = images[inds_est_lu_sorted]

    x = np.arange(0, 19, 2)
    for b in range(n_batches):
        fig = plt.figure(figsize=(8,8))

        for i in range(n_exps):
            ## draw the bar chart to the left
            pos = 2*i+1
            ax = plt.subplot(n_rows, n_cols, pos)

            ind = inds_est_lu_sorted[b*n_exps+i]
            probs_est = psx[ind]
            probs_hum = y_probs[ind].numpy()

            ax.bar(x-0.3, probs_hum, width=0.6, align='center', color='darkorange')
            ax.bar(x+0.3, probs_est, width=0.6, align='center', color='forestgreen')
            
            if pos // n_cols == 0:
                ax.legend(['hum', 'est'], bbox_to_anchor=(0.5, 1.17), fontsize=8, 
                        loc=9, ncol=3, frameon=False, columnspacing=0.5, handletextpad=0.3)

            ## define the xticks
            plt.xticks(x, range(10), fontsize=8)
            # if np.ceil(pos/n_cols) == n_rows:
            #     plt.xticks(x, range(10), fontsize=8)
            # else:
            #     plt.xticks([])
            ## define the yticks
            if pos % n_cols == 1:
                plt.yticks(np.arange(0, 1.01, 0.2), fontsize=8)
            else: 
                plt.yticks([])

            ## plot the cifar10 image to the right with lu score
            img = images_cl[ind].numpy().transpose(1,2,0)
            pos_row = i // (n_cols//2) + 1
            pos_col = i % (n_cols//2) + 1
        
            newax = fig.add_axes([-0.08+0.41*pos_col, 0.918-0.201*pos_row, 0.15, 0.15], anchor='NE')
            newax.imshow(img)
            newax.axis('off')

            textax = fig.add_axes([-0.08+0.41*pos_col, 0.9-0.201*pos_row, 0.15, 0.15], anchor='NE')
            textax.text(0, 0, 'hum: {:.2f}, est: {:.2f}'.format(lu[ind], est_lu[ind]), fontsize=9)
            textax.text(0, 1.16, 'CIFAR-10 label: {}'.format(labels_cl[ind]), color='blue', fontsize=8)
            textax.axis('off')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.5, 0.05, "0-airplane, 1-automobile, 2-bird, 3-cat, 4-deer, 5-dog, 6-frog, 7-horse, 8-ship, 9-truck", 
                horizontalalignment='center', verticalalignment='center', fontsize=12, transform=plt.gcf().transFigure, bbox=props)
        plt.savefig(args.img_dir+'/imgs_top_lu_batch_'+str(b)+'.png')
        plt.close(fig)

