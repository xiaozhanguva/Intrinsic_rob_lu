import numpy as np
import matplotlib.pyplot as plt
import utils
import copy

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td
import logging
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

logging.getLogger('matplotlib.font_manager').disabled = True

if not os.path.exists('./vis'):
    os.makedirs('./vis')

#### load the cifar10 test data and cifar10-h data
data_dir = '../00_data'
cifar_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transforms.ToTensor())
testloader = td.DataLoader(cifar_test, batch_size=10000, shuffle=False, pin_memory=True)

data_iter = iter(testloader)
images, labels = data_iter.next()

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
test_probs = np.load(data_dir+'/cifar10h-probs.npy')
y_probs = torch.tensor(test_probs)

#### compute the averaged label uncertainty for all cifar10 test data
probs_fstar = np.array( [y_probs[i, labels[i]].item() for i in range(len(labels))] )
y_probs_remain = copy.deepcopy(y_probs)
for i in range(len(labels)):
    y_probs_remain[i, labels[i]] = 0
probs_remain_top1 = torch.max(y_probs_remain, axis=1)[0].numpy()
label_uncertainty = 1 - (probs_fstar - probs_remain_top1)
print( '(uncertainty) mean: {:.4f}, 0.98 quantile: {:.4f}, 0.95 quantile: {:.4f}, 0.9 quantile: {:.4f}, 0.8 quantile: {:.4f}'.format(
         np.mean(label_uncertainty), np.quantile(label_uncertainty, 0.98), np.quantile(label_uncertainty, 0.95), 
         np.quantile(label_uncertainty, 0.9), np.quantile(label_uncertainty, 0.8)) )

#### plot the distribution of label uncertainty
plt.figure()
plt.hist(label_uncertainty, bins=50)
plt.xticks(np.arange(0, 2.01, 0.2))

plt.title('Distribution of Label Uncertainty (mean: {:.4f})'.format(np.mean(label_uncertainty)))
plt.xlabel("Example-wise Label Uncertainty", fontsize=14)
plt.ylabel("Frequency", fontsize=14)

plt.vlines(np.quantile(label_uncertainty, 0.98), 0, 6000, color='grey', linestyle='dashed', label='0.98 quantile')
plt.vlines(np.quantile(label_uncertainty, 0.9), 0, 6000, color='brown', linestyle='dashed', label='0.9 quantile')
plt.vlines(np.quantile(label_uncertainty, 0.8), 0, 6000, color='darkgreen', linestyle='dashed', label='0.8 quantile')
plt.legend()
plt.savefig('./vis/label_uncertainty_hist_all.png')


## plot the top uncertain images sorted by label uncertainty 
n_rows = 4
n_cols = 4
n_exps = n_rows * n_cols // 2   # number of examples per batch
n_batches = 50

inds_uncertain_sorted = np.argsort(-label_uncertainty)[:n_batches*n_exps]
imgs_uncertain_sorted = images[inds_uncertain_sorted]

x = np.arange(0, 19, 2)
for b in range(n_batches):
    fig = plt.figure(figsize=(8,8))

    for i in range(n_exps):
        ## draw the bar chart to the left
        pos = 2*i+1
        ax = plt.subplot(n_rows, n_cols, pos)

        ind = inds_uncertain_sorted[b*n_exps+i]
        probs = y_probs[ind].numpy()

        probs_fstar = np.zeros(len(probs))
        probs_fstar[labels[ind]] = 1
        
        ax.bar(x-0.3, probs, width=0.6, align='center')
        ax.bar(x+0.3, probs_fstar, width=0.6, align='center')
        
        if pos // n_cols == 0:
            ax.legend(['cifar10h', 'cifar10'], bbox_to_anchor=(0.5, 1.17), fontsize=8, 
                    loc=9, ncol=2, frameon=False, columnspacing=0.5, handletextpad=0.3)

        ## define the xticks
        plt.xticks(x, range(10), fontsize=8)
        if pos % n_cols == 1:
            plt.yticks(np.arange(0, 1.01, 0.2), fontsize=8)
        else: 
            plt.yticks([])

        ## plot the cifar10 image to the right with uncertainty score
        img = images[ind].numpy().transpose(1,2,0)
        pos_row = i // (n_cols//2) + 1
        pos_col = i % (n_cols//2) + 1
    
        newax = fig.add_axes([-0.08+0.41*pos_col, 0.918-0.201*pos_row, 0.16, 0.16], anchor='NE')
        newax.imshow(img)
        newax.axis('off')

        textax = fig.add_axes([-0.08+0.41*pos_col, 0.9-0.201*pos_row, 0.16, 0.16], anchor='NE')
        textax.text(0, 0, 'uncertainty: {:.3f}'.format(label_uncertainty[ind]), fontsize=10)
        textax.axis('off')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.5, 0.05, "0-airplane, 1-automobile, 2-bird, 3-cat, 4-deer, 5-dog, 6-frog, 7-horse, 8-ship, 9-truck", 
            horizontalalignment='center', verticalalignment='center', fontsize=12, transform=plt.gcf().transFigure, bbox=props)
    plt.savefig('./vis/imgs_uncertain_batch_'+str(b)+'.png')
    plt.close(fig)
