import waitGPU
waitGPU.wait(utilization=20, available_memory=10000, interval=20)

import numpy as np
import matplotlib.pyplot as plt
import utils
import copy
import models
import random

import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td
import os

#### specify parameters
batch_size_eval = 128
data_dir = '../00_data'

if not os.path.exists('./log'):
		os.makedirs('./log')

logging.getLogger('matplotlib.font_manager').disabled = True
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

#### load the cifar10 test data and cifar10-h data
cifar_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transforms.ToTensor())
test_batches = torch.utils.data.DataLoader(dataset=cifar_test, batch_size=batch_size_eval, shuffle=False, 
                            pin_memory=True, num_workers=2, drop_last=False)

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
label_confidence = probs_fstar
label_uncertainty = 1 - (probs_fstar - probs_remain_top1)


#### compute error stats for each saved standard-trained model
attack = 'none'
model_type_arr = ['resnet18']
# model_type_arr = ['small', 'large', 'resnet18', 'resnet50', 'wideresnet']

for model_type in model_type_arr:
    model_dir= './models/none/'+model_type
    print('========== model architecture:', model_type)
    
    err_log = open('./log/'+attack+'_'+model_type+'_err_stats.txt', "w")

    start_epoch, end_epoch, step = 9, 99, 10
    for epoch in np.arange(start_epoch, end_epoch+1, step):
        model = models.get_model(model_type)
        model = nn.DataParallel(model).cuda()
        model_path = model_dir+'/model_'+str(epoch)+'.pth'
        model.load_state_dict(torch.load(model_path))
        utils.model_eval(model)

        #### record the error image indices
        error_inds = []
        for i, (X, y) in enumerate(test_batches):   
            X, y = X.cuda(), y.cuda()

            output = model(utils.normalize(X))
            is_err = (output.max(1)[1] != y)
            inds_err = is_err.nonzero(as_tuple=False).squeeze().cpu().numpy()
            inds_err += i * batch_size_eval

            if type(inds_err.tolist()) == int:
                error_inds.append(inds_err.tolist())
            else:
                error_inds.extend(inds_err.tolist())

        #### compute and save the error region statistics
        ratio_err = len(error_inds) / y_probs.size(0)
        label_confidence_err = np.mean(label_confidence[error_inds])
        label_uncertainty_err = np.mean(label_uncertainty[error_inds])
        print('(error region) epoch: {}, measure: {:.4f}, confidence: {:.4f}, uncertainty: {:.4f}'.format(
                epoch, ratio_err, label_confidence_err, label_uncertainty_err)) 
        
        print(epoch, ratio_err, label_confidence_err, label_uncertainty_err, file=err_log)
        err_log.flush()


#### compute error stats for each saved adversarially-trained model (pgd, pgd-awp)
eps = 8.0
pgd_train_n_iters = 10
attack_arr = ['awp']
model_type_arr = ['resnet18']
# attack_arr = ['std', 'awp']
# model_type_arr = ['resnet18', 'wideresnet']

for model_type in model_type_arr:
    for attack in attack_arr:
        print('========== model architecture: {}, attack: {}'.format(model_type, attack))
        model_dir= './models/l_inf/'+model_type+'_pgd_'+attack
        err_log = open('./log/'+str(eps)+'_'+attack+'_'+str(pgd_train_n_iters)+'_'+model_type+'_err_stats.txt', "w")
        start_epoch, end_epoch, step = 9, 99, 10

        for epoch in np.arange(start_epoch, end_epoch+1, step):
            model = models.get_model(model_type)
            model = nn.DataParallel(model).cuda()
            model_path = model_dir+'/model_'+str(epoch)+'.pth'
            model.load_state_dict(torch.load(model_path))
            utils.model_eval(model)

            #### record the error image indices
            error_inds = []
            for i, (X, y) in enumerate(test_batches):   
                X, y = X.cuda(), y.cuda()

                output = model(utils.normalize(X))
                is_err = (output.max(1)[1] != y)
                inds_err = is_err.nonzero(as_tuple=False).squeeze().cpu().numpy()
                inds_err += i * batch_size_eval

                if type(inds_err.tolist()) == int:
                    error_inds.append(inds_err.tolist())
                else:
                    error_inds.extend(inds_err.tolist())

            #### compute and save the error region statistics
            ratio_err = len(error_inds) / y_probs.size(0)
            label_confidence_err = np.mean(label_confidence[error_inds])
            label_uncertainty_err = np.mean(label_uncertainty[error_inds])
            print('(error region) epoch: {}, measure: {:.4f}, confidence: {:.4f}, uncertainty: {:.4f}'.format(
                    epoch, ratio_err, label_confidence_err, label_uncertainty_err)) 
    
            print(epoch, ratio_err, label_confidence_err, label_uncertainty_err, file=err_log)
            err_log.flush()


#### randomly select error subsets and compute error statistics
alpha_lv = np.arange(5, 81, 5) / 100
rnd_log = open('./log/rnd_stats.txt', "w")

for alpha in alpha_lv:
    print('========== alpha:', alpha)
    k = int(alpha * labels.size(0))

    #### randomly generate center and its k nearest neighbors
    inds_center = random.sample(range(labels.size(0)), 1000)
    label_confidence_rnd, label_uncertainty_rnd = [], []

    for ind in inds_center:
        center = images[ind]
        _, inds_sorted = torch.sort(torch.sum((images - center)**2, axis=[1,2,3]))
        inds = inds_sorted[:k]

        label_confidence_rnd.append(np.mean(label_confidence[inds]))
        label_uncertainty_rnd.append(np.mean(label_uncertainty[inds]))
    
    print('(label confidence) mean: {:.4f}, std: {:.4f}'.format(
        np.mean(label_confidence_rnd), np.std(label_confidence_rnd)))
    print('(label uncertainty) mean: {:.4f}, std: {:.4f}'.format(
        np.mean(label_uncertainty_rnd), np.std(label_uncertainty_rnd)))

    print(alpha, np.mean(label_confidence_rnd), np.std(label_confidence_rnd), 
            np.mean(label_uncertainty_rnd), np.std(label_uncertainty_rnd), file=rnd_log)
    rnd_log.flush()
