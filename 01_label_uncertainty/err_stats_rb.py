import waitGPU
waitGPU.wait(utilization=20, available_memory=10000, interval=20)

import numpy as np
import matplotlib.pyplot as plt
import copy

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td
import os, json

from utils import load_model

#### specify parameters
data_dir = '../00_data'
batch_size_eval = 50
model_dir = './models/robustbench'
norm = 'Linf'
# norm = 'L2'

if not os.path.exists('./log'):
		os.makedirs('./log')

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
label_uncertainty = 1 - (probs_fstar - probs_remain_top1)

print( '(uncertainty) 0.95 quantile: {:.4f}, 0.9 quantile: {:.4f}, 0.8 quantile: {:.4f}, 0.7 quantile: {:.4f}'.format(
        np.quantile(label_uncertainty, 0.95), np.quantile(label_uncertainty, 0.9),
        np.quantile(label_uncertainty, 0.8), np.quantile(label_uncertainty, 0.7)) )


#### load all json files stored in robustbench
print('========== evaluating adversarially-trained models (in RobustBench)')
path_to_json = './model_info/'+norm
err_log = open('./log/robustbench_'+norm+'_err_stats.txt', "w")

json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
for index, js in enumerate(json_files):
    model_name = js[:-5]
    from model_zoo_old.models import model_dicts as all_models
    model_dicts = all_models[norm]
    if model_name not in model_dicts:   ## not all model_name saved
        continue
    else:
        model = load_model(model_name, model_dir, norm).cuda()

    ## record the error image indices
    error_inds = []
    for i, (X, y) in enumerate(test_batches):   
        X, y = X.cuda(), y.cuda()

        is_err = (model(X).max(1)[1] != y)
        inds_err = is_err.nonzero(as_tuple=False).squeeze().cpu().numpy()
        inds_err += i * batch_size_eval

        if type(inds_err.tolist()) == int:
            error_inds.append(inds_err.tolist())
        else:
            error_inds.extend(inds_err.tolist())
   
    #### compute and save the error region statistics
    clean_err = len(error_inds) / y_probs.size(0)

    label_uncertainty_err = np.mean(label_uncertainty[error_inds])

    #### keep the documented robust accuracy (evaluated using autoattack)
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)
        robust_acc = float(json_text['AA']) / 100      
        is_extra = json_text['additional_data']

    print('[{}] clean err: {:.4f}, uncertainty: {:.4f}, robust acc: {:.4f}, extra data: {}'.format(
            model_name, clean_err, label_uncertainty_err, robust_acc, is_extra)) 
    
    print(model_name, clean_err, label_uncertainty_err, robust_acc, is_extra, file=err_log)
    err_log.flush()

            


