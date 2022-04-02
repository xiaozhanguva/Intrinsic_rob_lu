# import waitGPU
# waitGPU.wait(utilization=50, available_memory=50000, interval=60)

import torch
import argparse
import numpy as np
import copy
import os

from utils import load_model, clean_accuracy
from data import load_cifar10
from autoattack import AutoAttack


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Carmon2019Unlabeled')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--n_ex', type=int, default=10000, help='number of examples to evaluate on')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size for evaluation')
    parser.add_argument('--data_dir', type=str, default='../00_data', help='where to store downloaded datasets')
    parser.add_argument('--model_dir', type=str, default='./models', help='where to store downloaded models')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for computations')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    device = torch.device(args.device)

    if not os.path.exists('./attack_log'):
            os.makedirs('./attack_log')
    attack_std_log = './attack_log/std_'+args.norm+'_'+args.model_name+'.npy'
    attack_rob_log = './attack_log/rob_'+args.norm+'_'+args.model_name+'.npy'

    x_test, y_test = load_cifar10(args.n_ex, args.data_dir)
    x_test, y_test = x_test.to(device), y_test.to(device)

    #### evaluate the clean and robust acc
    model = load_model(args.model_name, args.model_dir, args.norm).to(device).eval()

    acc = clean_accuracy(model, x_test, y_test, batch_size=args.batch_size, device=device)
    print('Clean accuracy: {:.2%}'.format(acc))

    adversary = AutoAttack(model, norm=args.norm, eps=args.eps, version='standard', device=device)
    _, normal_flags, robust_flags = adversary.run_standard_evaluation(x_test, y_test)
    
    np.save(attack_std_log, normal_flags.cpu().numpy())
    np.save(attack_rob_log, robust_flags.cpu().numpy())


