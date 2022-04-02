#### Adapt from the Code Repo of the paper 'Empirically Measuring Concentration: Fundamental Limits on Intrinsic Robustness'

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td

import argparse
import parser

import numpy as np 
import random
from pathlib import Path
# import setproctitle

import os
import torch
import copy
import time
import math

## soft dependencies for knn
try:
    import scipy.spatial.distance as _spd
    import scipy.special as _spspec
    _HAS_SCIPY = True
except:
    _HAS_SCIPY = False

try:
    import sklearn.neighbors as _sknbr
    _HAS_SKLEARN = True
except:
    _HAS_SKLEARN = False

#### knn graph construction (adapted from DeBaCl library)
def knn_graph(X, k, method='brute_force', leaf_size=30, metric='euclidean'):
	n, p = X.shape
	if method == 'kd_tree':
		if _HAS_SKLEARN:
			kdtree = _sknbr.KDTree(X, leaf_size=leaf_size, metric=metric)
			distances, neighbors = kdtree.query(X, k=k, return_distance=True,
												sort_results=True)
			radii = distances[:, -1]
		else:
			raise ImportError("The scikit-learn library could not be loaded." +
								" It is required for the 'kd-tree' method.")

	if method == 'ball_tree':
		if _HAS_SKLEARN:
			btree = _sknbr.BallTree(X, leaf_size=leaf_size, metric=metric)
			distances, neighbors = btree.query(X, k=k, return_distance=True,
												sort_results=True)
			radii = distances[:, -1]
		else:
			raise ImportError("The scikit-learn library could not be loaded." +
								" It is required for the 'ball-tree' method.")

	else:  # assume brute-force
		if not _HAS_SCIPY:
			raise ImportError("The 'scipy' module could not be loaded. " +
								"It is required for the 'brute_force' method " +
								"for building a knn similarity graph.")

		d = _spd.pdist(X, metric=metric)
		D = _spd.squareform(d)
		rank = np.argsort(D, axis=1)
		neighbors = rank[:, 0:k]
		k_nbr = neighbors[:, -1]
		radii = D[np.arange(n), k_nbr]

	return neighbors, radii


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
	parser.add_argument('--data_dir', type=str, default='../00_data', help='where to store downloaded datasets')
	parser.add_argument('--ratio', type=float, default=0.5, help='ratio of training dataset')
	parser.add_argument('--metric', type=str, default='L2', help='type of perturbations')
	parser.add_argument('--epsilon', type=float, default=0.5, help='perturbation strength')	
	parser.add_argument('--n_cluster', type=int, default=5, help='total number of balls allowed')

	parser.add_argument('--delta', type=float, default=0.05, help='additional training points to reduce overfitting')
	parser.add_argument('--alpha', type=float, default=0.05, help='risk threshold') 
	parser.add_argument('--gamma', type=float, default=0.17, help='label uncertainty threshold')
	parser.add_argument('--repeat', type=int, default=3, help='number of repeated trials')
	parser.add_argument('--verbose', type=int, default=20000)
	args = parser.parse_args()

	print('metric: {metric}, ' 'epsilon: {epsilon}, ' '#balls: {n_cluster}, ' 
			'alpha: {alpha}, ' 'gamma: {gamma} '.format(
			metric=args.metric, epsilon=args.epsilon, n_cluster=args.n_cluster,
			alpha=args.alpha, gamma=args.gamma))

	res_filepath = ('./results/'+args.metric+'/eps_'+str(args.epsilon)+'_alpha_'+str(args.alpha)
						+'_gamma_'+str(args.gamma)+'_n_clusters_'+str(args.n_cluster)+'.txt')
	if not os.path.exists(os.path.dirname(res_filepath)):
		os.makedirs(os.path.dirname(res_filepath))
	print("saving file to {}".format(res_filepath))
	res_file = open(res_filepath, "w")

	#### conduct repeated experiments
	risk_train, risk_test = np.zeros(args.repeat), np.zeros(args.repeat)
	advRisk_train, advRisk_test = np.zeros(args.repeat), np.zeros(args.repeat)
	err_reg_lu_train, err_reg_lu_test = np.zeros(args.repeat), np.zeros(args.repeat)

	for j in range(args.repeat):
		print('=============== Trial No.'+str(j)+ ' ===============')

		#### load the CIFAR-10 and CIFAR-10H datasets
		if args.dataset == 'cifar10':
			cifar10 = datasets.CIFAR10(args.data_dir, download=True, 
										train=False, transform=transforms.ToTensor())    
			data_loader = td.DataLoader(cifar10, batch_size=len(cifar10))
			X, y = iter(data_loader).next()
			inputs = X.view(-1, 3*32*32)
			labels = y

			soft_labels = np.load(args.data_dir+'/cifar10h-probs.npy')
			probs_fstar = np.array([ soft_labels[i, labels[i]] for i in range(len(labels)) ])

			soft_labels_copy = copy.deepcopy(soft_labels)
			for i in range(len(labels)):
				soft_labels_copy[i, labels[i]] = 0
			probs_remain_top1 = np.max(soft_labels_copy, axis=1)
			lu = 1 - (probs_fstar - probs_remain_top1)

			## generate training and testing datasets
			np.random.seed()

			n_total, n_train = len(cifar10), int(args.ratio*len(cifar10))
			train_inds = np.random.choice(len(cifar10), size=n_train, replace=False)
			test_inds = np.delete(np.arange(n_total), train_inds)

			print('ratio: {:.3f}, train size: {}, test size: {}'.format(
					args.ratio, len(train_inds), len(test_inds)))

			train_data, test_data = inputs[train_inds, :], inputs[test_inds, :]
			train_labels, test_labels = labels[train_inds], labels[test_inds]
			train_lu, test_lu = lu[train_inds], lu[test_inds]

			print('label uncertainty (train, test):({:.4f}, {:.4f})'.format(
					np.mean(train_lu), np.mean(test_lu)))
			print('')

		else:
			raise ValueError('specified dataset name not recognized.')

		#### obatin the k-nearest neighbours for each points in the training set
		n_train = train_data.shape[0]
		args.k = math.ceil(n_train*args.alpha)*2

		print('===== creating knn graph (k = '+str(args.k)+') ...')
		start_time = time.time()
		neighbor_ind_arr, _ = knn_graph(train_data, k=args.k, method='ball-tree', metric='euclidean')
		elapsed  = time.time() - start_time 
		print('knn graph created in {:.2f}s'.format(elapsed))

		## number of points to cover (add additional points to avoid overfitting)
		n_point_tot = math.ceil(n_train * args.alpha * (1 + args.delta))
		print('Total number of data points to cover:', n_point_tot)

		#### place the balls incrementally for the smallest expansion
		centroids = []
		radii = []
		index_init = []
		index_expand = []
		for t in range(args.n_cluster):
			print('')
			if len(index_init) >= n_point_tot:
				break
			start = time.time()

			## set the range of #covered points for the current ball
			n_lower = math.ceil((n_point_tot-len(index_init)) / (args.n_cluster-t))
			n_upper = n_point_tot-len(index_init)
			print('number of points to cover: ['+str(n_lower)+':'+str(n_upper)+']')

			if n_upper <= 1:
				break

			## record the indices for remaining data points
			ind_valid = np.array([x for x in range(n_train) if x not in index_init])
			ind_valid_expand = np.array([x for x in range(n_train) if x not in index_expand])

			n_expand_opt = n_train
			iter_count = 0
			
			## for each training data as center, find the minimum expansion
			for i in range(n_train):
				i_neighors = neighbor_ind_arr[i,:]
				neighbor_ind = i_neighors[~np.in1d(i_neighors, index_init)]

				## set the center at each training data point (faster, generalize better)
				center = train_data[i]
				dist = torch.sqrt(torch.sum((train_data[ind_valid] - center)**2, dim=1))
				dist_expand = torch.sqrt(torch.sum((train_data[ind_valid_expand] - center)**2, dim=1))

				## enumerate each value in [n_lower, n_upper], find the minimum expansion
				n_point_arr = range(n_lower-1,n_upper)
				for s in n_point_arr:
					iter_count += 1

					radius = torch.sqrt(torch.sum((train_data[neighbor_ind[s+1]] - center)**2)) - 1e-6		#reduce overfitting
					count_init = torch.sum(dist <= radius).double()
					count_expand = torch.sum(dist_expand <= radius + args.epsilon).double()				
					n_expand = count_expand - count_init

					## record the indices (initial coverage and expanded coverage)
					ind_init = ind_valid[torch.nonzero(dist <= radius).view(-1)]
					ind_expand_total = ind_valid[torch.nonzero(dist <= radius + args.epsilon).view(-1)]
					ind_expand = np.setdiff1d(ind_expand_total, index_expand)
					lu_err_reg = np.mean(train_lu[ind_init])	## record the error region label uncertainty

					## print the intermediate result
					if iter_count % args.verbose == 0:
						print('Progress: {prog:.2%}\t\t' 'Initial: {init}\t\t' 'Expanded: {expand}\t\t'
								'Uncertainty: {lu:.3f}\t\t' 'Radius: {radius:.2f}'.format(
								prog=float(iter_count) / float(n_train*len(n_point_arr)), init=len(ind_init), 
								expand=len(ind_expand), lu=lu_err_reg, radius=radius))
				
					## continue if error region label uncertainty is small
					if lu_err_reg < args.gamma:
						continue
					else:
						## record the statistics w.r.t. the smallest expansion
						if n_expand < n_expand_opt:	
							n_expand_opt = n_expand	
							init_opt = ind_init
							expand_opt = ind_expand
							center_opt = center
							radius_opt = radius
							lu_opt = lu_err_reg
				
			## record the optmal center and radius
			index_init.extend(init_opt)
			index_expand.extend(expand_opt)
			centroids.append(center_opt)
			radii.append(radius_opt)
			iter_time = time.time() - start

			print(' * Placed-balls [{0}/{1}]\t\t'
					'Time: {time:.2f}\t\t'
					'Initial: {init_opt}\t\t'
					'Expanded: {expand_opt}\t\t'
					'Uncertainty: {lu_opt:.3f}\t\t'
					'Radius: {radius:.4f}'.format(
						t+1, args.n_cluster, time=iter_time, 
						init_opt=len(init_opt), expand_opt=len(expand_opt), 
						lu_opt=lu_opt, radius=radius_opt))

		risk_train[j] = len(index_init) / float(n_train)
		advRisk_train[j] = len(index_expand) / float(n_train)
		err_reg_lu_train[j] = np.mean(train_lu[index_init])

		print('')
		print('Risk for train data:', '{:.2%}'.format(risk_train[j]))
		print('Adversarial risk for train data:', '{:.2%}'.format(advRisk_train[j]))
		print('Error region label uncertainty for train data:', '{:.3f}'.format(err_reg_lu_train[j]))

		#### test the generalization on testing data
		count= 0
		count_expand = 0
		err_inds = []
		n_test = len(test_data)

		print('========== testing ==========')
		for idx, data in enumerate(test_data):
			if idx % args.verbose == 0:
				print('Iteration [{0}/{1}]\t\t'.format(idx, len(test_data)))

			for num, center in enumerate(centroids):
				diff = torch.sqrt(torch.sum((data-center)**2))	
				if diff <= radii[num]:
					count += 1
					err_inds.append(idx)
					break
			for num, center in enumerate(centroids):
				diff = torch.sqrt(torch.sum((data-center)**2))	
				if diff <= radii[num]+args.epsilon:
					count_expand += 1
					break

		risk_test[j] = count / float(n_test)
		advRisk_test[j] = count_expand / float(n_test)
		err_reg_lu_test[j] = np.mean(test_lu[err_inds])

		print('Risk for test data:', '{:.2%}'.format(risk_test[j]))
		print('Adversarial risk for test data:', '{:.2%}'.format(advRisk_test[j]))
		print('Error region label uncertainty for test data:', '{:.3f}'.format(err_reg_lu_test[j]))

	#### save the results
	print(args.epsilon, args.alpha, args.gamma, args.n_cluster, 
		'{:.2%}'.format(np.mean(risk_train)), '({:.2%})'.format(np.std(risk_train)), 
		'{:.2%}'.format(np.mean(risk_test)), '({:.2%})'.format(np.std(risk_test)), 
		'{:.2%}'.format(np.mean(advRisk_train)), '({:.2%})'.format(np.std(advRisk_train)), 
		'{:.2%}'.format(np.mean(advRisk_test)), '({:.2%})'.format(np.std(advRisk_test)), 
		'{:.4f}'.format(np.mean(err_reg_lu_train)), '({:.4f})'.format(np.std(err_reg_lu_train)), 
		'{:.4f}'.format(np.mean(err_reg_lu_test)), '({:.4f})'.format(np.std(err_reg_lu_test)), file=res_file)
	res_file.flush()

