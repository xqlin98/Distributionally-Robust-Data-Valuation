import torch
from torch import nn, optim
from copy import deepcopy as dcopy

from utils.models_defined import CNN_Net, CNNCifar_10, Flower_LR

use_cuda = True
cuda_available = torch.cuda.is_available()

def update_gpu(args):
	if 'cuda' in str(args['device']):
		args['device'] = torch.device('cuda:{}'.format(args['gpu']))
	if torch.cuda.device_count() > 0:
		args['device_ids'] = [device_id for device_id in range(torch.cuda.device_count())]
	else:
		args['device_ids'] = []

mnist_args = {
	# system parameters
	'save_gpu':False,
	'gpu': 0,
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),

	# setting parameters
	'dataset': 'mnist',
	# 'sample_size_cap': 6000,
	'n_participants': 5,
	'batch_size' : 100, 
	# 'train_val_split_ratio': 0.9,
	'epochs':30, # epochs for each subset training
	'iteration': 200, # iterations for TMC-Shapley to converge 

	# sample size for different dataset
	'train_samples_size': 1000,
	'valid_samples_size': 1000,
	'test_samples_size': 1000,

	# model parameters
	'model_fn': CNN_Net, #MLP_Net, CNN_Net, MNIST_LogisticRegression
	'optimizer_fn': optim.Adam,
	'loss_fn': nn.NLLLoss, 
	'lr': 0.005,
}

flower_args = {
	# system parameters
	'save_gpu':False,
	'gpu': 0,
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),

	# setting parameters
	'dataset': 'flower',
	# 'sample_size_cap': 6000,
	'n_participants': 5,
	'batch_size' : 100, 
	# 'train_val_split_ratio': 0.9,
	'epochs': 30, # epochs for each subset training
	'iteration': 200, # iterations for TMC-Shapley to converge 

	# sample size for different dataset
	'train_samples_size': 1000,
	'valid_samples_size': 1000,
	'test_samples_size': 1000,

	# model parameters
	'model_fn': Flower_LR, #MLP_Net, CNN_Net, MNIST_LogisticRegression
	'optimizer_fn': optim.Adam,
	'loss_fn': nn.NLLLoss, 
	'lr': 0.01,
}




cifar_10_args = {
	# system parameters
	'gpu': 0,
	'device': torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"),
	'log_interval':20,
	
	# setting parameters
	'dataset': 'cifar10',
	'sample_size_cap': 20000,
	'n_participants': 10,
	'batch_size' : 128, 
	'train_val_split_ratio': 0.9,
	'epochs':5,
	
	# model parameters
	'model_fn': CNNCifar_10, #ResNet18_torch, CNNCifar_TF
	'optimizer_fn': optim.Adam,
	'loss_fn': nn.CrossEntropyLoss(),
	'lr':0.001,
	'gamma':0.977,
	'lr_decay':0.977,  #0.977**100 ~= 0.1
}
