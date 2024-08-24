import imp
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST


def get_train_valid_indices(n_samples, train_val_split_ratio, sample_size_cap=None):
	indices = list(range(n_samples))
	random.seed(1111)
	random.shuffle(indices)
	split_point = int(n_samples * train_val_split_ratio)
	train_indices, valid_indices = indices[:split_point], indices[split_point:]
	if sample_size_cap is not None:
		train_indices = indices[:min(split_point, sample_size_cap)]

	return  train_indices, valid_indices 

def get_dataset_indices(n_samples, train_samples_size, valid_samples_size, test_samples_size):
	indices = list(range(n_samples))
	random.seed(1111)
	random.shuffle(indices)
	train_indices, test_indices, valid_indices = indices[:train_samples_size], \
												indices[train_samples_size:(train_samples_size+test_samples_size)], \
												indices[(train_samples_size+test_samples_size):(train_samples_size+test_samples_size+valid_samples_size)]

	return  train_indices, valid_indices, test_indices 

def powerlaw(sample_indices, n_participants, alpha=1.65911332899, shuffle=False):
	from scipy.stats import powerlaw

	# the smaller the alpha, the more extreme the division
	if shuffle:
		random.seed(1234)
		random.shuffle(sample_indices)

	party_size = int(len(sample_indices) / n_participants)
	b = np.linspace(powerlaw.ppf(0.01, alpha), powerlaw.ppf(0.99, alpha), n_participants)
	shard_sizes = list(map(math.ceil, b/sum(b)*party_size*n_participants))
	indices_list = []
	accessed = 0
	for participant_id in range(n_participants):
		indices_list.append(sample_indices[accessed:accessed + shard_sizes[participant_id]])
		accessed += shard_sizes[participant_id]
	return indices_list

class Custom_Dataset(Dataset):

	def __init__(self, X, y, return_idx=True, device=None, transform=None):
		self.data = X.to(device)
		self.targets = y.to(device)
		self.count = len(X)
		self.device = device
		self.transform = transform
		self.return_idx = return_idx

	def __len__(self):
		return self.count

	def __getitem__(self, idx):
		if self.transform:
			data = self.transform(self.data[idx])
			targets = self.targets[idx]
		else:
			data = self.data[idx]
			targets = self.targets[idx]
		if self.return_idx:
			return data, targets, idx
		else:
			return data, targets

class FastMNIST(MNIST):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)		
		
		self.data = self.data.unsqueeze(1).float().div(255)
		from torch.nn import ZeroPad2d
		pad = ZeroPad2d(2)
		self.data = torch.stack([pad(sample.data) for sample in self.data])

		self.targets = self.targets.long()

		self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
		# self.data = self.data.sub_(0.1307).div_(0.3081)
		# Put both data and targets on GPU in advance
		self.data, self.targets = self.data, self.targets
		print('MNIST data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		return img, target

class FastCIFAR10(CIFAR10):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		# Scale data to [0,1]
		from torch import from_numpy
		self.data = from_numpy(self.data)
		self.data = self.data.float().div(255)
		self.data = self.data.permute(0, 3, 1, 2)

		self.targets = torch.Tensor(self.targets).long()


		# https://github.com/kuangliu/pytorch-cifar/issues/16
		# https://github.com/kuangliu/pytorch-cifar/issues/8
		for i, (mean, std) in enumerate(zip((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))):
			self.data[:,i].sub_(mean).div_(std)

		# Put both data and targets on GPU in advance
		self.data, self.targets = self.data, self.targets
		print('CIFAR10 data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		return img, target
