import argparse
import os
import random
from functools import reduce

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchtext.data import BucketIterator, Field, LabelField
from torchvision import transforms
from torchvision.datasets import CIFAR100

from utils.dataset import (Custom_Dataset, FastCIFAR10, FastMNIST, get_dataset_indices,
						   get_train_valid_indices, powerlaw)

from utils.utils import fake_minority_class, one_class_lo

class Data_Prepper:
	def __init__(self, name, n_participants, train_batch_size, train_samples_size, valid_samples_size=None, test_samples_size=None, device=None, args_dict=None):
		self.args_dict = args_dict
		self.name = name
		self.device = device
		self.train_samples_size = train_samples_size
		self.valid_samples_size = valid_samples_size
		self.test_samples_size = test_samples_size
		self.n_participants = n_participants
		self.n_class = None

		# setting the batch size 
		self.train_batch_size = train_batch_size
		self.valid_batch_size = 100
		self.test_batch_size = 100

		# get the training/testing/validation dataset
		self.train_dataset, self.validation_dataset, self.test_dataset = self.prepare_dataset(name)
		# show the size of three dataset we got
		print('------')
		print("Train to split size: {}. Validation size: {}. Test size: {}".format(len(self.train_dataset), len(self.validation_dataset), len(self.test_dataset)))
		print('------')

		self.valid_loader = DataLoader(self.validation_dataset, batch_size=self.valid_batch_size)
		self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size)

	def get_all_data_loaders(self, n_participants, unique_pattern='deterministic'):
		"""Return train loaders for different unique patterns we defined"""
		batch_size = self.train_batch_size

		train_data_indices = [torch.nonzero(self.train_dataset.targets == class_id).view(-1).tolist() for class_id in range(self.n_class)]
		valid_data_indices = [torch.nonzero(self.validation_dataset.targets == class_id).view(-1).tolist() for class_id in range(self.n_class)]

		if unique_pattern == "minority-one-class-lo":
			minority_class = np.random.choice(range(self.n_class), 1).item()
			train_data_indices = fake_minority_class(minority_class=minority_class,
													data_indices=train_data_indices,
													keep_ratio=0.3)
			valid_data_indices = fake_minority_class(minority_class=minority_class,
													data_indices=valid_data_indices,
													keep_ratio=0.3)
			n_data_points = sum([len(tmp) for tmp in train_data_indices]) // n_participants
			indices_list = one_class_lo(n_participants=n_participants,
										n_class=self.n_class,
										n_data_points=n_data_points,
										lo_class=minority_class,
										data_indices=train_data_indices,
										lo_ratio=0.3,
										lo_participant_percent=0.5)
			valid_loader = DataLoader(self.validation_dataset,
										batch_size=self.valid_batch_size,
										sampler=SubsetRandomSampler(np.concatenate(valid_data_indices,axis=0)))
			self.valid_loader = valid_loader

		elif unique_pattern == "one-class-lo":
			n_data_points = len(self.train_dataset) // n_participants
			lo_class = np.random.choice(range(self.n_class), 1).item()
			indices_list = one_class_lo(n_participants=n_participants,
										n_class=self.n_class,
										n_data_points=n_data_points,
										lo_class=lo_class,
										data_indices=train_data_indices,
										lo_ratio=0.2,
										lo_participant_percent=1.0)

		elif unique_pattern in ['uniform','equal']:
			from utils.Data_Prepper import random_split
			indices_list = random_split(sample_indices=list(range(len(self.train_dataset))), m_bins=n_participants, equal=True)

		elif unique_pattern == 'random':
			from utils.Data_Prepper import random_split
			indices_list = random_split(sample_indices=list(range(len(self.train_dataset))), m_bins=n_participants, equal=False)

		elif unique_pattern == 'deterministic':
			indices_list = [[tmp] for tmp in range(len(self.train_dataset))]
	  
		# individual trainining datasets created from the overall extracted dataset: self.train_dataset
		# this is so we can construct differentially private loaders
		# self.train_datasets = [Custom_Dataset(self.train_dataset.data[indices],self.train_dataset.targets[indices])  for indices in indices_list]
		self.shard_sizes = [len(indices) for indices in indices_list]

		participant_train_loaders = [DataLoader(self.train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(indices)) for indices in indices_list]
		self.train_loaders = participant_train_loaders
		return participant_train_loaders, self.valid_loader, self.test_loader

	def prepare_dataset(self, name='mnist'):
		if name == 'mnist':

			train = FastMNIST('datasets', train=True, download=True)
			test = FastMNIST('datasets', train=False, download=True)

			train_indices, valid_indices, test_indices = get_dataset_indices(len(train), 
														train_samples_size=self.train_samples_size,
														valid_samples_size=self.valid_samples_size,
														test_samples_size=self.test_samples_size)
			self.train_indices, self.valid_indices, self.test_indices  = train_indices, valid_indices, test_indices
			train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
			validation_set = Custom_Dataset(train.data[valid_indices],train.targets[valid_indices] , device=self.device)
			test_set = Custom_Dataset(train.data[test_indices], train.targets[test_indices], device=self.device)

			del test

			self.n_class = 10
			self.ori_data = {"train": {"data": train_set.data.cpu().numpy(), 
										"targets": train_set.targets.cpu().numpy()}, 
							"valid": {"data": validation_set.data.cpu().numpy(), 
									"targets": validation_set.targets.cpu().numpy()}}

			return train_set, validation_set, test_set

		elif name == 'flower':
			if not os.path.exists('datasets/tf_flowers.pt'):
				import tensorflow as tf
				import tensorflow_datasets as tfds
				# download tf_flower from tensorflow dataset
				flower_ds = tfds.load('tf_flowers', split='train', shuffle_files=True, data_dir='datasets')
				flower_ds = tfds.as_numpy(flower_ds)

				# resize the images
				transform_train = transforms.Compose([transforms.Resize(size=250),
										transforms.CenterCrop(size=(250,250))])
				images = []
				labels = []
				for sample in flower_ds:
					img = torch.tensor(sample['image']).transpose(0,2)
					trans_img = transform_train(img)
					images.append(trans_img)
					labels.append(sample['label'])
				images = torch.stack(images,axis=0).float() / 255
				labels = torch.tensor(labels)

				# extract features from inception v3
				# normalize the pixel values
				transform_train = transforms.Compose([transforms.Normalize( 
														mean=(0.4840, 0.4374, 0.3056) ,
														std=(0.2965, 0.2678, 0.2924))])
				trans_images = transform_train(images)
				inception_mdl = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).eval()
				from torchvision.models.feature_extraction import get_graph_node_names
				from torchvision.models.feature_extraction import create_feature_extractor
				# extract train and eval layers from the model
				train_nodes, eval_nodes = get_graph_node_names(inception_mdl)
				# remove the last layer
				return_nodes = eval_nodes[:-1]
				# create a feature extractor for each intermediary layer
				feat_inception = create_feature_extractor(inception_mdl, return_nodes=return_nodes)
				feat_inception = feat_inception.cuda().eval()
				img_loader = DataLoader(trans_images, batch_size=10)
				features = []
				for batch in img_loader:
					output_feat = feat_inception(batch.cuda())
					vec_feat = output_feat['flatten'].cpu().detach()
					features.append(vec_feat)
				features = torch.cat(features, axis=0)

				torch.save({'images': images, 'features': features, 'labels': labels}, 'datasets/tf_flowers.pt')
				del inception_mdl
			else:
				flower_ds = torch.load('datasets/tf_flowers.pt')
				images, features, labels = flower_ds['images'], flower_ds['features'], flower_ds['labels']

			train_indices, valid_indices, test_indices = get_dataset_indices(len(features), 
														train_samples_size=self.train_samples_size,
														valid_samples_size=self.valid_samples_size,
														test_samples_size=self.test_samples_size)
			self.train_indices, self.valid_indices, self.test_indices  = train_indices, valid_indices, test_indices
			train_set = Custom_Dataset(features[train_indices], labels[train_indices], device=self.device)
			validation_set = Custom_Dataset(features[valid_indices],labels[valid_indices] , device=self.device)
			test_set = Custom_Dataset(features[test_indices], labels[test_indices], device=self.device)

			self.n_class = 5
			self.ori_data = {"train": {"data": images[train_indices].cpu().numpy(), 
										"features": train_set.data.cpu().numpy(),
										"targets": train_set.targets.cpu().numpy()}, 
							"valid": {"data": images[valid_indices].cpu().numpy(), 
									"features": validation_set.data.cpu().numpy(),
									"targets": validation_set.targets.cpu().numpy()}}
			return train_set, validation_set, test_set

		elif name == 'cifar10':
			transform_train = transforms.Compose([
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])

			transform_test = transforms.Compose([
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])

			train = FastCIFAR10('datasets', train=True, download=True)#, transform=transform_train)
			test = FastCIFAR10('datasets', train=False, download=True)#, transform=transform_test)

			# transformation on data
			train.data = transform_train(train.data)
			test.data = transform_test(test.data)

			train_indices, valid_indices, test_indices = get_dataset_indices(len(train), 
														train_samples_size=self.train_samples_size,
														valid_samples_size=self.valid_samples_size,
														test_samples_size=self.test_samples_size)
			self.train_indices, self.valid_indices, self.test_indices  = train_indices, valid_indices, test_indices
			train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
			validation_set = Custom_Dataset(train.data[valid_indices],train.targets[valid_indices] , device=self.device)
			test_set = Custom_Dataset(train.data[test_indices], train.data[test_indices], device=self.device)

			del test

			self.n_class = 10
			return train_set, validation_set, test_set

		elif name == 'cifar100':
			transform_train = transforms.Compose([
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

			transform_test = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

			train = CIFAR100(root='datasets', train=True, download=True, transform=transform_train)
			test = CIFAR100(root='datasets', train=False, download=False, transform=transform_test)

			from torch import Tensor
			train.targets = Tensor(train.targets).long()
			test.targets = Tensor(test.targets).long()

			from torch import from_numpy
			train.data = from_numpy(train.data).permute(0, 3, 1, 2).float()
			test.data = from_numpy(test.data).permute(0, 3, 1, 2).float()

			train_indices, valid_indices, test_indices = get_dataset_indices(len(train), 
														train_samples_size=self.train_samples_size,
														valid_samples_size=self.valid_samples_size,
														test_samples_size=self.test_samples_size)

			train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
			validation_set = Custom_Dataset(train.data[valid_indices],train.targets[valid_indices] , device=self.device)
			test_set = Custom_Dataset(train.data[test_indices], train.data[test_indices], device=self.device)

			del train, test

			return train_set, validation_set, test_set

		elif name == "tiny_imagenet":

			pretrained_224 = False

			dataset_dir = "datasets/tiny-imagenet-200"
			train_dir = os.path.join(dataset_dir, 'train')
			val_dir = os.path.join(dataset_dir, 'val', 'images')
			kwargs = {'num_workers': 8, 'pin_memory': True}

			'''
			Separate validation images into separate sub folders
			'''
			val_dir = os.path.join(dataset_dir, 'val')
			img_dir = os.path.join(val_dir, 'images')

			fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
			data = fp.readlines()
			val_img_dict = {}
			for line in data:
				words = line.split('\t')
				val_img_dict[words[0]] = words[1]
			fp.close()

			# Create folder if not present and move images into proper folders
			for img, folder in val_img_dict.items():
				newpath = (os.path.join(img_dir, folder))
				if not os.path.exists(newpath):
					os.makedirs(newpath)
				if os.path.exists(os.path.join(img_dir, img)):
					os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

			# Pre-calculated mean & std on imagenet:
			# norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			# For other datasets, we could just simply use 0.5:
			# norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

			print('Preparing tiny_imagenet data ...')
			# Normalization
			norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
				if pretrained_224 else transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

			# Normal transformation
			if pretrained_224:
				train_trans = [transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224),
								transforms.ToTensor()]
				val_trans = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm]
			else:
				train_trans = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
				val_trans = [transforms.ToTensor(), norm]

			print('Preparing tiny_imagenet pytorch ImageFolders ...')
			from torchvision import datasets
			train_folder = datasets.ImageFolder(train_dir, transform=transforms.Compose(train_trans + [norm]))
			test_folder = datasets.ImageFolder(val_dir, transform=transforms.Compose(val_trans))

			return train_folder, test_folder

	def get_valid_loader(self):
		return self.valid_loader

	def get_test_loader(self):
		return self.test_loader


def random_split(sample_indices, m_bins, equal=True):
	sample_indices = np.asarray(sample_indices)
	if equal:
		indices_list = np.array_split(sample_indices, m_bins)
	else:
		split_points = np.random.choice(
			n_samples - 2, m_bins - 1, replace=False) + 1
		split_points.sort()
		indices_list = np.split(sample_indices, split_points)

	return indices_list
