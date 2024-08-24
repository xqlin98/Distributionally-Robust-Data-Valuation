# Distributionally robust data valuation
This repository contains the code for the experiments of the paper: Distributionally Robust Data Valuation. 

## 1. Preparing for the environment
Running the following command to install the required packages using the `environment.yml` file using conda:
```
conda env create -f environment.yml
```

## 2. Preparing for the dataset and model checkpoints
`HOUSING`: download the dataset from `https://www.kaggle.com/datasets/camnugent/california-housing-prices` and put the dataset under the folder `dataset/rideshare_kaggle.csv`.

`UBER`: download the dataset from `https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma` and put the dataset under the folder `dataset/Uber_lyft/rideshare_kaggle.csv`.

`DIABETES`: download the dataset from `https://www.kaggle.com/datasets/brandao/diabetes` and put the dataset under `datasets/diabetes/diabetic_data.csv`. Run the notebook `dataset/diabetes/prediction-on-hospital-readmission.ipynb` to get the data `datasets/diabetes/diabetic_data.npz`.

`MNIST`: does not need to download, will be automatically downloaded when running the code.

`CIFAR-10`: does not need to download, will be automatically downloaded when running the code.

One of the baselines called LAVA needs extra model checkpoints to conduct data valuation. The checkpoints can be downloaded from `https://github.com/ruoxi-jia-group/LAVA/tree/main` under the folder `checkpoint`. Put all the downloaded checkpoints into the local folder `LAVA/checkpoint`.

## 3. Parameters for the script
### 3.1 `kernel_regression_baselines.py`
This is the script for running the data removal experiments for the kernel regression model in the paper.
```
--dataset: the name of the dataset
--numdp: the number of data points in training dataset
--evaldp: the number of data points to be evaluated
--length_scale: the length scale for the RBF kernel
--thread_num: the thread number for multiprocessing
--epsilon: the epsilon for DRGE
--expname: the name of the experiment
--trials: the number of trials
--mu: the regularization parameter for the kernel ridge regression
--cluster: the number of clusters used to evaluate the worst-case model performance
```
### 3.2 `neural_network_allbaseline.py`
This is the script for running the data removal experiments for the neural network model in the paper.
```
--dataset: the name of the dataset
--numdp: the number of data points in training dataset
--evaldp: the number of data points to be evaluated
--thread_num: the thread number for multiprocessing
--thread_num_nn: the thread number for training neural network models
--epsilon: the epsilon for DRGE
--expname: the name of the experiment
--trials: the number of trials
--mu: the regularization parameter for the kernel ridge regression
--model: the neural network architecture used to train the model
--cpu: whether to use CPU to compute the neural tangent kernel or not, [0,1]
--batch_size: the batch size for training the model
--epochs: the number of epochs
--lr: the learning rate for model traininig
--gpus: specify the GPU ids used to perform the experiments, e.g., 0123
--cluster: the number of clusters used to evaluate the worst-case model performance
```

### 3.3 `subset_selection.py`
This is the script for running the data subset selection experiments for the kernel regression model in the paper.
```
--dataset: the name of the dataset
--numdp: the number of data points in training dataset
--length_scale: the length scale for the RBF kernel
--thread_num: the thread number for multiprocessing
--epsilon: the epsilon for DRGE
--expname: the name of the experiment
--trials: the number of trials
--mu: the regularization parameter for the kernel ridge regression
--cluster: the number of clusters used to evaluate the worst-case model performance
```

### 3.4 `subset_selection_nn.py`
This is the script for running the data subset selection experiments for the neural network model in the paper.
```
--dataset: the name of the dataset
--numdp: the number of data points in training dataset
--evaldp: the number of data points to be evaluated
--thread_num: the thread number for multiprocessing
--thread_num_nn: the thread number for training neural network models
--epsilon: the epsilon for DRGE
--expname: the name of the experiment
--trials: the number of trials
--mu: the regularization parameter for the kernel ridge regression
--model: the neural network architecture used to train the model
--cpu: whether to use CPU to compute the neural tangent kernel or not, [0,1]
--batch_size: the batch size for training the model
--epochs: the number of epochs
--lr: the learning rate for model traininig
--gpus: specify the GPU ids used to perform the experiments, e.g., 0123
--cluster: the number of clusters used to evaluate the worst-case model performance
```

## 4. Reproducing the experiments in the paper
Run the following command to reproduce the results for the kernel regression model:
```
bash kernel_regression.sh
```
Run the following command to reproduce the results for the neural network model:
```
bash neural_network.sh
```