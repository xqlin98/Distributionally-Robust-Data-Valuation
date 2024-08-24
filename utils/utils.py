import copy
import datetime
from functools import reduce
import os
import pickle
import random
import time

import numpy as np
from sklearn.cluster import KMeans
from utils.arguments import mnist_args, cifar_10_args, flower_args
from itertools import chain, combinations
from os.path import join
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import gurobipy as gp
from gurobipy import GRB
import contextlib
from joblib import Parallel, delayed
from tqdm import tqdm
import pytorch_influence_functions as ptif
from torch.utils.data import DataLoader
from utils.dataset import Custom_Dataset
import torch 
from torch import nn, optim
import copy
import joblib
from utils.models_defined import MLP_CLS_L, MLP_CLS_S, MLP_REGRESSION, MLP_REGRESSION_S, Model_Train
from torch_influence import BaseObjective, AutogradInfluenceModule, CGInfluenceModule
import torch.nn.functional as F
from functorch import make_functional, make_functional_with_buffers

from pathlib import Path
import sys

from utils.ntk import empirical_ntk_jacobian_contraction
parent_directory = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_directory))

from LAVA import lava
from LAVA.otdd.pytorch.datasets import get_dataset_transform, load_torchvision_data_keepclean, CustomTensorDataset1, CustomTensorDataset2

def update_model_args(cmd_args):
    if cmd_args.dataset == "mnist":
        args = copy.deepcopy(mnist_args)
    elif cmd_args.dataset == "cifar10":
        args = copy.deepcopy(cifar_10_args)
    elif cmd_args.dataset == "flower":
        args = copy.deepcopy(flower_args)
    args.update(vars(cmd_args))
    return args

def softmax(x, beta = 1.0):

    """Compute softmax values for each sets of scores in x."""
    x = np.array(x)
    e_x = np.exp(beta * x)
    return e_x / e_x.sum()


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def create_exp_dir(exp_name, dataset, results_dir='DataSV_Exp'):
    ''' Set up entire experiment directory '''
    str_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
    exp_dir = 'Exp_{}_{}'.format(exp_name, str_time)
    exp_dir = join(dataset, exp_dir)
    os.makedirs(join(results_dir, exp_dir), exist_ok=True)
    return join(results_dir, exp_dir)

def save_result(dir, sys_log, data):
    with open(join(dir, 'sys_log.txt'), 'w') as file:
        [file.write(key + ' : ' + str(value) + '\n') for key, value in sys_log.items()]

    with open(join(dir, 'sys_log.pickle'), 'wb') as f:
        pickle.dump(sys_log, f)
    
    with open(join(dir, "data.pickle"), "wb") as f:
        pickle.dump(data, f)
    # torch.save(data, join(dir,"data.pt"))

def fake_minority_class(minority_class, data_indices, keep_ratio):
    """
    To simulate the scenario of having a minority class

    :param int minority_class: the specified class to create minority sample
    :param list data_indices: the indices for each class in the dataset
    :param float keep_ratio: the ratio of data points remained in creating minority
    """
    minority_indices = data_indices[minority_class]
    minority_num = len(minority_indices)
    keep_num = int(minority_num * keep_ratio)
    sampled_minority_indices = np.random.choice(minority_indices, keep_num, replace=False)
    data_indices[minority_class] = list(sampled_minority_indices)
    return data_indices

def one_class_lo(n_participants, n_class, n_data_points, lo_class, data_indices, lo_ratio=0.3, lo_participant_percent=1.0):
    """
    To reserve one class and assign the one class data exclusively to some participants

    :param int n_participants: number of participants
    :param int n_class: number of classes in the label
    :param int n_data_points: number of data points in a normal participant
    :param int lo_class: the specified class to leave out
    :param list data_indices: the indices for each class in the dataset
    :param float lo_ratio: the ratio of participants that holds the exclusive data
    :param float lo_participant_percent: the ratio of number of data points for exclusive party
    """
    n_lo_participants = max(1,int(n_participants * lo_ratio))
    n_normal_participants = n_participants - n_lo_participants
    lo_indices = data_indices[lo_class]
    normal_indices = reduce(lambda a,b: a+b, [data_indices[i] for i in range(n_class) if i != lo_class])
    random.shuffle(normal_indices)
    indices_list = []
    end_point = None
    n_lo_data_points = len(lo_indices) // n_lo_participants
    lo_n_data_points = max(n_lo_data_points, int(n_data_points*lo_participant_percent))
    for i in range(n_normal_participants):
        indices_list.append(normal_indices[(i*n_data_points):((i+1)*n_data_points)])
        end_point = (i+1)*n_data_points
    for i in range(n_lo_participants):
        lo_part = lo_indices[(i*n_lo_data_points):((i+1)*n_lo_data_points)]
        if lo_n_data_points > n_lo_data_points:
            normal_part = normal_indices[(end_point + i*(lo_n_data_points-n_lo_data_points)):(end_point + (i+1)*(lo_n_data_points-n_lo_data_points))]
        indices_list.append(normal_part + lo_part)
    return indices_list

def duplicate_model_optimizer(model, optimizer_fn, lr, regularization=True):
    new_model = copy.deepcopy(model).cuda()
    new_optimizer = optimizer_fn(new_model.parameters(), lr=lr, weight_decay=5e-3 if regularization else 0)
    return new_model, new_optimizer

def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v

def merge_dataloaders(train_loaders):
    """Merge multiple dataloaders to a single loader"""
    indices = np.concatenate([tmp.sampler.indices for tmp in train_loaders],axis=0)
    batch_size = train_loaders[0].batch_size
    train_dataset = train_loaders[0].dataset

    return DataLoader(train_dataset,
                        batch_size=batch_size, 
                        sampler=SubsetRandomSampler(indices))

def drge_approximation(loss, epsilon):
    # solving quadratic programming problem
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model('quadratic',env=env) as guadratic_model:
            # guadratic_model = gp.Model('quadratic')
            guadratic_model.Params.LogToConsole = 0
            loss = np.array(loss)
            num_var = len(loss)

            x = guadratic_model.addMVar(shape=num_var, vtype=GRB.CONTINUOUS, lb=0, name="x")
            guadratic_model.setObjective(loss @ x, GRB.MAXIMIZE)
            guadratic_model.addConstr((num_var * x - 1) @ (num_var * x - 1) * 0.5 / num_var <= epsilon)
            guadratic_model.addConstr(x @ np.ones([num_var]) == 1)
            guadratic_model.optimize()
            # a = x.X
            obj_val = guadratic_model.objVal
    return obj_val

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# def calculate_influence_score(trdata, trlabel, regression=True, device='cpu'):
#     train_dataset = Custom_Dataset(torch.tensor(trdata), torch.tensor(trlabel), device=device, return_idx=False)
#     train_dataloader =  DataLoader(train_dataset, batch_size=128)

#     # initialize the model and optimizer
#     input_dim = trdata.shape[1]
#     model_fn = MLP_REGRESSION
#     optimizer_fn = optim.SGD
#     loss_fn = nn.MSELoss
#     lr = 0.001
#     batch_size = 128
#     epochs = 30
#     outdir = "./tmp_ifoutdir"
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_train = Model_Train(model_fn=model_fn,
#                             optimizer_fn=optimizer_fn,
#                             loss_fn=loss_fn,
#                             lr=lr,
#                             batch_size=batch_size,
#                             epochs=epochs,
#                             device=device,
#                             input_dim=input_dim)
#     model_train.simplefit(train_dataloader, epochs=epochs)
#     tmp_model = copy.deepcopy(model_train.model).cpu()
#     ptif.init_logging()
#     config = ptif.get_default_config()
#     config['gpu'] = -1
#     config['test_start_index'] = False
#     config['outdir'] = outdir
#     loss_func = 'mse' if regression else 'cross_entropy'
#     influences, _, _ = ptif.calc_img_wise(config, tmp_model, train_dataloader, train_dataloader, loss_func=loss_func)
#     all_influence = []
#     for idx in influences.keys():
#         all_influence.append(influences[idx]['influence'])
#     avg_influences = np.mean(np.array(all_influence), axis=0)
#     del tmp_model
#     return avg_influences

def calculate_influence_score_nn(trdata, trlabel, model_train_func, epochs, cpu=False):
    
    if cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn_ini = nn.MSELoss()
    batch_size = 128
    
    train_dataset = Custom_Dataset(torch.tensor(trdata, dtype=torch.float32), torch.tensor(trlabel, dtype=torch.float32), device=device, return_idx=False)
    train_dataloader =  DataLoader(train_dataset, batch_size=batch_size)

    # initialize the model and optimizer
    numdp = trdata.shape[0]

    class MyObjective(BaseObjective):

        def train_outputs(self, model, batch):
            return model(batch[0])

        def train_loss_on_outputs(self, outputs, batch):
            return loss_fn_ini(outputs, batch[1])  # mean reduction required

        def train_regularization(self, params):
            return 0

        # training loss by default taken to be 
        # train_loss_on_outputs + train_regularization

        def test_loss(self, model, params, batch):
            return loss_fn_ini(model(batch[0]), batch[1])  # no regularization in test loss
    
    model_train = model_train_func(device=device)
    model_train.simplefit(train_dataloader, epochs=epochs , init=True)
    
    print("Start calculating influence scores...")
    
    module = CGInfluenceModule(
            model=model_train.model,
            objective=MyObjective(),  
            train_loader=train_dataloader,
            test_loader=train_dataloader,
            device=device,
            damp=0.001,
            atol=1e-8,
            maxiter=1000
    )
    
    # influence scores of training points 1, 2, and 3 on test point 0
    scores = module.influences(np.arange(numdp), np.arange(numdp)).cpu().numpy().reshape(-1)
    return scores

def calculate_tracin_score_nn(trdata, trlabel, model_train_func, epochs, batch_size, lr, device='cpu'):

    train_dataset = Custom_Dataset(torch.tensor(trdata, dtype=torch.float32), torch.tensor(trlabel, dtype=torch.float32), device=device, return_idx=False)
    train_dataloader =  DataLoader(train_dataset, batch_size=batch_size)
    train_dataloader_sing = DataLoader(train_dataset, batch_size=1)

    criterion = nn.MSELoss()
    
    model = model_train_func(device=device).model
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
    state_dicts = []
    LR = lr
    
    for epoch in range(1, epochs):

        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

        if epoch % 3 == 0:
            # torch.save(model.state_dict(), f'model_v1_epoch_{epoch}')
            # weights.append(f'model_v1_epoch_{epoch}')
            state_dicts.append(copy.deepcopy(model.state_dict()))
    
    score_matrix = np.zeros((len(train_dataloader_sing), len(train_dataloader_sing)))
    
    gradient_all_sample = []
    for state in state_dicts:
        model = model_train_func(device=device).model
        model.load_state_dict(state)
        model.eval()
        gradient_all_sample = []
        for _, (x_train, y_train) in enumerate(train_dataloader_sing):
            y_pred = model(x_train) # pred
            loss = criterion(y_pred, y_train)
            loss.backward() # back
            train_grad = torch.cat([param.grad.reshape(-1) for param in model.parameters()])
            gradient_all_sample.append(train_grad)
        gradient_all_sample = torch.vstack(gradient_all_sample)
        gram_matrix = gradient_all_sample @ gradient_all_sample.transpose(0,1)
        score_matrix += LR * gram_matrix.cpu().numpy()

    trackin_scores = np.sum(score_matrix, axis=1).reshape(-1)
    return trackin_scores

def calculate_influence_score(trdata, trlabel, regression=True, class_num=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer_fn = optim.SGD
    if regression:
        model_fn = MLP_REGRESSION
    else:
        model_fn = MLP_CLS_S
    loss_fn = nn.MSELoss
    loss_fn_ini = loss_fn()
    lr = 0.001
    batch_size = 128
    epochs = 30
    
    train_dataset = Custom_Dataset(torch.tensor(trdata, dtype=torch.float32), torch.tensor(trlabel, dtype=torch.float32), device=device, return_idx=False)
    train_dataloader =  DataLoader(train_dataset, batch_size=batch_size)

    # initialize the model and optimizer
    input_dim = trdata.shape[1]
    numdp = trdata.shape[0]

    class MyObjective(BaseObjective):

        def train_outputs(self, model, batch):
            return model(batch[0])

        def train_loss_on_outputs(self, outputs, batch):
            return loss_fn_ini(outputs, batch[1])  # mean reduction required

        def train_regularization(self, params):
            return 0

        # training loss by default taken to be 
        # train_loss_on_outputs + train_regularization

        def test_loss(self, model, params, batch):
            return loss_fn_ini(model(batch[0]), batch[1])  # no regularization in test loss
    
    model_train = Model_Train(model_fn=model_fn,
                            optimizer_fn=optimizer_fn,
                            loss_fn=loss_fn,
                            lr=lr,
                            batch_size=batch_size,
                            epochs=epochs,
                            device=device,
                            input_dim=input_dim,
                            class_num=class_num)
    model_train.simplefit(train_dataloader, epochs=epochs)
    
    print("Start calculating influence scores...")
    # module = AutogradInfluenceModule(
    #     model=model_train.model,
    #     objective=MyObjective(),  
    #     train_loader=train_dataloader,
    #     test_loader=train_dataloader,
    #     device=device,
    #     damp=0.001
    # )
    
    module = CGInfluenceModule(
            model=model_train.model,
            objective=MyObjective(),  
            train_loader=train_dataloader,
            test_loader=train_dataloader,
            device=device,
            damp=0.001,
            atol=1e-8,
            maxiter=1000
    )
    
    # influence scores of training points 1, 2, and 3 on test point 0
    scores = module.influences(np.arange(numdp), np.arange(numdp)).cpu().numpy().reshape(-1)
    return scores

def calculate_tracin_score(trdata, trlabel, device='cpu', outdir="", regression=True, class_num=None):

    lr = 0.001
    epochs = 30
    batch_size = 64
    input_dim = trdata.shape[1]

    train_dataset = Custom_Dataset(torch.tensor(trdata, dtype=torch.float32), torch.tensor(trlabel, dtype=torch.float32), device=device, return_idx=False)
    train_dataloader =  DataLoader(train_dataset, batch_size=batch_size)
    train_dataloader_sing = DataLoader(train_dataset, batch_size=1)

    criterion = nn.MSELoss()
    
    if regression:
        MODEL = MLP_REGRESSION
    else:
        MODEL = MLP_CLS_S
    model = MODEL(input_dim, class_num=class_num).to(torch.float32)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
    weights = []
    state_dicts = []
    LR = lr
    
    for epoch in range(1, epochs):

        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

        if epoch % 3 == 0:
            # torch.save(model.state_dict(), f'model_v1_epoch_{epoch}')
            # weights.append(f'model_v1_epoch_{epoch}')
            state_dicts.append(copy.deepcopy(model.state_dict()))
    
    score_matrix = np.zeros((len(train_dataloader_sing), len(train_dataloader_sing)))
    
    gradient_all_sample = []
    for state in state_dicts:
        model = MODEL(input_dim, class_num=class_num).to(torch.float32)
        model.load_state_dict(state)
        model.eval()
        gradient_all_sample = []
        for _, (x_train, y_train) in enumerate(train_dataloader_sing):
            y_pred = model(x_train) # pred
            loss = criterion(y_pred, y_train)
            loss.backward() # back
            train_grad = torch.cat([param.grad.reshape(-1) for param in model.parameters()])
            gradient_all_sample.append(train_grad)
        gradient_all_sample = torch.vstack(gradient_all_sample)
        gram_matrix = gradient_all_sample @ gradient_all_sample.transpose(0,1)
        score_matrix += LR * gram_matrix.cpu().numpy()

    # for train_id, (x_train, y_train) in enumerate(train_dataloader_sing):
    #     if train_id % 50 == 0:
    #         print('Train:', round(train_id / 500 * 100), '%')
    #     for test_id, (x_test, y_test) in enumerate(train_dataloader_sing):
    #         grad_sum = 0
            
    #         for w in weights_paths:
    #             model = MLP_REGRESSION(input_dim)
    #             model.load_state_dict(torch.load(w)) # checkpoint
    #             model.eval()
    #             y_pred = model(x_train) # pred
    #             loss = criterion(y_pred, y_train)
    #             loss.backward() # back
    #             train_grad = torch.cat([param.grad.reshape(-1) for param in model.parameters()])
 
    #             model = Net()
    #             model.load_state_dict(torch.load(w)) # checkpoint
    #             model.eval()
    #             y_pred = model(x_test) # pred
    #             loss = criterion(y_pred, y_test)
    #             loss.backward() # back
    #             test_grad = torch.cat([param.grad.reshape(-1) for param in model.parameters()])

    #             grad_sum += LR * np.dot(train_grad, test_grad) # scalar mult, TracIn formula
            
    #         score_matrix[train_id][test_id] = grad_sum
    
    trackin_scores = np.sum(score_matrix, axis=1).reshape(-1)
    return trackin_scores


def rbf_mmd2(X, Y, length_scale=0, biased=True):
    """
    Computes squared MMD using a RBF kernel.
    
    Args:
        X, Y (Tensor): datasets that MMD is computed on
        length_scale (float): lengthscale of the RBF kernel
        biased (bool): whether to compute a biased mean
        
    Return:
        MMD squared
    """
    gamma = 1 / (2 * length_scale**2)
    
    XX = torch.matmul(X, X.T)
    XY = torch.matmul(X, Y.T)
    YY = torch.matmul(Y, Y.T)
    
    X_sqnorms = torch.diagonal(XX)
    Y_sqnorms = torch.diagonal(YY)
    
    K_XY = torch.exp(-gamma * (
            -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    K_XX = torch.exp(-gamma * (
            -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
    K_YY = torch.exp(-gamma * (
            -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    
    if biased:
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())
    return mmd2

def calculate_davinz_score(trdata, trlabel, thread_num, regression=True):
    """
    Computes the DaVinz score.
    
    Args:
        theta (Tensor): NTK matrix
        y_predict (Tensor): predicted labels   
        y_true (Tensor): true labels
        regression (bool): whether the task is regression or classification
    """
    
    # kernel regression on grand dataset
    input_dim = trdata.shape[1]
    thread_num = thread_num // 3
    device = torch.device('cpu')
    if regression:
        model_fn = MLP_REGRESSION
    else:
        model_fn = MLP_CLS_S
    optimizer_fn = optim.SGD
    loss_fn = nn.MSELoss
    lr = 0.01
    batch_size = 128
    epochs = 10
    class_num = 10 if not regression else None
    model_train_func = lambda device: Model_Train(model_fn=model_fn,
                            optimizer_fn=optimizer_fn,
                            loss_fn=loss_fn,
                            lr=lr,
                            batch_size=batch_size,
                            epochs=epochs,
                            device=device,
                            input_dim=input_dim,
                            class_num=class_num)

    model_train = model_train_func(device=device)    
    train_set = Custom_Dataset(torch.tensor(trdata, dtype=torch.float32), torch.tensor(trlabel, dtype=torch.float32), device=device, return_idx=False)
    train_dataloader =  DataLoader(train_set, batch_size=batch_size)
    _, y_predict = model_train.evaluate_point_loss(train_dataloader, regression=regression, return_logits=True)
    
    ini_model = model_train.get_init_model()
    fnet, params = make_functional(ini_model)
    
    kernel_fn_ = lambda x1, x2: empirical_ntk_jacobian_contraction(fnet, params, x1, x2, cpu=1, regression=regression)
    trdata_ = torch.tensor(trdata, dtype=torch.float32).to(device)
    theta = kernel_fn_(trdata_, trdata_)
    length_scale = 5 # default as the default code
    y_true = trlabel
    trdata_ = torch.tensor(trdata, dtype=torch.float32).to(device)
    num_dp = len(y_true)
    y_predict, y_true, trdata_ = torch.tensor(y_predict), torch.tensor(y_true), torch.tensor(trdata_)
    if len(trdata_.shape) > 2:
        trdata_ = trdata_.reshape(trdata_.shape[0], -1)

    if not regression:
        y_true_ = torch.argmax(y_true, dim=1)
        y_predict_ = y_predict[:,y_true_]
        delta_y = (1 - y_predict_).reshape(-1)
    else:
        delta_y = y_true - y_predict

    def cal_single_davinz_score(idx):
        if idx == None:
            selected_idx = np.array(range(num_dp))
        else:
            selected_idx = np.array(list(set(range(num_dp)) - set([idx])))

        selected_delta_y = delta_y[selected_idx]
        selected_theta = theta[np.ix_(selected_idx,selected_idx)]
        seleted_x = trdata_[selected_idx]

        ntk_score = np.sqrt(selected_delta_y.reshape(-1,1).transpose(0,1) @ selected_theta @ selected_delta_y.reshape(-1,1) / num_dp)
        mmd_score = rbf_mmd2(seleted_x, trdata_, length_scale)
        
        return [ntk_score.item(), mmd_score.item()]
    
    grand_ntk_score, grand_mmd_score = cal_single_davinz_score(None)
    with tqdm_joblib(tqdm(desc=f"Computing Davinz score", total=num_dp)) as progress_bar:
        ntk_mmd_scores = Parallel(n_jobs=thread_num)(delayed(cal_single_davinz_score)(idx) for idx in range(num_dp))
    ntk_mmd_scores = np.array(ntk_mmd_scores)
    subset_ntk_scores, subset_mmd_scores = ntk_mmd_scores[:,0], ntk_mmd_scores[:,1]
    
    kappa = np.mean(list(subset_mmd_scores) + [grand_mmd_score])/ np.mean(list(subset_ntk_scores) + [grand_ntk_score])
    combined_scores = - kappa * np.array(subset_ntk_scores) - np.array(subset_mmd_scores)
    grand_combined_scores = - kappa * grand_ntk_score - grand_mmd_score
    
    return grand_combined_scores - combined_scores

def calculate_davinz_score_nn(theta, y_predict, y_true, x, thread_num, regression=True):
    """
    Computes the DaVinz score.
    
    Args:
        theta (Tensor): NTK matrix
        y_predict (Tensor): predicted labels   
        y_true (Tensor): true labels
        evaluate_x (Tensor): evaluation dataset
        test_x (Tensor): test dataset
        regression (bool): whether the task is regression or classification
    """
    thread_num = thread_num // 3
    length_scale = 5 # default as the default code
    num_dp = len(y_true)
    y_predict, y_true, x = torch.tensor(y_predict), torch.tensor(y_true), torch.tensor(x)
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], -1)

    if not regression:
        y_true_ = torch.argmax(y_true, dim=1)
        y_predict_ = y_predict[:,y_true_]
        delta_y = (1 - y_predict_).reshape(-1)
    else:
        delta_y = y_true - y_predict

    def cal_single_davinz_score(idx):
        if idx == None:
            selected_idx = np.array(range(num_dp))
        else:
            selected_idx = np.array(list(set(range(num_dp)) - set([idx])))

        selected_delta_y = delta_y[selected_idx]
        selected_theta = theta[np.ix_(selected_idx,selected_idx)]
        seleted_x = x[selected_idx]

        ntk_score = np.sqrt(selected_delta_y.reshape(-1,1).transpose(0,1) @ selected_theta @ selected_delta_y.reshape(-1,1) / num_dp)
        mmd_score = rbf_mmd2(seleted_x, x, length_scale)
        
        return [ntk_score.item(), mmd_score.item()]
    
    grand_ntk_score, grand_mmd_score = cal_single_davinz_score(None)
    with tqdm_joblib(tqdm(desc=f"Computing Davinz score", total=num_dp)) as progress_bar:
        ntk_mmd_scores = Parallel(n_jobs=thread_num)(delayed(cal_single_davinz_score)(idx) for idx in range(num_dp))
    ntk_mmd_scores = np.array(ntk_mmd_scores)
    subset_ntk_scores, subset_mmd_scores = ntk_mmd_scores[:,0], ntk_mmd_scores[:,1]
    
    kappa = np.mean(list(subset_mmd_scores) + [grand_mmd_score])/ np.mean(list(subset_ntk_scores) + [grand_ntk_score])
    combined_scores = - kappa * np.array(subset_ntk_scores) - np.array(subset_mmd_scores)
    grand_combined_scores = - kappa * grand_ntk_score - grand_mmd_score
    
    return grand_combined_scores - combined_scores

def calculate_lava_score(dataset, trdata, trlabel, training_size, resize=32, device='cuda'):

    print("Start computing LAVA score")
    if dataset == "cifar10":
        ckpt_path = 'cifar10_embedder_preact_resnet18.pth'
        transform = get_dataset_transform(dataset.upper(), to3channels=False, resize=resize)
        train = CustomTensorDataset1(data=torch.tensor(trdata, dtype=torch.float32), targets=torch.tensor(trlabel), transform=transform)
    elif dataset == "mnist":
        # ckpt_path = 'preact_resnet18_test_mnist.pth'
        ckpt_path = 'cifar10_embedder_preact_resnet18.pth'
        transform = get_dataset_transform(dataset.upper(), to3channels=False, resize=resize)
        train = CustomTensorDataset2(tensors=(torch.tensor(np.repeat(np.expand_dims(trdata,1), 3, 1), dtype=torch.float32), torch.tensor(trlabel.reshape(-1))), transform=transform)
    # train = Custom_Dataset(X=torch.tensor(trdata, dtype=torch.float32), y=torch.tensor(trlabel))
    loaders, _ =  load_torchvision_data_keepclean(dataname=dataset.upper(), valid_size=0, splits=None, shuffle=False, stratified=False, random_seed=None, batch_size = 64,
                    resize=resize, to3channels=False, maxsize = None, maxsize_test=None, num_workers = 0, transform=None, data=(train, train))
    
    feature_extractor = lava.load_pretrained_feature_extractor(ckpt_path, device)
    dual_sol = lava.compute_dual_no_vis(feature_extractor, loaders['train'], loaders['test'], resize=resize)
    calibrated_gradient = lava.compute_values(dual_sol, training_size)
    print("Finish computing LAVA score")
    return np.array(calibrated_gradient)

def kmeans_idx_for_test(tedata, telabel, k=40):
    """
    Implements the k-means clustering algorithm for the test dataset. And return the indics of each cluster.
    """
    if len(tedata.shape) > 2:
        tedata_ = tedata.reshape(len(tedata), -1)
    else:
        tedata_ = tedata
    kmeans = KMeans(n_clusters=k, random_state=0).fit(tedata_)
    cluster_idx = [[] for _ in range(k)]
    for i in range(len(tedata)):
        cluster_idx[kmeans.labels_[i]].append(i)
    return cluster_idx

def worst_case_perform_test(test_predict, test_label, test_loss, cluster_idx, regression):
    """
    Calculate the worst performance on the model on the different clustered test subset (using cluster_idx as idx to split the test dataset). If the task is regression, the worst performance is the maximum loss on average loss of test subsets. If the task is classification, similarly the worst performance is the minimum accuracy and maximum loss (return both). test_predict is the logits of the model output, test_label is the one-hot label for classification task.
    """
    test_loss = np.array(test_loss)
    if regression:
        worst_loss = 0
        for idx in cluster_idx:
            worst_loss = max(worst_loss, test_loss[idx].mean())
        return None, worst_loss
    else:
        worst_acc = 1
        worst_loss = 0
        for idx in cluster_idx:
            worst_acc = min(worst_acc, sum(np.argmax(test_label[idx], axis=1) == np.argmax(test_predict[idx], axis=1))/len(idx))
            worst_loss = max(worst_loss, test_loss[idx].mean())
        return worst_acc, worst_loss

# summary of the results
LENGTH_SCALE_DICT = {'housing': 2, 'credit_card': 1.2, 'diabetes': 10.0, 'mnist': 50.0, 'cifar10': 600.0, 'uber_lyft': 3.0, 'rideshare': 2.0}
    