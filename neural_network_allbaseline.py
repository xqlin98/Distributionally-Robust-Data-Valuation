import argparse
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils.dataset import Custom_Dataset
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import jax
jax.config.update('jax_platform_name', 'cpu')

import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm
from utils.models_defined import CNN, MLP_REGRESSION, MLP_REGRESSION_L, MLP_REGRESSION_S, MLP_CLS_L, MLP_CLS_S, VGG, Model_Train, ResNet, deactivate_batchnorm

from utils.ntk import (compute_score, empirical_ntk_jacobian_contraction, get_full_kernel_matrix, linear_solver,
                       linear_solver_regression)
from utils.utils import calculate_davinz_score_nn, calculate_influence_score, calculate_influence_score_nn, calculate_tracin_score, calculate_lava_score, calculate_davinz_score, calculate_tracin_score_nn, drge_approximation, tqdm_joblib, kmeans_idx_for_test, worst_case_perform_test, LENGTH_SCALE_DICT
from utils.load_data import load_data, load_data_save
from joblib import Parallel, delayed
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from functorch import make_functional, make_functional_with_buffers
from functorch.experimental import replace_all_batch_norm_modules_
from torch.utils.data.sampler import SubsetRandomSampler

if __name__ == "__main__":
    ''' Parse cmd arguments '''
    parser = argparse.ArgumentParser(description='Process which dataset to run')
    parser.add_argument('-dataset', '--dataset', help='name of the dataset',
                        nargs='?',  type=str, default='rideshare')
    parser.add_argument('-seed', '--seed', help='random seed',
                        nargs='?',  type=int, default=123)
    parser.add_argument('-numdp', '--numdp', help='number of data points in training dataset',
                        nargs='?',  type=int, default=5000)
    parser.add_argument('-evaldp', '--evaldp', help='number of data points to be evaluated',
                        nargs='?',  type=int, default=None)
    parser.add_argument('-thread_num', '--thread_num', help='thread number for multiprocessing',
                        nargs='?',  type=int, default=10)
    parser.add_argument('-thread_num_nn', '--thread_num_nn', help='thread number for multiprocessing',
                        nargs='?',  type=int, default=10)
    parser.add_argument('-epsilon', '--epsilon', help='epsilon for drge',
                        nargs='?',  type=float, default=5)
    parser.add_argument('-expname', '--expname', help='name for the experiment',
                        nargs='?',  type=str, default='')
    parser.add_argument('-trial', '--trial', help='number of trial in the experiment',
                        nargs='?',  type=int, default=5)
    parser.add_argument('-mu', '--mu', help='mu',
                        nargs='?',  type=float, default=0)
    parser.add_argument('-model', '--model', help='model to use to compute NTK and training',
                        nargs='?',  type=str, default="MNIST_REGRESSION_RELU")
    parser.add_argument('-cpu', '--cpu', help='Use cpu to compute NTK or not',
                        nargs='?',  type=int, default=0)
    parser.add_argument('-batch_size', '--batch_size', help='batch size when training',
                        nargs='?',  type=int, default=256)
    parser.add_argument('-epochs', '--epochs', help='number of epochs',
                        nargs='?',  type=int, default=10)
    parser.add_argument('-lr', '--lr', help='learning rate',
                        nargs='?',  type=float, default=0.01)
    parser.add_argument('-gpus', '--gpus', help='which gpus to use',
                            nargs='?',  type=str, default="0")
    parser.add_argument('-restore', '--restore', help='path to restore the result',
                            nargs='?',  type=str, default=None)
    parser.add_argument('-cluster', '--cluster', help='number of clusters',
                        nargs='?',  type=int, default=50)
    
    cmd_args = vars(parser.parse_args())

    mu = cmd_args['mu']
    dataset = cmd_args['dataset']
    numdp = cmd_args['numdp']
    epsilon = cmd_args['epsilon']
    thread_num = cmd_args['thread_num']
    thread_num_nn = cmd_args['thread_num_nn']
    seed = cmd_args['seed']
    regression = False if dataset in ["mnist", 'cifar10'] else True
    class_num = 10 if dataset in ["mnist", 'cifar10'] else None
    in_channels = 3 if dataset in ["cifar10"] else 1
    linear_dim = 8 if dataset in ["cifar10"] else 7
    eval_step = 5
    evaldp = numdp if not cmd_args['evaldp'] else cmd_args['evaldp']
    expname = cmd_args['expname']
    model = cmd_args['model']
    cpu = cmd_args['cpu'] == 1
    gpus = [torch.device(f"cuda:{tmp}") for tmp in cmd_args['gpus']]
    
    # NN hyper
    if "CNN" in model:
        model_fn = lambda input_dim, class_num: CNN(input_dim=input_dim, in_channels=in_channels, class_num=class_num, linear_dim=linear_dim)
    elif "RESNET" in model:
        if "18" in model:
            num_blocks = [2, 2, 2, 2]
        elif "21" in model:
            num_blocks = [2, 3, 3, 3]
        elif "34" in model:
            num_blocks = [3, 4, 6, 3]
        model_fn = lambda input_dim, class_num: ResNet(num_blocks=num_blocks, in_channels=in_channels, class_num=class_num)
    elif "VGG" in model:
        model_fn = lambda input_dim, class_num: VGG(vgg_name=model, in_channels=in_channels, class_num=class_num)
    else:
        model_fn = eval(model)
    optimizer_fn = optim.SGD
    loss_fn = nn.MSELoss
    loss_fn_ini = loss_fn()
    lr = cmd_args['lr']
    batch_size = cmd_args['batch_size']
    epochs = cmd_args['epochs']
    device = torch.device("cpu")
    flatten = False if any(tmp in model for tmp in ["CNN", "RESNET", "VGG"]) else True
    save = True if dataset in ["mnist", 'cifar10'] else False
    save_name = f'NN/{dataset}-numdp-{numdp}-seed-{seed}-model-{model}-epsilon-{epsilon}-evaldp-{evaldp}-{expname}-allbaseline.npz'

    # load data
    if not save:
        trdata, tedata, trlabel, telabel = load_data(dataset=dataset, numdp=numdp, flatten=flatten)
    else:
        trdata, tedata, trlabel, telabel, trdata_raw, trlabel_raw = load_data_save(dataset=dataset, numdp=numdp, flatten=flatten)    
    trdata, tedata = torch.tensor(trdata, dtype=torch.float32).to(device), torch.tensor(tedata, dtype=torch.float32).to(device)

    np.random.seed(seed) # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # kernel regression on grand dataset
    input_dim = trdata.shape[1]
    train_set = Custom_Dataset(torch.tensor(trdata, dtype=torch.float32), torch.tensor(trlabel, dtype=torch.float32), device=device, return_idx=False)
    test_set = Custom_Dataset(torch.tensor(tedata, dtype=torch.float32), torch.tensor(telabel, dtype=torch.float32), device=device, return_idx=False)
    train_dataloader =  DataLoader(train_set, batch_size=batch_size)
    test_dataloader =  DataLoader(test_set, batch_size=batch_size)
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
    _, trlabel_predict = model_train.evaluate_point_loss(train_dataloader, regression=regression, return_logits=True)
    model_train.simplefit(train_dataloader, epochs=epochs, init=True)
    
    grand_tr_loss = model_train.evaluate(train_dataloader, regression=regression)
    grand_set_loss = model_train.evaluate_point_loss(test_dataloader, regression=regression)
    grand_drge = drge_approximation(grand_set_loss, epsilon)

    print(grand_tr_loss)
    print(np.mean(grand_set_loss))
    
    ini_model = model_train.get_init_model()
    if "RESNET" in model or "VGG" in model:
        replace_all_batch_norm_modules_(ini_model)
    fnet, params = make_functional(ini_model)
    
    kernel_fn = lambda x1, x2: empirical_ntk_jacobian_contraction(fnet, params, x1, x2, cpu=cpu, regression=regression)

    kernel_matrix = get_full_kernel_matrix(trdata, trdata, kernel_fn, sklearn=True)
    alpha = linear_solver_regression(kernel_matrix, trlabel, mu=mu)
    cluster_idx = kmeans_idx_for_test(tedata, telabel, k=cmd_args["cluster"])

    def compute_deviation_score_for_one(set_idx, regression=True, class_num=10):
        jax.config.update('jax_platform_name', 'cpu')

        set_selected = np.array(list(set(range(numdp)) - set([set_idx])))

        new_train_y = trlabel[set_selected]

        # kernel regression using new training dataset
        new_kernel_matrix_1 = kernel_matrix[np.ix_(set_selected,set_selected)]
        beta = linear_solver_regression(new_kernel_matrix_1, new_train_y, mu=mu)
        new_kernel_matrix_2 = kernel_matrix[np.ix_(np.array(range(numdp)),set_selected)]
        deviation_score = compute_score(alpha, beta, kernel_matrix, new_kernel_matrix_1, new_kernel_matrix_2, regression=regression, class_num=class_num)
        del new_kernel_matrix_1, new_kernel_matrix_2
        
        return deviation_score.item()
        # return [deviation_score.item(), loo_score.item(), marginal_drge]

    def compute_drge_score_for_one(idxs_list, device, show=False, desc=""):
        model_train = model_train_func(device)
        result = []
        if show == True:
            idxs_list = tqdm(idxs_list, desc=desc)
        for set_idx in idxs_list:
            set_selected = np.array(list(set(range(numdp)) - set([set_idx])))

            new_train_loader =  DataLoader(train_set, sampler=SubsetRandomSampler(set_selected), batch_size=batch_size)
            
            model_train.simplefit(new_train_loader, epochs=epochs, init=True)
            
            train_loss = model_train.evaluate(train_dataloader, regression=regression)
            loo_score = train_loss - grand_tr_loss.to(device)

            test_loss = model_train.evaluate_point_loss(test_dataloader, regression=regression)

            subset_drge = drge_approximation(test_loss, epsilon)
            marginal_drge = subset_drge - grand_drge
            result.append([loo_score.item(), marginal_drge])
        del model_train
        return np.array(result)

    if not regression:
        lava_scores = calculate_lava_score(dataset, trdata_raw, trlabel_raw, numdp)
    davinz_scores = calculate_davinz_score_nn(kernel_matrix, trlabel_predict, trlabel, trdata, thread_num, regression)
    influence_scores = calculate_influence_score_nn(copy.deepcopy(trdata), copy.deepcopy(trlabel),model_train_func=model_train_func, epochs=epochs)
    trackin_scores = calculate_tracin_score_nn(copy.deepcopy(trdata), copy.deepcopy(trlabel), model_train_func=model_train_func, epochs=epochs, batch_size=batch_size, lr=lr)
    # lava_scores=davinz_scores=influence_scores=trackin_scores= np.zeros(numdp)
    
    deviation_scores_m, loo_scores_m, marginal_drge_m, drge_deviation_score_m, drge_loo_score_m, drge_deviation_score_l2h_m, drge_loo_score_l2h_m = [],[],[],[],[],[],[]
    if_scores_m, trackin_scores_m, lava_scores_m, davinz_scores_m, drge_if_score_m, drge_trackin_score_m, drge_lava_score_m, drge_davinz_score_m, drge_random_score_m, drge_if_score_l2h_m, drge_trackin_score_l2h_m, drge_lava_score_l2h_m, drge_davinz_score_l2h_m, drge_random_score_l2h_m = [],[],[],[],[],[],[], [], [], [], [], [], [], []
    if cmd_args['restore']:
        result_restore = np.load(f"{cmd_args['restore']}")
        deviation_scores_m, loo_scores_m, marginal_drge_m, drge_deviation_score_m, drge_loo_score_m, drge_deviation_score_l2h_m, drge_loo_score_l2h_m = [tmp for tmp in result_restore['deviation_scores_m']], [tmp for tmp in result_restore['loo_scores_m']], [tmp for tmp in result_restore['marginal_drge_m']], [tmp for tmp in result_restore['drge_deviation_score_m']], [tmp for tmp in result_restore['drge_loo_score_m']], [tmp for tmp in result_restore['drge_deviation_score_l2h_m']], [tmp for tmp in result_restore['drge_loo_score_l2h_m']]
        if_scores_m, trackin_scores_m, lava_scores_m, davinz_scores_m, drge_if_score_m, drge_trackin_score_m, drge_lava_score_m, drge_davinz_score_m, drge_random_score_m, drge_if_score_l2h_m, drge_trackin_score_l2h_m, drge_lava_score_l2h_m, drge_davinz_score_l2h_m, drge_random_score_l2h_m = [tmp for tmp in result_restore['if_scores_m']], [tmp for tmp in result_restore['trackin_scores_m']], [tmp for tmp in result_restore['lava_scores_m']], [tmp for tmp in result_restore['davinz_scores_m']], [tmp for tmp in result_restore['drge_if_score_m']], [tmp for tmp in result_restore['drge_trackin_score_m']], [tmp for tmp in result_restore['drge_lava_score_m']], [tmp for tmp in result_restore['drge_davinz_score_m']], [tmp for tmp in result_restore['drge_random_score_m']], [tmp for tmp in result_restore['drge_if_score_l2h_m']], [tmp for tmp in result_restore['drge_trackin_score_l2h_m']], [tmp for tmp in result_restore['drge_lava_score_l2h_m']], [tmp for tmp in result_restore['drge_davinz_score_l2h_m']], [tmp for tmp in result_restore['drge_random_score_l2h_m']]
        trial_done = len(deviation_scores_m)
        print(f"Restored from {cmd_args['restore']}, {trial_done} trials done")
    for round in range(cmd_args['trial']):

        selected_dp = np.random.choice(range(numdp), evaldp, replace=False)

        if cmd_args['restore'] and round < trial_done:
            continue
            
        gpu_sequnce = gpus * (evaldp // len(gpus) + 1)
        tmp_idxs = np.array_split(np.arange(evaldp), thread_num_nn)
        scores_result = Parallel(n_jobs=thread_num_nn, max_nbytes=5000)(delayed(compute_drge_score_for_one)(selected_dp[tmp_idxs[tmp]], gpu_sequnce[tmp], show=(tmp==0), desc=f"Removing sets for loo and drge, round {round}") for tmp in range(thread_num_nn))
        scores_result = np.vstack(scores_result)
        loo_scores, marginal_drge = scores_result[:,0], scores_result[:,1]
        
        with tqdm_joblib(tqdm(desc=f"Removing sets for deviation score, round {round}", total=evaldp)) as progress_bar:
            scores_result = Parallel(n_jobs=thread_num, max_nbytes=5000)(delayed(compute_deviation_score_for_one)(selected_dp[tmp], regression=regression, class_num=class_num) for tmp in range(evaldp))
        deviation_scores = np.array(scores_result)
        # deviation_scores=loo_scores=marginal_drge = np.zeros(evaldp)
        
        deviation_scores_m += [deviation_scores]
        loo_scores_m += [loo_scores]
        marginal_drge_m += [marginal_drge]
        if_scores_m += [influence_scores[selected_dp]]
        trackin_scores_m += [trackin_scores[selected_dp]]
        if not regression:
            lava_scores_m += [lava_scores[selected_dp]]
        davinz_scores_m += [davinz_scores[selected_dp]]
        # result = np.load(save_name)
        # deviation_scores, loo_scores, marginal_drge = result['deviation_scores'], result['loo_scores'], result['marginal_drge']

        def remove_evaluation(sorted_idx,  idxs_list, device, show=False, desc=""):
            model_train = model_train_func(device)
            result = []
            if show:
                idxs_list = tqdm(idxs_list, desc=desc)
            for idx in idxs_list:
                set_idx = selected_dp[sorted_idx[:idx]]
                set_selected = np.array(list(set(range(numdp)) - set(set_idx)))

                new_train_loader = DataLoader(train_set, sampler=SubsetRandomSampler(set_selected), batch_size=batch_size)
                model_train.simplefit(new_train_loader, epochs=epochs, init=True)

                test_loss, test_predict = model_train.evaluate_point_loss(test_dataloader, regression=regression, return_logits=True)
                subset_drge = drge_approximation(test_loss, epsilon)

                wrost_acc, worst_loss = worst_case_perform_test(test_predict, telabel, test_loss, cluster_idx, regression)
                result.append([subset_drge, wrost_acc, worst_loss])
            del model_train
            return np.array(result)
        
        def remove_drge(sorted_idx, desc):
            tmp_idx = np.array_split(np.arange(evaldp), thread_num_nn)
            drge_score = Parallel(n_jobs=thread_num_nn)(delayed(remove_evaluation)(sorted_idx, tmp_idx[tmp], gpu_sequnce[tmp], show=(tmp==0), desc=desc) for tmp in range(thread_num_nn))
            drge_score = np.vstack(drge_score)
            return drge_score
        
        # data removal using deviation score
        sorted_idx = np.argsort(deviation_scores)[::-1]
        desc = f"Evaluation deviation, round {round}"
        drge_deviation_score_m += [remove_drge(sorted_idx, desc)]
        
        # data removal using loo score
        sorted_idx = np.argsort(loo_scores)[::-1]
        desc = f"Evaluation LOO, round {round}"
        drge_loo_score_m += [remove_drge(sorted_idx, desc)]
        
        # data removal using influence function
        sorted_idx = np.argsort(influence_scores[selected_dp])[::-1]
        desc = f"Evaluation influence function, round {round}"
        drge_if_score_m += [remove_drge(sorted_idx, desc)]
        
        # data removal using trackin score
        sorted_idx = np.argsort(trackin_scores[selected_dp])[::-1]
        desc = f"Evaluation trackin, round {round}"
        drge_trackin_score_m += [remove_drge(sorted_idx, desc)]

        # data removal using lave score, only when classification
        if not regression:
            sorted_idx = np.argsort(lava_scores[selected_dp])[::-1]
            desc = f"Evaluation lava, round {round}"
            drge_lava_score_m += [remove_drge(sorted_idx, desc)]
        
        # data removal using davinz score
        sorted_idx = np.argsort(davinz_scores[selected_dp])[::-1]
        desc = f"Evaluation davinz, round {round}"
        drge_davinz_score_m += [remove_drge(sorted_idx, desc)]

        # data removal using random
        sorted_idx = np.random.permutation(range(evaldp))
        desc = f"Evaluation random, round {round}"
        drge_random_score_m += [remove_drge(sorted_idx, desc)]

        # data removal using deviation score l2h
        sorted_idx = np.argsort(deviation_scores)
        desc = f"Evaluation deviation L2H, round {round}"
        drge_deviation_score_l2h_m += [remove_drge(sorted_idx, desc)]

        # data removal using loo score l2h
        sorted_idx = np.argsort(loo_scores)
        desc = f"Evaluation LOO L2H, round {round}"
        drge_loo_score_l2h_m += [remove_drge(sorted_idx, desc)]

        # data removal using influence function l2h
        sorted_idx = np.argsort(influence_scores[selected_dp])
        desc = f"Evaluation influence function L2H, round {round}"
        drge_if_score_l2h_m += [remove_drge(sorted_idx, desc)]

        # data removal using trackin l2h
        sorted_idx = np.argsort(trackin_scores[selected_dp])
        desc = f"Evaluation trackin L2H, round {round}"
        drge_trackin_score_l2h_m += [remove_drge(sorted_idx, desc)]

        # data removal using lave score l2h, only when classification
        if not regression:
            sorted_idx = np.argsort(lava_scores[selected_dp])
            desc = f"Evaluation lava L2H, round {round}"
            drge_lava_score_l2h_m += [remove_drge(sorted_idx, desc)]

        # data removal using davinz score l2h
        sorted_idx = np.argsort(davinz_scores[selected_dp])
        desc = f"Evaluation davinz L2H, round {round}"
        drge_davinz_score_l2h_m += [remove_drge(sorted_idx, desc)]

        # data removal using random l2h
        sorted_idx = np.random.permutation(range(evaldp))
        desc = f"Evaluation random L2H, round {round}"
        drge_random_score_l2h_m += [remove_drge(sorted_idx, desc)]

        np.savez(save_name, deviation_scores_m=np.array(deviation_scores_m), loo_scores_m=np.array(loo_scores_m), marginal_drge_m=np.array(marginal_drge_m), drge_deviation_score_m=np.array(drge_deviation_score_m), drge_loo_score_m=np.array(drge_loo_score_m), drge_deviation_score_l2h_m=np.array(drge_deviation_score_l2h_m), drge_loo_score_l2h_m=np.array(drge_loo_score_l2h_m), if_scores_m=np.array(if_scores_m), trackin_scores_m=np.array(trackin_scores_m), lava_scores_m=np.array(lava_scores_m), davinz_scores_m=np.array(davinz_scores_m), drge_if_score_m=np.array(drge_if_score_m), drge_trackin_score_m=np.array(drge_trackin_score_m), drge_lava_score_m=np.array(drge_lava_score_m), drge_davinz_score_m=np.array(drge_davinz_score_m), drge_random_score_m=np.array(drge_random_score_m), drge_if_score_l2h_m=np.array(drge_if_score_l2h_m), drge_trackin_score_l2h_m=np.array(drge_trackin_score_l2h_m), drge_lava_score_l2h_m=np.array(drge_lava_score_l2h_m), drge_davinz_score_l2h_m=np.array(drge_davinz_score_l2h_m), drge_random_score_l2h_m=np.array(drge_random_score_l2h_m))
