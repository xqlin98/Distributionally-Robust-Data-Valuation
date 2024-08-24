import argparse
import copy
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gurobipy as gp
import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')
import numpy as np
import pandas as pd
from gurobipy import GRB
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tqdm import tqdm
from utils.models_defined import MLP_REGRESSION, Model_Train

from utils.ntk import (compute_score, get_full_kernel_matrix, linear_solver,
                       linear_solver_regression)
from utils.utils import calculate_davinz_score, calculate_influence_score, calculate_lava_score, calculate_tracin_score, drge_approximation, kmeans_idx_for_test, tqdm_joblib, worst_case_perform_test
from utils.load_data import load_data, load_data_save
from joblib import Parallel, delayed
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
    parser.add_argument('-length_scale', '--length_scale', help='length scale for kernel',
                        nargs='?',  type=float, default=2)
    parser.add_argument('-thread_num', '--thread_num', help='thread number for multiprocessing',
                        nargs='?',  type=int, default=10)
    parser.add_argument('-epsilon', '--epsilon', help='epsilon for drge',
                        nargs='?',  type=float, default=5)
    parser.add_argument('-expname', '--expname', help='name for the experiment',
                        nargs='?',  type=str, default='allbaseline')
    parser.add_argument('-trial', '--trial', help='number of trial in the experiment',
                        nargs='?',  type=int, default=5)
    parser.add_argument('-mu', '--mu', help='mu',
                        nargs='?',  type=float, default=0.01)
    parser.add_argument('-cluster', '--cluster', help='number of clusters',
                        nargs='?',  type=int, default=50)
    cmd_args = vars(parser.parse_args())

    mu = cmd_args['mu']
    dataset = cmd_args['dataset']
    numdp = cmd_args['numdp']
    epsilon = cmd_args['epsilon']
    length_scale = cmd_args["length_scale"]
    thread_num = cmd_args['thread_num']
    seed = cmd_args['seed']
    regression = False if dataset in ["mnist", 'cifar10'] else True
    class_num = 10 if dataset in ["mnist", 'cifar10'] else None
    evaldp = numdp if not cmd_args['evaldp'] else cmd_args['evaldp']
    expname = cmd_args['expname']
    save = True if dataset in ["mnist", 'cifar10'] else False
    save_name = f'Kernel_Regression/{dataset}-numdp-{numdp}-seed-{seed}-length_scale-{length_scale}-epsilon-{epsilon}-mu-{mu}-evaldp-{evaldp}-{expname}.npz'
    
    # load data
    if not save:
        trdata, tedata, trlabel, telabel = load_data(dataset=dataset, numdp=numdp)
    else:
        trdata, tedata, trlabel, telabel, trdata_raw, trlabel_raw = load_data_save(dataset=dataset, numdp=numdp)    
    kernel_fn = RBF(length_scale=length_scale)

    np.random.seed(seed) # set random seed
    
    # kernel regression on grand dataset
    krr = KernelRidge(alpha=mu, kernel=RBF(length_scale=length_scale))
    krr.fit(trdata, trlabel)
    trlabel_predict = krr.predict(trdata)
    grand_tr_loss = mean_squared_error(trlabel_predict, trlabel)
    if regression:
        grand_set_loss = (krr.predict(tedata) - telabel)**2
    else:
        grand_set_loss = np.sum((krr.predict(tedata) - telabel)**2,axis=1)
    grand_drge = drge_approximation(grand_set_loss, epsilon)

    kernel_matrix = get_full_kernel_matrix(trdata, trdata, kernel_fn, sklearn=True)
    alpha = linear_solver_regression(kernel_matrix, trlabel, mu=mu)
    cluster_idx = kmeans_idx_for_test(tedata, telabel, k=cmd_args['cluster'])

    def compute_scores_for_one(set_idx, regression=True, class_num=10):
        jax.config.update('jax_platform_name', 'cpu')
        set_selected = np.array(list(set(range(numdp)) - set([set_idx])))
        
        new_train_x = trdata[set_selected]
        new_train_y = trlabel[set_selected]

        # kernel regression using new training dataset
        if regression:
            new_kernel_matrix_1 = kernel_matrix[np.ix_(set_selected,set_selected)]
            beta = linear_solver_regression(new_kernel_matrix_1, new_train_y.reshape(-1,1), mu=mu)
            new_kernel_matrix_2 = kernel_matrix[np.ix_(np.array(range(numdp)),set_selected)]
            deviation_score = compute_score(alpha, beta, kernel_matrix, new_kernel_matrix_1, new_kernel_matrix_2)
            del new_kernel_matrix_1, new_kernel_matrix_2
        else:
            new_kernel_matrix_1 = kernel_matrix[np.ix_(set_selected,set_selected)]
            beta = linear_solver_regression(new_kernel_matrix_1, new_train_y, mu=mu)
            new_kernel_matrix_2 = kernel_matrix[np.ix_(np.array(range(numdp)),set_selected)]
            deviation_score = compute_score(alpha, beta, kernel_matrix, new_kernel_matrix_1, new_kernel_matrix_2, regression=False, class_num=class_num)
            del new_kernel_matrix_1, new_kernel_matrix_2
            
        krr = KernelRidge(alpha=mu, kernel=RBF(length_scale=length_scale))
        krr.fit(new_train_x, new_train_y)
        
        train_loss = mean_squared_error(krr.predict(trdata),trlabel)
        loo_score = train_loss - grand_tr_loss

        if regression:
            test_loss = (krr.predict(tedata) - telabel)**2
        else:
            test_loss = np.sum((krr.predict(tedata) - telabel)**2,axis=1)
        subset_drge = drge_approximation(test_loss, epsilon)
        marginal_drge = subset_drge - grand_drge
        
        return [deviation_score.item(), loo_score, marginal_drge]

    if not regression:
        lava_scores = calculate_lava_score(dataset, trdata_raw, trlabel_raw, numdp)
    davinz_scores = calculate_davinz_score(trdata, trlabel, thread_num, regression)
    influence_scores = calculate_influence_score(copy.deepcopy(trdata), copy.deepcopy(trlabel),regression=regression, class_num=class_num)
    trackin_scores = calculate_tracin_score(copy.deepcopy(trdata), copy.deepcopy(trlabel), regression=regression, class_num=class_num)
    # lava_scores=davinz_scores=influence_scores=trackin_scores= np.zeros(numdp)
    
    deviation_scores_m, loo_scores_m, marginal_drge_m, drge_deviation_score_m, drge_loo_score_m, drge_deviation_score_l2h_m, drge_loo_score_l2h_m = [],[],[],[],[],[],[]
    if_scores_m, trackin_scores_m, lava_scores_m, davinz_scores_m, drge_if_score_m, drge_trackin_score_m, drge_lava_score_m, drge_davinz_score_m, drge_random_score_m, drge_if_score_l2h_m, drge_trackin_score_l2h_m, drge_lava_score_l2h_m, drge_davinz_score_l2h_m, drge_random_score_l2h_m = [],[],[],[],[],[],[], [], [], [], [], [], [], []
    for round in range(cmd_args['trial']):

        selected_dp = np.random.choice(range(numdp), evaldp, replace=False)

        with tqdm_joblib(tqdm(desc=f"Removing sets, round {round}", total=evaldp)) as progress_bar:
            scores_result = Parallel(n_jobs=thread_num)(delayed(compute_scores_for_one)(tmp, regression=regression, class_num=class_num) for tmp in selected_dp)
        scores_result = np.array(scores_result)
        deviation_scores, loo_scores, marginal_drge = scores_result[:,0], scores_result[:,1], scores_result[:,2]
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

        def remove_evaluation(set_idx, regression=True):
            set_selected = np.array(list(set(range(numdp)) - set(set_idx)))

            new_train_x = trdata[set_selected]
            new_train_y = trlabel[set_selected]


            krr = KernelRidge(alpha=mu, kernel=RBF(length_scale=length_scale))
            krr.fit(new_train_x, new_train_y)

            test_predict = krr.predict(tedata)
            if regression:
                test_loss = (test_predict - telabel)**2
            else:
                test_loss = np.sum((test_predict - telabel)**2,axis=1)
            subset_drge = drge_approximation(test_loss, epsilon)
            wrost_acc, worst_loss = worst_case_perform_test(test_predict, telabel, test_loss, cluster_idx, regression)
            return [subset_drge, wrost_acc, worst_loss]

        # data removal using deviation score
        sorted_idx = np.argsort(deviation_scores)[::-1]
        with tqdm_joblib(tqdm(desc=f"Evaluation deviation, round {round}", total=evaldp)) as progress_bar:
            drge_deviation_score = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
        drge_deviation_score_m += [drge_deviation_score]
        
        # data removal using loo score
        sorted_idx = np.argsort(loo_scores)[::-1]
        with tqdm_joblib(tqdm(desc=f"Evaluation LOO, round {round}", total=evaldp)) as progress_bar:
            drge_loo_score = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
        drge_loo_score_m += [drge_loo_score]
        
        # data removal using influence function
        sorted_idx = np.argsort(influence_scores[selected_dp])[::-1]
        with tqdm_joblib(tqdm(desc=f"Evaluation influence function, round {round}", total=evaldp)) as progress_bar:
            drge_if_score = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
        drge_if_score_m += [drge_if_score]
        
        # data removal using trackin score
        sorted_idx = np.argsort(trackin_scores[selected_dp])[::-1]
        with tqdm_joblib(tqdm(desc=f"Evaluation trackin, round {round}", total=evaldp)) as progress_bar:
            drge_trackin_score = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
        drge_trackin_score_m += [drge_trackin_score]

        # data removal using lave score, only when classification
        if not regression:
            sorted_idx = np.argsort(lava_scores[selected_dp])[::-1]
            with tqdm_joblib(tqdm(desc=f"Evaluation lava, round {round}", total=evaldp)) as progress_bar:
                drge_lava_score = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
            drge_lava_score_m += [drge_lava_score]
        
        # data removal using davinz score
        sorted_idx = np.argsort(davinz_scores[selected_dp])[::-1]
        with tqdm_joblib(tqdm(desc=f"Evaluation davinz, round {round}", total=evaldp)) as progress_bar:
            drge_davinz_score = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
        drge_davinz_score_m += [drge_davinz_score]

        # data removal using random
        sorted_idx = np.random.permutation(range(evaldp))
        with tqdm_joblib(tqdm(desc=f"Evaluation trackin, round {round}", total=evaldp)) as progress_bar:
            drge_random_score = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
        drge_random_score_m += [drge_random_score]
        



        # data removal using deviation score l2h
        sorted_idx = np.argsort(deviation_scores)
        with tqdm_joblib(tqdm(desc=f"Evaluation deviation L2H, round {round}", total=evaldp)) as progress_bar:
            drge_deviation_score_l2h = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
        drge_deviation_score_l2h_m += [drge_deviation_score_l2h]

        # data removal using loo score l2h
        sorted_idx = np.argsort(loo_scores)
        with tqdm_joblib(tqdm(desc=f"Evaluation LOO L2H, round {round}", total=evaldp)) as progress_bar:
            drge_loo_score_l2h = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
        drge_loo_score_l2h_m += [drge_loo_score_l2h]

        # data removal using influence function l2h
        sorted_idx = np.argsort(influence_scores[selected_dp])
        with tqdm_joblib(tqdm(desc=f"Evaluation influence function L2H, round {round}", total=evaldp)) as progress_bar:
            drge_if_score_l2h = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
        drge_if_score_l2h_m += [drge_if_score_l2h]

        # data removal using trackin l2h
        sorted_idx = np.argsort(trackin_scores[selected_dp])
        with tqdm_joblib(tqdm(desc=f"Evaluation trackin L2H, round {round}", total=evaldp)) as progress_bar:
            drge_trackin_score_l2h = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
        drge_trackin_score_l2h_m += [drge_trackin_score_l2h]

        # data removal using lave score l2h, only when classification
        if not regression:
            sorted_idx = np.argsort(lava_scores[selected_dp])
            with tqdm_joblib(tqdm(desc=f"Evaluation lava L2H, round {round}", total=evaldp)) as progress_bar:
                drge_lava_score_l2h = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
            drge_lava_score_l2h_m += [drge_lava_score_l2h]
        
        # data removal using davinz score l2h
        sorted_idx = np.argsort(davinz_scores[selected_dp])
        with tqdm_joblib(tqdm(desc=f"Evaluation davinz L2H, round {round}", total=evaldp)) as progress_bar:
            drge_davinz_score_l2h = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
        drge_davinz_score_l2h_m += [drge_davinz_score_l2h]
        
        # data removal using random l2h
        sorted_idx = np.random.permutation(range(evaldp))
        with tqdm_joblib(tqdm(desc=f"Evaluation random L2H, round {round}", total=evaldp)) as progress_bar:
            drge_random_score_l2h = Parallel(n_jobs=thread_num)(delayed(remove_evaluation)(selected_dp[sorted_idx[:tmp]],regression) for tmp in range(evaldp))
        drge_random_score_l2h_m += [drge_random_score_l2h]
        
        np.savez(save_name, deviation_scores_m=np.array(deviation_scores_m), loo_scores_m=np.array(loo_scores_m), marginal_drge_m=np.array(marginal_drge_m), drge_deviation_score_m=np.array(drge_deviation_score_m), drge_loo_score_m=np.array(drge_loo_score_m), drge_deviation_score_l2h_m=np.array(drge_deviation_score_l2h_m), drge_loo_score_l2h_m=np.array(drge_loo_score_l2h_m), if_scores_m=np.array(if_scores_m), trackin_scores_m=np.array(trackin_scores_m), lava_scores_m=np.array(lava_scores_m), davinz_scores_m=np.array(davinz_scores_m), drge_if_score_m=np.array(drge_if_score_m), drge_trackin_score_m=np.array(drge_trackin_score_m), drge_lava_score_m=np.array(drge_lava_score_m), drge_davinz_score_m=np.array(drge_davinz_score_m), drge_random_score_m=np.array(drge_random_score_m), drge_if_score_l2h_m=np.array(drge_if_score_l2h_m), drge_trackin_score_l2h_m=np.array(drge_trackin_score_l2h_m), drge_lava_score_l2h_m=np.array(drge_lava_score_l2h_m), drge_davinz_score_l2h_m=np.array(drge_davinz_score_l2h_m), drge_random_score_l2h_m=np.array(drge_random_score_l2h_m))
