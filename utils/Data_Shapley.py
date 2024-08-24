from collections import defaultdict
from math import comb
from tqdm import tqdm
import numpy as np

from utils.Adpt_Shapley import Adpt_Shapley
from utils.ntk import compute_score, get_full_kernel_matrix, linear_solver, linear_solver_incremental
from utils.utils import merge_dataloaders
import jax
import jax.numpy as jnp

class Data_Shapley(Adpt_Shapley):

    def __init__(self, model, train_loaders, val_loader, n_participants):
        Adpt_Shapley.__init__(self, model, train_loaders, val_loader, n_participants)
        self.sv_result = []

    def tmc_sv_from_mem(self):
        """To compute sv with TMC from memory"""
        sv_list = defaultdict(list)
        for coalition in self.memory:
            for i, idx in enumerate(self.memory[coalition]):
                sv_list[idx].extend(self.memory[coalition][idx])
        sv_result = []
        for i in range(self.n_participants):
            sv_result.append(np.mean(sv_list[i]))
        self.sv_result = sv_result
        return sv_result

    def exact_sv_from_mem(self):
        """To compute the sv with exact algo from memory"""
        gamma_vec = [1/comb(self.n_participants-1, k) for k in range(self.n_participants)]

        sv_list = defaultdict(list)
        for coalition in self.memory:
            if len(coalition) != self.n_participants:
                for i, idx in enumerate(self.memory[coalition]):
                    set_size = len(coalition)
                    if set_size < self.n_participants:
                        weighted_marginal = self.memory[coalition][idx] * gamma_vec[set_size] * (1/self.n_participants)
                        sv_list[idx].append(weighted_marginal)
        sv_result = []
        for i in range(self.n_participants):
            sv_result.append(np.sum(sv_list[i]))
        self.sv_result = sv_result
        return sv_result

    def run(self,
            method="tmc",
            iteration=2000,  
            tolerance=0.01,
            metric="accu",
            early_stopping=True):
            """Compute the sv with different method"""
            self.metric=metric
            self.memory = defaultdict()
            if method == "tmc":
                for iter in tqdm(range(iteration), desc='Computing data Shapley'):
                # for iter in range(iteration):
                    # if 100*(iter+1)/iteration % 1 == 0:
                    #     print('{} out of {} TMC_Shapley iterations.'.format(
                    #         iter + 1, iteration))
                    self.tmc_one_iteration(tolerance=tolerance, metric=metric, early_stopping=early_stopping)
                sv_result = self.tmc_sv_from_mem()
            elif method == "exact":
                self.exact_method(metric)
                sv_result = self.exact_sv_from_mem()
            return sv_result


class Data_Shapley_Deviation(Data_Shapley):

    def __init__(self,  train_x, train_y_number, kernel_fn, n_participants, class_num):
        Data_Shapley.__init__(self, None, None, None, n_participants)
        self.sv_result = []
        self.train_x = train_x
        self.train_y_number = train_y_number
        self.kernel_fn = kernel_fn
        self.train_num = n_participants
        self.class_num = class_num

    def tmc_one_iteration(self, mu=0):
        """Runs one iteration of TMC-Shapley algorithm."""
        idxs = np.random.permutation(self.train_num)
        marginal_contribs = np.zeros(self.train_num)
        
        train_y_onehot = jax.nn.one_hot(self.train_y_number, self.class_num).reshape(-1)
        kernel_matrix_full, kernel_matrix_ori = get_full_kernel_matrix(self.train_x, self.train_x, self.kernel_fn, self.class_num)
        alpha = linear_solver(kernel_matrix_ori, kernel_matrix_full, train_y_onehot.reshape(-1,1), self.class_num, mu=mu)
        
        selected_idx = []
        pre_kernel_inv = None
        all_scores = []
        rev_idxs = np.flip(idxs)
        for n, idx in tqdm(enumerate(rev_idxs), leave=False):
            selected_idx.append(idx)
            new_train_x = self.train_x[np.array(selected_idx)]
            new_train_y_number = self.train_y_number[np.array(selected_idx)]
            new_train_y = jax.nn.one_hot(new_train_y_number, self.class_num).reshape(-1)
            
            # kernel regression using new training dataset
            new_kernel_matrix_1, new_kernel_matrix_1_ori = get_full_kernel_matrix(new_train_x, new_train_x, self.kernel_fn, self.class_num)
            beta, pre_kernel_inv = linear_solver_incremental(new_kernel_matrix_1_ori, pre_kernel_inv, new_train_y.reshape(-1,1), self.class_num, mu=mu)
            new_kernel_matrix_2, _ = get_full_kernel_matrix(self.train_x, new_train_x, self.kernel_fn, self.class_num)
            score = compute_score(alpha, beta, kernel_matrix_full, new_kernel_matrix_1, new_kernel_matrix_2)
            del new_kernel_matrix_1, new_kernel_matrix_2, new_kernel_matrix_1_ori, _

            all_scores.append(score)

        for n, idx in enumerate(idxs):
            if n == self.train_num-1:
                score_full = np.matmul(np.matmul(alpha.transpose(), kernel_matrix_full), alpha)
                marginal_contribs[idx] = score_full - all_scores[(self.train_num-1-n)]
            else:
                marginal_contribs[idx] = all_scores[(self.train_num-1-n-1)] - all_scores[(self.train_num-1-n)]
                
        self.tmc_record(idxs=idxs,
                        marginal_contribs=marginal_contribs)

    def run(self,
            method="tmc",
            iteration=2000,
            mu=0):
        """Compute the sv with different method"""
        self.memory = defaultdict()
        if method == "tmc":
            for iter in tqdm(range(iteration), desc='Computing deviation score'):
                self.tmc_one_iteration(mu=mu)
            sv_result = self.tmc_sv_from_mem()
        # elif method == "exact":
        #     self.exact_method(metric)
        #     sv_result = self.exact_sv_from_mem()
        return sv_result
