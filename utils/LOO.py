from collections import defaultdict
from math import comb
from tqdm import tqdm
import numpy as np

from utils.Adpt_Shapley import Adpt_Shapley
from utils.ntk import compute_score, get_full_kernel_matrix, linear_solver
from utils.utils import merge_dataloaders
import jax
import jax.numpy as jnp

class LOO(Adpt_Shapley):

    def __init__(self, train_x=None, train_y_number=None, kernel_fn=None, model=None, train_loaders=None, val_loader=None, n_participants=None, class_num=None):
        Adpt_Shapley.__init__(self, model, train_loaders, val_loader, n_participants)
        self.train_x = train_x
        self.train_y_number = train_y_number
        self.kernel_fn = kernel_fn
        self.class_num = class_num

    def run(self,
            metric="accu",
            early_stopping=True):
        """Compute Leave-one-out score"""
        self.metric=metric
        loo_scores = np.zeros(self.n_participants)
        full_score = self.get_full_score()
        for idx in tqdm(range(self.n_participants), desc='Computing LOO score'):
            loo_train_loaders = [_ for _ in self.train_loaders]
            loo_train_loaders.pop(idx)

            loo_loader = merge_dataloaders(loo_train_loaders)
            loo_score = self.evaluate(loo_loader, metric, early_stopping=early_stopping)  

            loo_scores[idx] = full_score - loo_score
        self.loo_scores = loo_scores
        
        return loo_scores



class Single_Deviation(Adpt_Shapley):

    def __init__(self, train_x, train_y_number, kernel_fn, class_num):
        Adpt_Shapley.__init__(self, None, None, None, None)
        self.train_x = train_x
        self.train_y_number = train_y_number
        self.kernel_fn = kernel_fn
        self.class_num = class_num

    def run(self,
            mu=0):
        """Compute Leave-one-out score"""
        # compute the loo result of function
        train_y_onehot = jax.nn.one_hot(self.train_y_number, self.class_num).reshape(-1)
        kernel_matrix_full, kernel_matrix_ori = get_full_kernel_matrix(self.train_x, self.train_x, self.kernel_fn, self.class_num)
        alpha = linear_solver(kernel_matrix_ori, kernel_matrix_full, train_y_onehot.reshape(-1,1), self.class_num)
        train_num = len(self.train_y_number)
        loo_scores = []
        for i in tqdm(range(train_num), desc='Computing deviation score'):
            try:
                new_train_x = jnp.concatenate([self.train_x[0:i], self.train_x[(i+1):]], axis=0)  
                new_train_y_number = jnp.concatenate([self.train_y_number[0:i], self.train_y_number[(i+1):]], axis=0)  
                new_train_y = jax.nn.one_hot(new_train_y_number, self.class_num).reshape(-1)
                
                # kernel regression using new training dataset
                new_kernel_matrix_1, new_kernel_matrix_1_ori = get_full_kernel_matrix(new_train_x, new_train_x, self.kernel_fn, self.class_num)
                beta = linear_solver(new_kernel_matrix_1_ori, new_kernel_matrix_1, new_train_y.reshape(-1,1), self.class_num, mu=mu)
                new_kernel_matrix_2, _ = get_full_kernel_matrix(self.train_x, new_train_x, self.kernel_fn, self.class_num)
                score = compute_score(alpha, beta, kernel_matrix_full, new_kernel_matrix_1, new_kernel_matrix_2)
                loo_scores.append(score)
                del new_kernel_matrix_1, new_kernel_matrix_2, new_kernel_matrix_1_ori, _
            except:
                loo_scores.append(-np.infty)
        loo_scores = np.array(loo_scores).reshape([-1]).tolist()
        self.loo_scores = loo_scores
        return loo_scores
