import os
import numpy as np
import ntpath
import random ## seed function is called with this
import pprint
import matplotlib.pyplot as plt


from src.matrix.factorization import ksvd, MatrixFactorizationType, apply_mf
from sklearn.ensemble.forest import RandomForestRegressor, ExtraTreesRegressor
from src.misc import settings
from src.performance.rank import Rank
from src.performance.evaluate import EvaluationType, EvaluationMetricType
from src.main.cold_start import ColdStart
from sklearn.linear_model.bayes import BayesianRidge
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.coordinate_descent import Lasso
from sklearn.cross_decomposition.pls_ import PLSRegression
from sklearn.svm.classes import SVR
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.ensemble.weight_boosting import AdaBoostRegressor


class MappingMethodType:     
    RandomForest, Lasso, LinearRegression, PLSRegression, DecisionTreeRegressor, ExtraTreesRegressor = range(0, 6)

class LatentEstimateColdStart(ColdStart):
     
    def __init__(self, map_method_type, mf_rank, mf_type = MatrixFactorizationType.svd):
        self.map_method_type = map_method_type
        self.mf_rank = mf_rank
        self.mf_type = mf_type


    def train(self, ia_matrix, i_ft_matrix, mf_rank):
        
        max_rank = np.min(ia_matrix.shape)
        if mf_rank > max_rank:  
            mf_rank = max_rank
        
        Uk, sk, Vk, s = apply_mf(ia_matrix, self.mf_type, mf_rank)  
    
        map_method = self.get_map_method()
        map_method.fit(i_ft_matrix, Uk)

        return map_method, sk, Vk

    def get_map_method(self):
        
        map_method = None  
        if self.map_method_type == MappingMethodType.RandomForest:
            map_method = RandomForestRegressor(n_estimators=10, random_state=settings.__seed__) ##n_estimators = 10 (#trees) by default
        elif self.map_method_type == MappingMethodType.Lasso:
            map_method = Lasso()
        elif self.map_method_type == MappingMethodType.LinearRegression:
            map_method = LinearRegression()
        elif self.map_method_type == MappingMethodType.PLSRegression:
            map_method = PLSRegression()
        elif self.map_method_type == MappingMethodType.DecisionTreeRegressor:
            map_method = DecisionTreeRegressor()            
        elif self.map_method_type == MappingMethodType.ExtraTreesRegressor:     
            map_method = ExtraTreesRegressor(random_state=settings.__seed__)                     
                        
        return map_method


    def test(self, regr_model, i_ft_matrix):
        return regr_model.predict(i_ft_matrix)


    def predict(self, train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix):
        
        regr_model, sk, Vk = self.train(train_ia_rank_matrix, train_i_ft_matrix, self.mf_rank)
        Uk = self.test(regr_model, test_i_ft_matrix)
            
        if self.mf_rank == 1: ## just convert array (x,) to (x,1) not to have any error while multiplication
            Uk = np.reshape(Uk, (-1,1))    
            
        return np.dot(Uk, np.dot(np.diag(sk), Vk))

