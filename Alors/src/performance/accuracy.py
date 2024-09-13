'''
'''
import numpy as np
from random import randint

from src.performance.evaluate import Evaluate

class Accuracy(Evaluate):
    '''
    '''

    def __init__(self, ia_accuracy_matrix, pred_ia_rank_matrix = None):
        '''
        Constructor
        '''
        super(Accuracy, self).__init__("Accuracy", ia_data_matrix = ia_accuracy_matrix, pred_ia_rank_matrix = pred_ia_rank_matrix)
        if pred_ia_rank_matrix is None:
            self.pred_ia_rank_matrix = np.max(self.ia_data_matrix) - self.ia_data_matrix  
        

    def kth_oracle(self, k):
        '''
        '''
        accuracy = 0
        ##for inst_inx, i_row in enumerate(self.ia_data_matrix):
        for inst_inx, i_row in enumerate(self.pred_ia_rank_matrix):
            sorted_indices_arr = np.argsort(i_row)
            ##sorted_indices_arr = sorted_indices_arr[::-1]
            
            best_alg_inx = sorted_indices_arr[k-1]
            accuracy += self.ia_data_matrix[inst_inx][best_alg_inx]

        self.kth_oracle_perf_dict[k] = accuracy / float(len(self.ia_data_matrix))
            
        return self.kth_oracle_perf_dict[k]
    
    
    def kth_single_best(self, k):
        '''
        '''
        ##sum_accuracy = self.ia_data_matrix.sum(axis=0)
        sum_rank = self.pred_ia_rank_matrix.sum(axis=0)
        
        sorted_indices_arr_perf = np.argsort(sum_rank)
        ##sorted_indices_arr_perf = sorted_indices_arr_perf[::-1]
        
        ##self.kth_single_best_dict[k] = sum_accuracy[sorted_indices_arr_perf[k-1]] / float(len(self.ia_data_matrix))
        self.kth_single_best_dict[k] = np.sum(self.ia_data_matrix.T[sorted_indices_arr_perf[k-1]]) / float(len(self.ia_data_matrix))  
              
        return self.kth_single_best_dict[k]

    
    def random(self):
        
        avg_acc_per_inst = np.average(self.ia_data_matrix, 1)

        self.random_perf = np.sum(avg_acc_per_inst) / float(len(self.ia_data_matrix))
        
        return self.random_perf

