'''
'''
import numpy as np

from src.performance.evaluate import Evaluate

class NumSolved(Evaluate):
    '''
    '''

    def __init__(self, ia_runtime_matrix, cutoff_time, pred_ia_rank_matrix = None):
        '''
        Constructor
        '''
        super(NumSolved, self).__init__("#Solved Instances", ia_data_matrix = ia_runtime_matrix, pred_ia_rank_matrix = pred_ia_rank_matrix)
        self.cutoff_time = cutoff_time

        
    def kth_oracle(self, k):
        '''
        '''
        num_solved = 0

        for inst_inx, i_row in enumerate(self.pred_ia_rank_matrix):
            sorted_indices_arr = np.argsort(i_row)
                
            best_alg_inx = sorted_indices_arr[k-1]
            if self.ia_data_matrix[inst_inx][best_alg_inx] < self.cutoff_time:
                num_solved += 1
                
        self.kth_oracle_perf_dict[k] = num_solved
            
        return self.kth_oracle_perf_dict[k]
    
    
    def kth_single_best(self, k):
        '''
        '''
        num_solved = np.zeros((len(self.ia_data_matrix[0])))
        for (inst_inx, alg_inx), value in np.ndenumerate(self.ia_data_matrix):
            if self.ia_data_matrix[inst_inx][alg_inx] < self.cutoff_time:
                num_solved[alg_inx] += 1
        
        sorted_indices_arr_perf = np.argsort(num_solved)
        sorted_indices_arr_perf = sorted_indices_arr_perf[::-1]
            
        single_best_num_solved = num_solved[sorted_indices_arr_perf[k-1]] 
              
        self.kth_single_best_dict[k] = single_best_num_solved
            
        return self.kth_single_best_dict[k]
    
    def random(self):

        pi_num_solved = np.zeros((len(self.ia_data_matrix)))
        for (inst_inx, alg_inx), runtime in np.ndenumerate(self.ia_data_matrix):
            if runtime < self.cutoff_time:
                pi_num_solved[inst_inx] += 1
    
        pi_num_solved /= float(len(self.ia_data_matrix[0]))  ## per instance average    
        
        self.random_perf = np.sum(pi_num_solved)    
        
        return self.random_perf
