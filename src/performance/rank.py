'''
'''
import sys
import numpy as np
from random import randint

from src.performance.evaluate import Evaluate

class Rank(Evaluate):
    '''
        Calculate average rank
    '''

    def __init__(self, ia_rank_matrix, pred_ia_rank_matrix = None):
        '''
        Constructor
        '''
        super(Rank, self).__init__("Rank", ia_data_matrix = ia_rank_matrix, pred_ia_rank_matrix = pred_ia_rank_matrix)
    
    
    def kth_oracle(self, k):
        '''
        '''   
        self.kth_oracle_perf_dict[k] = 0

        for inst_inx, i_row in enumerate(self.pred_ia_rank_matrix):
            sorted_indices_arr = np.argsort(i_row)
                
            best_alg_inx = sorted_indices_arr[k-1]
            self.kth_oracle_perf_dict[k] += self.ia_data_matrix[inst_inx][best_alg_inx]
                
        self.kth_oracle_perf_dict[k] /= float(len(self.ia_data_matrix))
            
        return self.kth_oracle_perf_dict[k]
    
    
    def k_parallel_oracle(self, k = 3):
        '''
            kP parallel oracle performance
            for the actual oracle the result should be always same
            for a predicted matrix, kP oracle varies
        '''   
        if len(self.pred_ia_rank_matrix.T) < k:
            print "# parallel CPUs must be at most ", len(self.pred_ia_rank_matrix.T)
            print "# parallel CPUs ", k, " is changed as ", len(self.pred_ia_rank_matrix.T)
            k = len(self.pred_ia_rank_matrix.T)
            
        self.k_parallel_oracle_perf = 0
        ##for inst_inx, i_row in enumerate(self.ia_data_matrix):
        for inst_inx, i_row in enumerate(self.pred_ia_rank_matrix):
            sorted_indices_arr = np.argsort(i_row)
            
            best_val = sys.maxint
            for k_cpu in range(k):    
                best_alg_inx = sorted_indices_arr[k]
                if self.ia_data_matrix[inst_inx][best_alg_inx] < best_val:
                    best_val = self.ia_data_matrix[inst_inx][best_alg_inx]
                    
            self.k_parallel_oracle_perf += best_val
                
        self.k_parallel_oracle_perf /= float(len(self.ia_data_matrix))
            
        return self.k_parallel_oracle_perf
    
    
    def kth_single_best(self, k):
        '''
        '''
        sum_rank = self.pred_ia_rank_matrix.sum(axis=0)
        
        sorted_indices_arr_perf = np.argsort(sum_rank)
        self.kth_single_best_dict[k] = np.sum(self.ia_data_matrix.T[sorted_indices_arr_perf[k-1]]) / float(len(self.ia_data_matrix))
              
        return self.kth_single_best_dict[k]
        
        
    def k_parallel_single_best(self, k = 3):
        '''
        '''
        if len(self.pred_ia_rank_matrix.T) < k:
            print "# parallel CPUs must be at most ", len(self.pred_ia_rank_matrix.T)
            print "# parallel CPUs ", k, " is changed as ", len(self.pred_ia_rank_matrix.T)
            k = len(self.pred_ia_rank_matrix.T)
        
        sum_rank = self.pred_ia_rank_matrix.sum(axis=0)
        
        sorted_indices_arr_perf = np.argsort(sum_rank)
        
        self.k_parallel_single_best_perf = 0
        
        best_val = sys.maxint
        for k_cpu in range(k):
            if np.sum(self.ia_data_matrix.T[sorted_indices_arr_perf[k]]) < best_val:
                best_val = np.sum(self.ia_data_matrix.T[sorted_indices_arr_perf[k]])
        
        self.k_parallel_single_best_perf = best_val / float(len(self.ia_data_matrix))
              
        return self.k_parallel_single_best_perf
    
    
    def random(self):
        '''
        '''
        avg_rank_per_inst = np.average(self.ia_data_matrix, 1)
           
        self.random_perf = np.sum(avg_rank_per_inst) / float(len(self.ia_data_matrix))
        
        return self.random_perf
   

    def k_parallel_random(self, k = 3, num_samples = 1000):
        '''
        '''
        if len(self.pred_ia_rank_matrix.T) < k:
            print "# parallel CPUs must be at most ", len(self.pred_ia_rank_matrix.T)
            print "# parallel CPUs ", k, " is changed as ", len(self.pred_ia_rank_matrix.T)
            k = len(self.pred_ia_rank_matrix.T)
            
        avg_rank_per_inst = np.average(self.ia_data_matrix, 1)
           
        avg_rand_val = 0
        for iter_inx in range(num_samples): ## take num_samples random samples of k algorithms
            k_rand_algos = []
            while True:
                alg_inx = randint(0, len(self.pred_ia_rank_matrix.T)-1)
                if alg_inx not in k_rand_algos:
                    k_rand_algos.append(alg_inx)
            
            
            best_val = sys.maxint
            for alg_inx in k_rand_algos:
                if np.sum(self.ia_data_matrix.T[alg_inx]) < best_val:
                    best_val = np.sum(self.ia_data_matrix.T[alg_inx])
                
            avg_rand_val += best_val   
        
        avg_rand_val /= float(num_samples)  ## average of the samples
           
        self.random_perf = avg_rand_val / float(len(self.ia_data_matrix))
        
        return self.random_perf    
    