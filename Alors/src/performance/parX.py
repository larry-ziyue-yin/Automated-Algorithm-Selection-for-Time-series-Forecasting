'''
'''
import numpy as np

from src.performance.evaluate import Evaluate

class ParX(Evaluate):
    '''
        Calculate penalized-X runtime error 
    '''

    def __init__(self, ia_runtime_matrix, cutoff_time, X = 10, pred_ia_rank_matrix = None):
        '''
        Constructor
        '''
        super(ParX, self).__init__("Par"+str(int(X)), ia_data_matrix = ia_runtime_matrix, pred_ia_rank_matrix = pred_ia_rank_matrix)
        self.cutoff_time = cutoff_time
        self.X = X
        

    def kth_oracle(self, k):
        '''
        '''   
        self.kth_oracle_perf_dict[k] = 0
        for inst_inx, i_row in enumerate(self.pred_ia_rank_matrix):
            sorted_indices_arr = np.argsort(i_row)
                
            best_alg_inx = sorted_indices_arr[k-1]
            if self.ia_data_matrix[inst_inx][best_alg_inx] >= self.cutoff_time:
                self.kth_oracle_perf_dict[k] += self.X * self.cutoff_time  
            else:
                self.kth_oracle_perf_dict[k] += self.ia_data_matrix[inst_inx][best_alg_inx]
                
        self.kth_oracle_perf_dict[k] /= float(len(self.ia_data_matrix))
            
        return self.kth_oracle_perf_dict[k]
    
    
    def kth_single_best(self, k):
        '''
        '''
        parX = np.zeros((len(self.ia_data_matrix[0])))
        for (inst_inx, alg_inx), runtime in np.ndenumerate(self.ia_data_matrix):
            if runtime >= self.cutoff_time:
                parX[alg_inx] += self.X * self.cutoff_time    
            else:
                parX[alg_inx] += runtime  
        
        sorted_indices_arr_perf = np.argsort(parX)
        
        self.kth_single_best_dict[k] = parX[sorted_indices_arr_perf[k-1]] / float(len(self.ia_data_matrix))
              
        return self.kth_single_best_dict[k]
    
    
    def random(self):   
        
        pi_random = np.zeros((len(self.ia_data_matrix)))
        for (inst_inx, alg_inx), runtime in np.ndenumerate(self.ia_data_matrix):
            if runtime >= self.cutoff_time:
                pi_random[inst_inx] += self.X * self.cutoff_time  
            else:
                pi_random[inst_inx] += runtime
             
        pi_random /= float(len(self.ia_data_matrix[0])) ## per instance average parX
                                                                 
        self.random_perf = np.sum(pi_random) / float(len(self.ia_data_matrix))
        
        return self.random_perf
    