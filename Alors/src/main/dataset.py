'''
'''
import numpy as np
import ntpath

from src.dao.misc_io import load_data
from src.matrix.factorization import apply_mf
from src.misc import settings
from src.matrix.matrix_io import convert_to_rank_matrix

class Dataset(object):
    '''
        Class to keep all the info related to a given dataset
    '''

    def __init__(self, exp_setting_dict):
        '''
        Constructor
        '''
        self.name = None
        self.ia_perf_matrix = None
        self.ia_rank_matrix = None
        self.i_ft_matrix = None
        self.Uk = None
        self.sk = None
        self.Vk = None
        self.unsolved_inst_list = None
        self.sparsity_level = None
        self.missing_val = settings.missing_val
        self.higher_better = exp_setting_dict['higher_better']
        self.pre_process_list = None

        if 'perf_bound_val' in exp_setting_dict.keys():
            self.perf_bound_val = exp_setting_dict['perf_bound_val']
        else:
            self.perf_bound_val = settings.perf_bound_val
        
        ai_perf_matrix, self.i_ft_matrix, self.unsolved_inst_list = self.load(exp_setting_dict['ai_file_path'], 
                                                                               exp_setting_dict['ft_file_path'], 
                                                                               self.perf_bound_val, 
                                                                               exp_setting_dict['higher_better'])
        
        self.ia_perf_matrix = ai_perf_matrix.T
        
        decimal_level = -1  
        if 'decimal_level' in exp_setting_dict.keys():
            decimal_level = exp_setting_dict['decimal_level']  
            
        
        self.ia_rank_matrix = convert_to_rank_matrix(self.ia_perf_matrix, exp_setting_dict['higher_better'], decimal_level)
        
        self.Uk, self.sk, self.Vk, s = apply_mf(self.ia_rank_matrix, exp_setting_dict['mf_type'], exp_setting_dict['mf_rank'])
        
        if 'pre_process' in exp_setting_dict.keys():
            if exp_setting_dict['pre_process']:   
            
                self.pre_process_list = exp_setting_dict['pre_process']
                
                self.preprocess_data(exp_setting_dict['perf_bound_val'], 
                                     exp_setting_dict['mf_rank'], 
                                     exp_setting_dict['eval_metrics'][0],
                                     exp_setting_dict['higher_better'], 
                                     exp_setting_dict['pp_threshold'],
                                     1,
                                     self.name+"-hist",
                                     self.name+"-hist",  
                                     settings.__output_folder__)

        
        self.sparsity_level = self.calculate_sparsity_level()
        

        
    def load(self, alg_inst_perf_file, inst_ft_file, perf_bound_val, higher_better):
        '''
            Load and return algorithm-instance performance data  
                             and instance features
            
            :param alg_inst_perf_file: algorithm-instance performance file
            :param inst_ft_file: instance descriptive features file
            :return: numpy 2D array, numpy 2D array - algorithm-instance performance matrix,
                                                      instance features matrix
        '''
        
        self.name = ntpath.basename(alg_inst_perf_file)
        self.name = self.name[:self.name.index('.')]
        if self.pre_process_list is not None and len(self.pre_process_list) > 0:
            self.name = self.name + '-pre-process'      
        
        ai_perf_matrix = load_data(alg_inst_perf_file, True, True);
        i_ft_matrix = load_data(inst_ft_file, True, True);
        unsolved_inst_list = []
        
        per_inst_missing_val_arr = np.zeros((len(i_ft_matrix)))
        if higher_better:
            for (x,y), value in np.ndenumerate(ai_perf_matrix):
                if ai_perf_matrix[x][y] < perf_bound_val:
                    ai_perf_matrix[x][y] = perf_bound_val
                    
                    per_inst_missing_val_arr[y] += 1
        else:
            for (x,y), value in np.ndenumerate(ai_perf_matrix):
                if ai_perf_matrix[x][y] > perf_bound_val:
                    ai_perf_matrix[x][y] = perf_bound_val
                    
                    per_inst_missing_val_arr[y] += 1
                    
        for (y), value in np.ndenumerate(per_inst_missing_val_arr):
            if value == len(ai_perf_matrix): 
                unsolved_inst_list.append(y)
    
        
        ## remove unsolved instances from the dataset            
        ai_perf_matrix = np.delete(ai_perf_matrix, unsolved_inst_list, 1) ## remove columns as instances
        i_ft_matrix = np.delete(i_ft_matrix, unsolved_inst_list, 0) ## remove rows as instances
        
        return ai_perf_matrix, i_ft_matrix, unsolved_inst_list

        
    def calculate_sparsity_level(self):
        
        self.sparsity_level = 0
        for (inst_inx, alg_inx), value in np.ndenumerate(self.ia_perf_matrix):
            if value == self.missing_val:
                self.sparsity_level += 1
        
        num_insts, num_algs = self.ia_perf_matrix.shape
        self.sparsity_level /= float(num_insts * num_algs)
      
        