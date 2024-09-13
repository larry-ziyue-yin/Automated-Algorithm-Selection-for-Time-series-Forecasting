'''

'''

import numpy as np
import itertools

from scipy.stats._rank import rankdata

def dict2d_to_matrix(dict2d):
    '''
        Convert a given 2d/2 level dictionary to a 2d matrix
    '''
    keys_x = dict2d.keys()
    keys_y = dict2d[keys_x[0]].keys()
    
    result_matrix = np.zeros((len(keys_x), len(keys_y)))
    time_matrix = np.zeros((len(keys_x), len(keys_y)))
    
    for x in range(len(keys_x)):
        for y in range(len(keys_y)):
            result_matrix[x][y] = dict2d[keys_x[x]][keys_y[y]][0]
            time_matrix[x][y] = dict2d[keys_x[x]][keys_y[y]][1]
    
    return result_matrix, time_matrix, keys_x, keys_y


def convert_to_rank_matrix(ia_perf_matrix, higher_better, decimal_level = -1):
    
    if decimal_level >= 0:
        copy_ia_perf_matrix = np.round(ia_perf_matrix, decimal_level)
    else:
        copy_ia_perf_matrix = ia_perf_matrix 
    
    ia_rank_matrix = np.empty(copy_ia_perf_matrix.shape, dtype=float)
    for k, row in enumerate(copy_ia_perf_matrix):
        if higher_better:
            ia_rank_matrix[k] = len(row) - rankdata(row) + 1 ## from highest to lowest
        else:
            ia_rank_matrix[k] = rankdata(row)

    return ia_rank_matrix


def convert_to_ktop_rank_matrix(ia_perf_matrix, higher_better, num_top_ranks):
    '''
    '''
    ntop_ia_rank_matrix = np.empty(ia_perf_matrix.shape, dtype=float)
    for k, row in enumerate(ia_perf_matrix):
        sorted_indices_arr = np.argsort(row)
        if higher_better:
            sorted_indices_arr = sorted_indices_arr[::-1]
            ntop_ia_rank_matrix[k] = len(row) - rankdata(row) + 1 ## from highest to lowest
        else:
            ntop_ia_rank_matrix[k] = rankdata(row) 
            
        for ktop_inx in range(num_top_ranks):
            sorted_indices_arr = np.delete(sorted_indices_arr, 0)
            
        ## set all the ranks to the max rank (= #algorithms) except k-top algorithms' ranks
        ntop_ia_rank_matrix[k][sorted_indices_arr] = len(ia_perf_matrix[0])
        
    return ntop_ia_rank_matrix


def convert_to_issolved_matrix(ia_perf_matrix, higher_better, limit):
    '''
    '''
    ia_issolved_matrix = np.empty(ia_perf_matrix.shape, dtype=float)
    for (inst_inx, alg_inx), perf_value in np.ndenumerate(ia_perf_matrix):
        if higher_better:
            if perf_value > limit:
                ia_issolved_matrix[inst_inx][alg_inx] = 1
        else:
            if perf_value < limit:
                ia_issolved_matrix[inst_inx][alg_inx] = 1

    return ia_issolved_matrix

