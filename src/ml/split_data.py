import sys
import numpy as np

from random import randint
import pprint
from decimal import Decimal


def generate_splits(size, num_splits):
    '''
        Determine split indices
        
        @param size: upper bound to calculate splits
        @param num_splits: number of splits to return
    '''
    member_arr = range(size)
    
    single_split_size = size / num_splits
    
    split_list = []
    for split_inx in range(num_splits):
        single_split_list = []

        while member_arr is not None:
            inx_to_add = randint(0, len(member_arr)-1)
            single_split_list.append(member_arr[inx_to_add])
            del member_arr[inx_to_add]
    
            if member_arr is None:
                split_list.append(single_split_list)
                return split_list
            elif len(single_split_list) is single_split_size:
                
                if len(member_arr) < single_split_size:
                    for val in member_arr:
                        single_split_list.append(val)
                    del member_arr[:]
                
                split_list.append(single_split_list)
                break
    
    return split_list


def generate_mc_splits(matrix_dim_arr, num_splits, sparsity_level):
    '''
        Generate splits for matrix completion
        
        Returns a list of list of list
        indices to remove!!!   
    '''
    if sparsity_level <= 0:
        print 'sparsity_level should be > 0 !\nExiting...'
        sys.exit()
    elif sparsity_level >= 1:
        print 'sparsity_level should be < 1 !\nExiting...'
        sys.exit()
        
    split_row_sparsity_list = []
    
    per_row_sparsity =  (matrix_dim_arr[0] * matrix_dim_arr[1]) * sparsity_level  /  float(matrix_dim_arr[0])
  
    if per_row_sparsity == matrix_dim_arr[1]: ## size same as number of columns, keep at least one entry   
        per_row_sparsity -= 1

    per_row_sparsity_bounds = []
    if Decimal(per_row_sparsity) % 1 != 0:
        per_row_sparsity_bounds.append(int(np.floor(per_row_sparsity)))
        per_row_sparsity_bounds.append(int(np.ceil(per_row_sparsity)))
      
        if per_row_sparsity_bounds[0] >= matrix_dim_arr[1]:
            per_row_sparsity_bounds = []
            per_row_sparsity = matrix_dim_arr[1]-1
            
        elif per_row_sparsity_bounds[1] >= matrix_dim_arr[1]:
            per_row_sparsity_bounds = []
            per_row_sparsity = matrix_dim_arr[1]-1
            
    
    total_sparsity = int(np.ceil( (matrix_dim_arr[0] * matrix_dim_arr[1]) * sparsity_level ))
                         
    for split_inx in range(num_splits):
        
        item_num_tobe_removed = total_sparsity
        
        per_row_split_list = []
        for row_inx in range(matrix_dim_arr[0]):
            
            if len(per_row_sparsity_bounds) == 0:
                per_row_sparsity_indiv = per_row_sparsity
            else:  
                ## whether it is possible to meet the sparsity level, if not (or equal), then remove always from upper value
                if (matrix_dim_arr[0]-row_inx) * per_row_sparsity_bounds[1] <= item_num_tobe_removed:
                    per_row_sparsity_indiv = per_row_sparsity_bounds[1] 
                else:
                    per_row_sparsity_indiv = per_row_sparsity_bounds[randint(0, 1)]
            
            single_split_list = []
            while len(single_split_list) < per_row_sparsity_indiv:
                col_inx_to_add = randint(0, matrix_dim_arr[1]-1)    
                if col_inx_to_add not in single_split_list:
                    single_split_list.append(col_inx_to_add)
             
                    item_num_tobe_removed -= 1
                    if item_num_tobe_removed <= 0: ## to break inner loop
                        break  
                    
            per_row_split_list.append(single_split_list)
            
            if item_num_tobe_removed <= 0: ## to break outer loop
                break  

        ## print "for split_inx = ", split_inx, " | item_num_tobe_removed = |", item_num_tobe_removed, " from ", total_sparsity

        split_row_sparsity_list.append(per_row_split_list)
        
    return split_row_sparsity_list   


    