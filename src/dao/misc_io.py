import csv
import numpy as np
from src.misc import settings
from __builtin__ import str



def load_data(data_file, ignore_first_row = False, ignore_first_col = False, delimiter = ','):
    '''
        Read a given file and load it to a matrix
        
        :param data_file: 
    '''
    line_list = []  
    with open(data_file, 'rb') as f:
        reader = csv.reader(f)
         
        if ignore_first_row:
            next(reader, None)  # skip the header
         
        for row in reader:   
            if row: # skip if row (list) is empty
                if ignore_first_col:
                    line_list.append(row[1:len(row)])
                else:
                    line_list.append(row[0:len(row)])
                #print row
        
#       self.ia_perf_matrix = np.loadtxt(csv_file, delimiter=',', skiprows=1, usecols=range(1,...))
    
    
    ##print line_list
    data_matrix = np.array(line_list).astype(float)
    ##num_algs, num_insts = data_matrix.shape

    return data_matrix


def load_cs_time_data(data_file, ignore_first_row = False, ignore_first_col = False, delimiter = ','):
    '''
        Read a given file and load it to a matrix
        
        :param data_file: 
    '''
    line_list = []  
    with open(data_file, 'rb') as f:
        reader = csv.reader(f)
         
        if ignore_first_row:
            next(reader, None)  # skip the header
         
        for row in reader:   
            if row: # skip if row (list) is empty
                if ignore_first_col:
                    line_list.append(row[1:len(row)])
                else:
                    line_list.append(row[0:len(row)])
                #print row
        
#       self.ia_perf_matrix = np.loadtxt(csv_file, delimiter=',', skiprows=1, usecols=range(1,...))
    
    diff_line = None
    if len(line_list[0]) != len(line_list[1]):
        diff_line = line_list[0]
        line_list = line_list[1::] 
        
    
    ##print line_list
    data_matrix = np.array(line_list).astype(float)
    ##num_algs, num_insts = data_matrix.shape

    if diff_line != None:
        return data_matrix, diff_line
    else:
        return data_matrix
    


def load_data_with_titles(data_file, ignore_first_row = False, ignore_first_col = False, delimiter = ','):
    '''
        Read a given file and load it to a matrix
        
        :param data_file: 
    '''
    row_title_list = []
    col_title_list = []
    
    line_list = []  
    with open(data_file, 'rb') as f:
        reader = csv.reader(f)
         
        if ignore_first_row:
            next(reader, None)  # skip the header
            ##TODO: add col titles - col_title_list
         
        for row in reader:   
            if row: # skip if row (list) is empty
                if ignore_first_col:
                    row_title_list.append(row[0])
                    line_list.append(row[1:len(row)])
                else:
                    line_list.append(row[0:len(row)])
                #print row
        
#       self.ia_perf_matrix = np.loadtxt(csv_file, delimiter=',', skiprows=1, usecols=range(1,...))
    data_matrix = np.array(line_list).astype(float)
    ##num_algs, num_insts = data_matrix.shape
    
    return data_matrix, row_title_list, col_title_list


def load_netflix_format(data_file, num_users, num_items, ignore_first_row = False, missing_val = -99, delimiter = ','):
    '''
        Read a given netflix/imdb formatted file and load it to a matrix
        
        for algorithm selection user = instance, item = algorithm
        
        :param data_file: 
    '''
    data_matrix = np.zeros((num_users, num_items))  
    data_matrix[data_matrix == 0] = missing_val   
    
    with open(data_file, 'rb') as f:
        reader = csv.reader(f)
         
        if ignore_first_row:
            next(reader, None)  # skip the header
            ##TODO: add col titles - col_title_list
         
        for row in reader:   
            if row: 
                data_matrix[int(row[0])][int(row[1])] = float(row[2]) 
        
    return data_matrix


def load_matrix(data_file):
    '''
    '''
    f = open ( data_file , 'r')
    l = [ map(np.float32,line.split()) for line in f if line.strip() != "" ]
    
    return np.array(l)


def save_matrix(file_name, matrix, row_titles, col_titles, delimiter):
    '''
    '''                      
    with file(file_name, 'w') as f:
        
        if col_titles:
            col_titles_str = delimiter.join(map(str, col_titles))   
            f.write(col_titles_str+"\n")
            
        row_inx = 0
        for row in matrix:
            ##print "row: ", row
            if row_titles: 
                f.write(str(row_titles[row_inx])+delimiter)
            ##row_vals_str = delimiter.join(map(str, row.tolist()))
            row_vals_str = delimiter.join(map(str, row))   
            f.write(row_vals_str+"\n")
            row_inx += 1
          

def save_matrix_cofirank_format(file_name, ia_rank_matrix, missing_val):
    '''
        Write a given matrix in the cofirank format to a file
    '''
    with file(file_name, 'w') as f:
        
        first_added = False
        curr_inst_inx = 0
        for (inst_inx, alg_inx), rank in np.ndenumerate(ia_rank_matrix):
            
            if inst_inx == curr_inst_inx+1:
                curr_inst_inx = inst_inx
                first_added = False
                f.write("\n")
                
                
            if rank != missing_val:
                if first_added:
                    f.write(" ")

                f.write(str(alg_inx+1)+":"+str(rank)) ## algorithms start from 1??
                first_added = True
                
                
            
def load_matrix_cofirank_format(file_name, num_insts, num_algs, missing_val):
    '''
        Load given cofirank like formatted data as a matrix
        TODO: convert for not asking number of instances and algorithms 
    '''
    ia_rank_matrix = np.zeros((num_insts, num_algs))
    ia_rank_matrix.fill(missing_val)
    
    with open(file_name, 'rb') as f:

        inst_inx = 0
        
        reader = csv.reader(f, delimiter=" ")
        for row in reader:   
            if row:
                for entry in row:
                    entry_list = entry.split(":")
                    if len(entry_list) == 2:
                        ##print entry_list
                        ia_rank_matrix[inst_inx][int(entry_list[0]) - 1] = float(entry_list[1]) ## algorithms start from 1
                
            inst_inx += 1
    
    return ia_rank_matrix
  