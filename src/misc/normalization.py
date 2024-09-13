'''

'''
import numpy as np

def normalize_minmax(np_matrix, min_arr = None, max_arr = None):
    '''
        Normalize a given matrix with
        (np_matrix[i][j] - min_arr[j]) / (max_arr[j] - min_arr[j])

        :param np_matrix: numpy 2D matrix - input matrix to be normalized
        :param min_arr: numpy float array - per column minimum values used for normalization
        :param max_arr: numpy float array - per column maximum values used for normalization
        :return: numpy 2D matrix - normalized matrix (without changing the original matrix)
    '''
    num_rows = len(np_matrix)
    num_cols = len(np_matrix[0])

    pre_norm_info = False
    if min_arr is not None and max_arr is not None:
        pre_norm_info = True
    else:
        min_arr = np.zeros(shape=(num_cols))
        max_arr = np.zeros(shape=(num_cols))

    norm_matrix = np.zeros(shape=(num_rows, num_cols))

    for j in range(num_cols):

        if not pre_norm_info:
            min_arr[j] = np.min(np_matrix.T[j])
            max_arr[j] = np.max(np_matrix.T[j])

        for i in range(num_rows):

            if max_arr[j] == min_arr[j]:
                norm_matrix[i][j] = 0
            else:
                norm_matrix[i][j] = (np_matrix[i][j] - min_arr[j]) / (max_arr[j] - min_arr[j])

    return norm_matrix, min_arr, max_arr
