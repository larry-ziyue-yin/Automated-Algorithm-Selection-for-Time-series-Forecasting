'''

'''
import numpy as np

def percentile_based_outlier(data, threshold=75): ##95
    '''
        Determine the top outliers in a given array

        :param data: numpy array -
        :param threshold:
        :return:
    '''
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    # is_outlier_bool_arr = (data < minval) | (data > maxval)
    is_outlier_bool_arr = (data > maxval)

    num_outliers = np.sum(is_outlier_bool_arr)

    inx = 0
    outlier_inx_arr = np.zeros(shape=(num_outliers), dtype=np.int)
    outlier_arr = np.zeros(shape=(num_outliers))
    for i in range(len(is_outlier_bool_arr)):
        if is_outlier_bool_arr[i] == True:
            outlier_arr[inx] = data[i]
            outlier_inx_arr[inx] = i
            inx += 1

    return outlier_arr, outlier_inx_arr


def percentile_based_outlier_from_min(data, threshold=95):
    '''
        Determine the top outliers in a given array

        :param data: numpy array -
        :param threshold:
        :return:
    '''
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    is_outlier_bool_arr = (data < minval)
    is_outlier_bool_arr = [not i for i in is_outlier_bool_arr]

    num_outliers = np.sum(is_outlier_bool_arr)

    inx = 0
    outlier_inx_arr = np.zeros(shape=(num_outliers), dtype=np.int)
    outlier_arr = np.zeros(shape=(num_outliers))
    for i in range(len(is_outlier_bool_arr)):
        if is_outlier_bool_arr[i] == True:
            outlier_arr[inx] = data[i]
            outlier_inx_arr[inx] = i
            inx += 1

    return outlier_arr, outlier_inx_arr
