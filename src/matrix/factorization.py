import numpy as np
from src.misc.outlier_detector import percentile_based_outlier
from sklearn import decomposition
from src.misc import settings


class MatrixFactorizationType:
    svd, NNMF = range(0,2)
    
 
def apply_mf(ia_matrix, mf_type, k):    

    if mf_type == MatrixFactorizationType.svd:
        return ksvd(ia_matrix, k)
    elif mf_type == MatrixFactorizationType.NNMF:
        return NNMF(ia_matrix, k)
    else:
        return 


def ksvd(ia_matrix, k):
    '''
        Apply k-singular value decomposition (SVD)
        - Return matrices with k dimensions

        :param ia_matrix: numpy 2D array - (instance, algorithm) matrix
        :param k: int - SVD dimension
        :return: numpy 2D array, numpy array, numpy 2D array - Uk matrix representing rows,
                                                                 sk matrix (array) for singular values,
                                                                 Vk matrix representing columns
    '''
    max_k = min(len(ia_matrix), len(ia_matrix[0]))
    if k > max_k:
        k = max_k

    U, s, V = np.linalg.svd(ia_matrix, full_matrices=False)
    Uk = U[:,0:k]
    sk = s[0:k] #only diagonal values
    Vk = V[0:k,:]

    return Uk, sk, Vk, s


def NNMF(ia_matrix, k, max_iter=200):    
    model = decomposition.NMF(init="nndsvd", n_components=k, max_iter=max_iter, random_state=settings.__seed__)
    W = model.fit_transform(ia_matrix)
    H = model.components_    
    
    ## generate identity matrix to use the same convention with SVD
    s = np.zeros((k, k))
    s = np.diag(s)  
    s.fill(1)
    
    return W, s, H, s

