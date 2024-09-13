'''

'''
import sys
import numpy as np

from sklearn.cluster.k_means_ import KMeans
from src.misc import settings
from sklearn import metrics
from scipy.stats import rankdata


class ClusteringEvaluationMetrics:
    SLH, RankBasedPS = range(0, 2)
    
    
def kmeans_adaptive(row_matrix, k_max, init_k_max = 10, clst_eval_metric = ClusteringEvaluationMetrics.SLH, sensitivity = 0.01):
    '''
        kmeans clustering with automatically determining right k (number of clusters)

        :param row_matrix: numpy 2D matrix - matrix to be clustered (rows are data points)
        :param k_max: int - maximum k
        :param init_k_max: int - upper bound on k to test kmeans initially
        :param clst_eval_metric: ClusteringEvaluationMetrics - metric used to evaluate clustering results
        :return: clst, - centroids, labels, number of clusters
    '''
    best_score = -1
    best_clst = None
    best_k = -1

    if k_max < 2 or k_max > len(row_matrix):
        k_max = len(row_matrix)

    k_left = 2
    k_right = k_max-1

    score_left = -1
    score_right = -1

    if init_k_max < 2 or init_k_max > len(row_matrix):
        init_k_max = len(row_matrix)
        
    if init_k_max == 2:
        clst = None
        centroids = row_matrix
        labels = range(0, 2)
        num_clusters = 2
        return clst, centroids, labels, num_clusters
    

    # initially try first init_k_max values to speed up
    for k_val in xrange(2, init_k_max+1):
        clst = KMeans(n_clusters=k_val, random_state=settings.__seed__).fit(row_matrix)
        score = eval_clustering(row_matrix, clst, clst_eval_metric)

        if score > best_score:
            best_score = score
            best_clst = clst
            best_k = k_val
            
            ## not to cluster further if clustering is almost perfect
            if best_score >= 1.0 - sensitivity:
                init_k_max = k_max ## just to skip the further clustering 
                break


    if init_k_max < k_max: # adaptive

        score_left = best_score
        clst_left = best_clst
        k_left = init_k_max

        clst_right = KMeans(n_clusters=k_right, random_state=settings.__seed__).fit(row_matrix)
        score_right = eval_clustering(row_matrix, clst_right, clst_eval_metric)
        if score_right > best_score:
            best_score = score_right
            best_clst = clst_right
            best_k = k_right

        while True:
            
            ## not to cluster further if clustering is almost perfect
            if best_score >= 1.0 - sensitivity:
                break


            diff = (k_right - k_left) / 2
            if score_left > score_right:
                k_right = k_left + diff

                clst_right = KMeans(n_clusters=k_right, random_state=settings.__seed__).fit(row_matrix)
                score_right = eval_clustering(row_matrix, clst_right, clst_eval_metric)
                if score_right > best_score:
                    best_score = score_right
                    best_clst = clst_right
                    best_k = k_right
            else:
                k_left = k_left + diff

                clst_left = KMeans(n_clusters=k_left, random_state=settings.__seed__).fit(row_matrix)
                score_left = eval_clustering(row_matrix, clst_left, clst_eval_metric)
                if score_left > best_score:
                    best_score = score_left
                    best_clst = clst_left
                    best_k = k_left
                    
                    
            if k_left >= k_right-1: ## no more to check
                break

            # added to complete clustering earlier (can be removed??)
            if best_score > 0.5 and abs(score_left - score_right) <= sensitivity:
                break
            

    if best_k == len(row_matrix)-1: ## TODO: for now
        clst = None
        centroids = row_matrix
        labels = range(0, len(row_matrix))
        num_clusters = len(centroids)
    else:
        clst = best_clst
        centroids = best_clst.cluster_centers_
        labels = rankdata(best_clst.labels_, method='dense')-1  ## cluster indexes start from zero
        num_clusters = len(np.unique(labels))


    list_of_clst_insts_lists = get_clst_info(clst.cluster_centers_, clst.labels_, len(np.unique(labels)))    

    return list_of_clst_insts_lists, centroids, labels



def eval_clustering(row_data_matrix, clst, type, row_data_rank_matrix = None):
    '''
        Evaluate a clustering result

        :param row_data_matrix: numpy 2D matrix - data used for clustering
        :param clst: clustering model
        :param type: ClusteringEvaluationMetrics - clustering evaluation metric
        :param row_data_rank_matrix: numpy 2D matrix - rank matrix derived from row_data_matrix
        :return: float - clustering score
    '''
    score = 0

    ## to keep the cluster labels / numbers consecutive (starting from zero)
    labels = rankdata(clst.labels_, method='dense')-1

    if type == ClusteringEvaluationMetrics.SLH:
        score = metrics.silhouette_score(row_data_matrix, labels, metric='euclidean', random_state=settings.__seed__)
    elif type == ClusteringEvaluationMetrics.RankBasedPS:
        # num_clusters = len(np.unique(labels))

        if row_data_rank_matrix is None:
            row_data_rank_matrix = rankdata(row_data_matrix, method='dense')
        else:
            if np.array_equal(row_data_matrix, row_data_rank_matrix):
                print("row_data_matrix ", row_data_matrix.shape,  "and row_data_rank_matrix are in different shape", row_data_rank_matrix.shape, "\nExiting...")
                sys.exit()

        list_of_clst_insts_lists = get_clst_info(clst.cluster_centers_, clst.labels_, len(np.unique(labels)))
        avg_clst_rank_matrix = calc_avg_clst_rank_matrix(row_data_rank_matrix, list_of_clst_insts_lists)
        # print("avg_clst_rank_matrix: ", avg_clst_rank_matrix)
        score = calc_clst_rank_score(avg_clst_rank_matrix)

    return score




def get_clst_info(centroids, labels, num_clusters, labels_consecutive = False):
    '''
        Generate a list of lists regarding given clustering results,
        where each list refers to a cluster and includes the indices
        of the member data points

        :param centroids:
        :param labels:
        :param num_clusters: int - number of clusters
        :param labels_consecutive: boolean - whether cluster labels are consecutively numbered
        :return: list - cluster membership list
    '''
    clst_labels = None
    if labels_consecutive:
        clst_labels = rankdata(labels, method='dense')-1  ## cluster indices should start from zero
    else:
        clst_labels = labels

    clst_labels = clst_labels.astype(np.int)

    list_of_list = []
    for inx in range(num_clusters):
        list_of_list.append([])

    data_point_inx = 0
    for label in clst_labels:
        list_of_list[label].append(data_point_inx)
        data_point_inx += 1

    return list_of_list



def calc_avg_clst_rank_matrix(ia_rank_matrix, list_of_datapoint_lists):
    '''
        Calculate the average rank of each algorithm for each cluster, composed of instances

        :param ia_rank_matrix: numpy 2D matrix - (instance, algorithm) rank matrix
        :param list_of_datapoint_lists: list - cluster membership list including data points
        :return: numpy 2D matrix - average rank of each algorithm in each cluster
    '''
    avg_clst_rank_matrix = np.zeros(shape=(len(list_of_datapoint_lists), len(ia_rank_matrix.T)))
    clst_inx = 0
    for data_point_list in list_of_datapoint_lists:
        for data_point_inx in data_point_list:
            avg_clst_rank_matrix[clst_inx] = np.add(avg_clst_rank_matrix[clst_inx], ia_rank_matrix[data_point_inx])

        avg_clst_rank_matrix[clst_inx] /= len(data_point_list)

        clst_inx += 1

    return avg_clst_rank_matrix



def calc_clst_rank_score(avg_clst_rank_matrix):
    '''
        Calculate the rank score which indicates how the clusters differ
        w.r.t. algorithms' ranks on each cluster of instances
        (this metric is applicable only for instances)

        :param avg_clst_rank_matrix: numpy 2D matrix - average rank of each algorithm in each cluster
        :return: float - clustering rank score
    '''
    score = 0
    cnt = 0
    num_clusters = len(avg_clst_rank_matrix)
    for clst_1_inx in range(num_clusters):
        for clst_2_inx in xrange(clst_1_inx+1, num_clusters):
            score += np.sum(np.absolute(np.subtract(avg_clst_rank_matrix[clst_1_inx], avg_clst_rank_matrix[clst_2_inx])))
            cnt += 1
    score /= float(cnt)

    return score

