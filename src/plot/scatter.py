'''

'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib import rcParams

from src.misc import settings
from matplotlib.figure import figaspect
from matplotlib.pyplot import figure
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from src.ml.clustering.k_means import kmeans_adaptive,\
    ClusteringEvaluationMetrics
import itertools
 
 
def plot_alg_scatter_clustered(output_folder, title, a_latent_matrix_2D, cluster_labels): 
    plt.figure(1)   
    h, w = figaspect(1.)                       
    figure(figsize=(w,h))             
    
    rcParams.update({'font.size': 8})

    color_list = ['b','g','r','c','m','y','k']
    marker_list = ['o', '+', 's', 'x', 'v','^', '>', '<', 'p', '*', 'h', 'H', 'd']
    
    
    markers_ppoint = []
    for cluster_inx in cluster_labels:
        markers_ppoint.append( marker_list[int(cluster_inx) % len(marker_list)] )
        
        
    markers = ''
    for cluster_inx in cluster_labels:
        markers += marker_list[int(cluster_inx)]
    markers = itertools.cycle(markers)
    
    for inx in range(0, len(a_latent_matrix_2D)):
        point_marker = marker_list[int(cluster_labels[inx]) % len(marker_list)] 
        point_color = color_list[int(cluster_labels[inx]) % len(color_list)]
        plt.scatter(a_latent_matrix_2D[inx, 0], a_latent_matrix_2D[inx, 1], marker = point_marker, c = point_color, s=70, alpha=0.6)
    
    
    color = [str((item*100)/255.) for item in cluster_labels]
    
    ## print cluster_labels
    ## print color 
    
    plt.title('Algorithms-Latent')
    plt.axis('equal')
    

    plt.interactive(False)

    plt.tight_layout() ### http://matplotlib.org/users/tight_layout_guide.html
    
    pdf_to_save_path = os.path.join(output_folder, title+'-clustered.pdf')
    plt.savefig(pdf_to_save_path, format='pdf', dpi=settings.__fig_dpi__, bbox_inches='tight')
    ##plt.show() 
    
    plt.gcf().clear() 
        
    
def plot_inst_alg_scatter_clustered_4(output_folder, title, i_ft_matrix, i_latent_matrix, cluster_labels_init, cluster_labels_latent): 
    plt.figure(1)   
    ##plt.suptitle(title) ##, fontsize=18
    h, w = figaspect(1.)                       
    figure(figsize=(w,h))             
    
    rcParams.update({'font.size': 8})
    
    ## http://stackoverflow.com/questions/14324270/matplotlib-custom-marker-symbol
    markers = ['o', 's', '+','8', '>','<','p', '^','v']
    
    marker_list_init = []  
    for cluster_label in cluster_labels_init:
        marker_inx = int((len(markers)-1) % (cluster_label+1))
        marker_list_init.append(markers[marker_inx]) 
        
    marker_list_latent = []
    for cluster_label in cluster_labels_init:
        marker_inx = int((len(markers)-1) % (cluster_label+1))
        marker_list_latent.append(markers[marker_inx]) 
        
    plt.subplot(221)     
    plt.scatter(i_ft_matrix[:, 0], i_ft_matrix[:, 1], c = cluster_labels_init, s=22, alpha=0.6)
    plt.title('Instances-Initial')
    plt.axis('equal')
    
    plt.subplot(222)
    plt.scatter(i_latent_matrix[:, 0], i_latent_matrix[:, 1],  c = cluster_labels_init, s=22, alpha=0.6) ##marker = marker_list_init,
    plt.title('Instances-Latent')
    plt.axis('equal') 
    
    plt.subplot(223)   
    plt.scatter(i_ft_matrix[:, 0], i_ft_matrix[:, 1],  c = cluster_labels_latent, s=22, alpha=0.6) ##marker = marker_list_latent,
    ##plt.title('Instances-Initial')
    plt.axis('equal')
    
    plt.subplot(224)
    plt.scatter(i_latent_matrix[:, 0], i_latent_matrix[:, 1],  c = cluster_labels_latent, s=22, alpha=0.6)
    ##plt.title('Instances-Latent')
    plt.axis('equal')  

    plt.interactive(False)

    plt.tight_layout() ### http://matplotlib.org/users/tight_layout_guide.html
    
    pdf_to_save_path = os.path.join(output_folder, title+'-clustered-4.pdf')
    plt.savefig(pdf_to_save_path, format='pdf', dpi=settings.__fig_dpi__, bbox_inches='tight')
    ##plt.show() 
    
    plt.gcf().clear() 


def plot_inst_alg_scatter_clustered_4_markers(output_folder, title, i_ft_matrix, i_latent_matrix, cluster_labels_init, cluster_labels_latent): 
    plt.figure(1)   
    ##plt.suptitle(title) ##, fontsize=18
    h, w = figaspect(1.)                       
    figure(figsize=(w,h))             
    
    rcParams.update({'font.size': 8})
    
    color_list = ['b','g','r','c','m','y','k']
    marker_list = ['o', '+', 's', 'x', 'v','^', '>', '<', 'p', '*', 'h', 'H', 'd']
    
        
    plt.subplot(221)     
    for inx in range(0, len(i_ft_matrix)):
        point_marker = marker_list[int(cluster_labels_init[inx]) % len(marker_list)] 
        point_color = color_list[int(cluster_labels_init[inx]) % len(color_list)]
        plt.scatter(i_ft_matrix[inx, 0], i_ft_matrix[inx, 1], marker = point_marker, c = point_color,  s=22, alpha=0.6)
    plt.title('Instances-Initial')
    plt.axis('equal')
    
    plt.subplot(222)
    for inx in range(0, len(i_latent_matrix)):
        point_marker = marker_list[int(cluster_labels_init[inx]) % len(marker_list)] 
        point_color = color_list[int(cluster_labels_init[inx]) % len(color_list)]
        plt.scatter(i_latent_matrix[inx, 0], i_latent_matrix[inx, 1], marker = point_marker, c = point_color,  s=22, alpha=0.6)    
    plt.title('Instances-Latent')
    plt.axis('equal') 
    
    plt.subplot(223)   
    for inx in range(0, len(i_ft_matrix)):
        point_marker = marker_list[int(cluster_labels_latent[inx]) % len(marker_list)] 
        point_color = color_list[int(cluster_labels_latent[inx]) % len(color_list)]
        plt.scatter(i_ft_matrix[inx, 0], i_ft_matrix[inx, 1], marker = point_marker, c = point_color,  s=22, alpha=0.6)        
    ##plt.title('Instances-Initial')
    plt.axis('equal')
    
    plt.subplot(224)
    for inx in range(0, len(i_latent_matrix)):
        point_marker = marker_list[int(cluster_labels_latent[inx]) % len(marker_list)] 
        point_color = color_list[int(cluster_labels_latent[inx]) % len(color_list)]
        plt.scatter(i_latent_matrix[inx, 0], i_latent_matrix[inx, 1], marker = point_marker, c = point_color,  s=22, alpha=0.6)        
    ##plt.title('Instances-Latent')
    plt.axis('equal')  

    plt.interactive(False)

    plt.tight_layout() ### http://matplotlib.org/users/tight_layout_guide.html
    
    pdf_to_save_path = os.path.join(output_folder, title+'-clustered-4.pdf')
    plt.savefig(pdf_to_save_path, format='pdf', dpi=settings.__fig_dpi__, bbox_inches='tight')
    ##plt.show() 
    
    plt.gcf().clear() 
    
        

def get_cluster_labels(i_ft_matrix):
    '''
        @param i_ft_matrix: is firstly used as latent matrix = U matrix
    '''
    cluster_inst_belongto_list = [] ## list of lists - instances of each cluster
    
    cluster_inst_belongto_list, centroids, labels = kmeans_adaptive(i_ft_matrix, k_max = 10, init_k_max = 5, clst_eval_metric=ClusteringEvaluationMetrics.SLH)

    return labels


    