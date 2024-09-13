'''
'''
import os
import sys
import numpy as np
import random ## seed function is called with this
import pprint
import subprocess
import time

from datetime import datetime

from src.matrix.factorization import ksvd, NNMF, apply_mf
from src.misc import settings
from sklearn import manifold 
from src.importance.latent_importance import LatentImportance
from src.ml.split_data import generate_splits, generate_mc_splits
from src.plot.bar import plot_inst_ft_importance
from src.plot.scatter import plot_inst_alg_scatter, plot_inst_alg_heatmap,\
    plot_inst_alg_scatter_clustered, get_cluster_labels,\
    plot_inst_alg_scatter_clustered_4
from src.matrix.matrix_io import convert_to_rank_matrix
from src.plot.line import plot_oracle_singlebest_line_alors, plot_fold_perf,\
    plot_MC_perf
from src.performance.parX import ParX
from src.performance.num_solved import NumSolved
from src.performance.rank import Rank
from src.performance.evaluate import EvaluationType, EvaluationMetricType, Evaluate
from src.alors.dataset import Dataset
from src.alors.latent_estimate_cs import LatentEstimateColdStart,\
    MappingMethodType
from src.performance.accuracy import Accuracy
from src.alors import dataset_dict
from src.alors.rank_prediction_cs import RankPredictionColdStart
from src.alors.value_prediction_cs import ValuePredictionColdStart,\
    TransformationType
from src.performance.ratio_solved import RatioSolved
from src.alors.icluster_latent_estimate_cs import InstanceClusteringBasedLatentEstimateColdStart
from src.dao.misc_io import load_data, load_matrix, save_matrix,\
    save_matrix_cofirank_format, load_matrix_cofirank_format
from src.performance.regret_ac import RegretForAccuracy
from sklearn.manifold.spectral_embedding_ import spectral_embedding
from sklearn.metrics.pairwise import euclidean_distances
from src.matrix.mds_jsx import mds_jsa
from src.alors.knn_mc import kNNMatrixCompletion




class ALORS(object):
    '''
    '''
    ##def __init__(self, matrix_completion_type = MatrixCompletionType.kNN, cold_start_type = ):
    def __init__(self, exp_setting_dict):
        '''
        Constructor
        '''
#         self.matrix_completion_type = matrix_completion_type
#         self.cold_start_type = cold_start_type
        
        self.cold_start = LatentEstimateColdStart(exp_setting_dict['map_method'], exp_setting_dict['mf_rank'], exp_setting_dict['mf_type'])
        
        self.dataset = Dataset(exp_setting_dict)  ### unsolved instances are removed here !!
        self.mf_type = exp_setting_dict['mf_type']
        self.mf_rank = exp_setting_dict['mf_rank']
        self.eval_metrics = exp_setting_dict['eval_metrics']
        self.map_method_type = exp_setting_dict['map_method']
        self.missing_val = exp_setting_dict['missing_val']
        
        if 'cv_file_path' in exp_setting_dict.keys():
            
            i_fold_array = load_data(exp_setting_dict['cv_file_path'], False, False)[0]
            ## remove unsolved instances
            sorted_unsolved_inst_list = sorted(self.dataset.unsolved_inst_list)
            sorted_unsolved_inst_list = sorted_unsolved_inst_list[::-1] ##from larger to smaller !! 
            for inst_inx_to_rm in sorted_unsolved_inst_list:
                i_fold_array = np.delete(i_fold_array, inst_inx_to_rm)
            
            
            self.num_splits = len(np.unique(i_fold_array))
            self.inst_split_list = []
            for splt_inx in range(0, self.num_splits):
                self.inst_split_list.append([])  
            for inst_inx, fold_inx in np.ndenumerate(i_fold_array):
                inst_inx = inst_inx[0]  
                self.inst_split_list[int(fold_inx)].append(inst_inx)
            
        elif 'num_splits' in exp_setting_dict.keys():
            self.num_splits = exp_setting_dict['num_splits']
            self.inst_split_list = generate_splits(len(self.dataset.ia_perf_matrix), exp_setting_dict['num_splits'])
        else:
            print 'No split info!\nExiting...'
            sys.exit()
        
        
        self.sparsity_level = exp_setting_dict['sparsity_level'] ##TODO: matrix completion part


        
        self.pred_ia_rank_matrix = None
     
    
    def get_train_test_data(self, test_inst_split):   
        '''
        '''
        train_inst_split = range(len(self.dataset.ia_perf_matrix))
        for inst_inx in test_inst_split:
            train_inst_split.remove(inst_inx)
        
        train_ia_perf_matrix = self.dataset.ia_perf_matrix[train_inst_split]
        train_ia_rank_matrix = self.dataset.ia_rank_matrix[train_inst_split]
        train_i_ft_matrix = self.dataset.i_ft_matrix[train_inst_split]
        test_i_ft_matrix = self.dataset.i_ft_matrix[test_inst_split]
    
        return train_ia_perf_matrix, train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix
    
    
    def plot_inst_alg_scatter(self, output_folder, title): 
        '''
        '''
        ##Uk, sk, Vk, s = ksvd(self.dataset.ia_rank_matrix, self.mf_rank)
        Uk, sk, Vk, s = apply_mf(self.dataset.ia_rank_matrix, self.mf_type, self.mf_rank)
        
        mds = manifold.MDS(2, random_state = settings.__seed__)
        ##mds =  manifold.LocallyLinearEmbedding(10, 2, random_state = settings.__seed__)
        i_ft_matrix_2D = mds.fit_transform(self.dataset.i_ft_matrix)
        ##i_ft_matrix_2D = mds_jsa(euclidean_distances(self.dataset.i_ft_matrix))[0]
        Uk_2D = mds.fit_transform(Uk)
        VkT_2D = mds.fit_transform(Vk.T)  
        
        plot_inst_alg_scatter(output_folder, title, i_ft_matrix_2D, Uk_2D, VkT_2D)
        
        cluster_labels_init = get_cluster_labels(self.dataset.i_ft_matrix)
        cluster_labels_latent = get_cluster_labels(Uk)
        
        plot_inst_alg_scatter_clustered(output_folder, title, i_ft_matrix_2D, Uk_2D, cluster_labels_init)  
        plot_inst_alg_scatter_clustered_4(output_folder, title, i_ft_matrix_2D, Uk_2D, cluster_labels_init, cluster_labels_latent)
        
        ## no need, or to be improved ?
        ##plot_inst_alg_heatmap(output_folder, title+"-heatmap", i_ft_matrix_2D, Uk_2D, VkT_2D)
        
 
     
     
    def calculate_inst_ft_importance(self):
        
        ##Uk, sk, Vk, s = ksvd(self.dataset.ia_rank_matrix, self.mf_rank)
        Uk, sk, Vk, s = apply_mf(self.dataset.ia_rank_matrix, self.mf_type, self.mf_rank)
        
        ##print "@calculate_inst_ft_importance| s = ", s
        
        importance = LatentImportance()
        
        print "self.dataset.i_ft_matrix : ", self.dataset.i_ft_matrix.shape
        print "Uk : ", Uk.shape
        
        return importance.get_scores(self.dataset.i_ft_matrix, Uk, rf_n_estimators=10)   
        
    
    def dataset_evaluate_dict(self):
        '''
        '''
        evaluator_dict = {}
        for eval_m in self.eval_metrics:
            
            if eval_m == EvaluationMetricType.NumSolved:
                
                evaluator = NumSolved(self.dataset.ia_perf_matrix, self.dataset.perf_bound_val)
                legend_loc = 1

            elif eval_m == EvaluationMetricType.RatioSolved:
                
                evaluator = RatioSolved(self.dataset.ia_perf_matrix, self.dataset.perf_bound_val)
                legend_loc = 1
                                                        
            elif eval_m == EvaluationMetricType.ParX:
                
                evaluator = ParX(self.dataset.ia_perf_matrix, self.dataset.perf_bound_val, X = 10)
                legend_loc = 4
                
            elif eval_m == EvaluationMetricType.Rank:
                
                evaluator = Rank(self.dataset.ia_rank_matrix)
                legend_loc = 4
                
            elif eval_m == EvaluationMetricType.Accuracy:
                
                evaluator = Accuracy(self.dataset.ia_perf_matrix)
                legend_loc = 1

            elif eval_m == EvaluationMetricType.RegretAcc:
                
                evaluator = RegretForAccuracy(self.dataset.ia_perf_matrix)
                legend_loc = 1
                                
            else:
                return -1
            
            kth_oracle_perc_dict = evaluator.all_kth_oracle()
            kth_singlebest_perc_dict = evaluator.all_kth_single_best()
            evaluator_dict[eval_m] = {'name': evaluator.name, 
                                      'loc': legend_loc,
                                      'oracle': kth_oracle_perc_dict, 
                                      'single_best': kth_singlebest_perc_dict}
            
        return evaluator_dict
            
            
    def evaluate_prediction(self, eval_metrics, ia_perf_matrix, ia_rank_matrix, pred_ia_rank_matrix):
        '''
        '''
        evaluator_dict = {}
        for eval_m in eval_metrics:
            
            if eval_m == EvaluationMetricType.NumSolved:
                
                evaluator = NumSolved(ia_perf_matrix, self.dataset.perf_bound_val, pred_ia_rank_matrix)

            elif eval_m == EvaluationMetricType.RatioSolved:
                
                evaluator = RatioSolved(ia_perf_matrix, self.dataset.perf_bound_val, pred_ia_rank_matrix)
                                        
            elif eval_m == EvaluationMetricType.ParX:
                
                evaluator = ParX(ia_perf_matrix, self.dataset.perf_bound_val, 10, pred_ia_rank_matrix)
                
            elif eval_m == EvaluationMetricType.Rank:
                
                evaluator = Rank(ia_rank_matrix, pred_ia_rank_matrix)
                
            elif eval_m == EvaluationMetricType.Accuracy:
                
                evaluator = Accuracy(ia_perf_matrix, pred_ia_rank_matrix)
                
            elif eval_m == EvaluationMetricType.RegretAcc:
                
                evaluator = RegretForAccuracy(ia_perf_matrix, pred_ia_rank_matrix)
                
            else:
                return -1


            oracle = evaluator.oracle()
            single_best = evaluator.single_best()
            random = evaluator.random()
            evaluator_dict[eval_m] = {'name': evaluator.name, 
                                      'oracle': oracle,
                                      'single_best': single_best,
                                      'random': random}
            
        return evaluator_dict        
        
        
                      
    def analyse_dataset(self):
        '''
        '''
        if not os.path.exists(settings.__output_folder__):
            os.makedirs(settings.__output_folder__)
            
        
        ## save latent feature matrices  
        ##Uk, sk, Vk, s = ksvd(self.dataset.ia_rank_matrix, self.mf_rank)
        Uk, sk, Vk, s = apply_mf(self.dataset.ia_rank_matrix, self.mf_type, self.mf_rank)
        np.savetxt(settings.__output_folder__ + "/" + self.dataset.name+"-Instance-Latent-Features.txt", Uk, fmt='%1.6f', delimiter=',')
        np.savetxt(settings.__output_folder__ + "/" + self.dataset.name+"-Singular-Values.txt", sk, fmt='%1.6f', delimiter=',')
        np.savetxt(settings.__output_folder__ + "/" + self.dataset.name+"-Algorithm-Latent-Features.txt", Vk.T, fmt='%1.6f', delimiter=',')
        
              
        ##NNMF(self.dataset.ia_rank_matrix, k = self.mf_rank)

        ## generate three plots wrt instance fts, inst latent features, alg latent features 
        self.plot_inst_alg_scatter(settings.__output_folder__, self.dataset.name)

        
        
        ## TODO: add other ft importance methods
        ## generate a plot for instance feature importance (if Random Forest is used)
        if self.map_method_type == MappingMethodType.RandomForest or self.map_method_type == MappingMethodType.ExtraTreesRegressor:
            scores, top_ft_inx_arr, ft_importance_arr = self.calculate_inst_ft_importance()
            plot_inst_ft_importance(settings.__output_folder__, self.dataset.name+"-i-ft-importance", ft_importance_arr, top_ft_inx_arr)

        ## generate kth best oracle vs single best plots (for given evaluation metrics)
        evaluator_dict = self.dataset_evaluate_dict()
        plot_oracle_singlebest_line_alors(settings.__output_folder__, self.dataset.name+"-osb", evaluator_dict)
        
#         print "\nOriginal Dataset O-SB Performance"
#         pprint.pprint(evaluator_dict, width = 100)
        
     
     
#     def perform_matrix_completion(self):
#         '''
#         '''
#         per_sparsity_mc_results_dict = {}
#         ##per_sparsity_mc_perf_list = []
#         
#         for mc_inx in range(1, self.num_splits): ## for each sparsity level (from 0.1 to 0.9)
#             sparsity_level = mc_inx * 0.1
#             
#             ##per_sparsity_mc_perf_list.append([])
#             mc_perf_list = []
#              
#             matrix_dim_arr = np.zeros((2), dtype=np.int32)
#             matrix_dim_arr[0] = int(self.dataset.ia_rank_matrix.shape[0])
#             matrix_dim_arr[1] = int(self.dataset.ia_rank_matrix.shape[1])
#             
#             ## 10 groups for 10 random sparsity splits
#             split_row_sparsity_list = generate_mc_splits(matrix_dim_arr, self.num_splits, sparsity_level)
#              
#             for trail_inx in range(self.num_splits): 
#                 ## remove some elements (replace selected values by self.missing_val)
#                 copy_ia_rank_matrix = self.dataset.ia_rank_matrix.copy()
#                 ##for row_inx in range(len(copy_ia_rank_matrix)):
#                 for row_inx in range(len(split_row_sparsity_list[trail_inx])): ##sparsity may not present on some rows, so check split_row_sparsity_list's size !!
# #                     print trail_inx," - ", row_inx, " : "
# #                     if trail_inx == 2 and row_inx == 217:
# #                         split_row_sparsity_list[trail_inx][row_inx]
#                          
#                     copy_ia_rank_matrix[row_inx][split_row_sparsity_list[trail_inx][row_inx]] = self.missing_val
#           
#              
#                 ## complete matrix
#                 knn_mc = kNNMatrixCompletion(copy_ia_rank_matrix, self.missing_val)
# 
#                 
#                 start_ms = datetime.now()
#                 ##start_ms = time.time()
#                 
#                 ##copy_ia_rank_matrix = knn_mc.complete()
#                 copy_ia_rank_matrix = knn_mc.complete_with_k(10)
#                 
#                 end_ms = datetime.now()
#                 ##end_ms = time.time()
# 
#                 elapsed_ms = (end_ms - start_ms).total_seconds()
#                 
#                 print "Elapsed time for matrix completion: ", elapsed_ms
#             
#             
#                 mc_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
#                                                               self.dataset.ia_perf_matrix,  
#                                                               self.dataset.ia_rank_matrix, 
#                                                               copy_ia_rank_matrix) 
#                 
#                 
#                 print "MC sparsity = ", sparsity_level, " - trail_inx = ", trail_inx
#                 ##pprint.pprint(mc_evaluator_dict)  
#                 
#                 ##per_sparsity_mc_perf_list[mc_inx-1].append(mc_evaluator_dict)  ## starts from 1
#                 mc_perf_list.append(mc_evaluator_dict)  ## starts from 1  
#                 
#             
#             mc_eval_avg_std_arr = Evaluate.evaluate_prediction_from_folds(mc_perf_list, EvaluationType.Oracle)
#             print "mc_eval_avg_std_arr:\n", mc_eval_avg_std_arr
#             
#             
#             per_sparsity_mc_results_dict[sparsity_level] = [EvaluationType.Oracle,  mc_perf_list]  
#             
# #             if len(per_sparsity_mc_results_dict) == 2:
# #                 break ## for debugging
#             
#             
#         ## evaluate original dataset's fold performance
#         original_dataset_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
#                                                                   self.dataset.ia_perf_matrix,  
#                                                                   self.dataset.ia_rank_matrix, 
#                                                                   self.dataset.ia_rank_matrix,  
#                                                                   )
#         
#         
#         ## multiply list with self.num_splits in order to get line in the boxplot ??     
#         per_sparsity_mc_results_dict['Oracle'] =  [EvaluationType.Oracle, [original_dataset_evaluator_dict]*(self.num_splits-1)],
#         per_sparsity_mc_results_dict['Single Best'] = [EvaluationType.SingleBest, [original_dataset_evaluator_dict]*(self.num_splits-1)],
#         per_sparsity_mc_results_dict['Random'] = [EvaluationType.Random, [original_dataset_evaluator_dict]*(self.num_splits-1)]
# 
# 
#         pprint.pprint(per_sparsity_mc_results_dict)
# 
#         ## Plot matrix completion results    
#         plot_MC_perf(settings.__output_folder__, self.dataset.name+"-MC-Performance", per_sparsity_mc_results_dict)
        
 
 
 
 
    ##TODO
    def perform_matrix_completion_via_jar(self):
        '''
        '''
        per_sparsity_mc_results_dict = {}
        ##per_sparsity_mc_perf_list = [] 
        
        
        ## ========== for reporting ============== 
        ## for performance
        performance_matrix = np.zeros((len(self.eval_metrics), self.num_splits-1, self.num_splits))
        
        ## for elapsed time
        elapsed_time_list = []
        for inx in range(self.num_splits-1): 
            elapsed_time_list.append([])  
            
        row_titles = [] 
        col_titles = ['Sparsity/Sample#'] + range(1,11)   
        ## =============================================
        
        for mc_inx in range(1, self.num_splits): ## for each sparsity level (from 0.1 to 0.9)
            sparsity_level = mc_inx * 0.1
            
            row_titles.append(sparsity_level)
            
            ##per_sparsity_mc_perf_list.append([])
            mc_perf_list = []
             
            matrix_dim_arr = np.zeros((2), dtype=np.int32)
            matrix_dim_arr[0] = int(self.dataset.ia_rank_matrix.shape[0])
            matrix_dim_arr[1] = int(self.dataset.ia_rank_matrix.shape[1])
            
            ## 10 groups for 10 random sparsity splits
            split_row_sparsity_list = generate_mc_splits(matrix_dim_arr, self.num_splits, sparsity_level)
             
            for trail_inx in range(self.num_splits): 
                ## remove some elements (replace selected values by self.missing_val)
                copy_ia_rank_matrix = self.dataset.ia_rank_matrix.copy()
                ##for row_inx in range(len(copy_ia_rank_matrix)):
                
                missing_element_cnt = 0
                for row_inx in range(len(split_row_sparsity_list[trail_inx])): ##sparsity may not present on some rows, so check split_row_sparsity_list's size !!
#                     print trail_inx," - ", row_inx, " : "
#                     if trail_inx == 2 and row_inx == 217:
#                         split_row_sparsity_list[trail_inx][row_inx]
                         
                    copy_ia_rank_matrix[row_inx][split_row_sparsity_list[trail_inx][row_inx]] = self.missing_val

                    missing_element_cnt += len(split_row_sparsity_list[trail_inx][row_inx])
                  
             
                
                        
                ## ================================================================================
                ### CHECK and CORRECT SPARSITY ####
                ## e.g on CSP data set 2024 X 2, at most %50 sparsity is possible by keeping at least one entry per row 
                expected_num_miss_elements = matrix_dim_arr[0] * matrix_dim_arr[1] * sparsity_level
                num_to_be_removed_els = expected_num_miss_elements - missing_element_cnt 
                
                missing_row_cnt = 0
                if num_to_be_removed_els > 0:
                    row_inx = 0
                    for row in copy_ia_rank_matrix:
                        if bool(random.getrandbits(1)):
                            num_to_be_removed_els -= len(np.where(row == self.missing_val)[0])  
                            
                            ## set all values of the row to zero ??
                            copy_ia_rank_matrix[row_inx].fill(0) 
                            
                            missing_row_cnt += 1
                            
                            if num_to_be_removed_els <= 0:
                                break
                            
                        row_inx += 1
                        
                print "missing_row_cnt : ", missing_row_cnt
                ## ================================================================================
                            
                    
                    
             
             
             
             
             
             
                ## complete matrix
                ##knn_mc = kNNMatrixCompletion(copy_ia_rank_matrix, self.missing_val)
                
                np.savetxt(settings.main_project_folder + '/mc_in_matrix.txt', copy_ia_rank_matrix, delimiter='\t')
                
                ##start_ms = datetime.now()
                start_ms = time.time()
                
                
                
                
                ## Apply CofiRank ####################################
                # save matrix in the cofirank format
                save_matrix_cofirank_format(settings.main_project_folder + '/temp-cf.txt', copy_ia_rank_matrix, -99)
                
                ##copy_ia_rank_matrix = knn_mc.complete()
                cofirank_path = "/home/misir/Desktop/CofiRank/cofirank/dist/" + "/cofirank-deploy" 
                cofirank_config = settings.main_project_folder + "/default.cfg"

                p = subprocess.Popen([cofirank_path, cofirank_config])  
                out = p.communicate()[0]
                
                
                
                
                ##end_ms = datetime.now()
                end_ms = time.time()

                ##elapsed_ms = (end_ms - start_ms).seconds
                elapsed_ms = (end_ms - start_ms)  

                
                print "Elapsed time for matrix completion: ", elapsed_ms  
                
                elapsed_time_list[mc_inx-1].append(elapsed_ms)
                

                ##copy_ia_rank_matrix = load_matrix(settings.main_project_folder + '/src/alors/mc_in_matrix.txt-recommend.txt') ## , delimiter = '\\t'
                
                copy_ia_rank_matrix = load_matrix_cofirank_format(settings.main_project_folder + '/F.lsvm', len(copy_ia_rank_matrix), len(copy_ia_rank_matrix[0]), -99)
                
                
            
                #######################################################
                
                
             
                mc_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                              self.dataset.ia_perf_matrix,  
                                                              self.dataset.ia_rank_matrix, 
                                                              copy_ia_rank_matrix) 
                
                
                print "MC sparsity = ", sparsity_level, " - trail_inx = ", trail_inx
                ##pprint.pprint(mc_evaluator_dict)  
                
                ##per_sparsity_mc_perf_list[mc_inx-1].append(mc_evaluator_dict)  ## starts from 1
                mc_perf_list.append(mc_evaluator_dict)  ## starts from 1  
                
            
            mc_eval_avg_std_arr = Evaluate.evaluate_prediction_from_folds(mc_perf_list, EvaluationType.Oracle)
            print "mc_eval_avg_std_arr:\n", mc_eval_avg_std_arr
            
            ## TODO: this can be changed with evaluate_prediction_from_folds with additional outputs
            mc_eval_arr, mc_eval_metric_list = Evaluate.extract_performance_from_folds(mc_perf_list, EvaluationType.Oracle)
            for eval_mt_inx in range(len(self.eval_metrics)):
                performance_matrix[eval_mt_inx][mc_inx-1] = mc_eval_arr[eval_mt_inx]
             
            per_sparsity_mc_results_dict[sparsity_level] = [EvaluationType.Oracle,  mc_perf_list]  
            
#             if len(per_sparsity_mc_results_dict) == 2:
#                 break ## for debugging 
            
        
        print "elapsed_time_list: ", elapsed_time_list
        
        ## save elapsed time file      
        ##np.savetxt(settings.__output_folder__ + "/" + self.dataset.name+"-MC-ElapsedTime.txt", np.array(elapsed_time_list), fmt='%1.4f', delimiter=',')
        save_matrix(settings.__output_folder__ + "/" + self.dataset.name+"-MC-ElapsedTime.txt", np.array(elapsed_time_list), row_titles, col_titles, delimiter=',')    
        
#         for eval_mt_inx in range(len(self.eval_metrics)):  
#             save_matrix(settings.__output_folder__ + "/" + self.dataset.name+"-MC-"+mc_eval_metric_list[eval_mt_inx]+".txt", 
#                         performance_matrix[eval_mt_inx], row_titles, col_titles, delimiter=',')
        
         
        ## evaluate original dataset's fold performance
        original_dataset_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                                  self.dataset.ia_perf_matrix,  
                                                                  self.dataset.ia_rank_matrix, 
                                                                  self.dataset.ia_rank_matrix,  
                                                                  )
        
        
        ## multiply list with self.num_splits in order to get line (or same number of samples) in the boxplot ??     
        per_sparsity_mc_results_dict['Oracle'] =  [EvaluationType.Oracle, [original_dataset_evaluator_dict]*self.num_splits],
        per_sparsity_mc_results_dict['Single Best'] = [EvaluationType.SingleBest, [original_dataset_evaluator_dict]*self.num_splits],
        per_sparsity_mc_results_dict['Random'] = [EvaluationType.Random, [original_dataset_evaluator_dict]*self.num_splits]


        pprint.pprint(per_sparsity_mc_results_dict)

        ## Plot matrix completion results    
        plot_MC_perf(settings.__output_folder__, self.dataset.name+"-MC-Performance", per_sparsity_mc_results_dict)

        
           
        
    def perform_cold_start(self):
        '''
        '''
        fold_perf_list = []
        
        icluster_lecs_fold_perf_list = []
        
        rankp_fold_perf_list = []
        valuep_fold_perf_list = []
        valuep_log2_fold_perf_list = []
        valuep_logE_fold_perf_list = []
        original_dataset_fold_perf_list = []
        original_TRAIN_dataset_fold_perf_list = []
        


        self.pred_ia_rank_matrix = np.zeros(self.dataset.ia_rank_matrix.shape)
        
#        iclusteR_lecs_pred_ia_rank_matrix = np.zeros(self.dataset.ia_rank_matrix.shape)
        rank_pred_ia_rank_matrix = np.zeros(self.dataset.ia_rank_matrix.shape)
        value_pred_ia_rank_matrix = np.zeros(self.dataset.ia_rank_matrix.shape)
        value_pred_log2_ia_rank_matrix = np.zeros(self.dataset.ia_rank_matrix.shape)
        value_pred_logE_ia_rank_matrix = np.zeros(self.dataset.ia_rank_matrix.shape)
        
        
        rankp_cs = RankPredictionColdStart()
        valuep_cs = ValuePredictionColdStart()
        
        icluster_lecs = InstanceClusteringBasedLatentEstimateColdStart(self.map_method_type, self.mf_rank) 
        
        
        
                
        for test_inst_split in self.inst_split_list:
            
            train_ia_perf_matrix, train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix = self.get_train_test_data(test_inst_split)
            
            ##TODO: apply matrix completion
            
            
            
            ## apply cold start
            ##a = self.cold_start.predict(train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix) ##??
            self.pred_ia_rank_matrix[test_inst_split] = self.cold_start.predict(train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix);
            
#             b = icluster_lecs.predict(train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix);
#             iclusteR_lecs_pred_ia_rank_matrix[test_inst_split] = icluster_lecs.predict(train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix);

            rank_pred_ia_rank_matrix[test_inst_split] = rankp_cs.predict(train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix)
 
            value_pred_ia_perf_matrix = valuep_cs.predict(train_ia_perf_matrix, train_i_ft_matrix, test_i_ft_matrix)
            value_pred_ia_rank_matrix[test_inst_split] = convert_to_rank_matrix(value_pred_ia_perf_matrix, self.dataset.higher_better)
                       
            value_pred_log2_ia_perf_matrix = valuep_cs.predict(train_ia_perf_matrix, train_i_ft_matrix, test_i_ft_matrix, TransformationType.Log2)
            value_pred_log2_ia_rank_matrix[test_inst_split] = convert_to_rank_matrix(value_pred_log2_ia_perf_matrix, self.dataset.higher_better)
                                                        
            value_pred_logE_ia_perf_matrix = valuep_cs.predict(train_ia_perf_matrix, train_i_ft_matrix, test_i_ft_matrix, TransformationType.LogE)
            value_pred_logE_ia_rank_matrix[test_inst_split] = convert_to_rank_matrix(value_pred_logE_ia_perf_matrix, self.dataset.higher_better)


            ## evaluate ALORS' prediction performance 
            evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                      self.dataset.ia_perf_matrix[test_inst_split],  
                                                      self.dataset.ia_rank_matrix[test_inst_split], 
                                                      self.pred_ia_rank_matrix[test_inst_split]) 
            
#              ## evaluate Clustering based ALORS' prediction performance 
#             icluster_lecs_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
#                                                       self.dataset.ia_perf_matrix[test_inst_split],  
#                                                       self.dataset.ia_rank_matrix[test_inst_split], 
#                                                       iclusteR_lecs_pred_ia_rank_matrix[test_inst_split])      
                        
            
            ## evaluate RANK PREDICTION's prediction performance 
            rankp_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                              self.dataset.ia_perf_matrix[test_inst_split],  
                                                              self.dataset.ia_rank_matrix[test_inst_split], 
                                                              rank_pred_ia_rank_matrix[test_inst_split]) 
 
            ## evaluate VALUE PREDICTIONS's prediction performance 
            valuep_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                              self.dataset.ia_perf_matrix[test_inst_split],  
                                                              self.dataset.ia_rank_matrix[test_inst_split], 
                                                              value_pred_ia_rank_matrix[test_inst_split]) 
            
            ## evaluate VALUE LOG2 PREDICTIONS's prediction performance 
            valuep_log2_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                              self.dataset.ia_perf_matrix[test_inst_split],  
                                                              self.dataset.ia_rank_matrix[test_inst_split], 
                                                              value_pred_log2_ia_rank_matrix[test_inst_split]) 
 
            ## evaluate VALUE LOGE PREDICTIONS's prediction performance 
            valuep_logE_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                              self.dataset.ia_perf_matrix[test_inst_split],  
                                                              self.dataset.ia_rank_matrix[test_inst_split], 
                                                              value_pred_logE_ia_rank_matrix[test_inst_split])
                       
            ## evaluate original dataset's fold performance
            original_dataset_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                                      self.dataset.ia_perf_matrix[test_inst_split],  
                                                                      self.dataset.ia_rank_matrix[test_inst_split], 
                                                                      self.dataset.ia_rank_matrix[test_inst_split]
                                                                      )
            
            
            
            ## add ALORS' fold performance results
            fold_perf_list.append(evaluator_dict)
            
#             ## add Clustering based ALORS' fold performance results
#             icluster_lecs_fold_perf_list.append(icluster_lecs_evaluator_dict)
            
            ## add RANK PREDICTION's fold performance results
            rankp_fold_perf_list.append(rankp_evaluator_dict)

            ## add VALUE PREDICTION's fold performance results
            valuep_fold_perf_list.append(valuep_evaluator_dict)
            
            ## add VALUE LOG2 PREDICTION's fold performance results
            valuep_log2_fold_perf_list.append(valuep_log2_evaluator_dict)
            
            ## add VALUE LOGE PREDICTION's fold performance results
            valuep_logE_fold_perf_list.append(valuep_logE_evaluator_dict)
                                    
            ## add original dataset's fold performance results
            original_dataset_fold_perf_list.append(original_dataset_evaluator_dict)   
            
             
        ##pprint.pprint(fold_perf_list, width=100)
        
        ## calculate avg performance + std
        fold_eval_avg_std_arr = Evaluate.evaluate_prediction_from_folds(fold_perf_list, EvaluationType.Oracle)
        print "ALORS_fold_eval_avg_std_arr:\n", fold_eval_avg_std_arr
 
#         icluster_lecs_fold_eval_avg_std_arr = Evaluate.evaluate_prediction_from_folds(icluster_lecs_fold_perf_list, EvaluationType.Oracle)
#         print "icluster_lecs_fold_eval_avg_std_arr:\n", icluster_lecs_fold_eval_avg_std_arr
               
        rankp_fold_eval_avg_std_arr = Evaluate.evaluate_prediction_from_folds(rankp_fold_perf_list, EvaluationType.Oracle)
        print "rankp_fold_eval_avg_std_arr:\n", rankp_fold_eval_avg_std_arr

        valuep_fold_eval_avg_std_arr = Evaluate.evaluate_prediction_from_folds(valuep_fold_perf_list, EvaluationType.Oracle)
        print "valuep_fold_eval_avg_std_arr:\n", valuep_fold_eval_avg_std_arr
        
        valuep_log2_fold_eval_avg_std_arr = Evaluate.evaluate_prediction_from_folds(valuep_log2_fold_perf_list, EvaluationType.Oracle)
        print "valuep_log2_fold_eval_avg_std_arr:\n", valuep_log2_fold_eval_avg_std_arr
        
        valuep_logE_fold_eval_avg_std_arr = Evaluate.evaluate_prediction_from_folds(valuep_logE_fold_perf_list, EvaluationType.Oracle)
        print "valuep_logE_fold_eval_avg_std_arr:\n", valuep_logE_fold_eval_avg_std_arr
        
        original_dataset_oracle_fold_eval_avg_std_arr = Evaluate.evaluate_prediction_from_folds(original_dataset_fold_perf_list, EvaluationType.Oracle)
        print "original_dataset_oracle_fold_eval_avg_std_arr:\n", original_dataset_oracle_fold_eval_avg_std_arr
        
        original_dataset_sb_fold_eval_avg_std_arr = Evaluate.evaluate_prediction_from_folds(original_dataset_fold_perf_list, EvaluationType.SingleBest)
        print "original_dataset_singlebest_fold_eval_avg_std_arr:\n", original_dataset_sb_fold_eval_avg_std_arr

        
        
        fold_results_dict = {     
                             'ALORS': [EvaluationType.Oracle, fold_perf_list], 
                             ##'IClsALORS': [EvaluationType.Oracle, icluster_lecs_fold_perf_list],
                             'Rank Prediction': [EvaluationType.Oracle, rankp_fold_perf_list],
                             'Value Prediction': [EvaluationType.Oracle, valuep_fold_perf_list],     
                             'Value Log2 Prediction': [EvaluationType.Oracle, valuep_log2_fold_perf_list],       
                             'Oracle': [EvaluationType.Oracle, original_dataset_fold_perf_list],
                             'Single Best': [EvaluationType.SingleBest, original_dataset_fold_perf_list],
                             'Random': [EvaluationType.Random, original_dataset_fold_perf_list]
                             }  
        
        
        plot_fold_perf(settings.__output_folder__, self.dataset.name+"-Fold-Performance", fold_results_dict)
  
  
        print "\nOverall Prediction Matrix Evaluation\n----------------------------------------"
        evaluator_dict = self.evaluate_prediction(self.eval_metrics, 
                                                  self.dataset.ia_perf_matrix, 
                                                  self.dataset.ia_rank_matrix, 
                                                  self.pred_ia_rank_matrix)    
        
        pprint.pprint(evaluator_dict, width=100)
        
        
        print "\nOverall Original Matrix Evaluation\n----------------------------------------"
        original_datasaet_evaluator_dict = self.evaluate_prediction(self.eval_metrics, 
                                                  self.dataset.ia_perf_matrix, 
                                                  self.dataset.ia_rank_matrix, 
                                                  self.dataset.ia_rank_matrix)    
        
        pprint.pprint(original_datasaet_evaluator_dict, width=100)
        
 
 
#      ##TODO 
#     def perform_mc_cs(self):
#         '''
#             Perform matrix completion first, then cold start
#         '''     
# 
#         for test_inst_split in self.inst_split_list: ## for each fold
#     
#             train_ia_perf_matrix, train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix = self.get_train_test_data(test_inst_split)
#             
#             for mc_inx in range(1, self.num_splits): ## for each sparsity level (from 0.1 to 0.9)
#                 sparsity_level = mc_inx * 0.1
#                 
#                 ##per_sparsity_mc_perf_list.append([])
#                 mc_perf_list = []
#                  
#                 matrix_dim_arr = np.zeros((2), dtype=np.int32)
#                 matrix_dim_arr[0] = int(train_ia_rank_matrix.shape[0])
#                 matrix_dim_arr[1] = int(train_ia_rank_matrix.shape[1])
#                 
#                 ## 10 groups for 10 random sparsity splits
#                 split_row_sparsity_list = generate_mc_splits(matrix_dim_arr, self.num_splits, sparsity_level)
#                  
#                 for trail_inx in range(self.num_splits): 
#                     ## remove some elements (replace selected values by self.missing_val)
#                     copy_ia_rank_matrix = train_ia_rank_matrix.copy()
#                     ##for row_inx in range(len(copy_ia_rank_matrix)):
#                     for row_inx in range(len(split_row_sparsity_list[trail_inx])): ##sparsity may not present on some rows, so check split_row_sparsity_list's size !!
#                              
#                         copy_ia_rank_matrix[row_inx][split_row_sparsity_list[trail_inx][row_inx]] = self.missing_val
# 
#                  
#                         ##########################
#                         ## apply matrix completion
#                         ##########################
#         
#                         knn_mc = kNNMatrixCompletion(copy_ia_rank_matrix, self.missing_val)
#                         copy_ia_rank_matrix = knn_mc.complete()
#                 
#                 
#                 
#                         ##########################
#                         ## apply cold start
#                         ##########################
#                         a = self.cold_start.predict(copy_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix)
#                         self.pred_ia_rank_matrix[test_inst_split] = self.cold_start.predict(train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix);
#                         
#                         
#                         ##TODO
            
            
              
    ##TODO 
    def perform_mc_cs_with_jar(self):
        '''
        '''

        
        self.pred_ia_rank_matrix = np.zeros(self.dataset.ia_rank_matrix.shape)

        
        ## alors performance with cs (or + mc with zero sparsity)
        alors_fold_perf_list = []
        
        ## alors performance with mc + cs
        alors_fold_perf_list_per_mc = []
        for inx in range(self.num_splits):
            alors_fold_perf_list_per_mc.append([])
            
        ## original dataset performance (e.g. oracle + single best + random etc.)
        original_dataset_fold_perf_list = []

        
        per_sparsity_mc_results_dict = {}
        ##per_sparsity_mc_perf_list = [] 
        
        
        ## ========== for reporting ============== 
        ## for performance
        ## eval metrics -> matrix completion sparsity -> cold start -> matrix completion sparsity trail
        performance_matrix = np.zeros( (len(self.eval_metrics), self.num_splits-1, len(self.inst_split_list), self.num_splits) )
        
        
        ## for elapsed time
        elapsed_time_list = []
        for cs_inx in range(len(self.inst_split_list)): ## for each cold start part
            elapsed_time_list.append([])
            ##for inx in range(self.num_splits-1): ## for each matrix completion part
            for inx in range(self.num_splits): ## for each matrix completion part including 0.0 sparsity
                elapsed_time_list[cs_inx].append([])
                  
            
        row_titles = [] 
        col_titles = ['Sparsity/Sample#'] + range(1,11)   
        ## =============================================
        
        
        cs_inx = 0           
        for test_inst_split in self.inst_split_list: ## For each COLD START split
            
            train_ia_perf_matrix, train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix = self.get_train_test_data(test_inst_split)
            
            
            original_dataset_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                          self.dataset.ia_perf_matrix[test_inst_split],  
                                                          self.dataset.ia_rank_matrix[test_inst_split], 
                                                          self.dataset.ia_rank_matrix[test_inst_split]
                                                          )
            
            
            start_ocs = time.time()                                     
            
            alors_pred_test_ia_rank_matrix = self.cold_start.predict(train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix);

            end_ocs = time.time()
            
            elapsed_ocs = end_ocs - start_ocs
            
            elapsed_time_list[cs_inx][0].append(elapsed_ocs)
            
            row_titles.append(0.0)


            
            ## evaluate ALORS' prediction performance 
            alors_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                              self.dataset.ia_perf_matrix[test_inst_split],  
                                                              self.dataset.ia_rank_matrix[test_inst_split], 
                                                              alors_pred_test_ia_rank_matrix) 

            

            
            
            for mc_inx in range(1, self.num_splits): ## for each sparsity level (from 0.1 to 0.9)
                sparsity_level = mc_inx * 0.1   
                
                row_titles.append(sparsity_level)
                
                ##per_sparsity_mc_perf_list.append([])
                mc_cs_perf_list = []
                 
                matrix_dim_arr = np.zeros((2), dtype=np.int32)
                matrix_dim_arr[0] = int(train_ia_perf_matrix.shape[0])
                matrix_dim_arr[1] = int(train_ia_perf_matrix.shape[1])
                
                ## 10 groups for 10 random sparsity splits
                split_row_sparsity_list = generate_mc_splits(matrix_dim_arr, self.num_splits, sparsity_level)
                 
                for trail_inx in range(self.num_splits): 
                    
                    ## remove some elements (replace selected values by self.missing_val)
                    copy_train_ia_rank_matrix = train_ia_rank_matrix.copy()
                    
                    copy_train_i_ft_matrix = train_i_ft_matrix.copy()
                    
                    
                    
                    missing_element_cnt = 0
                    for row_inx in range(len(split_row_sparsity_list[trail_inx])): ##sparsity may not present on some rows, so check split_row_sparsity_list's size !!
                             
                        copy_train_ia_rank_matrix[row_inx][split_row_sparsity_list[trail_inx][row_inx]] = self.missing_val
                        
                        missing_element_cnt += len(split_row_sparsity_list[trail_inx][row_inx])
                    
                    
                    ## ================================================================================
                    ### CHECK and CORRECT SPARSITY ####
                    ## e.g on CSP data set 2024 X 2, at most %50 sparsity is possible by keeping at least one entry per row 
                    expected_num_miss_elements = matrix_dim_arr[0] * matrix_dim_arr[1] * sparsity_level
                    num_to_be_removed_els = expected_num_miss_elements - missing_element_cnt 
                    
                    
                    rows_tobe_kept = range(0, len(copy_train_ia_rank_matrix))                   
                    if num_to_be_removed_els > 0:
                        row_inx = 0
                        for row in copy_train_ia_rank_matrix:
                            if bool(random.getrandbits(1)):
                                num_to_be_removed_els -= len(np.where(row == self.missing_val)[0])  
                                rows_tobe_kept.remove(row_inx)
                                
                                if num_to_be_removed_els <= 0:
                                    break
                                
                            row_inx += 1
                            
                            
                    ## update train data TODO
                    copy_train_ia_rank_matrix = copy_train_ia_rank_matrix[rows_tobe_kept]
                    copy_train_i_ft_matrix = copy_train_i_ft_matrix[rows_tobe_kept]
                        
                    print "rows_tobe_kept : ", len(rows_tobe_kept), " out of ", len(train_ia_rank_matrix)
                    ## ================================================================================
                    
                      
                      
                      
                      
                 
                    ## complete matrix
                    ##knn_mc = kNNMatrixCompletion(copy_ia_rank_matrix, self.missing_val)
                    
                    np.savetxt(settings.main_project_folder + '/mc_in_matrix.txt', copy_train_ia_rank_matrix, delimiter='\t')
                    
                    ##start_ms = datetime.now()
                    start_ms = time.time()
                    
                    
                    ##copy_train_ia_rank_matrix = knn_mc.complete()
                    subprocess.call(['java', '-jar', 
                                     settings.main_project_folder + '/ARS.jar', 'M', 
                                     settings.main_project_folder + '/mc_in_matrix.txt'])  
                    
                    ##end_ms = datetime.now()
                    end_ms = time.time()
    
    
                    ##elapsed_ms = (end_ms - start_ms).seconds  
                    elapsed_ms = (end_ms - start_ms)
                    
                    print "Elapsed time for matrix completion: ", elapsed_ms  
                    
                       
                    copy_train_ia_rank_matrix = load_matrix(settings.main_project_folder + '/src/alors/mc_in_matrix.txt-recommend.txt') ## , delimiter = '\\t'
                    
                    
                    
                    
                    ##start_ms = datetime.now()
                    start_ms = time.time()
                    
                    ####### APPLY COLD START ##################  
                    ##self.pred_ia_rank_matrix[test_inst_split] = self.cold_start.predict(copy_train_ia_rank_matrix, train_i_ft_matrix, test_i_ft_matrix);
                    self.pred_ia_rank_matrix[test_inst_split] = self.cold_start.predict(copy_train_ia_rank_matrix, copy_train_i_ft_matrix, test_i_ft_matrix);
                    ###########################################
                    
                    ##end_ms = datetime.now()
                    end_ms = time.time()
                    
                    
                    ##elapsed_ms += (end_ms - start_ms).seconds ## add time elapsed for cold start
                    elapsed_ms += (end_ms - start_ms)  
                    
                    elapsed_time_list[cs_inx][mc_inx].append(elapsed_ms)
                    
                    
                    
                 
                    mc_cs_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                                  self.dataset.ia_perf_matrix[test_inst_split],  
                                                                  self.dataset.ia_rank_matrix[test_inst_split], 
                                                                  self.pred_ia_rank_matrix[test_inst_split])    
                    
                    
                    print "CS fold #", cs_inx," MC sparsity = ", sparsity_level, " - trail_inx = ", trail_inx
                    ##pprint.pprint(mc_evaluator_dict)  
                    
                    ##per_sparsity_mc_perf_list[mc_inx-1].append(mc_evaluator_dict)  ## starts from 1
                    mc_cs_perf_list.append(mc_cs_evaluator_dict)  ## starts from 1  
                    
                    
                    ## to maintain per sparsity results
                    ##alors_fold_perf_list_per_mc[mc_inx-1].append(mc_cs_evaluator_dict)
                    
                    ##print "alors_fold_perf_list_per_mc - mc_inx: ", mc_inx, " - ", len(alors_fold_perf_list_per_mc[mc_inx-1])
                
                
                combined_dict = Evaluate.combine_dict_in_list(mc_cs_perf_list)
                alors_fold_perf_list_per_mc[mc_inx-1].append(combined_dict)
                
                
                
                mc_eval_avg_std_arr = Evaluate.evaluate_prediction_from_folds(mc_cs_perf_list, EvaluationType.Oracle)
                print "mc_eval_avg_std_arr:\n", mc_eval_avg_std_arr
                
                ## TODO: this can be changed with evaluate_prediction_from_folds with additional outputs
                mc_eval_arr, mc_eval_metric_list = Evaluate.extract_performance_from_folds(mc_cs_perf_list, EvaluationType.Oracle)
                for eval_mt_inx in range(len(self.eval_metrics)):
                    performance_matrix[eval_mt_inx][mc_inx-1][cs_inx] = mc_eval_arr[eval_mt_inx]  

                 
                per_sparsity_mc_results_dict[sparsity_level] = [EvaluationType.Oracle,  mc_cs_perf_list]  
                
    #             if len(per_sparsity_mc_results_dict) == 2:
    #                 break ## for debugging 
            
            
            
        
            print "elapsed_time_list: ", elapsed_time_list
            
            ## save elapsed time file - TODO: open later      
            save_matrix(settings.__output_folder__ + "/" + self.dataset.name+"-CS-"+str(cs_inx+1)+"-MC-ElapsedTime.txt", np.array(elapsed_time_list[cs_inx]), row_titles, col_titles, delimiter=',')    

            
            ##TODO - to output results or write all together to a single file??
#             for eval_mt_inx in range(len(self.eval_metrics)):  
#                 save_matrix(settings.__output_folder__ + "/" + self.dataset.name+"-CS-"+str(cs_inx+1)+"-MC-"+mc_eval_metric_list[eval_mt_inx]+".txt", 
#                             performance_matrix[eval_mt_inx][cs_inx], row_titles, col_titles, delimiter=',')
            
            
            
            
                
            ## evaluate original dataset's fold performance
            original_dataset_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                                      self.dataset.ia_perf_matrix[test_inst_split],  
                                                                      self.dataset.ia_rank_matrix[test_inst_split], 
                                                                      self.dataset.ia_rank_matrix[test_inst_split]
                                                                      )

            ## add 0.0 sparsity alors results   
            alors_fold_perf_list.append(alors_evaluator_dict)
                         
            ## add original dataset's fold performance results  
            original_dataset_fold_perf_list.append(original_dataset_evaluator_dict)
#             for inx in range(self.num_splits-1): ## ad same result for each sparsity ??
#                 original_dataset_fold_perf_list.append(original_dataset_evaluator_dict)  
#                 alors_fold_perf_list.append(alors_evaluator_dict)
                 
                 
                 
                 
            cs_inx += 1
            
                
        print "performance_matrix = ", performance_matrix

        
             
        ## add alors evaluations on mc+cs
        fold_results_dict = {     
                         '0.0': [EvaluationType.Oracle, alors_fold_perf_list], 
                         'Oracle': [EvaluationType.Oracle, original_dataset_fold_perf_list],
                         'Single Best': [EvaluationType.SingleBest, original_dataset_fold_perf_list],
                         'Random': [EvaluationType.Random, original_dataset_fold_perf_list]
                         }  
         
        for inx in range(1, self.num_splits):    
            ###if len(alors_fold_perf_list_per_mc[inx]) > 0: ## just for test
            fold_results_dict[str(inx * 0.1)] = [EvaluationType.Oracle, alors_fold_perf_list_per_mc[inx-1]]     
#             else:
#                 print "\ninx = ", inx, " ==> Missing results !!!\n" 
         
        print "\n============= RES ==================\n"
           
        pprint.pprint(fold_results_dict)
           
        print "\n============= RES ==================\n"
         
        plot_fold_perf(settings.__output_folder__, self.dataset.name+"-MC-CS-Fold-Performance", fold_results_dict)
            
            
        
    def run(self):
        '''
        '''
        ## analyse original dataset (plots + reports)
##        self.analyse_dataset()  

       
        
        ### ONLY MATRIX COMPLETION ##########################
        ##self.perform_matrix_completion()    
        self.perform_matrix_completion_via_jar()
        
        
        
        ### MATRIX COMPLETION + COLD START ######################## sparsity_level should be > 0 !
##        self.perform_mc_cs_with_jar()
        
        

        ### ONLY COLD START ###################
##        self.pred_ia_rank_matrix = self.perform_cold_start()
                               
        
        
            
        return self.pred_ia_rank_matrix
    
        
        
if __name__ == "__main__":   
    
    random.seed(settings.__seed__)        
    
    num_test_cases = len(dataset_dict.exp_setting_dict)
    for test_case_inx in range(num_test_cases):
        alors = ALORS(dataset_dict.exp_setting_dict[test_case_inx])    
        alors.run()
