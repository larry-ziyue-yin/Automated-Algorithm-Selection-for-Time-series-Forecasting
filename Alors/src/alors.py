'''
'''
import os
import sys
import numpy as np
import random
import pprint
import subprocess
import time
import warnings

from datetime import datetime

from src.matrix.factorization import ksvd, NNMF, apply_mf,\
    MatrixFactorizationType
from src.misc import settings
from sklearn import manifold 
from src.ml.split_data import generate_splits, generate_mc_splits
from src.plot.scatter import get_cluster_labels,\
    plot_inst_alg_scatter_clustered_4, plot_alg_scatter_clustered,\
    plot_inst_alg_scatter_clustered_4_markers
from src.matrix.matrix_io import convert_to_rank_matrix
from src.plot.line_box import plot_fold_perf,\
    plot_MC_perf, plot_fold_perf_for_matrix_rank_k
from src.performance.parX import ParX
from src.performance.num_solved import NumSolved
from src.performance.rank import Rank
from src.performance.evaluate import EvaluationType, EvaluationMetricType, Evaluate
from src.main.dataset import Dataset
from src.main.latent_estimate_cs import LatentEstimateColdStart,\
    MappingMethodType
from src.performance.accuracy import Accuracy
from src.performance.ratio_solved import RatioSolved
from src.dao.misc_io import load_data, load_matrix, save_matrix
from src.performance.regret_ac import RegretForAccuracy
from sklearn.manifold.spectral_embedding_ import spectral_embedding
from sklearn.metrics.pairwise import euclidean_distances


warnings.filterwarnings("ignore") ## supress all the warnings


class ALORS(object):
    '''
    '''
    def __init__(self, exp_setting_dict):
        '''
            Constructor
        '''
        self.cold_start = LatentEstimateColdStart(exp_setting_dict['map_method'], exp_setting_dict['mf_rank'], exp_setting_dict['mf_type'])
        
        self.dataset = Dataset(exp_setting_dict)  
        self.mf_type = exp_setting_dict['mf_type']
        self.mf_rank = exp_setting_dict['mf_rank']
        self.eval_metrics = exp_setting_dict['eval_metrics']
        self.map_method_type = exp_setting_dict['map_method']
        self.missing_val = -99
        
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
        Uk, sk, Vk, s = apply_mf(self.dataset.ia_rank_matrix, self.mf_type, self.mf_rank)
        
        mds = manifold.MDS(2, random_state = settings.__seed__)
        i_ft_matrix_2D = mds.fit_transform(self.dataset.i_ft_matrix)
        Uk_2D = mds.fit_transform(Uk)
        VkT_2D = mds.fit_transform(Vk.T)  
        
        ## plot instance clusters ------------------------------------------
        cluster_labels_init = get_cluster_labels(self.dataset.i_ft_matrix)
        cluster_labels_latent = get_cluster_labels(Uk)
        
        plot_inst_alg_scatter_clustered_4_markers(output_folder, title, i_ft_matrix_2D, Uk_2D, cluster_labels_init, cluster_labels_latent)
        
        ## plot algorithm clusters
        cluster_labels_alg_latent = get_cluster_labels(Vk.T)
        plot_alg_scatter_clustered(output_folder, title+"-Algorithms-Latent", VkT_2D, cluster_labels_alg_latent)

    
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
            General dataset analysis
        '''
        if not os.path.exists(settings.__output_folder__):
            os.makedirs(settings.__output_folder__)
            
        ## save latent feature matrices  
        Uk, sk, Vk, s = apply_mf(self.dataset.ia_rank_matrix, self.mf_type, self.mf_rank)
        np.savetxt(settings.__output_folder__ + "/" + self.dataset.name+"-Instance-Latent-Features.txt", Uk, fmt='%1.6f', delimiter=',')
        np.savetxt(settings.__output_folder__ + "/" + self.dataset.name+"-Singular-Values.txt", sk, fmt='%1.6f', delimiter=',')
        np.savetxt(settings.__output_folder__ + "/" + self.dataset.name+"-Algorithm-Latent-Features.txt", Vk.T, fmt='%1.6f', delimiter=',')
        
        ## generate three plots wrt instance fts, inst latent features, alg latent features 
        self.plot_inst_alg_scatter(settings.__output_folder__, self.dataset.name)


    def perform_matrix_completion_via_jar(self):
        '''
            Matrix completion is performed by calling ARS.jar
        '''
        per_sparsity_mc_results_dict = {}
        
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
            
            mc_perf_list = []
             
            matrix_dim_arr = np.zeros((2), dtype=np.int32)
            matrix_dim_arr[0] = int(self.dataset.ia_rank_matrix.shape[0])
            matrix_dim_arr[1] = int(self.dataset.ia_rank_matrix.shape[1])
            
            ## 10 groups for 10 random sparsity splits
            split_row_sparsity_list = generate_mc_splits(matrix_dim_arr, self.num_splits, sparsity_level)
             
            for trail_inx in range(self.num_splits): 
                ## remove some elements (replace selected values by self.missing_val)
                copy_ia_rank_matrix = self.dataset.ia_rank_matrix.copy()
                
                missing_element_cnt = 0
                for row_inx in range(len(split_row_sparsity_list[trail_inx])): ##sparsity may not present on some rows, so check split_row_sparsity_list's size !!
                         
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
                        
                ## print "missing_row_cnt : ", missing_row_cnt
                ## ================================================================================
                            
                    
                    
                ## complete matrix
                ##knn_mc = kNNMatrixCompletion(copy_ia_rank_matrix, self.missing_val)
                
                np.savetxt(settings.main_project_folder + '/mc_in_matrix.txt', copy_ia_rank_matrix, delimiter='\t')
                
                ##start_ms = datetime.now()
                start_ms = time.time()
                
                ##copy_ia_rank_matrix = knn_mc.complete()
                subprocess.call(['java', '-jar', 
                                 settings.main_project_folder + '/ARS.jar', 'M', 
                                 settings.main_project_folder + '/mc_in_matrix.txt'])  
                
                ##end_ms = datetime.now()
                end_ms = time.time()

                ##elapsed_ms = (end_ms - start_ms).seconds
                elapsed_ms = (end_ms - start_ms)
                
                ## print "Elapsed time for matrix completion: ", elapsed_ms  
                
                elapsed_time_list[mc_inx-1].append(elapsed_ms)
                
                   
                copy_ia_rank_matrix = load_matrix(settings.main_project_folder + '/src/alors/mc_in_matrix.txt-recommend.txt') ## , delimiter = '\\t'
                
            
             
                mc_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                              self.dataset.ia_perf_matrix,  
                                                              self.dataset.ia_rank_matrix, 
                                                              copy_ia_rank_matrix) 
                
                
                ## print "MC sparsity = ", sparsity_level, " - trail_inx = ", trail_inx
                ##pprint.pprint(mc_evaluator_dict)  
                
                mc_perf_list.append(mc_evaluator_dict)  ## starts from 1  
                
            
            mc_eval_avg_std_arr = Evaluate.evaluate_prediction_from_folds(mc_perf_list, EvaluationType.Oracle)
            ## print "mc_eval_avg_std_arr:\n", mc_eval_avg_std_arr
            
            mc_eval_arr, mc_eval_metric_list = Evaluate.extract_performance_from_folds(mc_perf_list, EvaluationType.Oracle)
            for eval_mt_inx in range(len(self.eval_metrics)):
                performance_matrix[eval_mt_inx][mc_inx-1] = mc_eval_arr[eval_mt_inx]
             
            per_sparsity_mc_results_dict[sparsity_level] = [EvaluationType.Oracle,  mc_perf_list]  
            
            
        
        ## print "elapsed_time_list: ", elapsed_time_list
        
        ## save elapsed time file      
        save_matrix(settings.__output_folder__ + "/" + self.dataset.name+"-MC-ElapsedTime.txt", np.array(elapsed_time_list), row_titles, col_titles, delimiter=',')    

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


    def perform_mc_cs_with_jar(self):
        '''
        '''
        self.pred_ia_rank_matrix = np.zeros(self.dataset.ia_rank_matrix.shape)
        
        ## alors performance with cs (or + mc with zero sparsity)
        alors_fold_perf_list = []
        
        ## alors performance with mc + cs
        alors_fold_perf_list_per_mc = []
        for inx in range(10): ## for each sparsity level (0.1 -> 0.9)  
            alors_fold_perf_list_per_mc.append([])
            
        ## original dataset performance (e.g. oracle + single best + random etc.)
        original_dataset_fold_perf_list = []

        
        per_sparsity_mc_results_dict = {}
        ##per_sparsity_mc_perf_list = [] 
        
        
        ## ========== for reporting ============== 
        ## for performance
        ## eval metrics -> matrix completion sparsity -> cold start -> matrix completion sparsity trail
        performance_matrix = np.zeros( (len(self.eval_metrics), 9, len(self.inst_split_list), 10) )
        
        
        ## for elapsed time
        elapsed_time_list = []
        for cs_inx in range(len(self.inst_split_list)): ## for each cold start part
            elapsed_time_list.append([])
            ##for inx in range(self.num_splits-1): ## for each matrix completion part
            for inx in range(10): ## for each matrix completion part including 0.0 sparsity (to evaluate only cold start's spent time)
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

            
            
            for mc_inx in range(1, 10): ## for each sparsity level (from 0.1 to 0.9)
                sparsity_level = mc_inx * 0.1   
                
                row_titles.append(sparsity_level)
                
                ##per_sparsity_mc_perf_list.append([])
                mc_cs_perf_list = []
                 
                matrix_dim_arr = np.zeros((2), dtype=np.int32)
                matrix_dim_arr[0] = int(train_ia_perf_matrix.shape[0])
                matrix_dim_arr[1] = int(train_ia_perf_matrix.shape[1])
                
                ## 10 groups for 10 random sparsity splits
                split_row_sparsity_list = generate_mc_splits(matrix_dim_arr, 10, sparsity_level)
                 
                for trail_inx in range(10): 
                    
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
                        
                    ## print "rows_tobe_kept : ", len(rows_tobe_kept), " out of ", len(train_ia_rank_matrix)
                    ## ================================================================================
                    
                 
                    ## complete matrix #########################
                    
                    np.savetxt(settings.main_project_folder + '/mc_in_matrix.txt', copy_train_ia_rank_matrix, delimiter='\t')
                    
                    ##start_ms = datetime.now()
                    start_ms = time.time()
                    
                    ##apply matrix completion from ARS.jar
                    subprocess.call(['java', '-jar', 
                                     settings.main_project_folder + '/ARS.jar', 'M', 
                                     settings.main_project_folder + '/mc_in_matrix.txt'])  
                    
                    ##end_ms = datetime.now()
                    end_ms = time.time()
    
                    ##elapsed_ms = (end_ms - start_ms).seconds
                    elapsed_ms = (end_ms - start_ms)  
                    
                    ## print "Elapsed time for matrix completion: ", elapsed_ms  
                    
                       
                    ##copy_train_ia_rank_matrix = load_matrix(settings.main_project_folder + '/src/main/mc_in_matrix.txt-recommend.txt') ## , delimiter = '\\t'
                    copy_train_ia_rank_matrix = load_matrix(settings.main_project_folder + '/src/mc_in_matrix.txt-recommend.txt') ## , delimiter = '\\t'
                    
                    
                    
                    
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
                    
                    ##elapsed_time_list[cs_inx][mc_inx-1].append(elapsed_ms)
                    elapsed_time_list[cs_inx][mc_inx].append(elapsed_ms)
                    
                    
                    
                 
                    mc_cs_evaluator_dict = self.evaluate_prediction(self.eval_metrics,
                                                                  self.dataset.ia_perf_matrix[test_inst_split],  
                                                                  self.dataset.ia_rank_matrix[test_inst_split], 
                                                                  self.pred_ia_rank_matrix[test_inst_split])    
                    
                    
                    print ">> CS fold #", cs_inx,"MC sparsity", sparsity_level, "trail", trail_inx, "is completed !"
                    ##pprint.pprint(mc_evaluator_dict)  
                    
                    ##per_sparsity_mc_perf_list[mc_inx-1].append(mc_evaluator_dict)  ## starts from 1
                    mc_cs_perf_list.append(mc_cs_evaluator_dict)  ## starts from 1  
                    
                    
                    ## to maintain per sparsity results
                    ##alors_fold_perf_list_per_mc[mc_inx-1].append(mc_cs_evaluator_dict)
                    
                    ##print "alors_fold_perf_list_per_mc - mc_inx: ", mc_inx, " - ", len(alors_fold_perf_list_per_mc[mc_inx-1])
                
                
                combined_dict = Evaluate.combine_dict_in_list(mc_cs_perf_list)
                alors_fold_perf_list_per_mc[mc_inx-1].append(combined_dict)
                
                
                
                mc_eval_avg_std_arr = Evaluate.evaluate_prediction_from_folds(mc_cs_perf_list, EvaluationType.Oracle)
                ## print "mc_eval_avg_std_arr:\n", mc_eval_avg_std_arr
                
                ## TODO: this can be changed with evaluate_prediction_from_folds with additional outputs
                mc_eval_arr, mc_eval_metric_list = Evaluate.extract_performance_from_folds(mc_cs_perf_list, EvaluationType.Oracle)
                for eval_mt_inx in range(len(self.eval_metrics)):
                    performance_matrix[eval_mt_inx][mc_inx-1][cs_inx] = mc_eval_arr[eval_mt_inx]  

                 
                per_sparsity_mc_results_dict[sparsity_level] = [EvaluationType.Oracle,  mc_cs_perf_list]  
                
        
            ## print "elapsed_time_list: ", elapsed_time_list
            
            ## save elapsed time file
            save_matrix(settings.__output_folder__ + "/" + self.dataset.name+"-CS-"+str(cs_inx+1)+"-MC-ElapsedTime.txt", np.array(elapsed_time_list[cs_inx]), row_titles, col_titles, delimiter=',')    
            
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
                 
            cs_inx += 1
            
             
        ## add alors evaluations on mc+cs
        fold_results_dict = {     
                         '0.0': [EvaluationType.Oracle, alors_fold_perf_list], 
                         'Oracle': [EvaluationType.Oracle, original_dataset_fold_perf_list],
                         'Single Best': [EvaluationType.SingleBest, original_dataset_fold_perf_list],
                         'Random': [EvaluationType.Random, original_dataset_fold_perf_list]
                         }  
         
        for inx in range(1, 10): ## for each sparsity level (0.1 -> 0.9)      
            fold_results_dict[str(inx * 0.1)] = [EvaluationType.Oracle, alors_fold_perf_list_per_mc[inx-1]]     
         
        plot_fold_perf(settings.__output_folder__, self.dataset.name+"-MC-CS-Fold-Performance", fold_results_dict)
            
            
        
    def run(self):
        '''
            Main function to run ALORS
        '''
        print "ALORS is running ..."
        
        
        ## analyse original dataset (plots + reports)
        print ">> Performing general data analysis "
        self.analyse_dataset()
        print ">> General data analysis is completed !"
       
        
        ### ONLY MATRIX COMPLETION ##########################
        ##self.perform_matrix_completion()    
##        self.perform_matrix_completion_via_jar()
##        print ">> Matrix compleletion predictions are done "        
        
        
        ### MATRIX COMPLETION + COLD START ######################## sparsity_level should be > 0 !
        print ">> Working on cold start predictions "
        self.perform_mc_cs_with_jar()
        print ">> Cold start predictions are done "
        
        print "Check", settings.__output_folder__, "for all the output"
            
        return self.pred_ia_rank_matrix
    
        
        
if __name__ == "__main__":   
    
    random.seed(settings.__seed__)        
    
    ## main_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "as_datasets/")
    main_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "as_datasets/")
    
    exp_setting_dict = {'ai_file_path': main_folder_path + "/SAT11_HAND-ai-perf.csv", ## Algorithm-Instance performance data file
                        'ft_file_path': main_folder_path + "/SAT11_HAND-features.txt", ## Instance feature file
                        'cv_file_path': main_folder_path + "/SAT11_HAND-cv.txt",  ## Cross-validation data: if this is not given, folds will be automatically generated
                        'perf_bound_val': 5000, ## Performance value bound (e.g. runtime limit for SAT... ignore if you don't have such a limit) 
                        'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank], ## performance metrics to evaluate / report
                        'mf_type': MatrixFactorizationType.svd, 
                        'mf_rank': 10, 
                        'higher_better': False,
                        'map_method': MappingMethodType.RandomForest,      
                        'num_splits': 10, ## doesn't affect if a cv file is given   
                         }
    

    alors = ALORS(exp_setting_dict)    
    alors.run()
        
