'''
'''
import numpy as np
import pprint

from random import randint

class EvaluationType:
    ## TODO: Random, kPSingleBest, kPRandom
    Oracle, SingleBest, Random, kPSingleBest, kPRandom = range(0, 5)
    
    @staticmethod
    def get_eval_type_name(eval_type):
        if eval_type == EvaluationType.Oracle:
            return "oracle"
        elif eval_type == EvaluationType.SingleBest:
            return "single_best"
        elif eval_type == EvaluationType.Random:
            return "random"
        else:
            return NotImplementedError ##TODO : add remaining conditions
        
        
    @staticmethod
    def get_eval_type_num(eval_type_name):
        if eval_type_name == "oracle":
            return EvaluationType.Oracle
        elif eval_type_name == "single_best":
            return EvaluationType.SingleBest
        elif eval_type_name == "random":
            return EvaluationType.Random
        else:
            return NotImplementedError ##TODO : add remaining conditions
            
                     
    
class EvaluationMetricType:
    Accuracy, Rank, RegretAcc, ParX, NumSolved, RatioSolved, NDCGk = range(0, 7)    
    
    @staticmethod
    def get_eval_metric_num(eval_metric_name):
        if eval_metric_name == "Accuracy":
            return EvaluationMetricType.Accuracy
        elif eval_metric_name == "Regret":
            return EvaluationMetricType.RegretAcc
        elif eval_metric_name == "Rank":
            return EvaluationMetricType.Rank
        elif eval_metric_name == "Par10": 
            return EvaluationMetricType.ParX
        elif eval_metric_name == "Solved Instances Ratio":
            return EvaluationMetricType.NumSolved
        elif eval_metric_name == "RatioSolved":
            return EvaluationMetricType.RatioSolved
        elif eval_metric_name == "NDCG@3":
            return EvaluationMetricType.NDCGk
                                        

class Evaluate(object):
    '''
    classdocs
    '''

    def __init__(self, name, ia_data_matrix, pred_ia_rank_matrix = None):
        '''
            Constructor
            :param name: performance metric's name
            :param ia_data_matrix: true performance data matrix (instance x algorithm)
            :param ia_rank_matrix: predicted rank matrix (instance x algorithm)
        '''
        self.name = name
        self.ia_data_matrix = ia_data_matrix
        self.pred_ia_rank_matrix = pred_ia_rank_matrix
        if self.pred_ia_rank_matrix is None:
            self.pred_ia_rank_matrix = ia_data_matrix
        
        self.oracle_perf = None
        self.single_best_perf = None
        self.random_perf = None
        ##self.regret_perf = None
        self.kth_oracle_perf_dict = {}
        self.kth_single_best_dict = {}
        ## paralelization performance
        self.k_parallel_oracle_perf = None
        self.k_parallel_single_best_perf = None
        self.k_parallel_random_perf = None
        
    def oracle(self):
        self.oracle_perf = self.kth_oracle(1)  
        return self.oracle_perf
    
    def single_best(self):
        self.single_best_perf = self.kth_single_best(1)
        return self.single_best_perf
    
    def k_parallel_oracle(self):
        raise NotImplementedError
    
    def k_parallel_single_best(self):
        raise NotImplementedError
    
    def k_parallel_random(self):
        raise NotImplementedError
    
    def random(self):
        raise NotImplementedError
    
    def kth_oracle(self, k):
        raise NotImplementedError
    
    def kth_single_best(self, k):
        raise NotImplementedError
    
    def all_kth_oracle(self):
        for k in range(1,len(self.ia_data_matrix[0])+1):
            if k not in self.kth_oracle_perf_dict.keys():
                self.kth_oracle(k)
        
        return self.kth_oracle_perf_dict
            
    def all_kth_single_best(self):
        for k in range(1,len(self.ia_data_matrix[0])+1):
            if k not in self.kth_single_best_dict.keys():
                self.kth_single_best(k)
                
        return self.kth_single_best_dict
    
    def pprint(self, evaluation_type = None):
        
        if evaluation_type == EvaluationType.Oracle:
            
            if self.oracle_perf is None:
                self.oracle()
                
            print self.name+": oracle = ", self.oracle_perf
        
        elif evaluation_type == EvaluationType.SingleBest:
 
            if self.single_best_perf is None:
                self.single_best()    
                
            print self.name+": single_best = ", self.single_best_perf
            
        else:
            
            if self.oracle_perf is None:
                self.oracle()
            
            if self.single_best_perf is None:
                self.single_best()    
                
            print self.name+": oracle = ", self.oracle_perf, " | single_best = ", self.single_best_perf
        
        
    def pprint_k(self, evaluation_type = None, calculate_all = True):
        
        if evaluation_type == EvaluationType.Oracle:
            if calculate_all and len(self.kth_oracle_perf_dict) < len(self.ia_data_matrix[0]):
                self.all_kth_oracle()
                
            print "\n", self.name,"- kth Oracle performance\n--------------------"
            pprint.pprint(self.kth_oracle_perf_dict, width=1)
        
        elif evaluation_type == EvaluationType.SingleBest:
            if calculate_all and len(self.kth_single_best_dict) < len(self.ia_data_matrix[0]):
                self.all_kth_single_best()  

            print "\n", self.name,"- kth Single Best performance\n--------------------"   
            pprint.pprint(self.kth_single_best_dict, width=1)
                            
        else:
            if calculate_all and len(self.kth_oracle_perf_dict) < len(self.ia_data_matrix[0]):
                self.all_kth_oracle()
            
            if calculate_all and len(self.kth_single_best_dict) < len(self.ia_data_matrix[0]):
                self.all_kth_single_best()   
                
            print "\n", self.name,"- kth Oracle performance\n--------------------"
            pprint.pprint(self.kth_oracle_perf_dict, width=1)
            
            print "\n", self.name,"- kth Single Best performance\n--------------------"   
            pprint.pprint(self.kth_single_best_dict, width=1)
    
    
    
    
    @staticmethod
    def evaluate_prediction_from_folds(fold_perf_list, eval_type):
        '''
        '''
        num_eval_metrics = len(fold_perf_list[0])
        
        eval_arr = np.zeros((num_eval_metrics, len(fold_perf_list))) 
        eval_metric_list = []
        
        fold_inx = 0 
        for eval_dic in fold_perf_list:
            eval_inx = 0
            for e_key, e_dict in eval_dic.iteritems():
                if fold_inx == 0:
                    eval_metric_list.append(e_dict['name'])
                
                if eval_type == EvaluationType.Oracle:
                    eval_arr[eval_inx][fold_inx] = e_dict['oracle']
                elif eval_type == EvaluationType.SingleBest:
                    if "NDCG" not in e_dict['name']: 
                        eval_arr[eval_inx][fold_inx] = e_dict['single_best']
                                                     
                eval_inx += 1
                
            fold_inx += 1
            
        eval_avg_std_arr = np.zeros((num_eval_metrics, 2))
        for eval_inx in range(len(eval_metric_list)):
            eval_avg_std_arr[eval_inx][0] = np.average(eval_arr[eval_inx])
            eval_avg_std_arr[eval_inx][1] = np.std(eval_arr[eval_inx])
            
        return eval_avg_std_arr
    
    
    @staticmethod
    def combine_dict_in_list(fold_perf_list):
        '''
            TODO
        '''
        
        eval_dict = {}
        
        num_eval_metrics = len(fold_perf_list[0])
        num_eval_types = len(fold_perf_list[0][fold_perf_list[0].keys()[0]])-1
        
        eval_arr = np.zeros((num_eval_metrics, num_eval_types, len(fold_perf_list))) 
        eval_metric_list = []
        eval_type_name_list = []
        
        fold_inx = 0 
        for eval_dic in fold_perf_list:
            eval_inx = 0
            for e_key, e_dict in eval_dic.iteritems():
                
                if fold_inx == 0:
                    eval_metric_list.append(e_dict['name'])
                    eval_type_name_list = e_dict.keys()
                    
                e_dict_keys = e_dict.keys()
                e_dict_keys.remove('name')
                

                eval_type_inx = 0                
                for eval_type_name in e_dict_keys:
                    
                    eval_type = EvaluationType.get_eval_type_num(eval_type_name)
                    
                    eval_arr[eval_inx][eval_type_inx][fold_inx] = e_dict[eval_type_name]
                        
                    eval_type_inx += 1
                                                     
                eval_inx += 1
                
            fold_inx += 1
            
        eval_avg_std_arr = np.zeros((num_eval_metrics, num_eval_types, 2))
        for eval_inx in range(num_eval_metrics):
            
            eval_metric_inx = EvaluationMetricType.get_eval_metric_num(eval_metric_list[eval_inx])
            
            eval_dict[eval_metric_inx] = {}
            
            eval_dict[eval_metric_inx]['name'] = eval_metric_list[eval_inx]

            for eval_type_inx in range(num_eval_types):
                
                eval_avg_std_arr[eval_inx][eval_type_inx][0] = np.average(eval_arr[eval_inx][eval_type_inx])
            
                eval_dict[eval_metric_inx][eval_type_name_list[eval_type_inx]] = eval_avg_std_arr[eval_inx][eval_type_inx][0]
            
                        
        return eval_dict    
    
    
    
    @staticmethod
    def extract_performance_from_folds(fold_perf_list, eval_type):
        '''
        '''
        num_eval_metrics = len(fold_perf_list[0])
        
        eval_arr = np.zeros((num_eval_metrics, len(fold_perf_list))) 
        eval_metric_list = []
        
        fold_inx = 0 
        for eval_dic in fold_perf_list:
            eval_inx = 0
            for e_key, e_dict in eval_dic.iteritems():
                if fold_inx == 0:
                    eval_metric_list.append(e_dict['name'])
                
                if eval_type == EvaluationType.Oracle:
                    eval_arr[eval_inx][fold_inx] = e_dict['oracle']
                elif eval_type == EvaluationType.SingleBest:
                    eval_arr[eval_inx][fold_inx] = e_dict['single_best']
                                                     
                eval_inx += 1
                
            fold_inx += 1
            
        eval_avg_std_arr = np.zeros((num_eval_metrics, 2))
        for eval_inx in range(len(eval_metric_list)):
            eval_avg_std_arr[eval_inx][0] = np.average(eval_arr[eval_inx])
            eval_avg_std_arr[eval_inx][1] = np.std(eval_arr[eval_inx])
            
        return eval_arr, eval_metric_list
    

    @staticmethod
    def extract_performance(eval_dic, eval_type):
        '''
        '''
        for e_key, e_dict in eval_dic.iteritems():
            if eval_type == EvaluationType.Oracle:
                return e_dict['oracle']
            elif eval_type == EvaluationType.SingleBest:
                return e_dict['single_best']
                                                 
    
            
