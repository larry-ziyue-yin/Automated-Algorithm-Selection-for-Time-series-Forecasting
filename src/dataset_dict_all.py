import os

from src.performance.evaluate import EvaluationMetricType
from src.main.latent_estimate_cs import MappingMethodType
from src.matrix.factorization import MatrixFactorizationType


main_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "as_datasets/")
    

exp_setting_dict = {} 

exp_setting_dict[0] = {'ai_file_path': main_folder_path + "/ASP_POTASSCO-ai-perf.csv", 
                    'ft_file_path': main_folder_path + "/ASP_POTASSCO-features.txt", 
                    'cv_file_path': main_folder_path + "/ASP_POTASSCO-cv.txt",  
                    'perf_bound_val': 600,  
                    'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank, EvaluationMetricType.NDCGk],
                    'mf_type': MatrixFactorizationType.svd,
                    'mf_rank': 10, ##10      
                    'higher_better': False,
                    'map_method': MappingMethodType.RandomForest,      
                    'num_splits': 10}
     

exp_setting_dict[1] = {'ai_file_path': main_folder_path + "/PROTEUS_2014-ai-perf.csv", 
                    'ft_file_path': main_folder_path + "/PROTEUS_2014-features.txt", 
                    'cv_file_path': main_folder_path + "/PROTEUS_2014-cv.txt",  
                    'perf_bound_val': 3600,    
                    'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank],
                    'mf_type': MatrixFactorizationType.svd,
                    'mf_rank': 10, ##10      
                    'higher_better': False,
                    'map_method': MappingMethodType.RandomForest,      
                    'num_splits': 10}


exp_setting_dict[2] = {'ai_file_path': main_folder_path + "/QBF_2011-ai-perf.csv", 
                    'ft_file_path': main_folder_path + "/QBF_2011-features.txt", 
                    'cv_file_path': main_folder_path + "/QBF_2011-cv.txt",     
                    'perf_bound_val': 3600,      
                    'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank],
                    'mf_type': MatrixFactorizationType.svd,
                    'mf_rank': 10, ##10      
                    'higher_better': False,
                    'map_method': MappingMethodType.RandomForest,      
                    'num_splits': 10}


     
exp_setting_dict[3] = {'ai_file_path': main_folder_path + "/CSP_2010-ai-perf.csv", 
                    'ft_file_path': main_folder_path + "/CSP_2010-features.txt", 
                    'cv_file_path': main_folder_path + "/CSP_2010-cv.txt",  
                    'perf_bound_val': 5000,  
                    'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank],
                    'mf_type': MatrixFactorizationType.svd,
                    'mf_rank': 10, ##10      
                    'higher_better': False,
                    'map_method': MappingMethodType.RandomForest,      
                    'num_splits': 10}     


exp_setting_dict[4] = {'ai_file_path': main_folder_path + "/MAXSAT12_PMS-ai-perf.csv", 
                    'ft_file_path': main_folder_path + "/MAXSAT12_PMS-features.txt", 
                    'perf_bound_val': 2100,  
                    'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank],
                    'mf_type': MatrixFactorizationType.svd,
                    'mf_rank': 10, ##10      
                    'higher_better': False, 
                    'map_method': MappingMethodType.RandomForest,      
                    'num_splits': 10}   

 
exp_setting_dict[5] = {'ai_file_path': main_folder_path + "/PREMARSHALLING_ASTAR_2015-ai-perf.csv", 
                    'ft_file_path': main_folder_path + "/PREMARSHALLING_ASTAR_2015-features.txt", 
                    'perf_bound_val': 3600,      
                    'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank],
                    'mf_type': MatrixFactorizationType.svd, 
                    'mf_rank': 10, ##10      
                    'higher_better': False,
                    'map_method': MappingMethodType.RandomForest,      
                    'num_splits': 10}

  
exp_setting_dict[6] = {'ai_file_path': main_folder_path + "/SAT11_HAND-ai-perf.csv", 
                    'ft_file_path': main_folder_path + "/SAT11_HAND-features.txt",
                    'cv_file_path': main_folder_path + "/SAT11_HAND-cv.txt",  
                    'perf_bound_val': 5000,
                    'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank, EvaluationMetricType.NDCGk], ### NDCGk for test
                    'mf_type': MatrixFactorizationType.svd, 
                    'mf_rank': 10, ##10      
                    'higher_better': False,
                    'map_method': MappingMethodType.RandomForest,      
                    'num_splits': 10, ## doesn't affect if a cv file is given   
                     }
  
  
exp_setting_dict[7] = {'ai_file_path': main_folder_path + "/SAT11_RAND-ai-perf.csv", 
                    'ft_file_path': main_folder_path + "/SAT11_RAND-features.txt",
                    'cv_file_path': main_folder_path + "/SAT11_RAND-cv.txt",   
                    'perf_bound_val': 5000,
                    'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank],
                    'mf_type': MatrixFactorizationType.svd, 
                    'mf_rank': 10, ##10      
                    'higher_better': False,
                    'map_method': MappingMethodType.RandomForest,      
                    'num_splits': 10, ## doesn't affect if a cv file is given      
                     }


exp_setting_dict[8] = {'ai_file_path': main_folder_path + "/SAT11_INDU-ai-perf.csv", 
                    'ft_file_path': main_folder_path + "/SAT11_INDU-features.txt",
                    'cv_file_path': main_folder_path + "/SAT11_INDU-cv.txt",   
                    'perf_bound_val': 5000,
                    'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank],
                    'mf_type': MatrixFactorizationType.svd, 
                    'mf_rank': 10, ##10      
                    'higher_better': False,
                    'map_method': MappingMethodType.RandomForest,      
                    'num_splits': 10, ## doesn't affect if a cv file is given      
                     }
 


exp_setting_dict[9] = {'ai_file_path': main_folder_path + "/SAT12_ALL-ai-perf.csv", 
                    'ft_file_path': main_folder_path + "/SAT12_ALL-features.txt",
                    'cv_file_path': main_folder_path + "/SAT12_ALL-cv.txt",   
                    'perf_bound_val': 1200,
                    'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank],
                    'mf_type': MatrixFactorizationType.svd,     
                    'mf_rank': 10, ##10      
                    'higher_better': False,
                    'map_method': MappingMethodType.RandomForest,      
                    'num_splits': 10, ## doesn't affect if a cv file is given   
                     }



exp_setting_dict[10] = {'ai_file_path': main_folder_path + "/SAT12_HAND-ai-perf.csv", 
                    'ft_file_path': main_folder_path + "/SAT12_HAND-features.txt",
                    'cv_file_path': main_folder_path + "/SAT12_HAND-cv.txt",   
                    'perf_bound_val': 1200,
                    'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank],
                    'mf_type': MatrixFactorizationType.svd, 
                    'mf_rank': 10, ##10
                    'higher_better': False,
                    'map_method': MappingMethodType.RandomForest,      
                    'num_splits': 10, ## doesn't affect if a cv file is given    
                     }


exp_setting_dict[11] = {'ai_file_path': main_folder_path + "/SAT12_INDU-ai-perf.csv", 
                    'ft_file_path': main_folder_path + "/SAT12_INDU-features.txt",
                    'cv_file_path': main_folder_path + "/SAT12_INDU-cv.txt",   
                    'perf_bound_val': 1200,
                    'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank],
                    'mf_type': MatrixFactorizationType.svd, 
                    'mf_rank': 10, ##10      
                    'higher_better': False,
                    'map_method': MappingMethodType.RandomForest,      
                    'num_splits': 10, ## doesn't affect if a cv file is given    
                     }


exp_setting_dict[12] = {'ai_file_path': main_folder_path + "/SAT12_RAND-ai-perf.csv", 
                    'ft_file_path': main_folder_path + "/SAT12_RAND-features.txt",
                    'cv_file_path': main_folder_path + "/SAT12_RAND-cv.txt",   
                    'perf_bound_val': 1200,
                    'eval_metrics': [EvaluationMetricType.ParX, EvaluationMetricType.RatioSolved, EvaluationMetricType.Rank],
                    'mf_type': MatrixFactorizationType.svd, 
                    'mf_rank': 10, ##10      
                    'higher_better': False,
                    'map_method': MappingMethodType.RandomForest,      
                    'num_splits': 10, ## doesn't affect if a cv file is given     
                     }


exp_setting_dict[13] = {'ai_file_path': main_folder_path + "/openml-ai-bagging-accuracy.csv",
                    'ft_file_path': main_folder_path + "/openml-features.txt",  ##already normalized
                    'perf_bound_val': -1000, ## should be low since higher_better = True  
                    'eval_metrics': [EvaluationMetricType.RegretAcc, EvaluationMetricType.Rank],
                    'mf_type': MatrixFactorizationType.svd,   
                    'mf_rank': 10,
                    'higher_better': True,
                    'map_method': MappingMethodType.RandomForest, 
                    'num_splits': 10   
                    }


exp_setting_dict[14] = {'ai_file_path': main_folder_path + "/openml-ai-AdaBoost-accuracy.csv",
                        'ft_file_path': main_folder_path + "/openml-features.txt",  ##already normalized
                        'perf_bound_val': -1000, ## should be low since higher_better = True  
                        'eval_metrics': [EvaluationMetricType.RegretAcc, EvaluationMetricType.Rank],
                        'mf_type': MatrixFactorizationType.svd,   
                        'mf_rank': 10,
                        'higher_better': True,
                        'map_method': MappingMethodType.RandomForest, 
                        'num_splits': 5      
                        }


exp_setting_dict[15] = {'ai_file_path': main_folder_path + "/openml-ai-j48-accuracy.csv",
                        'ft_file_path': main_folder_path + "/openml-features.txt",  ##already normalized
                        'perf_bound_val': -1000, ## should be low since higher_better = True  
                        'eval_metrics': [EvaluationMetricType.RegretAcc, EvaluationMetricType.Rank],   
                        'mf_type': MatrixFactorizationType.svd,   
                        'mf_rank': 10,
                        'higher_better': True,
                        'map_method': MappingMethodType.RandomForest, 
                        'num_splits': 5   
                        }



exp_setting_dict[16] = {'ai_file_path': main_folder_path + "/openml-ai-accuracy.csv",
                        'ft_file_path': main_folder_path + "/openml-features.txt",  ##already normalized
                        'perf_bound_val': -1000, ## should be low since higher_better = True  
                        'eval_metrics': [EvaluationMetricType.RegretAcc, EvaluationMetricType.Rank],   
                        'mf_type': MatrixFactorizationType.svd,   
                        'mf_rank': 10,
                        'higher_better': True,
                        'map_method': MappingMethodType.RandomForest, 
                        'num_splits': 5    ##10
                        }
 
