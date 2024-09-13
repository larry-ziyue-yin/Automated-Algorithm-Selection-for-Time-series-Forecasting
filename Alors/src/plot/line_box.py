'''
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pprint


from matplotlib import rcParams

from src.misc import settings
from src.performance.evaluate import EvaluationType
from matplotlib.figure import figaspect
from src.dao.misc_io import save_matrix
from matplotlib.patches import Rectangle

def plot_oracle_singlebest_line(title, oracle_dict_list, singlebest_dict_list):
    
    k_arr = range(1, len(oracle_dict_list)+1)
    
    oracle_perf_arr = np.zeros((len(oracle_dict_list),2))
    inx = 0
    for oracle_dict in oracle_dict_list:
        oracle_perf_arr[inx][0] = oracle_dict['accuracy']
        oracle_perf_arr[inx][1] = oracle_dict['rank']
        inx += 1  
    
    sb_perf_arr = np.zeros((len(singlebest_dict_list),2))
    inx = 0
    for sb_dict in singlebest_dict_list:
        sb_perf_arr[inx][0] = sb_dict['accuracy']
        sb_perf_arr[inx][1] = sb_dict['rank']
        inx += 1
    
    
    rcParams.update({'font.size': 12})

    plt.figure(1)
    ##plt.suptitle(title)  
    
    plt.subplot(121)
    plt.plot(k_arr, oracle_perf_arr[:,0], marker='.', linestyle = '--', color='r', label='Oracle')
    plt.plot(k_arr, sb_perf_arr[:,0], marker='.', linestyle = '--', color='b', label='Single Best')
    plt.xlabel('k')
    plt.ylabel('Performance')
    ##plt.title('Title')
    plt.legend(loc=3)
    
    plt.xlim([1, len(k_arr)+1])
    plt.grid()
      
    
    plt.subplot(122)
    plt.plot(k_arr, oracle_perf_arr[:,1], marker='.', linestyle = '--', color='r', label='Oracle')
    plt.plot(k_arr, sb_perf_arr[:,1], marker='.', linestyle = '--', color='b', label='Single Best')
    plt.xlabel('k')
    plt.ylabel('Rank')
    ##plt.title('Title')
    plt.legend(loc=4)
    
    plt.xlim([1, len(k_arr)+1])
    plt.grid()
    
    plt.tight_layout() ### http://matplotlib.org/users/tight_layout_guide.html
    
    pdf_to_save_path = os.path.join(settings.__output_folder__, title+'.pdf')
    plt.savefig(pdf_to_save_path, format='pdf', dpi=settings.__fig_dpi__, bbox_inches='tight')
    ##plt.show() 

    plt.gcf().clear()
    

def plot_MC_perf(output_folder, title, per_sparsity_mc_results_dict):
    '''
    '''
    num_lines_toplot = len(per_sparsity_mc_results_dict)
    num_eval_metrics = len(per_sparsity_mc_results_dict[0.1][1][0])  
    num_folds = len(per_sparsity_mc_results_dict[0.1][1])   
    
    eval_method_fold_perf_arr = np.zeros((num_eval_metrics, num_lines_toplot, num_folds))
    
    method_name_list = per_sparsity_mc_results_dict.keys()   

    if 'Oracle' in method_name_list:
        index_of = method_name_list.index('Oracle')
        method_name_list[0], method_name_list[index_of] = method_name_list[index_of], method_name_list[0]  

    if 'Single Best' in method_name_list:
        index_of = method_name_list.index('Single Best')
        method_name_list[1], method_name_list[index_of] = method_name_list[index_of], method_name_list[1]  
        
    if 'Random' in method_name_list:
        index_of = method_name_list.index('Random')
        method_name_list[2], method_name_list[index_of] = method_name_list[index_of], method_name_list[2] 
        
        
    ## sort rest of the method names
    method_name_list[3:] = sorted(method_name_list[3:])

    method_name_arr = np.array(method_name_list)
    
    
    eval_name_list = []
    
    ##print "method_name_arr: ", method_name_arr  
    
    ## extract data to plot
    ##method_inx = 0
    for res_key, res_dict in per_sparsity_mc_results_dict.iteritems(): ## for each test method
        if type(res_dict) is tuple:     
            res_dict = res_dict[0] ## since it is a tuple
              
        eval_type = res_dict[0]
        
        fold_inx = 0  
        for eval_dict in res_dict[1]: ## for each fold 
            
            
            eval_inx= 0
            for eval_key, eval_dict in eval_dict.iteritems(): ## for each evaluation metric

                ## get evaluation metric names 
                if len(eval_name_list) < num_eval_metrics:
                    eval_name_list.append(eval_dict['name'])
                
                method_inx = method_name_list.index(res_key)
                
                if eval_type == EvaluationType.Oracle:
                    eval_method_fold_perf_arr[eval_inx][method_inx][fold_inx] = eval_dict['oracle']
                elif eval_type == EvaluationType.SingleBest:
                    eval_method_fold_perf_arr[eval_inx][method_inx][fold_inx] = eval_dict['single_best']
                elif eval_type == EvaluationType.Random:
                    eval_method_fold_perf_arr[eval_inx][method_inx][fold_inx] = eval_dict['random']  
                else:
                    print 'Unidentified evaluation type: ', eval_type, '\nExiting...'
                    sys.exit()
                      
                    
                eval_inx += 1
            
            fold_inx += 1
            
        ##method_inx += 1
        
    
    ###################
    #### boxplot ####
    ###################         
    ##fig, ax1 = plt.subplots(1, num_eval_metrics)     
    ## http://stackoverflow.com/questions/14907062/matplotlib-aspect-ratio-in-subplots-with-various-y-axes   
    h, w = figaspect(2.)  ### make a figure twice as wide as it is tall    
    fig, ax1 = plt.subplots(nrows=1, ncols=num_eval_metrics, figsize=(w,h)) ##figsize=(2*num_eval_metrics,num_eval_metrics)            
    

    for plt_num in range(1, num_eval_metrics+1):     
        plt_inx =  100 + (num_eval_metrics * 10) + plt_num
        
        add_fold_boxplot(plt_inx, eval_name_list[plt_num-1], 4, eval_method_fold_perf_arr[plt_num-1], method_name_arr)
        
        
        ## =====================================================================
        ## save plot data to a file for further processing
        res_matrix_title = os.path.join(output_folder, title+'-'+eval_name_list[plt_num-1]+'.txt')  
        save_matrix(res_matrix_title, eval_method_fold_perf_arr[plt_num-1], method_name_arr.tolist(), None, ",")  
        ## =====================================================================
        
        
        
    plt.tight_layout() ### http://matplotlib.org/users/tight_layout_guide.html

    pdf_to_save_path = os.path.join(output_folder, title+'-boxplot.pdf')
    plt.savefig(pdf_to_save_path, format='pdf', dpi=settings.__fig_dpi__, bbox_inches='tight')
    ##plt.show() 

    plt.gcf().clear()
    
    
def plot_fold_perf_for_matrix_rank_k(output_folder, title, fold_results_dict):
    '''
    '''
    num_lines_toplot = len(fold_results_dict)
    num_eval_metrics = len(fold_results_dict[fold_results_dict.keys()[0]][1][0])
    num_folds = len(fold_results_dict[fold_results_dict.keys()[0]][1])
    
    ## print "\n@plot_fold_perf: ", num_lines_toplot, " - ", num_eval_metrics, " - ", num_folds
    
    ## pprint.pprint(fold_results_dict)
    
    eval_method_fold_perf_arr = np.zeros((num_eval_metrics, num_lines_toplot, num_folds))
    
    method_name_list = fold_results_dict.keys()   
    #### re-order method names (i.e. Oracle, Single Best, ALORS)
    ## http://stackoverflow.com/questions/2493920/how-to-switch-position-of-two-items-in-a-python-list
    num_order_changed = 0
    if 'Oracle' in method_name_list:
        index_of = method_name_list.index('Oracle')
        method_name_list[0], method_name_list[index_of] = method_name_list[index_of], method_name_list[0] 
        num_order_changed += 1 

    if 'Single Best' in method_name_list:
        index_of = method_name_list.index('Single Best')
        method_name_list[1], method_name_list[index_of] = method_name_list[index_of], method_name_list[1]
        num_order_changed += 1  
        
    if 'Random' in method_name_list:
        index_of = method_name_list.index('Random')
        method_name_list[2], method_name_list[index_of] = method_name_list[index_of], method_name_list[2]
        num_order_changed += 1 
        
    if 'ALORS' in method_name_list:
        index_of = method_name_list.index('ALORS')
        method_name_list[3], method_name_list[index_of] = method_name_list[index_of], method_name_list[3]
        num_order_changed += 1  


    ## sort rest of the method names  
    method_name_list[num_order_changed:] = sorted(method_name_list[num_order_changed:])

    ## print "method_name_list : ", method_name_list
    
    method_name_arr = np.array(method_name_list)
    
    
    eval_name_list = []
    
    ##print "method_name_arr: ", method_name_arr  
    
    ## extract data to plot
    ##method_inx = 0
    for res_key, res_dict in fold_results_dict.iteritems(): ## for each test method
        eval_type = res_dict[0]
        
        fold_inx = 0  
        for eval_dict in res_dict[1]: ## for each fold 
            
            eval_inx= 0
            for eval_key, eval_dict in eval_dict.iteritems(): ## for each evaluation metric

                ##print method_inx,"-",fold_inx,"-",eval_inx," :: ", eval_dict
                
                ## get evaluation metric names 
                if len(eval_name_list) < num_eval_metrics:
                    eval_name_list.append(eval_dict['name'])
                
                method_inx = method_name_list.index(res_key)
                
                if eval_type == EvaluationType.Oracle:
                    eval_method_fold_perf_arr[eval_inx][method_inx][fold_inx] = eval_dict['oracle']
                elif eval_type == EvaluationType.SingleBest:
                    eval_method_fold_perf_arr[eval_inx][method_inx][fold_inx] = eval_dict['single_best']
                elif eval_type == EvaluationType.Random:
                    eval_method_fold_perf_arr[eval_inx][method_inx][fold_inx] = eval_dict['random']  
                else:
                    print 'Unidentified evaluation type: ', eval_type, '\nExiting...'
                    sys.exit()
                      
                    
                eval_inx += 1
            
            fold_inx += 1
            
        ##method_inx += 1
    
    
    
    ###################
    #### boxplot ####
    ###################         
    ##fig, ax1 = plt.subplots(1, num_eval_metrics)     
    ## http://stackoverflow.com/questions/14907062/matplotlib-aspect-ratio-in-subplots-with-various-y-axes   
    h, w = figaspect(2.)  ### make a figure twice as wide as it is tall    
    fig, ax1 = plt.subplots(nrows=1, ncols=num_eval_metrics, figsize=(w,h)) ##figsize=(2*num_eval_metrics,num_eval_metrics)            
    

    for plt_num in range(1, num_eval_metrics+1):     
        plt_inx =  100 + (num_eval_metrics * 10) + plt_num
        
        add_fold_boxplot_for_matrix_rank_k(plt_inx, eval_name_list[plt_num-1], 4, eval_method_fold_perf_arr[plt_num-1], method_name_arr)
        
        
        ## =====================================================================
        ## save plot data to a file for further processing
        res_matrix_title = os.path.join(output_folder, title+'-'+eval_name_list[plt_num-1]+'.txt')  
        save_matrix(res_matrix_title, eval_method_fold_perf_arr[plt_num-1], method_name_arr.tolist(), None, ",")  
        ## =====================================================================
        
        
        
    plt.tight_layout() ### http://matplotlib.org/users/tight_layout_guide.html

    pdf_to_save_path = os.path.join(output_folder, title+'-boxplot.pdf')
    plt.savefig(pdf_to_save_path, format='pdf', dpi=settings.__fig_dpi__, bbox_inches='tight')
    ##plt.show() 

    plt.gcf().clear()
    
    
    
    

def plot_fold_perf(output_folder, title, fold_results_dict):
    '''
    '''
    num_lines_toplot = len(fold_results_dict)
    num_eval_metrics = len(fold_results_dict[fold_results_dict.keys()[0]][1][0])
    num_folds = len(fold_results_dict[fold_results_dict.keys()[0]][1])
    
    ## print "\n@plot_fold_perf: ", num_lines_toplot, " - ", num_eval_metrics, " - ", num_folds
    
    ## pprint.pprint(fold_results_dict)
    
    eval_method_fold_perf_arr = np.zeros((num_eval_metrics, num_lines_toplot, num_folds))
    
    method_name_list = fold_results_dict.keys()   
    #### re-order method names (i.e. Oracle, Single Best, ALORS)
    ## http://stackoverflow.com/questions/2493920/how-to-switch-position-of-two-items-in-a-python-list
    num_order_changed = 0
    if 'Oracle' in method_name_list:
        index_of = method_name_list.index('Oracle')
        method_name_list[0], method_name_list[index_of] = method_name_list[index_of], method_name_list[0] 
        num_order_changed += 1 

    if 'Single Best' in method_name_list:
        index_of = method_name_list.index('Single Best')
        method_name_list[1], method_name_list[index_of] = method_name_list[index_of], method_name_list[1]
        num_order_changed += 1  
        
    if 'Random' in method_name_list:
        index_of = method_name_list.index('Random')
        method_name_list[2], method_name_list[index_of] = method_name_list[index_of], method_name_list[2]
        num_order_changed += 1 
        
    if 'ALORS' in method_name_list:
        index_of = method_name_list.index('ALORS')
        method_name_list[3], method_name_list[index_of] = method_name_list[index_of], method_name_list[3]
        num_order_changed += 1  


    ## sort rest of the method names  
    method_name_list[num_order_changed:] = sorted(method_name_list[num_order_changed:])

    ## print "method_name_list : ", method_name_list
    
    method_name_arr = np.array(method_name_list)
    
    
    eval_name_list = []
    
    ##print "method_name_arr: ", method_name_arr  
    
    ## extract data to plot
    ##method_inx = 0
    for res_key, res_dict in fold_results_dict.iteritems(): ## for each test method
        eval_type = res_dict[0]
        
        fold_inx = 0  
        for eval_dict in res_dict[1]: ## for each fold 
            
            eval_inx= 0
            for eval_key, eval_dict in eval_dict.iteritems(): ## for each evaluation metric

                ##print method_inx,"-",fold_inx,"-",eval_inx," :: ", eval_dict
                
                ## get evaluation metric names 
                if len(eval_name_list) < num_eval_metrics:
                    eval_name_list.append(eval_dict['name'])
                
                method_inx = method_name_list.index(res_key)
                
                if eval_type == EvaluationType.Oracle:
                    eval_method_fold_perf_arr[eval_inx][method_inx][fold_inx] = eval_dict['oracle']
                elif eval_type == EvaluationType.SingleBest:
                    if "NDCG"not in eval_dict['name']: ## not to call single best function for NDCG
                        eval_method_fold_perf_arr[eval_inx][method_inx][fold_inx] = eval_dict['single_best']
                elif eval_type == EvaluationType.Random:
                    if "NDCG"not in eval_dict['name']: ## not to call random function for NDCG
                        eval_method_fold_perf_arr[eval_inx][method_inx][fold_inx] = eval_dict['random']  
                else:
                    print 'Unidentified evaluation type: ', eval_type, '\nExiting...'
                    sys.exit()
                      
                    
                eval_inx += 1

            fold_inx += 1
            
        ##method_inx += 1
        
    
    
    ###################
    #### boxplot ####
    ###################         
    ##fig, ax1 = plt.subplots(1, num_eval_metrics)     
    ## http://stackoverflow.com/questions/14907062/matplotlib-aspect-ratio-in-subplots-with-various-y-axes   
    h, w = figaspect(2.)  ### make a figure twice as wide as it is tall    
    fig, ax1 = plt.subplots(nrows=1, ncols=num_eval_metrics, figsize=(w,h)) ##figsize=(2*num_eval_metrics,num_eval_metrics)            
    

    for plt_num in range(1, num_eval_metrics+1):     
        plt_inx =  100 + (num_eval_metrics * 10) + plt_num
        
        if "NDCG" in eval_name_list[plt_num-1]: ## just for NDCG plotting
            method_inx_list = range(len(method_name_arr))
            method_inx_list.remove(0) ## remove single best index  
            method_inx_list.remove(1) ## remove single best index
            method_inx_list.remove(2) ## remove random index   
            
            new_method_name_arr = method_name_arr[:] ## get copy of the name array of will be plotted methods
            new_method_name_arr = np.delete(new_method_name_arr, 2) ## remove random name  (start from the larger index)  
            new_method_name_arr = np.delete(new_method_name_arr, 1) ## remove single best name
            new_method_name_arr = np.delete(new_method_name_arr, 0) ## remove oracle name (since this is always 1 for NDCG@k)  
            add_fold_boxplot(plt_inx, eval_name_list[plt_num-1], 4, eval_method_fold_perf_arr[plt_num-1][method_inx_list], new_method_name_arr)
        else:
            add_fold_boxplot(plt_inx, eval_name_list[plt_num-1], 4, eval_method_fold_perf_arr[plt_num-1], method_name_arr)
        
        
        ## =====================================================================
        ## save plot data to a file for further processing
        res_matrix_title = os.path.join(output_folder, title+'-'+eval_name_list[plt_num-1]+'.txt')  
        save_matrix(res_matrix_title, eval_method_fold_perf_arr[plt_num-1], method_name_arr.tolist(), None, ",")  
        ## =====================================================================
        
        
        
        
    plt.tight_layout() ### http://matplotlib.org/users/tight_layout_guide.html

    pdf_to_save_path = os.path.join(output_folder, title+'-boxplot.pdf')
    plt.savefig(pdf_to_save_path, format='pdf', dpi=settings.__fig_dpi__, bbox_inches='tight')
    ##plt.show() 

    plt.gcf().clear()
    
    

def add_fold_boxplot(plt_inx, perf_name, legend_loc, method_fold_perf_arr, method_name_arr):
    '''
    '''                
    ax = plt.subplot(plt_inx)
      
    ##xtickNames = plt.setp(ax, xticklabels=method_name_arr)
    ##plt.setp(xtickNames, rotation=90) ##rotation=45, fontsize=8
    ax.set_xticklabels(method_name_arr, rotation=90)  
    plt.boxplot(method_fold_perf_arr.T)      
    
    plt.ylabel(perf_name) 
    
    ##plt.xlim([0, len(method_name_arr)+1])    
    ##plt.ylim([np.min(method_fold_perf_arr), np.max(method_fold_perf_arr)])
    plt.grid()
    
    ## TODO: axes to be fixed
    ######forceAspect(ax,aspect=1)


def add_fold_boxplot_for_matrix_rank_k(plt_inx, perf_name, legend_loc, method_fold_perf_arr, method_name_arr):
    '''
    '''                
    ax = plt.subplot(plt_inx)
      
    ##xtickNames = plt.setp(ax, xticklabels=method_name_arr)
    ##plt.setp(xtickNames, rotation=90) ##rotation=45, fontsize=8
    ax.set_xticklabels(method_name_arr)  
    plt.boxplot(method_fold_perf_arr.T)      
    
    plt.xlabel('k')   
    plt.ylabel(perf_name) 
    
    ##plt.xlim([0, len(method_name_arr)+1])    
    ##plt.ylim([np.min(method_fold_perf_arr), np.max(method_fold_perf_arr)])
    plt.grid()
    
    ## TODO: axes to be fixed
    ######forceAspect(ax,aspect=1)
    
    

def add_fold_line_plot(plt_inx, perf_name, legend_loc, method_fold_perf_arr, method_name_arr):
    '''
    '''
    fold_inx_arr = range(1, len(method_fold_perf_arr[0])+1)
    
    plt.subplot(plt_inx)
    
    for method_inx, method_name in np.ndenumerate(method_name_arr):  
        method_inx = method_inx[0] 
        ##clr_val = str(1.0 / float(method_inx+1))       
        ##plt.plot(fold_inx_arr, method_fold_perf_arr[method_inx][:], marker='.', linestyle = '--', color=clr_val, label=method_name)
        plt.plot(fold_inx_arr, method_fold_perf_arr[method_inx][:], marker='.', linestyle = '--', label=method_name)     
    
    ##plt.xlabel('k')
    plt.ylabel(perf_name)
    ##plt.title('Title')
    ##plt.legend(loc=4)
    plt.legend(loc=legend_loc)  
    
    plt.xlim([1, len(fold_inx_arr)])
    plt.grid()
   
   
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
     

def time_plot(output_folder, title, ratio_avg_matrix, dataset_name_list, row_vals, _xlabel, _ylabel):
    '''
        Plot line plots for showing elapsed time ration between two methods   
    '''
    fig = plt.figure()
    ax = plt.subplot(111)


    color_list = ['b','g','r','c','m','y','k']
    marker_list = ['o', '+', 's', 'x', 'v','^', '>', '<', 'p', '*', 'h', 'H', 'd']
    
    row_inx = 0
    for row in ratio_avg_matrix:
        marker_ = marker_list[row_inx % len(marker_list)]
        color_ = color_list[row_inx % len(color_list)]
        plt.plot(row_vals, row, marker=marker_, linestyle = '--', color=color_, markersize = 10, label=dataset_name_list[row_inx])
        row_inx += 1
 
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    
    ax.set_xlim([np.min(row_vals), np.max(row_vals)])
 
    plt.ylabel(_ylabel)
    plt.xlabel(_xlabel)  
    
    plt.grid()
    
    plt.tight_layout() ### http://matplotlib.org/users/tight_layout_guide.html  
  
    pdf_to_save_path = os.path.join(output_folder, title+'.pdf')
    plt.savefig(pdf_to_save_path, format='pdf', dpi=settings.__fig_dpi__, bbox_inches='tight')
    ##plt.show() 


    plt.gcf().clear()
    