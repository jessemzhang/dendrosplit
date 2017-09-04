from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from utils import *
from evaluation import *
import numpy as np
import matplotlib.pyplot as plt
import itertools

# import rpy2.robjects as ro
# from rpy2.robjects.numpy2ri import numpy2ri

# # Some functions for scoring how good a split is
# def sigclust(X,y):
#     if len(np.unique(y)) != 2: 
#         print 'ERROR: can only have 2 unique labels'
#         return
#     y = str_labels_to_ints(y)
#     ro.r('library(sigclust)')
#     ro.r('''g<-function(X,y){
#                 p = sigclust(X,100,labflag=1,label=y,icovest=3)
#                 return(p@pval)
#             }''')
#     g = ro.globalenv['g']
#     p = g(numpy2ri(X),numpy2ri(y))
#     return np.nan_to_num(-np.log10(p[0]))

# Feature selection using random forest 
def skRandomForest(X,Y):
    clf = RandomForestClassifier()
    clf.fit(X,Y)
    return clf

def select_genes_using_RF(X,Y,return_score=False,verbose=False,return_comparisons=False,score_function=None):
    # find most important features using random forest
    n = 100 # number of times to repeat random forest
    m = int(np.round(X.shape[1]/2)) # number of features to output at the end
    if m > n: m = n
    # Save the number of times each gene appears in the top n most important features
    gene_scores = np.zeros(X.shape[1])
    # For each fitting, score how well the fitting classifies the data
    if return_score and score_function is None: scores = np.zeros(n)
    for i in range(0,n):    
        clf = skRandomForest(X,Y)
        importances = clf.feature_importances_
        y_clf = clf.predict(X)
        if return_score and score_function is None: scores[i] = NMI(Y,y_clf)
        gene_ranks = np.flipud(np.argsort(importances))
        gene_scores[gene_ranks[0:m]] += 1
    # Rank genes by how often it appeared in the top n most important features
    gene_ranks = np.flipud(np.transpose(np.argsort(np.transpose(gene_scores))))[0:m]
    gene_scores /= float(n)
    outputs = [gene_ranks,gene_scores[gene_ranks]]
    if verbose: print 'Mean prediction score: %.3f'%(np.mean(scores))
    if return_score: 
        if score_function is None: outputs.append(np.mean(scores))
        else: outputs.append(score_function(X,Y))
    if return_comparisons: outputs.append(compare_feature_means(X,Y)[gene_ranks])
    return tuple(outputs)

def log_select_genes_using_RF(X,Y,**kwargs):
    return select_genes_using_RF(np.log(1+X),Y,**kwargs)

def log_select_genes_using_RF_sigclust(X,Y,**kwargs):
    return select_genes_using_RF(np.log(1+X),Y,score_function=sigclust,**kwargs)

# Feature selection using L1-regularized logistic regression
def skLogisticRegression(X,Y):
    lr = LogisticRegression(penalty='l1')
    lr.fit(X,Y)
    return lr

def select_genes_using_LR(X,Y,return_score=False,verbose=False,return_comparisons=False,score_function=None):
    clf = skLogisticRegression(X,Y)
    y_clf = clf.predict(X)
    gene_scores = np.abs(clf.coef_)[0]
    gene_ranks = np.flipud(np.argsort(gene_scores))
    outputs = [gene_ranks,gene_scores[gene_ranks]]
    if verbose: print 'Score: %.2f'%(score)
    if return_score: 
        if score_function is None: score = compute_clustering_accuracy(Y,y_clf)
        else: score = score_function(X,Y)
        outputs.append(score)
    if return_comparisons: outputs.append(compare_feature_means(X,Y)[gene_ranks])
    return tuple(outputs)

def log_select_genes_using_LR(X,Y,**kwargs):
    return select_genes_using_LR(np.log(1+X),Y,**kwargs)

def log_select_genes_using_LR_sigclust(X,Y,**kwargs):
    return select_genes_using_LR(np.log(1+X),Y,score_function=sigclust,**kwargs)

# Feature selection using Welch's t-test
def select_genes_using_Welchs(X,Y,return_score=False,verbose=False,return_comparisons=False,
                              score_function=None,return_tp=False,equal_var=False):
    if len(np.unique(Y)) > 2: 
        print 'ERROR: Y should only have 2 unique values'
    N,M = np.shape(X)
    y = str_labels_to_ints(Y)
    X1,X0 = X[y==1,:],X[y==0,:]
    # Numerical error edge case handling
    epsilon = 1e-8
    num_err_fix = np.abs(np.mean(X1,0)-np.mean(X0,0)) < epsilon
    X1[:,num_err_fix] = 0
    X0[:,num_err_fix] = 0
    # Perform t-test
    t_orig,p_orig = ttest_ind(X1,X0,equal_var=equal_var)
    gene_inds = np.array(range(M))
    finite_inds = np.isfinite(t_orig) # To account for cases where gene has 0 expression in both populations
    t,p,gene_inds = t_orig[finite_inds],p_orig[finite_inds],gene_inds[finite_inds]
    # Argsort first by p-value, then by -np.abs(t-value)
    keep_inds = np.array([j[2] for j in sorted([(p[i],-np.abs(t[i]),i) for i in range(len(t))])])
    gene_ranks = gene_inds[keep_inds]
    gene_scores = np.nan_to_num(-np.log10(p[keep_inds]))
    outputs = [gene_ranks,gene_scores]
    if verbose: print 'Score: %.2f'%(score)
    if return_score: 
        if score_function is None: score = gene_scores[0]
        else: score = score_function(X,Y)
        outputs.append(score)
    if return_comparisons: outputs.append((t>0).astype(int)[keep_inds])
    if return_tp: 
        outputs.append(t_orig)
        outputs.append(p_orig)
    # Note: For dendrosplit, each split involves exactly two clusters. 'L' will be mapped
    #   to 0, and 'R' will mapped to 1. A positive t statistic indicates that the 1-clust
    #   has a higher median than the 0-clust (i.e. 'L' has less of the feature than 'R' on
    #   average). Therefore return_comparisons will index the features that are more greater
    #   in 1-clust (or 'R'-clust). In the visualize_history function, 0's are red, 1's are
    #   green.
    return tuple(outputs)

def log_select_genes_using_Welchs(X,Y,**kwargs):
    return select_genes_using_Welchs(np.log(1+X),Y,**kwargs)

def log_select_genes_using_Welchs_sigclust(X,Y,**kwargs):
    return select_genes_using_Welchs(np.log(1+X),Y,score_function=sigclust,**kwargs)

# Visualization scripts
def one_from_rest_gene_selection(X,Y,feature_selector=log_select_genes_using_Welchs):
    selected_genes_for_each_cluster = []
    for ctype in np.unique(Y):
        selected_genes_for_each_cluster.append(feature_selector(X,Y == ctype,return_score=True))
    return selected_genes_for_each_cluster
        
def one_from_rest_visualize_genes(X,genes,x1,x2,y,num_genes=3,
                                  feature_selector=log_select_genes_using_Welchs):
    selected_genes_for_each_cluster = one_from_rest_gene_selection(X,y,feature_selector)
    unique_clusts = np.unique(y)
    for k,selected_genes in enumerate(selected_genes_for_each_cluster):
        plt.figure(figsize=(4*(num_genes+1),3))
        plt.subplot(1,num_genes+1,1)
        plt.scatter(x1,x2,c=y==unique_clusts[k],edgecolors='none')
        plt.title('Clust: %d, Acc: '%(unique_clusts[k])+sn(selected_genes[2]))
        _ = plt.axis('off')
        for i in range(num_genes):
            g = genes[selected_genes[0][i]]
            plt.subplot(1,num_genes+1,i+2)
            plt.scatter(x1,x2,c=X[:,genes == g],edgecolors='none')
            plt.title(g+' '+sn(selected_genes[1][i]))      
            _ = plt.axis('off')

# Note that if you use save_name while show_plots is True, all the plots will be saved as well
def save_more_highly_expressed_genes_in_one_clust(X,genes,y,x1=None,x2=None,num_genes=3,verbose=True,
                                                  feature_selector=log_select_genes_using_Welchs,save_name=None,
                                                  show_plots=True,pval_cutoff=10):
    if show_plots:
        if x1 is None or x2 is None: print 'NEED TO PASS IN x1, x2 FOR PLOTTING'
    if save_name is not None: 
        f = open(save_name+'_cluster_features.txt','w')
        print >> f, 'cluster\tgene\tpvalue\tmean_of_expr\tfold_change_of_expr'
    for c in np.unique(y):
        c1 = X[y == c,:]
        c2 = X[y != c,:]
        keep_genes = np.mean(c1,0) > np.mean(c2,0) 
        g_temp = genes[keep_genes]
        if len(g_temp) == 0: continue
        if verbose: print 'Cluster %d: %d/%d genes kept'%(c,np.sum(keep_genes),len(keep_genes))
        num_genes_c = np.min([np.sum(keep_genes),num_genes])
        gene_ranks,gene_scores,score = feature_selector(X[:,keep_genes],y==c,return_score=True)
        if show_plots and x1 is not None and x2 is not None:
            plt.figure(figsize=(4*(num_genes_c+1),3))
            plt.subplot(1,num_genes_c+1,1)
            plt.scatter(x1,x2,c=y==c,edgecolors='none')
            plt.title('Clust: %d, Score: '%(c)+sn(score))
            _ = plt.axis('off')
        for i in range(num_genes_c):
            g = g_temp[gene_ranks[i]]
            g_ind = np.where(genes==g_temp[gene_ranks[i]])[0]
            if show_plots and x1 is not None and x2 is not None:
                plt.subplot(1,num_genes_c+1,i+2)
                plt.scatter(x1,x2,c=X[:,genes==g],edgecolors='none')
                plt.title(g+' '+sn(gene_scores[i]))
                _ = plt.axis('off')
            if save_name is not None and gene_scores[i] > pval_cutoff:
                g_mean_in = np.mean(X[y==c,g_ind])
                g_mean_out = np.mean(X[y!=c,g_ind])
                s = '\t'.join([str(c),g_temp[gene_ranks[i]],sn(gene_scores[i],j=10),
                               sn(g_mean_in,j=10),sn(g_mean_in/g_mean_out,j=10)])
                print >> f, s
        if show_plots and x1 is not None and x2 is not None and save_name is not None: 
            plt.savefig(save_name+'_cluster'+str(c)+'.png', format='png', dpi=300)
    if save_name is not None: f.close()

# Note that if you use save_name while show_plots is True, all the plots will be saved as well
def pairwise_cluster_comparison(X,genes,y,x1=None,x2=None,num_genes=20,verbose=True,
                                feature_selector=log_select_genes_using_Welchs,save_name=None,
                                show_plots=True,pval_cutoff=10):
    if show_plots:
        if x1 is None or x2 is None: print 'NEED TO PASS IN x1, x2 FOR PLOTTING'
    if save_name is not None:
        f = open(save_name+'_pairwise_cluster_features.txt','w')
        print >> f, 'c1\tc2\tgene\tpvalue\tlarger_clust\tfold_change_of_exp_for_larger_clust'
    for (i,j) in itertools.combinations(np.unique(y),2):
        labels_ij = np.logical_or(y==i,y==j)
        gene_ranks,gene_scores,score = feature_selector(X[labels_ij,:],y[labels_ij],return_score=True)
        if verbose: print '\nComparing cluster %s and %s'%(i,j)
        if show_plots and x1 is not None and x2 is not None:
            plt.figure(figsize=(4*(num_genes+1),3))
            plt.subplot(1,num_genes+1,1)
            plot_labels_legend(x1[labels_ij],x2[labels_ij],y[labels_ij])
            plt.title('Clust %s v. clust %s, Score: '%(i,j)+sn(score))
            _ = plt.axis('off')
        for k in range(np.min([len(gene_ranks),num_genes])):
            g_mean_i = np.mean(X[y==i,gene_ranks[k]])
            g_mean_j = np.mean(X[y==j,gene_ranks[k]])
            if g_mean_i > g_mean_j: 
                c = i
                fold = g_mean_i/g_mean_j
            else: 
                c = j
                fold = g_mean_j/g_mean_i
            if verbose: print '%s (more expressed in %s) '%(genes[gene_ranks[k]],c)+sn(gene_scores[k])
            if show_plots and x1 is not None and x2 is not None:
                plt.subplot(1,num_genes+1,k+2)
                plt.scatter(x1[labels_ij],x2[labels_ij],c=X[labels_ij,gene_ranks[k]],edgecolors='none')
                plt.title(genes[gene_ranks[k]]+' '+sn(gene_scores[k]))
                _ = plt.axis('off')
                if save_name is not None: plt.savefig()
            if save_name is not None and gene_scores[k] > pval_cutoff:
                s = '\t'.join([str(i),str(j),genes[gene_ranks[k]],
                               sn(gene_scores[k],j=10),str(c),sn(fold,j=10)])
                print >> f, s
        if show_plots and x1 is not None and x2 is not None and save_name is not None:
            plt.savefig(save_name+'_c'+str(i)+'_v_c'+str(j)+'.png', format='png', dpi=300)  
