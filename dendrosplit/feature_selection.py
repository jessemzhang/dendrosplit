from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Feature selection using random forest 
def skRandomForest(X,Y):
    clf = RandomForestClassifier()
    clf.fit(X,Y)
    return clf

def select_genes_using_RF(X,Y,return_score=False,verbose=False):
    # find most important features using random forest
    n = 100 # number of times to repeat random forest
    m = int(np.round(X.shape[1]/2)) # number of features to output at the end
    if m > n: m = n
    # Save the number of times each gene appears in the top n most important features
    gene_scores = np.zeros(X.shape[1])
    # For each fitting, score how well the fitting classifies the data
    scores = np.zeros(n)
    for i in range(0,n):    
        clf = skRandomForest(X,Y)
        importances = clf.feature_importances_
        y_clf = clf.predict(X)
        scores[i] = NMI(Y,y_clf)
        gene_ranks = np.flipud(np.argsort(importances))
        gene_scores[gene_ranks[0:m]] += 1
    # Rank genes by how often it appeared in the top n most important features
    gene_ranks = np.flipud(np.transpose(np.argsort(np.transpose(gene_scores))))[0:m]
    gene_scores /= float(n)
    if verbose:
        print 'Mean prediction score: %.3f'%(np.mean(scores))
    if return_score:
        return gene_ranks,gene_scores[gene_ranks],np.mean(scores)
    return gene_ranks,gene_scores[gene_ranks]

# Feature selection using logistic regression
def skLogisticRegression(X,Y):
    lr = LogisticRegression(penalty='l1')
    lr.fit(X,Y)
    return lr

def select_genes_using_LR(X,Y,return_score=False,verbose=False):
    clf = skLogisticRegression(X,Y)
    y_clf = clf.predict(X)
    gene_scores = np.abs(clf.coef_)[0]
    gene_ranks = np.flipud(np.argsort(gene_scores))
    score = compute_clustering_accuracy(Y,y_clf)
    if verbose: print 'Score: %.2f'%(score)
    if return_score: return gene_ranks,gene_scores[gene_ranks],score
    return gene_ranks,gene_scores[gene_ranks]

# Feature selection using Welch's t-test
def select_genes_using_Welchs(X,Y,return_score=False,verbose=False):
    if len(np.unique(Y)) > 2: 
        print 'ERROR: Y should only have 2 unique values'
    y = Y == Y[0]
    t,p = ttest_ind(X[y==1,:],X[y==0,:],equal_var=False)
    gene_ranks = np.argsort(-np.abs(t))
    gene_scores = np.nan_to_num(-np.log10(p[gene_ranks]))
    score = gene_scores[0]
    if verbose: print 'Score: %.2f'%(score)
    if return_score: return gene_ranks,gene_scores,score
    return gene_ranks,gene_scores

# Visualization scripts
def one_from_rest_gene_selection(X,Y,feature_selector=select_genes_using_RF):
    selected_genes_for_each_cluster = []
    for ctype in np.unique(Y):
        selected_genes_for_each_cluster.append(feature_selector(X,Y == ctype,return_score=True))
    return selected_genes_for_each_cluster
        
def one_from_rest_visualize_genes(X,genes,x1,x2,y,num_genes=3,
                                  feature_selector=select_genes_using_RF):
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

def visualize_more_highly_expressed_genes_in_one_clust(X,genes,x1,x2,y,num_genes=3,verbose=True,
                                                       feature_selector=select_genes_using_RF,save_name=None):
    for c in np.unique(y):
        c1 = X[y == c,:]
        c2 = X[y != c,:]
        keep_genes = np.mean(c1,0) > np.mean(c2,0) 
        g_temp = genes[keep_genes]
        if verbose: print 'Cluster %d: %d/%d genes kept'%(c,np.sum(keep_genes),len(keep_genes))
        gene_ranks,gene_scores,score = feature_selector(X[:,keep_genes],y==c,return_score=True)
        plt.figure(figsize=(4*(num_genes+1),3))
        plt.subplot(1,num_genes+1,1)
        plt.scatter(x1,x2,c=y==c,edgecolors='none')
        plt.title('Clust: %d, Acc: '%(c)+sn(score))
        _ = plt.axis('off')
        for i in range(num_genes):
            g = g_temp[gene_ranks[i]]
            plt.subplot(1,num_genes+1,i+2)
            plt.scatter(x1,x2,c=X[:,genes==g],edgecolors='none')
            plt.title(g+' '+sn(gene_scores[i]))
            _ = plt.axis('off')
        if save_name is not None: plt.savefig(save_name+'_cluster'+str(c)+'.png', format='png', dpi=300)
