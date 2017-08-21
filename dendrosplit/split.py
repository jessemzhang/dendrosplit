from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import numpy as np
import matplotlib.pyplot as plt
import time,colorsys

from preprocessing import *
from feature_selection import *
from utils import *

# Using the linkage matrix Z, iteratively perform the following from the top:
#    1. Split into two clusters
#    2. If one of the clusters are too small (< min_clust_size), mark all points in it as outliers/error. remove it
#    3. Else, use random forest to classify the two types against each other. If high enough accuracy, keep clusters
#    4. Repeat on subclusters that aren't marked as outliers/error

# Given an inner node, split the two subtrees into two groups 
def node_to_leaf_labels(Z,i,ltypes=['0','1'],defaultlabel='_'):
    # Z: hierarchical clustering result returned by scipy linkage function
    # i: row of Z to perform splitting on 
    # ltypes: 2x1 vector. labels given to subtrees
    n = int(Z[-1,3])
    labels = [defaultlabel for _ in range(n)]
    def recursion(j,label):
        # base case 1: leaf node
        if j < n: 
            labels[j] = label
        # base case 2: parent of two leaf nodes
        elif Z[j-n,3] == 2:
            labels[int(Z[j-n,0])] = label
            labels[int(Z[j-n,1])] = label
        # recursive case: 
        else:
            recursion(int(Z[j-n,0]),label)
            recursion(int(Z[j-n,1]),label)
    recursion(int(Z[i,0]),ltypes[0])
    recursion(int(Z[i,1]),ltypes[1])
    return np.array(labels)

# Label updating algorithm for dendrosplit
def update_labels(y,labels,background='_'):
    label = None
    for i in range(len(y)):
        if y[i] != background:
            if label == None: label = labels[i]
            labels[i] += y[i] # Update labels 
    return label,labels

# Dendrosplit algorithm
def dendrosplit(X,preprocessing=log_correlation,min_clust_size=2,score_threshold=10,
                method='complete',verbose=False,split_evaluator=log_select_genes_using_Welchs,
                disband_percentile=100):
    '''
        Perform iterative hierarchical clustering. Obtain a dendrogram using a distance matrix. 
        Starting from the top, iteratively split the tree into two subtrees. Perform a test to 
        see if the two subtrees are different enough.
        
        Inputs:
            X: Matrix of counts 
            preprocessing: function to convert X to a distance matrix. This argument
                can be set to 'precomputed', in which case X should be a tuple (D,X)
                where D is the already-computed distance matrix
            min_clust_size: min size of cluster in order for a cluster split to be considered
            score_threshold: min score achieved by split required to keep the split
            method: linkage method for hierarchical clustering
            verbose: whether or not to turn on print statements while the algorithm runs
            split_evaluator: function to score how good a split is
            disband_percentile: if both subtrees have less than min_clust_size samples or if 
                a candidate split does not achieve high enough a score, look at the pairwise 
                distances amongst samples in this final cluster. if all are greater than 
                this percentile of the distances, then mark all of the points as singletons
            
        Outputs:
            y: cluster labels 
            history: list of tuples of information generated at each split. Use the functions
                'print_history' and 'visualize_history' with this output
    '''
    
    start_time = time.time()
    
    # Catch edge cases
    if preprocessing != 'precomputed': 
        if np.sum(np.sum(X,0) == 0) > 1:
            print 'Please remove genes that sum to 0 (try split.filter_genes())'
            return
#         if not check_if_all_entries_are_whole(X):
#             print 'Please make sure X is a valid distance matrix'
#             return
        D = preprocessing(X)
        if verbose: print 'Preprocessing took %.3f s'%(time.time()-start_time)
    elif type(X) != tuple: 
        print 'preprocessing = precomputed. Feed in (D,X) tuple as first argument.'
        return
    elif not np.all(np.isclose(X[0],X[0].T)):
        print 'Need a valid distance matrix.'
        return
    else: 
        D,X = X[0],X[1]
        if np.sum(np.sum(X,0) == 0) > 1:
            print 'Please remove columns that sum to 0 (try split.filter_genes())'
            return
#         if not check_if_all_entries_are_whole(X):
#             print 'Please make sure X is a valid distance matrix'
#             return

    # Perform the hierarchical clustering
#    D = (D+D.T)/2
    Ds = squareform(D)
    Z = linkage(Ds,method=method)

    N = len(X)

    # Compute disband percentile
    disband_threshold = np.percentile(flatten_distance_matrix(D),disband_percentile)
    
    # start with the label 'r' (for 'root') for all samples
    labels = ['r' for i in range(N)]

    # Stuff we may want to remember about each stage of splitting:
    #  - labels before the split and # of labels
    #  - labels after the split and # of each labels
    #  - splitting score
    #  - important features for distingushing the two subtypes
    #  - which cluster has greater expression of each feature
    history,scount = [],[0]
    
    def recursion(j,split_depth,labels):

        j = int(j)

        # Check number of leaves below node exceeds required threshold
        # Or if we're at a leaf node
        if j < 0: return
        if Z[j,3] < min_clust_size: return
        
        # Split subtrees and get labels
        y_node = node_to_leaf_labels(Z,j,ltypes=['L','R'])
      
        lsize = np.sum(y_node == 'L')
        rsize = np.sum(y_node == 'R')
        if verbose: print 'Potential split result: %d and %d'%(lsize,rsize)
        
        # If the split produces a cluster that's too small, ignore the smaller cluster
        if lsize < min_clust_size and rsize < min_clust_size: 
            # if both clusters are too small, check distances. If all distances are above
            # the disband_percentile overall distance, then make all samples singletons
            if np.sum(flatten_distance_matrix(D,y_node!='_') < disband_threshold) == 0:
                if verbose: print 'Disbanding (points in cluster too far from each other)'
                for i in range(N):
                    # guarantee that each sample has a unique label
                    if y_node[i] != '_': labels[i] += y_node[i]+str(i)
            return

        if lsize < min_clust_size: 
            label,labels = update_labels(y_node,labels)
            history.append(({label: np.sum(y_node != '_'),
                             label+'L': np.sum(y_node == 'L'),
                             label+'R': np.sum(y_node == 'R')},
                            y_node,Z[j,2],split_depth))
            recursion(Z[j,1]-N,split_depth,labels)
            return
        if rsize < min_clust_size: 
            label,labels = update_labels(y_node,labels)
            history.append(({label: np.sum(y_node != '_'),
                             label+'L':np.sum(y_node == 'L'),
                             label+'R':np.sum(y_node == 'R')},
                            y_node,Z[j,2],split_depth))
            recursion(Z[j,0]-N,split_depth,labels)
            return
        
        # Given a data matrix and some labels, fit a classifier. Evaluate classification accuracy
        scount[0] += 1
        genes,rankings,score,comparisons = split_evaluator(X[y_node!='_',:],y_node[y_node!='_'],
                                                           return_score=True,return_comparisons=True)
        if verbose: print ' Split score '+sn(score)

        # Decide if we should keep the split based on the score
        if score > score_threshold:
            label,labels = update_labels(y_node,labels)
            history.append(({label: np.sum(y_node != '_'),
                             label+'L': np.sum(y_node == 'L'),
                             label+'R': np.sum(y_node == 'R')},
                             y_node, Z[j,2], split_depth+1,
                             genes, rankings, score, comparisons))
            recursion(Z[j,0]-N,split_depth+1,labels)
            recursion(Z[j,1]-N,split_depth+1,labels)
        else: 
            if np.sum(flatten_distance_matrix(D,y_node!='_') < disband_threshold) == 0:
                if verbose: print 'Disbanding (points in cluster too far from each other)'
                for i in range(N):
                    if y_node[i] != '_': labels[i] += y_node[i]+str(i)
            return
        
    recursion(len(Z)-1,0,labels)
    if verbose: 
        print '# of times score function was called: %d'%(scount[0])
        print 'Total computational time was %.3f s'%(time.time()-start_time)
    return np.array(labels),history

# calculate full dendrogram
def plot_dendro(D,sample_names=None,p=None,labels=None,save_name=None,method='complete',lr=90,return_cell_order=False,fig_dim=(100,10),font_size=7):
    '''
    Calculate the full dendrogram based on a distance matrix D
    
    Inputs: 
    D: distance matrix (NxN)
    sample_names: name for each sample (uses the index by default, length-N vector)
    p: depth to cut the tree (goes to leaves by default)
    labels: a label for each sample (also a length-N vector, good for indicating cluster IDs)
    method: linkage methods for scipy's hierarchical clustering algorithm
    lr: leaf rotation
    return_cell_order: boolean. If true, returns the indices of the samples in the order of the dendrogram
    fig_dim: tuple indicating the size of the figure
    font_size: font size of leaves (7 by default)
    '''
    if p is None: p = len(D)
    if sample_names is None: sample_names = range(len(D))
    if labels is not None:
        Ncolors = len(np.unique(labels))
        HSVs = [(x*1.0/Ncolors, 0.8, 0.9) for x in range(Ncolors)]
        RGBs = map(lambda x: colorsys.hsv_to_rgb(*x), HSVs)
        color_map = {i:RGBs[j] for j,i in enumerate(np.unique(labels))}
        colors = map(lambda x:color_map[x],labels)
    # Convert n-by-n D to (n choose 2)-by-1 vector for scipy's linkage function
    Ds = squareform(D)
    # 'complete' merges clusters based on farthest points in two clusters
    # 'single' merges clusters based on closest points in two clusters
    # 'average' merges clusters based on the average pairwise distance between all points in two clusters
    # 'weighted' merges clusters A and B based on the average distance of the two clusters that were merged to become A
    Z = linkage(Ds,method=method)
    plt.figure(figsize=fig_dim)
    D = dendrogram(Z,truncate_mode='lastp',p=p,leaf_rotation=lr,leaf_font_size=font_size,labels=sample_names,)
    ax = plt.gca()
    if labels != None:
        xlbls = ax.get_xmajorticklabels()
        for i in range(len(xlbls)):
            xlbls[i].set_color(colors[int(float(xlbls[i].get_text()[0:]))])
    if save_name is not None: plt.savefig(save_name+'.png', format='png', dpi=300)
    if return_cell_order: return np.array([int(i.get_text()) for i in ax.get_xmajorticklabels()])

        
# Visualize how the splitting progresses
def visualize_history(X,x1,x2,genes,history,save_name=None,axisoff=False,
                      markersize=5,select_inds=None,thresh=0,save_separately=False):
    prefix = 'Split'
    ii = 1
    if type(history[0]) != tuple: 
        history = history[2:]
        prefix = 'Merge'
    for i in history: 
        if len(i) < 8: continue
        if i[-2] < thresh: continue
        if save_separately:
            plt.figure(figsize=(5,5))
        else:
            plt.figure(figsize=(16,4))
            plt.subplot(1,3,1)
        plot_labels_legend(x1,x2,str_labels_to_ints(i[1]),markersize=markersize,select_inds=select_inds)
        plt.title('%s %d: '%(prefix,ii)+'  '.join(['%s(%d): '%(genes[i[4][j]],i[7][j])+sn(i[5][j]) for j in range(len(i[7])) if j < 3]))
        if save_separately:
            if save_name is not None:
                plt.savefig(save_name+'_candidateclusts_'+str(ii)+'.png', format='png', dpi=300, bbox_inches='tight')
            plt.figure(figsize=(5,5))
        else: plt.subplot(1,3,2)
        if select_inds is not None:
            expr = X[select_inds,i[4][0]]
        else: expr = X[:,i[4][0]]
        plt.scatter(x1,x2,edgecolors='none',c=expr)
        plt.title('Expression of '+genes[i[4][0]])
        _ = plt.axis('off')
        if save_separately:
            if save_name is not None:
                plt.savefig(save_name+'_expr_'+str(ii)+'.png', format='png', dpi=300, bbox_inches='tight')
            plt.figure(figsize=(5,5))
        else:
            plt.subplot(1,3,3)
        v0 = X[i[1]=='L',i[4][0]]
        v1 = X[i[1]=='R',i[4][0]]
        bins = np.histogram(np.hstack((v0,v1)), bins=20)[1]
        data = [v0,v1]
        plt.hist(data,bins,label=['0','1'],alpha=0.8,color=['r','g'],normed=True,edgecolor='none')
        plt.xlabel('expression')
        if axisoff: plt.axis('off')
        plt.title('Expression of '+genes[i[4][0]])
        ii += 1
        if save_name is not None: 
            if save_separately:
                plt.savefig(save_name+'_exprhist_'+str(ii)+'.png', format='png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(save_name+'_'+str(ii)+'.png', format='png', dpi=300, bbox_inches='tight')
        
# Print the history in an interpretable way
def print_history(genes,history):
    prefix = 'Pre-split'
    if type(history[0]) != tuple:
        print '%d of %d samples are singletons'%(np.sum(np.bincount(str_labels_to_ints(history[0]))==1),len(history[0]))
        print_singleton_merging_result(history[0],history[1])
        history = history[2:]
        prefix = 'Post-merge'
    for i in history:
        if len(i) < 8: continue
        if type(i[0]) == dict:
            nP = max(i[0].values())
            for key in i[0]:
                if i[0][key] < nP and key[-1] == 'L': nL = i[0][key]
                if i[0][key] < nP and key[-1] == 'R': nR = i[0][key]
            print prefix+": %-5s  L: %-5s  R: %-5s  Score: "%(nP,nL,nR)+ \
                  sn(i[6])+"  Top Gene: %-10.10s  Top Gene Score: "%(genes[i[4][0]])+sn(i[5][0])

# Print more information on how singletons were handled
def print_singleton_merging_result(L1,L2):
    # Make sure L1 is the result of merging singletons (has less unique clusters)
    if len(np.unique(L1)) > len(np.unique(L2)): L1,L2 = L2,L1
    outliers = [str(i) for i in np.where(L1 == -1)[0]]
    print 'Singleton(s) '+', '.join(outliers)+' marked as outliers (N = %d)'%(len(outliers))
    for c in np.unique(L1):
        if c != -1:
            s = ''
            large_clust = None
            Ltemp = L2[L1 == c]
            if len(np.unique(Ltemp)) > 1:
                for j in np.unique(Ltemp): 
                    if np.sum(Ltemp == j) == 1: s += str(j)+', ' 
                    else: large_clust = j
                if large_clust is not None:
                    print 'Singleton(s) '+s[:-2]+' merged with cluster %d (N = %d) to form cluster %d (N = %d)' \
                          %(large_clust,np.sum(Ltemp == large_clust),c,np.sum(L1==c))
                else:
                    print 'Singleton(s) '+s[:-2]+' merged to form cluster %d (N = %d)' \
                          %(c,np.sum(L1==c))

# Analyze a split
def analyze_split(X,x1,x2,genes,history,split_num,num_genes=12,clust=None,show_background=True):
    prefix = 'Split'
    if type(history[0]) != tuple: 
        history = history[2:]
        prefix = 'Merge'

    ii = 0
    for k,i in enumerate(history):
        if len(i) == 8: ii += 1
        if ii == split_num: break

    i = history[k]
    y = str_labels_to_ints(i[1])
    important_gene_indices = i[4]
    scores = i[5]

    comparisons= i[7]

    if clust is not None:
        if num_genes > np.sum(comparisons == clust): 
            print 'num_genes is greater than the number of genes more highly expressed in clust %d'%(clust)
            num_genes = np.sum(comparisons == clust)
        important_gene_indices = important_gene_indices[comparisons == clust]
        scores = scores[comparisons == clust]
        comparisons = comparisons[comparisons == clust]

    plot_labels_legend(x1,x2,y)

    if show_background is False:
        x1 = x1[y != 2]
        x2 = x2[y != 2]
        X = X[y != 2,:]
        
    plt.title('%s %d: '%(prefix,split_num))
    for j in range(num_genes):
        if j%4 == 0: plt.figure(figsize=(16,4))
        plt.subplot(1,4,j%4+1)
        plt.scatter(x1,x2,edgecolors='none',c=X[:,important_gene_indices[j]])
        plt.title(genes[important_gene_indices[j]]+' (%d): '%(comparisons[j])+sn(scores[j]))
        _ = plt.axis('off')

# Select features from history
def feature_extraction_via_split_depth(history):
    feature_lists = []
    for i in history:
        feature_lists.append(i[1][:np.max([20-i[-1],1])])
    return reduce(np.union1d,feature_lists)

# Remove entries from shistory that would not be there if the splitting step was performed 
# with the specified threshold
def filter_out_extraneous_steps(shistory,threshold):
    shistory_filt = [i for i in shistory if len(i) < 8 or i[6] > threshold]
    a = [sorted(i[0].keys(),key=lambda x:len(x))[0] for i in shistory_filt]
    flags = [False for i in range(len(a))]
    keep_entries = {i:None for i in a}
    # Given a list of strings, for each string s of length N, remove it if s[:-1] is not in the list
    prev_len = len(keep_entries)
    while True:
        for j,i in enumerate(keep_entries.keys()):
            if len(i) > 1 and i[:-1] not in keep_entries: keep_entries.pop(i)
        if prev_len == len(keep_entries): break
        prev_len = len(keep_entries)
    
    for j,i in enumerate(a):
        if i not in keep_entries: flags[j] = True
        
    return [i for j,i in enumerate(shistory_filt) if not flags[j]]

# Perform parameter sweeping using the above function
def get_clusters_from_history(D,shistory,threshold,disband_percentile):
    shistory = filter_out_extraneous_steps(shistory,threshold)
    
    # For a given threshold, recover the clustering from D
    N = len(D)
    labels = ['r' for i in range(N)]
    for q,i in enumerate(shistory): _,labels = update_labels(i[1],labels)
    labels = np.array(labels).astype('S%d'%(max([len(i) for i in labels])+len(str(N))))

    # Remove singletons
    disband_threshold = np.percentile(flatten_distance_matrix(D),disband_percentile)
    for i in np.unique(labels):
        inds = labels == i
        if np.sum(inds) > 1 and np.sum(flatten_distance_matrix(D,inds) < disband_threshold) == 0:
            for j in range(N):
                if inds[j]: labels[j] = labels[j]+str(j)
                    
    return labels
