from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import numpy as np
import matplotlib.pyplot as plt
import time,itertools

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

# Dendrosplit algorithm
def dendrosplit(X,preprocessing=log_correlation,min_clust_size=5,score_threshold=0.9,
                method='complete',verbose=False,split_evaluator=select_genes_using_LR):
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
            
        Outputs:
            y: cluster labels 
            history: list of tuples of information generated at each split. Use the functions
                'print_history' and 'visualize_history' with this output
    '''
    
    start_time = time.time()
    
    # Get linkage array
    if preprocessing != 'precomputed': 
        D = preprocessing(X)
        if verbose: print 'Preprocessing took %.3f s'%(time.time()-start_time)
    elif type(X) != tuple: 
        print 'preprocessing = precomputed. Feed in (D,X) tuple as first argument.'
        return
    elif not np.all(np.isclose(X[0],X[0].T)):
        print 'Need a valid distance matrix.'
        return
    else: D,X = X[0],X[1]
    D = (D+D.T)/2
    Ds = squareform(D)
    Z = linkage(Ds,method=method)
    N = len(X)
    
    # start with the label 'r' (for 'root') for all samples
    labels = ['r' for i in range(N)]
    
    # Stuff we may want to remember about each stage of splitting:
    #  - labels before the split and # of labels
    #  - labels after the split and # of each labels
    #  - splitting score
    #  - important features for distingushing the two subtypes
    #  - if a cluster is deemed too small
    history,scount = [],[0]
    
    def recursion(j,label):
        j = int(j)
        
        # Check number of leaves below node exceeds required threshold
        # Or if we're at a leaf node
        if j < 0: return
        if Z[j,3] < min_clust_size: return
        
        # 1. Split subtrees and get labels
        y_node = node_to_leaf_labels(Z,j,ltypes=['L','R'])
        
        lsize = np.sum(y_node == 'L')
        rsize = np.sum(y_node == 'R')
        if verbose: print 'Potential split result: %d and %d'%(lsize,rsize)
        
        # If the split produces a cluster that's too small, ignore the smaller cluster
        if lsize < min_clust_size and rsize < min_clust_size: return
        if lsize < min_clust_size: 
            recursion(Z[j,1]-N,label)
            return
        if rsize < min_clust_size: 
            recursion(Z[j,0]-N,label)
            return
        
        # 2. Given a data matrix and some labels, fit a classifier. Evaluate classification accuracy
        scount[0] += 1
        genes,rankings,score = split_evaluator(np.log(1+X[y_node!='_',:]),y_node[y_node!='_'],return_score=True)
        if verbose: print ' Split score '+sn(score)

        # 3. Decide if we should keep the split based on the score
        if score > score_threshold:
            for i in range(N): 
                if y_node[i] != '_': 
                    labels[i] += y_node[i] # Update labels
            history.append(({label: np.sum(y_node != '_'),
                             label+'L': np.sum(y_node == 'L'),
                             label+'R': np.sum(y_node == 'R')},
                             genes, rankings, score, y_node))
            recursion(Z[j,0]-N,label+'L')
            recursion(Z[j,1]-N,label+'R')
        else: 
            return
        
    recursion(len(Z)-1,'r')
    if verbose: 
        print '# of times score function was called: %d'%(scount[0])
        print 'Total computational time was %.3f s'%(time.time()-start_time)
    return np.array(labels),history

# calculate full dendrogram
def plot_dendro(D,p,labels,colors=None,save_name=None,method='complete',lr=0):
    # Convert n-by-n D to (n choose 2)-by-1 vector for scipy's linkage function
    Ds = squareform(D)
    # 'complete' merges clusters based on farthest points in two clusters
    # 'single' merges clusters based on closest points in two clusters
    # 'average' merges clusters based on the average pairwise distance between all points in two clusters
    # 'weighted' merges clusters A and B based on the average distance of the two clusters that were merged to become A
    Z = linkage(Ds,method=method)
    plt.figure(figsize=(100,10))
    D = dendrogram(Z,truncate_mode='lastp',p=p,leaf_rotation=lr,leaf_font_size=7.,labels=labels,)
    ax = plt.gca()
    if colors != None:
        xlbls = ax.get_xmajorticklabels()
        for i in range(len(xlbls)):
            xlbls[i].set_color(colors[int(float(xlbls[i].get_text()[0:]))])
    if save_name is not None: plt.savefig(save_name+'.png', format='png', dpi=300)
        
def str_labels_to_inds(y_str):
    y_int = np.zeros(len(y_str))
    for i,label in enumerate(np.unique(y_str)):
        y_int[y_str == label] = i
    return y_int.astype(int)

# Visualize how the splitting progresses
def visualize_history(X,x1,x2,genes,history,save_name=None):
    for ii,i in enumerate(history): 
        plt.figure(figsize=(16,6))
        plt.subplot(1,2,1)
        plot_labels_legend(x1,x2,str_labels_to_inds(i[4]))
        plt.title('  '.join(['%s: '%(genes[i[1][j]])+sn(i[2][j]) for j in range(len(i[1])) if j < 4]))
        plt.subplot(1,2,2)
        plt.scatter(x1,x2,edgecolors='none',c=X[:,i[1][0]])
        plt.title('Expression of '+genes[i[1][0]])
        _ = plt.axis('off')
        if save_name is not None: plt.savefig(save_name+'_'+str(ii)+'.png', format='png', dpi=300)
        
# Print the history in an interpretable way
def print_history(genes,history):
    for i in history:
        if type(i[0]) == dict:
            nP = max(i[0].values())
            for key in i[0]:
                if i[0][key] < nP and key[-1] == 'L': nL = i[0][key]
                if i[0][key] < nP and key[-1] == 'R': nR = i[0][key]
            print "Pre-split: %-5s  Left: %-5s  Right: %-5s  Score: "%(nP,nL,nR)+ \
                  sn(i[3])+"  Top Gene: %-10.10s  Top Gene Score: "%(genes[i[1][0]])+sn(i[2][0])

# Clean-up step: For each cluster that's too small, find the cluster that it's closest to and 
# merge those clusters together

# Approach 1: Pairwise distances between clusters. 
# Distance = mininum p-value between the two?
# If two clusters are similar, they should have a small distance. There will be few genes that distinguish the two
# and a small -log10 p-value will be generated (distributions of all genes are similar). 

# 1. Look at the pool of small clusters. 
# 2. Choose the one that has the lowest distance to any cluster. 
# 3. Merge those clusters. Recompute the matrix
# 4. Repeat

def find_min(D):
    n = len(D)
    D = np.multiply(D,np.tril(np.ones([n,n]),-1))+np.max(D)*np.triu(np.ones([n,n]))
    min_inds = np.where(D == np.min(D)) # in case of ties, just pick the first one
    return np.sort([min_inds[0][0],min_inds[1][0]])

def clust_dist(X1,X2):
    # Singleton case: return largest int (let singleton remain as outlier)
    if len(X1) == 1 or len(X2) == 1: return np.nan_to_num(np.inf)
    t,p = ttest_ind(np.log(1+X1),np.log(1+X2),equal_var=False)
    t[np.isnan(t)] = 0
    t[np.isinf(t)] = 0 # Ignore cases where t is NaN or Inf due to variance issues
    t = np.abs(t)
    i = np.where(t == np.max(t))[0][0]
    return np.nan_to_num(-np.log10(p[i]))

def remap_labels(L):
    c = np.unique(L)
    mapper = {c[i]:i for i in range(len(c))}
    return np.array([mapper[i] for i in L])

def merge_cleanup(X,y,score_threshold,verbose=False):
    z = str_labels_to_inds(y)
    Nc = len(np.unique(z))
    Dc = np.zeros((Nc,Nc))
    start_time = time.time()
    
    # Compute distances for all clusters
    for i,j in itertools.combinations(range(Nc),2):
        Dc[i,j] = clust_dist(X[z==i,:],X[z==j,:])
        Dc[j,i] = Dc[i,j]
        
    if verbose: print '%.3f s to generate Dc'%(time.time()-start_time)

    while True:

        # Merge together the two clusters with the smallest distance
        merge_inds = find_min(Dc)
        if Dc[merge_inds[0],merge_inds[1]] > score_threshold: break

        if verbose:
            print 'Before the merge: %d clusters'%(Nc)
            print 'Merging labels %d (N = %d) and %d (N = %d) with distance %.3f' \
                  %(merge_inds[0],np.sum(z==merge_inds[0]),merge_inds[1],
                    np.sum(z==merge_inds[1]),Dc[merge_inds[0],merge_inds[1]])

        # Delete the rows/columns of the distance matrix corresponding to the two merged clusters
        Dc = np.delete(np.delete(Dc,merge_inds,1),merge_inds,0)

        # Merge the clusters
        z[z == merge_inds[0]] = Nc
        z[z == merge_inds[1]] = Nc
        z = remap_labels(z)
        Nc -= 1

        # Create new distance matrix
        dc = np.zeros(Nc-1)
        for i in range(Nc-1):
            dc[i] = clust_dist(X[z==i,:],X[z==Nc-1,:])
        dc = np.matrix(dc)
        Dc = np.vstack((np.hstack((Dc,dc.T)),np.hstack((dc,np.matrix(0)))))

        if verbose: print 'After the merge: %d clusters\n'%(Nc)

    return z
