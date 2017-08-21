import numpy as np
import networkx as nx
import community,time,itertools,pickle
from scipy.stats import ttest_ind
from scipy.spatial.distance import squareform
from utils import *
from preprocessing import *
from feature_selection import *

# Clean-up step: For each cluster that's too small, find the cluster that it's closest to and 
# merge those clusters together

# Approach: Pairwise distances between clusters. 
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
    # in case of ties, just pick the first one that's not along a diagonal
    min_inds = np.where(D == np.min(D)) 
    for i in range(len(min_inds[0])):
        if min_inds[0][i] != min_inds[1][i]:
            return np.sort([min_inds[0][i],min_inds[1][i]])

# Compute the distance between clusters using minimum p-value for any gene t-test
def cdist_log_Welchs(X,D,i,j,z,cache):
    X1 = X[z == i]
    X2 = X[z == j]
    # Singleton case: return largest int (let singleton remain as outlier)
    if len(X1) == 1 or len(X2) == 1: return np.nan_to_num(np.inf)
    X = np.vstack((X1,X2))
    y = np.zeros(len(X))
    y[len(X1):] = 1
    _,_,score = log_select_genes_using_Welchs(X,y,return_score=True)
    return cache,score

# Compute the distance between clusters using sigclust
def cdist_log_sigclust(X,D,i,j,z,cache):
    X1 = np.log(1+X[z==i])
    X2 = np.log(1+X[z==j])
    X = np.vstack((X1,X2))
    y = np.zeros(len(X))
    y[len(X1):] = 1
    score = sigclust(X,y)
    return cache,score

# Given bins and some samples, compute the distribution over those bins
def dist_over_bins(x,bins):
    num_bins = len(bins)-1
    dist = np.bincount(np.digitize(x,bins),minlength=num_bins+1)[1:]
    return dist.astype(float)/np.sum(dist)

# Compute the distance between clusters using distribution of correlation bins after merge
def cdist_corr_distributions(X,D,i,j,z,cache,show_plot=False):
    
    num_bins = 20
    if 'bins' not in cache:
        print 'computing bins'
        d = flatten_distance_matrix(D)
        bins = np.percentile(d,np.linspace(0,100,num_bins+1))
        bins[-1] = np.inf
        cache['bins'] = bins
        
    dM = flatten_distance_matrix(D,np.logical_or(z==i,z==j))
    distM = dist_over_bins(dM,cache['bins'])
    distance = np.sum(distM[-num_bins/2:])
    
    if show_plot:
        plt.stem(distM)
        plt.xlim((-1,num_bins))
        plt.title('nL:%d, nR:%d, nM:%d, dist: ' \
                  %(np.sum(z==i),np.sum(z==j),np.sum(z!='_'))+sn(distance))
    
    return cache,distance

# Compute the distance between clusters using proportion of clustering in the bottom percentile
# Does the same thing as above function, only a bit faster
def cdist_corr_percentile(X,D,i,j,z,cache):
    if 'cutoff' not in cache:
        d = flatten_distance_matrix(D)
        cache['cutoff'] = np.percentile(D,50)
    dM = flatten_distance_matrix(D,np.logical_or(z==i,z==j))
    distance = np.sum(dM[dM > cache['cutoff']])
    return cache,distance

# visualize distributions of distances in each cluster
def visualize_within_cluster_distance_distributions(D,y,num_bins=20,num_plots_per_row=3,show_D_dist=False):
    d = flatten_distance_matrix(D)
    bins = np.percentile(d,np.linspace(0,100,num_bins+1))
    if show_D_dist:
        plt.figure(figsize=(16,4))
        h = plt.hist(d,bins=num_bins)
        for i in range(num_bins+1):
            plt.plot([bins[i],bins[i]],[0,np.max(h[0])*1.1],'r')
        plt.ylim([0,np.max(h[0])*1.1])
        plt.title('Histogram of distances for all pairs of points')
    bins[-1] = np.inf
    j = 0
    for i in np.unique(y):
        if j%num_plots_per_row == 0: plt.figure(figsize=(16,2))
        if np.sum(y==i) == 1: continue
        plt.subplot(1,num_plots_per_row,j%num_plots_per_row+1)
        dM = flatten_distance_matrix(D,y==i)
        distM = dist_over_bins(dM,bins)
        plt.stem(distM)
        plt.xlim((-1,num_bins))
        plt.title('C:%d, N=%d, score='%(i,np.sum(y==i))+sn(np.sum(distM[-num_bins/2:])))
        j += 1

# Map labels to integers between 0 and #labels-1
def remap_labels(L):
    c = np.unique(L)
    mapper = {c[i]:i for i in range(len(c))}
    return np.array([mapper[i] for i in L])

# Merge function for handling singletons: assign to same cluster as nearest neighbor
def assign_singletons(X,y,preprocessing=log_correlation,verbose=False,
                      allow_singleton_pairs=True,threshold_percentile=None):
    z = str_labels_to_ints(y)
    z_counts = np.bincount(z)
    Ns = np.sum(z_counts == 1)
    if verbose: print '%d of %d samples are singletons'%(Ns,len(y))
    if Ns == 0: return z
    if preprocessing != 'precomputed': 
        D = preprocessing(X)
        D_backup = np.copy(D)
    else: D = X

    # Save a list of outliers (points that are too far from all other points according to some threshold)
    # by putting them in cluster "-1"
    if threshold_percentile is None: threshold = np.max(D)
    else: threshold = np.percentile(flatten_distance_matrix(D),threshold_percentile)
    if verbose: print 'Outlier threshold is '+sn(threshold)

    # Truncate distance matrix so that is a Ns-by-N matrix of distances
    sing_ind = np.zeros(len(y))
    for i in range(len(z_counts)):
        if z_counts[i] == 1: sing_ind[z == i] = 1
    sing_ind = sing_ind.astype(bool)
    sing_to_orig_map = np.array(range(len(y)))[sing_ind]
    if allow_singleton_pairs:
        for i in range(len(D)): D[i,i] = np.inf # Ignore diagonal
    D = D[sing_ind,:]

    # Account for case where we don't want a singleton being assigned to another singleton
    if not allow_singleton_pairs:
        nonsing_to_orig_map = np.array(range(len(y)))[np.invert(sing_ind)]
        D = D[:,np.invert(sing_ind)]

    # Grab nearest neighbor for each singleton
    nneighbor = np.argmin(D,1)

    def is_singleton_str(x,sid):
        if sid[x]: return ' (singleton)'
        return ''

    if allow_singleton_pairs:
        nneighbor_dists = np.min(D,1)
        for(d,i) in sorted([(dn,j) for j,dn in enumerate(nneighbor_dists)]):
            if verbose: print '%d\'s nearest neighbor: %d%s in cluster %d (D = %.3f)' \
                              %(sing_to_orig_map[i],nneighbor[i],
                                is_singleton_str(nneighbor[i],sing_ind),
                                z[nneighbor[i]],D[i,nneighbor[i]])
            if D[i,nneighbor[i]] < threshold:
                z[z == z[sing_to_orig_map[i]]] = z[nneighbor[i]]
            else: z[z == z[sing_to_orig_map[i]]] = -1

    else:
        for i in range(len(nneighbor)):
            if verbose: print '%d\'s nearest neighbor: %d%s in cluster %d (D = %.3f)' \
                              %(sing_to_orig_map[i],nneighbor[i],
                                is_singleton_str(nneighbor[i],sing_ind),
                                z[nneighbor[i]],D[i,nneighbor[i]])
            if D[i,nneighbor[i]] < threshold:
                z[sing_to_orig_map[i]] = z[nonsing_to_orig_map[nneighbor[i]]]
            else: z[z == z[sing_to_orig_map[i]]] = -1
    if verbose: print 'Total number of outliers: %d'%(np.sum(z==-1))
    z_out = remap_labels(z)
    if np.sum(z==-1) > 0: z_out -= 1
    if preprocessing != 'precomputed': return D_backup,z_out
    return z_out
    
def dendromerge(X,y,score_threshold=None,preprocessing=log_correlation,clust_dist=cdist_log_Welchs,
                verbose=False,split_evaluator=log_select_genes_using_Welchs,save_prefix=None,
                outlier_threshold_percentile=100,perform_community_detection=False,return_Dc_history=False):
    '''
    Clean-up step following the splitting step. Merges clusters basd on some criteria.

    Inputs:
    X: input matrix of molecule counts. (D,X) tuple if distance matrix precomputed
    y: labels of clusters 
    score_threshold: threshold of score to determine if we should stop merging. i.e. if all clusters
       are at least this far away from each other, stop merging clusters
    preprocessing: function to generate distance matrix from X
    clust_dist: function to determine the distance between clusters
    verbose: whether or not to print the details of the merging phase
    split_evaluator: function to determine how well separated two clusters are (used for history)
    save_prefix: if not None, then the pairwise cluster distance matrix is saved
    outlier_threshold_percentile: if a singleton's distance to its closest point is in the top
       1-outlier_threshold_percentile of overall distances, then mark the singleton as an outlier
    perform_community_detection: boolean to determine whether or not to merge using community
       detection on the pairwise cluster distance matrix instead

    Outputs:
    ym: labels after merging
    history: first entry is the set of original labels, second entry is the set of labels after
       assigning singletons. Every entry after that tracks some information about each merge in the
       same way as the splitting step.
    '''

    start_time = time.time()

    if score_threshold is None and perform_community_detection is False:
        print 'Please enter a score threshold for the score_threshold keyword'
        return

    if preprocessing != 'precomputed':
        D,z = assign_singletons(X,y,preprocessing=preprocessing,verbose=verbose,
                                threshold_percentile=outlier_threshold_percentile)
        if verbose: print 'Preprocessing took %.3f s'%(time.time()-start_time)
    elif type(X) != tuple:
        print 'preprocessing = precomputed. Feed in (D,X) tuple as first argument.'
        return
    elif not np.all(np.isclose(X[0],X[0].T)):
        print 'Need a valid distance matrix.'
        return
    else: 
        D,X = X[0],X[1]
        z = assign_singletons(np.copy(D),y,preprocessing=preprocessing,verbose=verbose,
                              threshold_percentile=outlier_threshold_percentile)
    if verbose: print 'Singletons assigned (%.3f s)'%(time.time()-start_time)

    # Handle outliers (0's from previous step)
    outliers = np.where(z==-1)[0]
    non_outliers = np.where(z!=-1)[0]
    D = cut_matrix_along_both_axes(D,z!=-1)
    X = X[z!=-1,:]
    z = remap_labels(z[z!=-1])

    # Save useful things to not have to recompute
    cache = {}

    N_samples = len(outliers)+len(non_outliers)
    Nc = len(np.unique(z))
    Dc = np.zeros((Nc,Nc))
    N,M = np.shape(X)

    # Map outlier-only labels to original labels
    def readjust_labels(L,padval):
        L_adjust = np.array([padval for i in range(N_samples)])
        for i,non_outlier in enumerate(non_outliers):
            L_adjust[non_outlier] = L[i]
        return L_adjust

    history = [str_labels_to_ints(y),readjust_labels(z,-1)]

    # Compute distances for all clusters
    for i,j in itertools.combinations(range(Nc),2):
        cache,Dc[i,j] = clust_dist(X,D,i,j,z,cache)
        Dc[j,i] = Dc[i,j]

    if save_prefix is not None: pickle.dump(Dc,file(save_prefix+'_Dc.pickle','w'))
    if verbose: print 'Dc generated (%.3f s)'%(time.time()-start_time)

    # Edge case
    if len(Dc) < 2: return readjust_labels(z,-1),history

    # Merge using community detection
    if perform_community_detection:

            
        # Convert distance matrix to edge tuples for networkx
        def D_to_squareform_with_thresholding(D,thresh=None):
            indices = list(itertools.combinations(range(0,len(D)),2))
            if thresh is None: thresh = np.percentile(flatten_distance_matrix(D),100./np.sqrt(len(D)))
            else: thresh = np.percentile(flatten_distance_matrix(D),thresh)
            D_squareform = []
            for j,i in enumerate(squareform(D)):
                if i < thresh: D_squareform.append(indices[j])
            return D_squareform
        
        edge_weights = D_to_squareform_with_thresholding(Dc)
        G = nx.Graph()
        G.add_nodes_from(np.unique(z))
        G.add_edges_from(edge_weights)
        if verbose: print 'Graph constructed with %d nodes and %d edges.. (%.3f s)' \
                    %(len(G.nodes()),len(G.edges()),time.time()-start_time)
        partition = community.best_partition(G)
        if verbose: print 'Merging clusters took %.3f s'%(time.time()-start_time)
        return readjust_labels(np.array([partition[i] for i in z]),-1),history

    # Original merge algorithm: iteratively merge the two clusters with the smallest distance
    merge_depth = 1
    while len(Dc) > 1:
        merge_inds = find_min(Dc)
        if Dc[merge_inds[0],merge_inds[1]] > score_threshold: break

        if verbose:
            print 'Before the merge: %d clusters'%(Nc)
            print 'Merging labels %d (N = %d) and %d (N = %d) with distance ' \
                  %(merge_inds[0],np.sum(z==merge_inds[0]),merge_inds[1],
                    np.sum(z==merge_inds[1])) + sn(Dc[merge_inds[0],merge_inds[1]])
        
        # Get info for history
        y_node = np.array(['_' for i in range(N)])
        y_node[z == merge_inds[0]] = 'L'
        y_node[z == merge_inds[1]] = 'R'
        genes,rankings,score,comparisons = split_evaluator(X[y_node!='_',:],y_node[y_node!='_'],
                                                           return_score=True,return_comparisons=True)
        history.append(({'m': np.sum(y_node != '_'),
                         'mL': np.sum(y_node == 'L'),
                         'mR': np.sum(y_node == 'R')},
                         readjust_labels(y_node,'_'),Dc[merge_inds[0],merge_inds[1]],merge_depth,
                         genes, rankings, score, comparisons))
            
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
            cache,dc[i] = clust_dist(X,D,i,Nc-1,z,cache)
        dc = np.matrix(dc)
        Dc = np.vstack((np.hstack((Dc,dc.T)),np.hstack((dc,np.matrix(0)))))
        
        merge_depth += 1

    # Readjust z to include the original outliers
    if verbose: print 'Merging clusters took %.3f s'%(time.time()-start_time)
    return readjust_labels(z,-1),history
