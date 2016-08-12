import random,string,pyclust,itertools,sys,os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import SpectralClustering,DBSCAN,KMeans,AgglomerativeClustering,AffinityPropagation
from sklearn.neighbors import kneighbors_graph

# CLUSTERING ALGORITHMS

# eps: maximum distance allowed between two points in the same neighborhood
# min_samples: # of samples in a neighborhood for a point to be considered "core"
def skDBSCAN(D,eps=0.05,min_samples=5):
    dbscan = DBSCAN(eps=eps,min_samples=min_samples,metric='precomputed')
    return dbscan.fit_predict(D)

# D NEEDS TO BE A SIMILARITY/AFFINITY MATRIX 
# pref: as preference increases, so does the # of exemplars (and # of clusters)
# damp: higher damping = higher weight on older messages 
def AffinityProp(D,pref,damp):
    aff=AffinityPropagation(affinity='precomputed',preference=pref,damping=damp)
    labels=aff.fit_predict(D)
    return labels

# hierarchical clustering with connectivity information
def hclust(D,k,compute_connectivity_info = False):
    if compute_connectivity_info:
        knn_graph = kneighbors_graph(D, 7, include_self=False, metric='precomputed')
        hc = AgglomerativeClustering(n_clusters=k,connectivity=knn_graph,
                                     affinity='precomputed',linkage='complete')
    else:
        hc = AgglomerativeClustering(n_clusters=k,affinity='precomputed',linkage='complete')
    return hc.fit_predict(D)

# This one is a bit different. Cannot use a distance matrix
def kMedoids(X,k,d_metric='euclidean'):
    km = pyclust.KMedoids(n_clusters=k,distance=d_metric)
    return km.fit_predict(X)

# Kmeans
def kMeans(X,k):
    km = KMeans(n_clusters=k)
    return km.fit_predict(X)

# SNN-Cliq clustering algorithm
def SNN_Cliq(D,r=0.7,m=0.5,k=5,save_dist=True,verbose=False,deletelabels=True,SNNdir=None):
    if SNNdir is None:
        print 'Need to set SNNdir to the directory where SNN.R and Cliq.py are saved'
        print '(see directory where this module is saved)'
        return
    suffix = generate_random_string(10)
    distfile='tempDistFile_'+suffix
    edgefile='tempEdgeFile_'+suffix
    labelsfile='labels_'+str(k)+'_'+suffix
    if save_dist: np.savetxt(SNNdir+distfile,D)
    start_time = time.time()
    if verbose: print 'Building SNN graph..'
    os.system('Rscript '+SNNdir+'SNN.R '+SNNdir+distfile+' '+SNNdir+edgefile+' '+str(k))
    if verbose: print 'SNN graph built (%.3f s)'%(time.time()-start_time)
    start_time = time.time()
    if verbose: print 'Finding clusters..'
    os.system('python '+SNNdir+'Cliq.py -i '+SNNdir+edgefile+' -o '+SNNdir+labelsfile
              +' -r '+str(r)+' -m '+str(m)+' -n '+str(len(D)))
    if verbose: print 'Clusters found (%.3f s)'%(time.time()-start_time)
    os.system('rm '+SNNdir+distfile+' '+SNNdir+edgefile)
    y = np.loadtxt(SNNdir+labelsfile,dtype=int)
    os.system('rm '+SNNdir+labelsfile)
    return y

def generate_random_string(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

# ------------------------------------------------------------------------------
# CONSENSUS

# all_labels list of labels. This function finds a consensus amongst them by counting how
# often two points are put in the same cluster
def build_consensus_matrix(all_labels):
    n = len(all_labels[0])
    D = np.zeros((n,n))
    cIDs = np.array(range(n))
    for i,labels in enumerate(all_labels):
        sys.stdout.write("\rbuilding consensus matrix.. %d/%d"%(i+1,len(all_labels)))
        sys.stdout.flush()
        D = dist_update(D,labels,cIDs,np.max(labels)+1)
    # Map similarity matrix to distance matrix
    D = D/float(len(all_labels)) + np.diag(np.ones(len(D))) 
    return 1-D

# function to update similarity matrix counts based on which samples are in the same cluster
def dist_update(D,y,cIDs,k):
    for clustID in range(k):
        for i,j in itertools.combinations(cIDs[y == clustID],2):
            D[i,j] += 1
            D[j,i] += 1
    return D
