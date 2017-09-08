import sys,time
import numpy as np
from dendrosplit import split,merge,utils
from scipy.sparse import coo_matrix

# Distance computation
def generate_log_corr_D(X,verbose=True,standardize=False): 
    if verbose: start = time.time()
    if standardize: X = split.log_standardize(X)
    else: X = np.log(X+1)
    D = pairwise_distances(X,metric='correlation',n_jobs=32)
    D[np.isnan(D)] = 2. # Set all nan values to max distance
    D = np.multiply(D,np.ones(np.shape(D))-np.diag(np.ones(len(D)))) # Make sure diagonal is 0
    D = (D+D.T)/2 # Make sure symmetric
    if verbose: print('Distance matrix computed in %.3f s.'%(time.time()-start))
    return D

from sklearn.preprocessing import normalize
def generate_l1_D(X,verbose=True):
    if verbose: start = time.time()
    X = normalize(X,norm='l1')
    D = pairwise_distances(X,metric='l1',n_jobs=32)
    D[np.isnan(D)] = 2. # Set all nan values to max distance
    D = np.multiply(D,np.ones(np.shape(D))-np.diag(np.ones(len(D)))) # Make sure diagonal is 0
    D = (D+D.T)/2 # Make sure symmetric
    if verbose: print('Distance matrix computed in %.3f s.'%(time.time()-start))
    return D
    
# Get embedding
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
def generate_tsne(D,verbose=True):
    if verbose: start = time.time()
    tsne = TSNE(metric="precomputed")
    xtsne = tsne.fit_transform(D)
    if verbose: print('tSNE embedding computed in %.3f s.'%(time.time()-start))
    return xtsne[:,0],xtsne[:,1]

# Compute clusters
def get_clusters(D,X,sthresh=30,mthresh=10,verbose=True,dp=50,split_evaluator=split.log_select_genes_using_Welchs):
    if verbose: start = time.time()
    ys,shistory = split.dendrosplit((D,X),
                                    preprocessing='precomputed',
                                    score_threshold=sthresh,
                                    verbose=verbose,
                                    disband_percentile=dp,
                                    split_evaluator=split_evaluator)
    if mthresh is None: 
        if verbose: print('Clusters (only split step) computed in %.3f s.'%(time.time()-start))
        return ys, shistory, None, None
    ym,mhistory = merge.dendromerge((D,X),ys,score_threshold=mthresh,preprocessing='precomputed',
                                    verbose=verbose,outlier_threshold_percentile=90)
    if verbose: print('Clusters computed in %.3f s.'%(time.time()-start))
    return ys,shistory,ym,mhistory

# Feature (for comparing two clusters of interest)
def selected_tcc_to_genes(tcc,d):
    orig_ec = d['NZEC_to_EC'][int(tcc)]
    ensg = d['EC_to_ENSG'][orig_ec]
    return np.unique([d['ENSG_to_GSYM'][i][0] for i in ensg])

def gene_selection_pairwise(X,tccnames,labels,clusts_of_interest,num_genes=10):
    inds = np.logical_or(labels == clusts_of_interest[0], labels == clusts_of_interest[1])
    tcc_ranks,tcc_scores = split.log_select_genes_using_Welchs(X[inds,:],labels[inds])
    for i,tcc in enumerate(tcc_ranks[0:num_genes]):
        print('TCC %s with score %.3f. Aligns with '%(tcc,tcc_scores[i])+' '.join(selected_tcc_to_genes(int(tccnames[tcc]),d)))
        
def gene_selection_1_vs_rest(X,tccnames,labels,num_genes=10):
    for i in np.unique(labels):
        print('-'*80+'\nLook at label '+str(i))
        labels_temp = (labels == i).astype(int)
        gene_selection_pairwise(X,tccnames,labels_temp,[0,1],num_genes=10)

import pandas as pd
from scipy.stats import itemfreq
def confusion_matrix(yhat,ytrue):
    d = {i:{j:0 for j in np.unique(yhat)} for i in np.unique(ytrue)}
    for i in np.unique(ytrue):
        s = {k:int(l) for k,l in itemfreq(yhat[ytrue == i])}
        for j in s:
            d[i][j] += s[j]
    return pd.DataFrame(d)
