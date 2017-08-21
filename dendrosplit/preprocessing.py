import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import FastICA,NMF
from sklearn.preprocessing import StandardScaler

# TRANSFORMATIONS

# Remove genes with greater than some number of expression across all cells
# Also remove genes with 'MT' in its name (these are mitochondrial genes)
def filter_genes(X,genes,thresh=0):
    keep_inds_thresh = np.where(np.sum(X,0) > thresh)[0]
    keep_inds_MT = np.array([i for i in range(len(genes)) if 'MT-' not in genes[i].upper()])
    keep_inds = np.intersect1d(keep_inds_thresh,keep_inds_MT)
    print 'Kept %d features for having > %d counts across all cells'%(len(keep_inds),thresh)
    return X[:,keep_inds],genes[keep_inds]

# Downsample the number of cells
def samp_data(X,percent=50):
    n = len(X)
    inds = np.random.choice(n,int(percent/100.*n),replace=False)
    return X[inds,:]

# Map features to rankings
def feature_ranking_map(X):
    X += np.random.normal(0,0.01,np.shape(X))
    Y = np.zeros_like(X)
    for i in range(len(Y.T)):
        Y[:,i] = stats.rankdata(X[:,i], "average")
    return Y

# Normalize rows of data matrix (such that they sum to 1)
def normalize(Z): return np.nan_to_num(np.array(Z/np.matrix(np.sum(Z,1)).T))

# Take log, then normalize columns of data matrix (such that their L2 norms are 1)
def log_unit_norm_features(X):
    return X/np.linalg.norm(X,axis=0).reshape(1,-1)

# Take log, then normalize the variance of each column to 1
def log_stdvar(X):
    Z = np.log(X+1)
    return Z/np.sqrt(np.var(Z,axis=0).reshape(1,-1))

# Take log, the zero-mean and std var each column
def log_standardize(X):
    Z = np.log(X+1)
    ss = StandardScaler()
    return ss.fit_transform(Z)

# First normalize columns, then normalize rows
def normalize_all(X):
    Z = normalize(X.T)
    return normalize(Z.T)

# 10x approach: normalize across cells, multiply each cell with median, log,
#     standardize each gene
def normalizecounts_multiplymedian_log_standardize(X):
    med = np.median(np.sum(X,1))
    Z = normalize(X)*med
    return log_standardize(Z)

# For dropseq approach, do above after performing gene selection
def preprocessing_dropseq(X):
    keep_inds = dropseq_gene_selection(np.log(X+1))
    return normalizecounts_multiplymedian_log_standardize(X[:,keep_inds])

# For 10X approach, do both the above in addition to PCA
def preprocessing_10x(X,state_var_prop=False):
    Z = preprocessing_dropseq(X)
    zpca = sk_pca(X)
    if state_var_prop:
        vs = np.sum(np.square(np.linalg.norm(zpca,axis=1)))
        s = np.sum(np.square(np.linalg.norm(zpca[:,:50],axis=1)))/vs
        print 'Variance explained by first 50 pcs: %.3f'%(s)
    return zpca[:,:50]

# Dropseq gene selection 
def assign_to_bins(v,bins=20):
    last_ind = len(v)-len(v)%bins
    sorted_inds = np.argsort(v)
    bin_assignments = np.split(sorted_inds[:last_ind],bins)
    bin_assignments[-1] = np.append(bin_assignments[-1],sorted_inds[last_ind:])
    assignments = np.zeros(len(v))
    for i,b in enumerate(bin_assignments):
        assignments[b] = i
    return assignments

def dropseq_gene_selection(X,z_cutoff=1.7,bins=20):
    m = np.mean(X,0)
    assignments = assign_to_bins(m,bins=bins)
    dispersion = np.divide(np.var(X,0),m)
    # select genes to keep based on zscore within assigned bins
    keep_inds = np.zeros(np.shape(X)[1])
    for i in range(bins):
        zscores = stats.mstats.zscore(dispersion[assignments == i])
        keep_inds[assignments == i] = zscores > z_cutoff
    return keep_inds.astype('bool')

# ------------------------------------------------------------------------------
# DIMENSIONALITY REDUCTION

# sklearn's ICA
def sk_ica(X,nc=10):
    ica = FastICA(n_components=nc)
    return ica.fit_transform(X)

# sklearn's tSNE 
def sk_tsne(X,showplot=False,colors=None,title=None):
    model = TSNE(verbose=False)
    X_tSNE = model.fit_transform(X)
    if showplot:
        plot_embedding(X_tSNE[:,0],X_tSNE[:,1],colors,title+' (First two tSNE components)')
    return X_tSNE

# sklearn's tSNE with distance precomputing
def sk_tsne_precompute(D,showplot=False,colors=None,title=None):
    model = TSNE(verbose=False,metric='precomputed')
    X_tSNE = model.fit_transform(D)
    if showplot:
        plot_embedding(X_tSNE[:,0],X_tSNE[:,1],colors,title+' (First two tSNE components)')
    return X_tSNE

# Custom implementation of PCA (using SVD)
def sk_pca(X,showplot=False,colors=None,title=None):
    X_ = np.array(X)
    X_ -= np.mean(X_,0)
    U,s,V = np.linalg.svd(X_,full_matrices=0)
    pcs = np.dot(U,np.diag(s))
    if showplot:
        plt.figure()
        plt.stem(s)
        plt.xlim([0,20]) # Show first 20 singular values
        plt.title(title+' (SVD singular values)')
        plot_embedding(pcs[:,0],pcs[:,1],colors,title+' (First two PCs)')
    return pcs

# Low dimensional embedding using tSNE and PCA
def low_dimensional_embedding(X,pcs=10):
    xpca = sk_pca(X,showplot=False)
    xtsne = sk_tsne(xpca[:,:pcs],showplot=False)
    return xtsne[:,0],xtsne[:,1]

# Function for visualizing tsne or pca
def plot_embedding(x,y,colors,title):
    plt.figure()
    plt.scatter(x,y,edgecolors='none',c=colors)
    plt.colorbar()
    plt.title(title)

def compute_pcs_needed_to_explain_variance(X,percvar,verbose=False):
    U,S,V = np.linalg.svd(X)
    S = np.square(S) # Squared singular value = % variance explained by each pc
    cs = np.cumsum(S)
    thresh = cs[-1]*percvar/float(100)
    if thresh < cs[0]:
        if verbose: print 'First pc explains %.3f%% of the variance'%(100*cs[0]/cs[-1])
        return 1
    else:
        npcs = np.where(cs < thresh)[0][-1] + 1
        if verbose: print 'Number of pcs to explain %d%% of the variance: %d'%(percvar,npcs)  
        return npcs

# ------------------------------------------------------------------------------
# DISTANCE METRICS

def pairwise_L2(X): # L2
    return pairwise_distances(X)

def pairwise_L2_after_PCA(X): # L2 after a dimensionality reduction
    Xpca = sk_pca(X,showplot=False)
    return pairwise_distances(Xpca[:,:5])

def pairwise_L1(X): # L1 after normalizing each sample
    Xnorm = np.nan_to_num(normalize(X))
    return pairwise_distances(Xnorm,metric='l1')

def pairwise_correlation(X): # Pearson correlation 
    return pairwise_distances(X,metric='correlation')

def get_D_percentile(D,percentile):
    return np.percentile(D.reshape(1,-1),percentile)

# ------------------------------------------------------------------------------
# FULL PREPROCESSING OPTIONS (counts --> distance function)

# Log, standardize, PCA, remove PCS correlated with proportion of genes expressed
def log_std_PCA_filterpcs(X):
    X = log_standardize(X)
    # Compute proportion of genes expressed
    cdr = np.sum(X > 0,1)
    # PCA
    U,S,V = np.linalg.svd(X)
    pcs = np.dot(U,np.diag(S))
    # Compute correlation between each PC and CDR
    pc_cdr_corr = np.ndarray.flatten(np.abs(1-pairwise_distances(U.T,cdr.reshape(1,-1),metric='correlation')))
    # Only keep PCs which do not have high correlation with CDR
    S = S[pc_cdr_corr < 0.8]
    pcs = pcs[:,pc_cdr_corr < 0.8]
    print '%d pcs removed for having high correlation with proportion of genes expressed'%(len(U)-len(S))
    cs = np.cumsum(np.square(S))
    thresh = cs[-1]*0.8 # Arbitrarily chose 80% of variance explained
    npcs = np.where(cs < thresh)[0][-1] + 1
    X = pcs[:,:npcs]
    print '%d pcs kept to explain 80%% of the variance'%(np.shape(X)[1])
    return X

# Log, standardize variance, NMF, euclidean
def log_stdvar_NMF_L2(X):
    X = log_stdvar(X)
    k = compute_pcs_needed_to_explain_variance(X,50)
    nmf = NMF(n_components=k)
    Xrd = nmf.fit_transform(X)
    return pairwise_distances(Xrd)

# Log, standardize, PCA, L2
def log_PCA_L2(X):
    X = log_standardize(X)
    X = sk_pca(X,showplot=False)[:,:10]
    return pairwise_distances(X)

# Log, PCA, L2
def log_standardize_PCA_L2(X):
    X = np.log(X+1)
    X = sk_pca(X,showplot=False)[:,:10]
    return pairwise_distances(X)

# Normalize counts, L1
def normalizecounts_L1(X):
    X = normalize(X)
    return pairwise_L1(X)

def log_std_PCA_filterpcs_correlation(X):
    return pairwise_correlation(log_std_PCA_filterpcs(X))

def log_std_PCA_filterpcs_L2(X):
    return pairwise_L2(log_std_PCA_filterpcs(X))

# Log, L1
def log_L1(X):
    return pairwise_L1(np.log(X+1))

# Log, normalize, L1
def log_norm_L1(X):
    X = normalize(np.log(X+1))
    return pairwise_L1(X)

# Log, standardize, L1
def log_standardize_L1(X):
    X = log_standardize(X)
    return pairwise_L1(X)

# Log, unit var, L1
def log_stdvar_L1(X):
    X = log_stdvar(X)
    return pairwise_L1(X)

# Log, correlation
def log_correlation(X):
    return pairwise_correlation(np.log(X+1))

# Log, normalize, correlation
def log_norm_correlation(X):
    X = normalize(np.log(X+1))
    return pairwise_correlation(X)

# Log, standardize, correlation
def log_standardize_correlation(X):
    X = log_standardize(X)
    return pairwise_correlation(X)

# Log, unit var, correlation
def log_stdvar_correlation(X):
    X = log_stdvar(X)
    return pairwise_correlation(X)
