import colorsys
import numpy as np
import matplotlib.pyplot as plt

# Load csv file (skipcol = 4 for resolve)
def load_csv(filename,skipcol=4):
    X = np.genfromtxt(filename,delimiter=',',skip_header=1)
    X = X[:,skipcol:]
    print 'We have %d samples and %d features per sample.'%np.shape(X)
    print '%.3f%% of the entries are 0.'%(100*np.sum(X == 0)/float(np.prod(np.shape(X))))
    return X

# Check if all entries in a numpy matrix are whole numbers
def all_entries_are_whole_nums(X):
    a = np.product(np.shape(X))
    b = sum([1 for i in np.ndarray.flatten(X) if float.is_integer(i)])
    return a == b

# When printing, decide whether to use scientific notation (if value gets too big)
def sn(i,j=2):
    if i > 1000 or i < 0.01: return '%.1E'%(i)
    s = '%.'+str(j)+'f'
    return s%(i)

# Simple way to plot colors and labels with legend                                        
#  -if x1, x2 is just a subsample of all the points, need select_inds to indicate
#   which points they correspond to
def plot_labels_legend(x1,x2,y_,labels=None,save_name=None,label_singletons=True,
                       show_axes=False,xlog_scale=False,ylog_scale=False,
                       legend_pos=(1.4, 1.0),markersize=5,select_inds=None):

    if select_inds is not None: y = y_[select_inds]
    else: y = y_ 

    N = len(np.unique(y))
    Ncolors = N+1
    if label_singletons:
        for i,c in enumerate(np.unique(y)):
            if np.sum(y == c) == 1: 
                y[y == c] = -1
                Ncolors -= 1
    if labels == None: labels = np.unique(y)
    if -1 in labels or '-1' in labels: Ncolors -= 1
    HSVs = [(x*1.0/Ncolors, 0.8, 0.9) for x in range(Ncolors)]
    RGBs = map(lambda x: colorsys.hsv_to_rgb(*x), HSVs)
    j = 0
    for i in labels:
        if i != -1 and i != '-1':
            plt.plot(x1[y==i],x2[y==i],'.',c=RGBs[j],label=str(i)+' ('+str(np.sum(y_==i))+')',
                     markersize=markersize)
            j += 1
        else:
            plt.plot(x1[y==i],x2[y==i],'*',c='w',label='Singletons ('+str(np.sum(y_==i))+')',
                     markersize=markersize,markeredgecolor='k')
    if not show_axes: _ = plt.axis('off')
    if xlog_scale: plt.xscale('log')
    if ylog_scale: plt.yscale('log')
    if legend_pos is not None: plt.legend(bbox_to_anchor=legend_pos)
    if save_name is not None: plt.savefig(save_name+'.png', format='png', dpi=300)

# For each feature, determine the index of the cluster with the greatest expression
def compare_feature_means(X,Y):
    Nc = len(np.unique(Y))
    N,M = np.shape(X)
    m = np.zeros((Nc,M))
    for i,c in enumerate(np.unique(Y)):
        m[i,:] = np.mean(X[Y == c,:],0)
    return np.argmax(m,0)

# Map labels to integers
def str_labels_to_ints(y_str):
    y_int = np.zeros(len(y_str))
    for i,label in enumerate(np.unique(y_str)):
        y_int[y_str == label] = i
    return y_int.astype(int)

# Get all off-diagonal entries in a distance matrix
def flatten_distance_matrix(D,inds=None):
    if inds is not None: D2 = cut_matrix_along_both_axes(D,inds)
    else: D2 = D
    d = D2.reshape(-1,1)
    return np.delete(d,np.where(d==0)[0])

# index a 2D matrix along both axes
def cut_matrix_along_both_axes(X,inds):
    Z = X[inds,:]
    return Z[:,inds]

