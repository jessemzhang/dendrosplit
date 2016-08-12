import colorsys
import numpy as np
import matplotlib.pyplot as plt

# Load csv file for resolve
def load_csv(filename,skipcol=4):
    X = np.genfromtxt(filename,delimiter=',',skip_header=1)
    X = X[:,skipcol:]
    print 'We have %d samples and %d features per sample.'%np.shape(X)
    print '%.3f%% of the entries are 0.'%(100*np.sum(X == 0)/float(np.prod(np.shape(X))))
    return X

# Check if all entries in a numpy matrix are whole numbers
def check_if_all_entries_are_whole(X):
    a = np.product(np.shape(X))
    b = sum([1 for i in np.ndarray.flatten(X) if float.is_integer(i)])
    return a == b

# When printing, decide whether to use scientific notation (if value gets too big)
def sn(i):
    if i > 1000: return '%.2E'%(i)
    return '%.3f'%(i)

# Simple way to plot colors and labels with legend                                        
def plot_labels_legend(x1,x2,y,labels=None,save_name=None):
    N = len(np.unique(y))
    if labels == None: labels = range(N)
    HSVs = [(x*1.0/N, 0.8, 0.9) for x in range(N)]
    RGBs = map(lambda x: colorsys.hsv_to_rgb(*x), HSVs)
    for i in np.unique(y):
        plt.plot(x1[y==i],x2[y==i],'.',c=RGBs[i],label = str(labels[i])+' ('+str(np.sum(y==i))+')')
    _ = plt.axis('off')
    plt.legend(bbox_to_anchor=(1.4, 1.0))
    if save_name is not None: plt.savefig(save_name+'.png', format='png', dpi=300)
