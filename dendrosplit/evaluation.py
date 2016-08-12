import numpy as np
import networkx as nx

# NMI
from sklearn.metrics import normalized_mutual_info_score
def NMI(x,y):
    return normalized_mutual_info_score(x,y)

# Max weight matching
# gets max weight matching of a biparetite graph with row_label x column_label
# (weights are given by weight_matrix)
def get_max_wt_matching(row_label,column_label, weight_matrix):
    # Create a bipartite graph where each group has |unique labels| nodes 
    G = nx.complete_bipartite_graph(len(row_label), len(column_label))
    # Weight each edge by the weight in weight matrix.. 
    for u,v in G.edges(): G[u][v]["weight"]=weight_matrix[u,v-len(row_label)]
    # Perform weight matching using Kuhn Munkres
    H=nx.max_weight_matching(G)
    max_wt=0
    for u,v in H.items(): max_wt+=G[u][v]["weight"]/float(2)
    return max_wt

def compute_clustering_accuracy(label1, label2):
    uniq1,uniq2 = np.unique(label1),np.unique(label2)
    # Create two dictionaries. Each will store the indices of each label
    entries1,entries2 = {},{}
    for label in uniq1: entries1[label] = set(np.flatnonzero((label1==label)))
    for label in uniq2: entries2[label] = set(np.flatnonzero((label2==label)))
    # Create an intersection matrix which counts the number of entries that overlap for each label combination        
    W = np.zeros((len(uniq1),len(uniq2)))
    for i,j in itertools.product(range(len(uniq1)),range(len(uniq2))):
        W[i,j]=len(entries1[uniq1[i]].intersection(entries2[uniq2[j]]))
    # find the max weight matching
    match_val = get_max_wt_matching(uniq1,uniq2,W)
    # return the error rate
    return match_val/float(len(label1))
