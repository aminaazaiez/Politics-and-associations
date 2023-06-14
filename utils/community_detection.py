from sknetwork.clustering import*
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import entropy
from utils.general import array2dict
from collections import Counter
import pandas as pd

import sys 
sys.path.insert(1, '/home/azaiez/Documents/Cours/These/Politics and Associations/Programs')
from utils.general import check_algo_name


## Clustering


def Clustering(adjacency_matrix, algo_name, res =1, random_state =0, get_edge_label = False):
    '''  '''
    louvain = Louvain(resolution = res, modularity ='Newman' , shuffle_nodes = True, random_state = random_state)
    # get partition
    louvain.fit(adjacency_matrix)
    if algo_name == 'Louvain_b':
        if get_edge_label :
            return(louvain.labels_row_ , louvain.labels_col_)
        else:
            return(louvain.labels_row_ )
    elif algo_name == 'Louvain_g':
        return(louvain.labels_)
        
def modularity(adjacency_matrix, algo_name , partition):
    if algo_name == 'Louvain_b':
        return(get_modularity(adjacency_matrix, partition[0], partition[1]))
    elif algo_name =='Louvain_g':
        return(get_modularity(adjacency_matrix, partition))
    else : print('Error')

def partition_with_highest_mod(H, algo_name, nb_itt, return_idx = False):
    ''' Partition the nodes and edges into clusters using the modularity function for bipartite graph proposed by Barder (2007) and the Newmann modularity function for the clique expansion 
    Parameters 
    ---------
    H :
        Hypergraph
    algo_name :str (Louvain_b, Louvian_g)
        Louvain_b for hyergraph clustering
        Louvain_g for clique expansion clustering
    nb_itt: int
    
    Returns 
    ---------
    Partition: 1D array or tuple of 1D array
        the selected partition is the one with the highest modularity'''

    
    partitions = pd.DataFrame(columns = ['seed', 'algo_name', 'modularity' ])
    adjacency_matrix = check_algo_name(H, algo_name)
    for i in range (nb_itt):
        partition = Clustering(adjacency_matrix , algo_name, random_state = i , get_edge_label = True) 
        partitions = pd.concat([partitions, pd.DataFrame({'seed' : i,  'algo_name' :algo_name,  'modularity' : [modularity(adjacency_matrix, algo_name , partition)] } ) ], ignore_index=True )
    max_index = partitions.iloc[partitions['modularity'].idxmax()]['seed']
    if return_idx :
        return(max_index)
    return(Clustering(adjacency_matrix , algo_name, random_state = max_index ))

    
##  Mutual information  
def mutual_inofmation_btw_equal_sized_partitions(partitions):
    mis=[]
    qs =[]
    for q in  set(partitions['q']):
        p_gs = partitions.query("q == %d & algo_name == 'Louvain_g'"%q)['Partition']
        p_bs = partitions.query("q == %d & algo_name == 'Louvain_b'"%q)['Partition']
        for p_g in p_gs:
            for p_b in p_bs:
                
                mis.append(normalized_mutual_info_score(p_g, p_b ))
                qs.append(q)
    return(qs, mis)
## Standard deviation of clusters' volume     
def std_cluster_vol(H, partition, nodes_labels):
    clusters_vol = Counter()
    for i in range (len(partition)):
        try:
            clusters_vol[partition[i]] += H.nodes[nodes_labels[i]].strength
        except:           
            clusters_vol[partition[i]] = H.nodes[nodes_labels[i]].strength

            
    return(np.std([ vol for vol in clusters_vol.values()]) )

    