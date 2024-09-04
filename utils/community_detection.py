from collections import Counter
from sknetwork.clustering import Louvain, get_modularity
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
from utils.general import generate_adjacency_matrix


## Clustering


def clustering(adjacency_matrix, algo_name, res =1, random_state =0, get_edge_label = False):
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
    adjacency_matrix = generate_adjacency_matrix(H, algo_name)
    for i in range (nb_itt):
        partition = clustering(adjacency_matrix , algo_name, random_state = i , get_edge_label = True) 
        partitions = pd.concat([partitions, pd.DataFrame({'seed' : i,  'algo_name' :algo_name,  'modularity' : [modularity(adjacency_matrix, algo_name , partition)] } ) ], ignore_index=True )
    max_index = partitions.iloc[partitions['modularity'].idxmax()]['seed']
    if return_idx :
        return(max_index)
    return(clustering(adjacency_matrix , algo_name, random_state = max_index ))

    
##  Mutual information  
def mutual_information_btw_equaly_sized_partitions(partitions):
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
