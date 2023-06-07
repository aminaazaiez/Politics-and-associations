from sknetwork.data import from_edge_list
from sknetwork.clustering import*
import itertools
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import entropy
from utils.general import array2dict



import sys 
sys.path.insert(1, '/home/azaiez/Documents/Cours/These/Politics and Associations/Programs')
from utils.general import volume
## Create a sknetwork bipartite network and the clique expansion of the hypergraph

def create_sknetwork_bipartite(H):
    edge_list= []
    for e in H.edges():
        for agent in H.edges[e]:
            edge_list.append( ( agent, int(e) ,H.edges[e].weight))
    graph = from_edge_list(edge_list, bipartite=True, sum_duplicates =True)  
    return graph 

def create_sknetwork_graph_n(H):    
    edge_list=[]    
    for e in H.edges():
        for u, v in itertools.combinations(H.edges[e],2):
            edge_list.append((u,v, H.edges[e].weight))
    graph = from_edge_list(edge_list, sum_duplicates = True)            
    return(graph)
    
def create_sknetwork_graph_e(H):
    edge_list=[]    
    for agent in H.nodes():
        for u, v in itertools.combinations(H.nodes.memberships[agent],2):
            edge_list.append((u, v))
    graph = from_edge_list(edge_list,  bipartite =False,  sum_duplicates = True, matrix_only = False)   
    return(graph)

## Clustering
def check_random_state( seed , random_state):
    if type(random_state) == np.random.mtrand.RandomState:
        return(random_state)
    elif type(random_state) == list:
        return(int(random_state[seed]))
    
def check_algo_name(H, algo_name):
    if algo_name == 'Louvain_b':
        network =  create_sknetwork_bipartite(H)
        adjacency_matrix = network.biadjacency
        
    elif algo_name == 'Louvain_g':
        network =  create_sknetwork_graph_n(H)
        adjacency_matrix = network.adjacency
    else :
        print('Error, choose algo name between Louvain_b and Louvain_g')
    return(network , adjacency_matrix)

def partitions_random_shuffle_nodes(H, algo_name, nb_itt, res =1, random_state= np.random.RandomState()):
    '''  '''
    #Create the sknetwork according to  H
    partitions_=[]
    network , adjacency_matrix = check_algo_name(H, algo_name)
    # Clustring    
    for i in range(nb_itt):
        # random state
        rand = check_random_state( i ,random_state)
        louvain = Louvain(resolution = res, modularity ='Newman' , shuffle_nodes = True, random_state = rand)
        # get partition
        louvain.fit(adjacency_matrix)
        if algo_name == 'Louvain_b':
            partitions_.append(louvain.labels_row_)
        elif algo_name == 'Louvain_g':
            partitions_.append(louvain.labels_)
    return(partitions_)

def partition_with_highest_mod(H, algo_name, nb_itt):
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
    network , adjacency_matrix = check_algo_name(H, algo_name)
    rand = list(np.arange(nb_itt)) # random seeds
    partitions =  partitions_random_shuffle_nodes(H, algo_name, nb_itt, res =1, edge_label = False, random_state= rand )
    if algo_name == 'Louvain_b':
        mods = [ get_modularity( adjacency_matrix, partitions[j][0],partitions[j][1] ) for j in range(len(partitions)) ]
    elif algo_name == 'Louvain_g':
        mods = [ get_modularity( adjacency_matrix, partitions[j] ) for j in range(len(partitions)) ]
    index =  mods.index(max(mods))
    return (partitions[index])

    
##  Mutual information  
def mutual_inofmation_btw_equal_sized_partitions(partitions_g,partitions_b):
    mi_=[]
    nb_clusters_ =[]
    for c_g in partitions_g:
        for c_b in partitions_b:
            if np.max(c_g) == np.max(c_b): # if the parititions c_g and c_b have the same number of clusters
                mi_.append(normalized_mutual_info_score(c_g, c_b ))
                nb_clusters_.append( np.max(c_g) +1)
                
    return(nb_clusters_, mi_)
## Standard deviation of clusters' volume     
def std_cluster_vol(H, partition, nodes_labels):
    clusters = array2dict(partition, nodes_labels)
    return(len(clusters) , np.std([ volume(H, clusters[c]) for c in clusters.keys()]) )

    