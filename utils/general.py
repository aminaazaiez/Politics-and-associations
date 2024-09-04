''' General functions  '''
import itertools
import numpy as np
from sknetwork.data import from_edge_list
import networkx as nx

def gini(A):
    """Calculate the Gini coefficient of a numpy array."""

    array = np.array(A)
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = array + 0.0000001
    array = np.sort(array)
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) 

def ccdf (data):
    ''' Complementary cumulative distribution function plot'''
    data = list(data)
    N = len(data)
    x = np.sort(data)
    y = 1 - (np.arange(N) / float(N))
    return x,y 
    
def part2dict(A):
    """
    Given a partition (list of sets), returns a dictionary mapping the part for each vertex
    """
    x = []
    for i in range(len(A)):
        x.extend([(a, i) for a in A[i]])
    return {k: v for k, v in x}
    
def array2dict(A, node_labels):
    ''' Using a 1-D array where A[i] is the cluster to which agent i belongs, return the dictionnary of partition
        Paramters 
        ---------
        A : 1D array shape (nb_nodes,) 
        
        node_labels: 1D array shape (nb_nodes,) 
        
        Returns
        -------
        clusters_: dict
            clusters_[i]: list of agent belonging to the cluster i'''
    n= len(set(A)) # = number of clusters
    # clusters of nodes
    clusters_ ={i : [] for i in range(n)}
    for c , p in zip( A, node_labels):
        clusters_[c].append(p)
    return(clusters_)
    

    
## clique_expansion using networkx 
def nx_clique_expansion(H):
    I = nx.Graph()
    edge = {}
    for e in H.edges():
        for u, v in itertools.combinations(H.edges[e],2):
            try:
                edge[(u,v)]+= H.edges[e].weight
            except Exception:
                edge[(u,v)]= H.edges[e].weight
    for (u,v), w in edge.items():
        I.add_edge(u,v, weight =w )
    return(I)
## clique_expansion using sknetwork 

def create_sknetwork_graph_n(H):    
    edge_list=[]    
    for e in H.edges():
        for u, v in itertools.combinations(H.edges[e],2):
            edge_list.append((u,v, H.edges[e].weight))
    graph = from_edge_list(edge_list, sum_duplicates = True)            
    return(graph)

## clique_expansion using networkx 

def create_sknetwork_bipartite(H):
    edge_list= []
    for e in H.edges():
        for agent in H.edges[e]:
            edge_list.append( ( agent, int(e) ,H.edges[e].weight))
    graph = from_edge_list(edge_list, bipartite=True, sum_duplicates =True)  
    return graph 


def generate_adjacency_matrix(H, algo_name):
    if algo_name == 'Louvain_b':
        network =  create_sknetwork_bipartite(H)
        adjacency_matrix = network.biadjacency
        
    elif algo_name == 'Louvain_g':
        network =  create_sknetwork_graph_n(H)
        adjacency_matrix = network.adjacency
    else :
        print('Error, choose algo name among Louvain_b and Louvain_g')
    return( adjacency_matrix)