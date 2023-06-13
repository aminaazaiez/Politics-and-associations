import numpy as np
from sknetwork.data import from_edge_list
import networkx as nx
import itertools


def cdf (ax, data, label):
    N=len(data)
    # sort the data in ascending order
    x = np.sort(data)
    # get the cdf values of y
    y = 1 - (np.arange(N) / float(N))
    # plotting
    ax.set_xlabel(label)
    ax.set_ylabel('CDF of %s' %label.lower())
    ax.plot(x, y, marker='o', markersize = 1.5, linewidth =0)
    #ax.title('cdf of %s in log log scale' %label)
    ax.set_yscale('log')
    ax.set_xscale('log')
    return(ax)
    
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
    dict = {}
    for e in H.edges():
        for u, v in itertools.combinations(H.edges[e],2):
            try:
                dict[(u,v)]+= H.edges[e].weight
            except:
                dict[(u,v)]= H.edges[e].weight
    for (u,v), w in dict.items():
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


#
def check_algo_name(H, algo_name):
    if algo_name == 'Louvain_b':
        network =  create_sknetwork_bipartite(H)
        adjacency_matrix = network.biadjacency
        
    elif algo_name == 'Louvain_g':
        network =  create_sknetwork_graph_n(H)
        adjacency_matrix = network.adjacency
    else :
        print('Error, choose algo name between Louvain_b and Louvain_g')
    return( adjacency_matrix)