from sknetwork.data import from_edge_list
from sknetwork.clustering import*
import itertools
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import entropy

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
            edge_list.append((u, v   ))
    graph = from_edge_list(edge_list,  bipartite =False,  sum_duplicates = True, matrix_only = False)   
    return(graph)

## Clustering
def louvain_clustering_bipartite( H, res =1):
    graph = create_sknetwork_bipartite(H)
    louvain = Louvain( resolution = res, modularity ='Newman', shuffle_nodes= True)
    louvain.fit(graph.biadjacency)
    n= len(set(louvain.labels_row_)) # = number of clusters
    # clusters of nodes
    clusters_n ={i : [] for i in range(n)}
    for c , p in zip( louvain.labels_row_, graph.names_row):
        clusters_n[c].append(p)
    # clusters of edges
    clusters_e = {i : [] for i in range(n)}
    for c , e in zip( louvain.labels_col_, graph.names_col):
        clusters_e[c].append(int(e))
    return clusters_n , clusters_e
    
def louvain_clustering_graph(H,  res =1):
    graph_n = create_sknetwork_graph_n(H)
    graph_e = create_sknetwork_graph_e(H)
    louvain_n = Louvain( resolution = res, modularity ='Newman')
    louvain_n.fit(graph_n.adjacency)
    n= len(set(louvain_n.labels_))
    clusters_n ={i : [] for i in range(n)}
    
    for c , p in zip( louvain_n.labels_, graph_n.names):
        clusters_n[c].append(p)
        
    louvain_e = Louvain( resolution = res, modularity ='Newman')
    louvain_e.fit(graph_e.adjacency)
    n= len(set(louvain_e.labels_))
    clusters_e ={i : [] for i in range(n)}
    for e, c in enumerate(louvain_e.labels_):
        clusters_e[c].append(int(e))
    return(clusters_n, clusters_e)
    
##  Mutual information  
def mutual_inofmation(H,rs):
    bi_graph = create_sknetwork_bipartite(H)
    graph_n = create_sknetwork_graph_n(H)
    
    clusters_b =[]
    clusters_g=[]
    
    for r in rs :
            
        louvain_b = Louvain( resolution = r, modularity ='Newman')
        louvain_b.fit(bi_graph.biadjacency)
        clusters_b.append(louvain_b.labels_row_)
        
        louvain_g_n = Louvain( resolution = r, modularity ='Newman')
        louvain_g_n.fit(graph_n.adjacency)
        clusters_g.append(louvain_g_n.labels_)
    # compute mutual inofmation of partition of the same size    
    mi_=[]
    nb_clusters_ =[]
    for c_g in clusters_g:
        for c_b in clusters_b:
            if np.max(c_g) == np.max(c_b): # if the clustering c_g and c_b have the same number of clusters
                mi_.append(normalized_mutual_info_score(c_g, c_b ))
                nb_clusters_.append( np.max(c_g))
    return(nb_clusters_, mi_)