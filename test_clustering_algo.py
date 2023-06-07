import itertools
import matplotlib.pyplot as plt
import pandas as pd
from sknetwork.clustering import*
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import sys 
sys.path.insert(1, '/home/azaiez/Documents/Cours/These/Politics and Associations/Programs')
from utils.load_data import create_hypergraph
from utils.community_detection import create_sknetwork_bipartite 
from utils.community_detection import create_sknetwork_graph_n
from utils.general import volume

path='/home/azaiez/Documents/Cours/These/Politics and Associations/Programs/'
##load data 
path='/home/azaiez/Documents/Cours/These/Politics and Associations/Programs/'

activities = pd.read_excel( path+'activities.ods',engine='odf', index_col = 'Activity')

w= [F for F in activities['Frequency']] # weights
edges = [s.split('/') for s in list( activities ['Individuals'])]
H = create_hypergraph(edges, w)


##test mutual information between different relization of node shuffling

def random_shuffling_nodes(H, algo_name, nb_itt, res =1, edge_label = False, random_state= np.random.RandomState()):
    '''  '''
    #Create the sknetwork according to the H
    partitions_=[]
    if algo_name == 'Louvain_b':
        network =  create_sknetwork_bipartite(H)
        adjacency_matrix = network.biadjacency
    elif algo_name == 'Louvain_g':
        network =  create_sknetwork_graph_n(H)
        adjacency_matrix = network.adjacency
        
    # Clustring    
    for i in range(nb_itt):
        
        # random state
        if type(random_state) == np.random.mtrand.RandomState:
            louvain = Louvain(resolution = res, modularity ='Newman' , shuffle_nodes = True, random_state = random_state)
        elif type(random_state) == list:
            louvain = Louvain(resolution = res, modularity ='Newman' , shuffle_nodes = True, random_state = int(random_state[i]))
        # get partition

        louvain.fit(adjacency_matrix)
        if algo_name == 'Louvain_b' and edge_label ==False :
            partitions_.append(louvain.labels_row_)
        elif algo_name == 'Louvain_b' and edge_label ==True :
            partitions_.append((louvain.labels_row_, louvain.labels_col_))
        elif algo_name == 'Louvain_g':
            partitions_.append(louvain.labels_)
    return(partitions_)
    
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

def clustering(H,algo_name, kwargs_={}):
    if algo_name == 'Louvain_b':
        network =  create_sknetwork_bipartite(H)
        adjacency_matrix = network.biadjacency
        louvain = Louvain( modularity ='Newman', **kwargs_ )
        louvain.fit(adjacency_matrix)
        return(louvain.labels_row_, louvain.labels_col_)
    elif algo_name == 'Louvain_g':
        network =  create_sknetwork_graph_n(H)
        adjacency_matrix = network.adjacency
        louvain = Louvain( modularity ='Newman', **kwargs_ )
        louvain.fit(adjacency_matrix)
        return(louvain.labels_)
##
nb_itt = 20
fig, ax= plt.subplots(figsize=(5,5))
for algo_name in ['Louvain_b', 'Louvain_g']:
    partitions = random_shuffling_nodes(H, algo_name, nb_itt)
    mi= []
    for (p1, p2) in itertools.combinations(partitions,2):
        mi.append(normalized_mutual_info_score(p1,p2))
    step = 0.01
    bins = np.arange ( 0, 1+step , step)
    kwargs = {'bins' : bins, 'alpha' :0.5}
    ax.hist(mi, **kwargs, label= algo_name)
ax.legend()
fig.savefig(path + 'Figures/test_clustering/MI_random_shufflig_nodes.pdf')
plt.close(fig)
''' Test : method with more robustness? No difference betw hypergraph and clique expansion clustering '''

##test mutual information between hypergraph and graph clustering for the same cluster size
#Run clustering algos
nb_itt = 20
start , stop, step = 0 , 5, 1
rs = np.arange (start, stop, step)
partitions_b = []
partitions_g = []
for r in rs:
    partitions_b += partitions_random_shuffle_nodes(H, 'Louvain_b', nb_itt, res=r)
    partitions_g += partitions_random_shuffle_nodes(H, 'Louvain_g', nb_itt, res =r)
# Compute mutual information
mi_=[]
nb_clusters_ =[]
for c_g in partitions_g:
    for c_b in partitions_b:
        if np.max(c_g) == np.max(c_b[0]): # if the parititions c_g and c_b have the same number of clusters
            mi_.append(normalized_mutual_info_score(c_g, c_b[0] ))
            nb_clusters_.append( np.max(c_g))
# Plot                  
fig,ax = plt.subplots( figsize = (5,5))
ax.scatter(nb_clusters_, mi_, s =10, alpha =0.1)
ax.set_xlabel('$q$')
ax.set_ylabel(r'$I_{norm}$')
plt.tight_layout()
fig.savefig(path +'Figures/test_clustering/MI_vs_nb_clusters.pdf')

''' Test : run both algorithms several times and compare the mutual information of pair of partitions (P1, P2) where p1 is obtained using hypergraph clustering and P2 is obtained using clique expansion clustering whit the number of clusters of P1 et P2 being equal.   
    Test : Run one algo for different resolution param and compute mutual information between partitions of the same size. Result : The hyperraph clustering seems more stable'''
    
## Distribution of q_g and q_h  std_g std_h for r=1
nb_itt = 200
network =  create_sknetwork_bipartite(H)
q =[[],[]]
std =[[],[]]
for i , algo_name in enumerate (['Louvain_b', 'Louvain_g']):
    partitions = random_shuffling_nodes(H, algo_name, nb_itt, res =1)
    q[i] = [np.max(c_) for c_ in partitions] # list of partition size (#clusters) for different random realization
    for partition in partitions:
        clusters = array2dict(partition, network.names)
        std[i].append(np.std([ volume(H, clusters[c]) for c in clusters.keys()]) )
##

fig,ax = plt.subplots(1,2, figsize = (10,5))

min_q = min(q[0]+q[1])
max_q = max(q[0]+q[1])
step_q =1
bins_q = np.arange(min_q , max_q+ step , step)

min_std = min(std[0]+std[1])
max_std = max(std[0]+std[1])
step_std =10
bins_std = np.arange(min_std , max_std+ step_std , step_std)

kwargs = {'density' : True}

for i, algo_name in enumerate(['Louvain_b', 'Louvain_g']):
    ax[0].hist(q[i], bins = bins_q,  label= algo_name, **kwargs)
    ax[1].hist(std[i], bins = bins_std, label= algo_name, **kwargs)
        
    ax[0].set_xlabel('q')
    ax[1].set_xlabel('std')
ax[0].legend()
ax[1].legend()
fig.savefig(path +'Figures/test_clustering/hist_q.pdf')
plt.close(fig)    

'''Test : run both algorithm several times with resultion parameter r = 1, and plot the histograms of q = #number of clusters and std = standard deviation of cluster size (volumes).
    Result : hypergraph clustering provide more clusters whith more balance sizes. Clique expansion clustering provides less clusters, with grater '''
## Compute modularity for r = 1 for different realizations
nb_itt = 200
def compute_modularity(H, algo_name_, nb_itt=1, random_state= np.random.RandomState()):
    #Clique expansion
    if algo_name_ == 'Louvain_g':
        adj =  create_sknetwork_graph_n(H).adjacency 
        mod_ =[]
        partitions = random_shuffling_nodes(H, algo_name_, nb_itt, res =1, random_state = random_state )
        for j in range(len(partitions)):
            mod_.append(get_modularity( adj, partitions[j]))
        return(mod_)
    #bipartite    
    elif algo_name_ == 'Louvain_b':     
        adj =  create_sknetwork_bipartite(H).biadjacency 
        mod_ =[]
        seeds= [np.random.RandomState()for _ in range (nb_itt)]
        partitions = random_shuffling_nodes(H, algo_name_, nb_itt, res =1, edge_label = True, random_state = random_state)
        for j in range(len(partitions)):
            mod_.append(get_modularity( adj, partitions[j][0],partitions[j][1] ))    
        return(mod_)    
        
#plot
fig,ax = plt.subplots(figsize = (5,5))
step =0.001
#'bins' : np.arange(0.5, 1+step, step)
kwargs = {'density' : True  }
for  algo_name in ['Louvain_b', 'Louvain_g']:
    mod = compute_modularity(H, nb_itt = nb_itt,  algo_name)
    ax.hist(mod, label= algo_name, **kwargs)
ax.legend()
fig.savefig(path + 'Figures/test_clustering/hist_mod.pdf')
plt.close(fig)

''' Test : Compute the modularity of partitions obtained whit r =1  
    Returns : For hypergraphs clustering the (hyper)modularity have a larger value than the modularity for the clique expanstion clustering'''
## Community connectedness
import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)
def community_connectedness(H,clusters_):
    ''' 
        resturn : number of connected components per cluster'''
    nb_connected_components =[] #number of connected commponents per cluster
    for (i , c ) in clusters_.items() : 
        H_c = H.restrict_to_nodes(c)
        nb_connected_components.append(len(list(H_c.s_components(edges=False)))) 
    return (nb_connected_components)

nb_itt =100
cc =[]
partitions = random_shuffling_nodes(H, 'Louvain_g', nb_itt, res =1)
for i in range (len(partitions)):
    print(i)
    clusters  = array2dict(partitions[i], network.names)
    cc+=community_connectedness(H,clusters)

for nb_compo in cc:
    if nb_compo !=1:
        print('there is disconneted community', nb_compo)
        
''' Test: are clusters connected. i,e, if we restrict the hypergraphes to the set of nodes present in the cluster, is the hypergraph connected ? 
    Return: for hypergraph clustering, the clusters are connected. For clique expansion it happens ones for one cluster (with 2 components)
    '''        
##Choose the partition whit the highest modularity

nb_itt = 2000
r = list(np.arange(nb_itt))
fig,ax = plt.subplots(figsize = (5,5))

for  algo_name in ['Louvain_b', 'Louvain_g']:
    mod = compute_modularity(H, algo_name, nb_itt, random_state = r)
    print([ i for i in range (len(mod)) if mod[i ] == max(mod)])
    ax.plot(r, mod, label= algo_name)
ax.legend()
fig.savefig(path + 'Figures/test_clustering/mod_vs_seed.pdf')
plt.close(fig)

##
def choose_partition (H, algo_name):
    mod = compute_modularity(H, algo_name, nb_itt, random_state = r)
    index =  mod.index(max(mod))
    partitions = random_shuffling_nodes(H, algo_name, nb_itt, res =1, random_state =r)
    q = [np.max(c_) for c_ in partitions]
    
    labels = clustering(H , algo_name , kwargs_ ={'shuffle_nodes' : True, 'random_state' : index})
    return (labels)

## dipersition of cluster sizes
def algo(algo_name):
    if algo_name == 'Louvain_b':
        return( louvain_clustering_bipartite)
    elif algo_name == 'Louvain_g':
        return( louvain_clustering_graph)
        
def std_comm_size( H,algo_name, rs):    
    cluster_vol_std =[]
    nb_clusters =[]
    for r in rs :
        clustering_algo = algo(algo_name)
        clusters_n , clusters_e = clustering_algo(H,r)
        cluster_vol_std.append(np.std( [ sum([H.nodes[i].strength for i in c]) for c in clusters_n.values()] ))
        nb_clusters.append(len(clusters_n))
    return(cluster_vol_std , nb_clusters) 
    
    
## std of cluster volume vs nb_clusters

start , stop, step = 0 , 10, 1
rs = np.arange (start, stop, step)
# 
# for r in rs:
#     partitions_b +=partitions_random_shuffle_nodes(H, 'Louvain_b', nb_itt, res=r)
#     partitions_g += partitions_random_shuffle_nodes(H, 'Louvain_g', nb_itt, res=r)        
# partitions_b_nodes = [partition[0] for partition in partitions_b] 
nb_clusters_g , std_g , nb_clusters_b, std_b =[],[],[],[]
for partition_g, partition_b in zip(partitions_g, partitions_b_nodes):
    
    nb_clusters_g_ , std_g_ = std_cluster_vol(partition_g, network.names) 
    nb_clusters_g.append(nb_clusters_g_)
    std_g.append(std_g_)
    nb_clusters_b_ , std_b_ = std_cluster_vol(partition_b, network.names) 
    nb_clusters_b.append(nb_clusters_b_)
    std_b.append(std_b_)
    
fig,ax = plt.subplots( figsize = (5,5))
kwargs = { 'marker' : 'o' , 'markersize' : 1 , 'alpha': 0.5}
ax.plot( nb_clusters_g, std_g, label='Graph' ,**kwargs)
ax.plot( nb_clusters_b, std_b, label='Hypergraph' , **kwargs)
#ax.set_title('volume std')
ax.set_xlabel(r'$q$')    
ax.set_ylabel('std')    
plt.close(fig)
ax.legend()
plt.tight_layout()
fig.savefig(path+'Figures/std_comm_vol_vs_nb_clusters.pdf')



####

for r in rs:
    partitions_b +=partitions_random_shuffle_nodes(H, 'Louvain_b', nb_itt, res=r)
    partitions_g += partitions_random_shuffle_nodes(H, 'Louvain_g', nb_itt, res=r)        
partitions_b_nodes = [partition[0] for partition in partitions_b] 
 
nb_clusters_g , std_g = [std_cluster_vol(partition) for partition in partitions_g]
nb_clusters_b , std_b = [std_cluster_vol(partitions) for partition in partitions_b_nodes]





