import pandas as pd
import hypernetx as hnx
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt

import sys 
sys.path.insert(1, '/home/azaiez/Documents/Cours/These/Politics and Associations/Programs')
from utils.load_data import create_hypergraph
from utils.general import cdf
from utils.general import array2dict


from utils.community_detection import partition_with_highest_mod
from utils.community_detection import mutual_inofmation_btw_equal_sized_partitions
from utils.community_detection import create_sknetwork_bipartite
from utils.community_detection import partitions_random_shuffle_nodes
from utils.community_detection import std_cluster_vol


from utils.cluster_composition import cumulated_membership_per_cluster
from utils.cluster_composition import cluster_composition_bar_plot
from utils.cluster_composition import vectorization_agents_orga
from utils.cluster_composition import vectorization_agents_cat
from utils.cluster_composition import intra_similarity
from utils.cluster_composition import inter_similarity
from utils.cluster_composition import association_graph
from utils.cluster_composition import plot_association_graph
from utils.cluster_composition import random_categorization


### Load Data and create the Hypergraph

path='/home/azaiez/Documents/Cours/These/Politics and Associations/Programs/'

activities = pd.read_excel( path+'activities.ods',engine='odf', index_col = 'Activity')
individuals = pd.read_excel( path+'individual_carac.ods',engine='odf', index_col = 'Individual')
orga = pd.read_excel( path+ 'orga_carac.ods',engine='odf', index_col = 'Orga')
    
w= [F for F in activities['Frequency']] # weights
edges = [s.split('/') for s in list( activities ['Individuals'])]
H = create_hypergraph(edges, w)

orga_cat = { orga_id : cat for orga_id, cat in zip (orga.index, orga['Category'])} 
FM = { individual_id : list(formal_membership.split("/")) for individual_id, formal_membership in zip (individuals.index, individuals['Membership'])} 

### Cumulative distribution functions of degree strength weight cardinality

features = [[H.degree(agent) for agent in H.nodes()] , [H.nodes[agent].strength for agent in H.nodes()], [H.size(e) for e in H.edges()],  [H.edges[e].weight for e in H.edges()]]
labels = ['Node degree', 'Node strength', 'Edge cardinality', 'Edge weigth']
notations =['k_i', 's_i', 'd_e', 'w_e']

for feature,label,notation in zip(features,labels, notations):
    fig , ax = plt.subplots(figsize = (3,3))
    ax = cdf(ax,feature, label)
    plt.tight_layout()
    plt.savefig(path +'Figures/%s.pdf' %label)
    plt.close(fig)
############# Community detection ###############
#Generate random partitions with different resultution parameters
nb_itt = 5
start , stop, step = 0 , 5, 1
rs = np.arange (start, stop, step)
partitions_b = []
partitions_g = []
for r in rs:
    print(r)
    partitions_b += partitions_random_shuffle_nodes(H, 'Louvain_b', nb_itt, res=r)
    partitions_g += partitions_random_shuffle_nodes(H, 'Louvain_g', nb_itt, res=r)        
partitions_b_nodes = [partition[0] for partition in partitions_b] 

#Mutual information between partitions of the same size

nb_clusters, mi = mutual_inofmation_btw_equal_sized_partitions(partitions_g , partitions_b_nodes)

fig,ax = plt.subplots( figsize = (5,5))
ax.scatter(nb_clusters, mi, s =10, alpha =0.1)
ax.set_xlabel('$q$')
ax.set_ylabel(r'$I_{norm}$')
plt.tight_layout()
fig.savefig(path +'Figures/MI_vs_nb_clusters.png')

#Standard deviation of clusters' volume
network = create_sknetwork_bipartite(H)
nb_clusters_g , std_g , nb_clusters_b, std_b =[],[],[],[]
for i, (partition_g, partition_b) in enumerate( zip(partitions_g, partitions_b_nodes)):
    print(i)
    nb_clusters_g_ , std_g_ = std_cluster_vol(H, partition_g, network.names) 
    nb_clusters_g.append(nb_clusters_g_)
    std_g.append(std_g_)
    nb_clusters_b_ , std_b_ = std_cluster_vol(H, partition_b, network.names) 
    nb_clusters_b.append(nb_clusters_b_)
    std_b.append(std_b_)
##    
fig,ax = plt.subplots( figsize = (5,5))
kwargs = { 'marker' : 'o' , 's' : 2 , 'alpha': 0.1}
ax.scatter( nb_clusters_g, std_g, label='Graph' , **kwargs)
ax.scatter( nb_clusters_b, std_b, label='Hypergraph' , **kwargs)
#ax.set_title('volume std')
ax.set_xlabel(r'$q$')    
ax.set_ylabel('std')    
plt.close(fig)
ax.legend()
plt.tight_layout()
fig.savefig(path+'Figures/std_comm_vol_vs_nb_clusters.pdf')
plt.close(fig)


############# Cluster Composition ###############

## Partition of nodes 
nb_itt = 700
algo_name = 'Louvain_b'
clusters , clusters_e = partition_with_highest_mod(H, algo_name, nb_itt)
clusters_n = array2dict(clusters, create_sknetwork_bipartite(H).names)    
    
## Standard deviation individuals' strength per cluster

std_intra = [np.std([ H.nodes[agent].strength for agent in c]) for c in clusters_n.values()]
mean_intra = [np.mean([ H.nodes[agent].strength for agent in c]) for c in clusters_n.values()]

for i in range(len(( clusters_n))):
    print( '\t ',i+1, '\t &',  '%.2f'%mean_intra[i], '\t &', '%.2f'%std_intra[i], '\\\\')

## Graph of clusters 
G = association_graph (H , clusters_n )
vol_c = {c:  sum([H.nodes[i].strength for i in H.nodes() if i in clusters_n[c]])  for c in range(len(clusters_n))}

fig,ax = plt.subplots( )

ax = plot_association_graph(G, ax, vol_c)
fig.savefig(path +'Figures/comm_graph.pdf')

## Category of memberships per cluster
cat_c = cumulated_membership_per_cluster(clusters_n , FM, orga_cat) #dictionary of counters
categories = ['Political','Art', 'Educational', 'Sport', 'Human Service', 'Recreational', 'Environmental-Protection', 'Occupational', 'Professional']
result={}        # dictionary of list of lenght = len(categorie). 
for c in clusters_n.keys():
    result[c] =[]
    for cat in categories:
        if cat in cat_c[c]:
            result[c].append(cat_c[c][cat])
        else:
            result[c].append(0)

fig, ax = plt.subplots(figsize=(9.2, 5))
ax = cluster_composition_bar_plot(ax, result, categories )
ax.legend(ncol =2, loc='upper right')
ax.set_title('Clustering of nodes')
fig.savefig(path +'Figures/categorial composition.pdf')
## Similarity between agents

#Compute orga_similarity 
  
M=vectorization_agents_orga(clusters_n, FM , list(orga_cat.keys()))
#M = vectorization_agents_cat(clusters_n, FM , orga_cat)

cos_intra = intra_similarity(M)
cos_inter = inter_similarity(M)
#Plot orga sim
step = 0.05
bins = np.arange ( 0, 1+step , step)
fig, ax = plt.subplots(figsize=[5, 5])
kwargs = {'bins' :bins, 'density': True, 'alpha' : 0.5, 'edgecolor' : 'k'  }
ax.hist(cos_inter, label = 'Inter', color = 'darkcyan', **kwargs)
ax.hist(cos_intra,  label = 'Intra', color = 'darkorange', **kwargs)
plt.xlabel('Similarity')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
fig.savefig(path +'Figures/orga_sim.pdf')
##
#Compute category similarity with random assignation of category for organizations
nb_itt =200
c_inter=Counter()
c_intra=Counter()

for itt in range (nb_itt):
    print(itt)
    r_orga_cat = random_categorization(orga_cat)
    M = vectorization_agents_cat(clusters_n, FM , r_orga_cat)
    cos_intra = intra_similarity(M)
    cos_inter = inter_similarity(M)
    c_inter += Counter({i : sim for i , sim in enumerate(cos_inter) })
    c_intra += Counter({i : sim for i , sim in enumerate(cos_intra) })
    
for couple in c_inter.keys():
    c_inter[couple]/=nb_itt
for couple in c_intra.keys():
    c_intra[couple]/=nb_itt
# empirical case    
M=vectorization_agents_cat(clusters_n, FM , orga_cat)
cos_intra_emp = intra_similarity(M)
cos_inter_emp = inter_similarity(M)
#plot
step = 0.07
bins = np.arange ( 0, 1+step , step)
fig, axs =plt.subplots(1,2 , sharey = True, figsize = (10,5))

for ax, similarity , sim_type , colors in zip (axs,[ [cos_inter_emp,c_inter.values() ],[cos_intra_emp,c_intra.values()]], ['Inter-similarity', 'Intra-similarity'], [['darkcyan','aliceblue' ],['darkorange','moccasin' ]]):  
    ax.hist(similarity[0], label = 'empirical', **kwargs, color =colors[0])
    ax.hist(similarity[1], label = 'random', **kwargs, color = colors[1])
    ax.set_title(sim_type)
    ax.set_xlabel('Similarity')
    ax.set_ylabel('Density')
    ax.legend()
fig.savefig(path +'Figures/random_cat_sim.pdf')

