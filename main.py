import pandas as pd
import hypernetx as hnx
import numpy as np
from collections import Counter
from sknetwork.ranking import Betweenness
from sknetwork.ranking import Closeness
from scipy import stats

import matplotlib.pyplot as plt

import sys 
sys.path.insert(1, '/home/azaiez/Documents/Cours/These/Politics and Associations/Programs')
from utils.load_data import *
from utils.general import *
from utils.community_detection import *
from utils.cluster_composition import *
from utils.centrality import *

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
nb_itt = 100
start , stop, step = 0 , 5, 0.1
rs = np.arange (start, stop, step)

partitions = pd.DataFrame(columns = ['seed', 'r','Partition', 'q', 'algo_name' ])
for algo_name in ['Louvain_b', 'Louvain_g']:
    adjacency_matrix = check_algo_name(H, algo_name)
    for r in rs:
        print('r = %.2f')
        for i in range (nb_itt):
            partition = Clustering(adjacency_matrix , algo_name, res=r, random_state = i ) 
            partitions = pd.concat([partitions, pd.DataFrame({'seed' : i, 'r' : r, 'Partition' : [partition], 'algo_name' :algo_name, 'q' : np.max(partition) +1}) ], ignore_index=True )
            
##
#Mutual information between partitions of the same size

nb_clusters, mi =  mutual_inofmation_btw_equal_sized_partitions(partitions)

fig,ax = plt.subplots( figsize = (5,5))
ax.scatter(nb_clusters, mi, s =10, alpha =0.1)
ax.set_xlabel('$q$')
ax.set_ylabel(r'$I_{norm}$')
plt.tight_layout()
fig.savefig(path +'Figures/MI_vs_nb_clusters.png')

#Standard deviation of clusters' volume
network = create_sknetwork_bipartite(H)
std = []
for i , partition in enumerate(partitions['Partition']):
    print(i)
    std.append(std_cluster_vol(H, list(partition), network.names)) 
partitions['std_vol']= std
#save partition to json file
#partitions.to_csv(path+'random_partitions.csv')
##
#compute mean and standard of mean of std_vol

fig, ax =plt.subplots(figsize =(5,5))

for algo_name in ['Louvain_b', 'Louvain_g']:
    df = pd.DataFrame( partitions.query("algo_name == '%s'" %algo_name).groupby(['q'])['std_vol'].mean())
    df['error'] = partitions.query("algo_name == '%s'" %algo_name).groupby(['q'])['std_vol'].sem(ddof = 0)

    df['std_vol'].plot(ax =ax , label = algo_name )
    ax.fill_between(df.index, df['std_vol']+df['error'],  df['std_vol']-df['error'], alpha = 0.5)
    
ax.legend()
ax.set_ylabel('std')    
plt.tight_layout()
fig.savefig(path+'Figures/std_comm_vol_vs_nb_clusters.pdf')
plt.close(fig)

############# Cluster Composition ###############

## Partition of nodes 
nb_itt = 700
algo_name = 'Louvain_b'
clusters = array2dict(partition_with_highest_mod(H, algo_name, nb_itt), create_sknetwork_bipartite(H).names)   
    
## Standard deviation individuals' strength per cluster

std_intra = [np.std([ H.nodes[agent].strength for agent in c]) for c in clusters.values()]
mean_intra = [np.mean([ H.nodes[agent].strength for agent in c]) for c in clusters.values()]

for i in range(len(( clusters))):
    print( '\t ',i+1, '\t &',  '%.2f'%mean_intra[i], '\t &', '%.2f'%std_intra[i], '\\\\')


## Category of memberships per cluster
cat_c = cumulated_membership_per_cluster(clusters , FM, orga_cat) #dictionary of counters
categories = ['Political','Art', 'Educational', 'Sport', 'Human Service', 'Recreational', 'Environmental-Protection', 'Occupational', 'Professional']
result={}        # dictionary of list of lenght = len(categorie).
for c in clusters.keys():
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
  
M=vectorization_agents_orga(clusters, FM , list(orga_cat.keys()))
#M = vectorization_agents_cat(clusters, FM , orga_cat)

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
    M = vectorization_agents_cat(clusters, FM , r_orga_cat)
    cos_intra = intra_similarity(M)
    cos_inter = inter_similarity(M)
    c_inter += Counter({i : sim for i , sim in enumerate(cos_inter) })
    c_intra += Counter({i : sim for i , sim in enumerate(cos_intra) })
    
for couple in c_inter.keys():
    c_inter[couple]/=nb_itt
for couple in c_intra.keys():
    c_intra[couple]/=nb_itt
# empirical case    
M=vectorization_agents_cat(clusters, FM , orga_cat)
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

######################### Centrality ##########################
nb_itt = 700
algo_name = 'Louvain_b'
clusters = array2dict(partition_with_highest_mod(H, algo_name, nb_itt), create_sknetwork_bipartite(H).names)    
#Degree centralities
individuals['Strength'] = [H.nodes[agent].strength for agent in individuals.index]
individuals['Diversity'] = diversity(H , clusters , individuals.index)
#Eigenvector
individuals['EV linear'] = eigenvector(H,'linear', individuals.index)
individuals['EV log expo'] = eigenvector(H, 'log exp', individuals.index)
individuals['EV max']  = eigenvector(H, 'max' , individuals.index)
#Core to periphery 
individuals['Core to Periphery']  = core_to_periphery(H,clusters, individuals.index)
#clique expansion 
network  = create_sknetwork_graph_n(H)
btw_ = Betweenness().fit_predict(network.adjacency)
cls_ =  Closeness().fit_predict(network.adjacency)

btw = { agent : btw_[i] for i, agent in enumerate(network.names)}
cls = { agent : cls_[i] for i, agent in enumerate(network.names)}
individuals['Betweenness'] = [ btw[agent]  for agent in individuals.index]
individuals['Closeness'] = [ cls[agent]  for agent in individuals.index]

# using networkx
I =nx_clique_expansion(H)
btw_weighted_nx = nx.betweenness_centrality(I, normalized = True, weight = 'weight')
closeness = nx.closeness_centrality(I, distance = 'weight')
individuals['Betweenness'] = [ btw_weighted_nx[agent]  for agent in individuals.index]
individuals['Closeness'] = [ closeness[agent]  for agent in individuals.index]
#Political participation
pol = Political_Body(orga_cat = orga_cat, individuals = individuals)
individuals['Political Participation'] = political_participation(H , individuals,  individuals.index)
# print correlation btw political participation and centralities
centralities_label  = [ 'Strength','Diversity', 'EV linear', 'EV log expo' , 'EV max','Core to Periphery' , 'Betweenness', 'Closeness' ]

for label in centralities_label:
    P_corr = individuals.groupby(['Political Body']).corr(numeric_only = True)['Political Participation']['Y'][label]
    n_P_corr = individuals.groupby(['Political Body']).corr(numeric_only = True)['Political Participation']['N'][label]
    print('%s		&	%.3f &		%.3f \\\ '  %(label, P_corr, n_P_corr) )
#print correlation btw president and centralities
individuals = individuals.replace({'President': { 'P' : 1, 'N' : 0 } })
pointbiserial = individuals.groupby(['Political Body'])[['Political Participation' ]+ centralities_label].corrwith(individuals['President'], method=stats.pointbiserialr)

for cent in ['Political Participation' ] + centralities_label: 
    print(cent, '&  %.3f'%pointbiserial[cent]['N'][0],  '& %.3f'%pointbiserial[cent]['N'][1], '\\\\')
    
############# Export gephi file for vizualizartion #############
#Clustering to arritube cluster belonging to nodes
nb_itt = 700
algo_name = 'Louvain_b'
I = create_sknetwork_bipartite(H)
idx = partition_with_highest_mod(H, algo_name, nb_itt, return_idx = True)
clusters  = Clustering( I.biadjacency, 'Louvain_b',  random_state = idx, get_edge_label = True )
#create nx bipartite graph
s_edges, s_weights = split_data(edges , w)

B = nx.Graph()

# Add nodes with the node attribute "bipartite"
for e , c in zip(I.names_col , clusters[1]):
    B.add_node(e, bipartite= 'edge', cluster = c)
for a , c in zip(I.names , clusters[0]):
    B.add_node(a, bipartite= 'node', cluster = c)

for edge , node , w_ in zip (s_edges[0] , s_edges[1] , s_weights):
    B.add_edge( edge, node , weight =w_)


nx.write_gexf(B, path+"export2gephi/test.gexf")
    