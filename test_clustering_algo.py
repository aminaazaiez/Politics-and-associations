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

cluster_vol_std_g, nb_clusters_g = std_comm_size(H,  'Louvain_g', rs)
cluster_vol_std_b, nb_clusters_b = std_comm_size(H,  'Louvain_b', rs)

fig,ax = plt.subplots( figsize = (5,5))
kwargs = { 'marker' : 'o' , 'markersize' : 1 , 'alpha': 0.5}
algo
ax.plot( nb_clusters_g, cluster_vol_std_g, label='Graph' ,**kwargs)
ax.plot( nb_clusters_b, cluster_vol_std_b, label='Hypergraph' , **kwargs)
#ax.set_title('volume std')
ax.set_xlabel(r'$q$')    
ax.set_ylabel('std')    
plt.close(fig)
ax.legend()
plt.tight_layout()
fig.savefig(path+'Figures/std_comm_vol_vs_nb_clusters.pdf')