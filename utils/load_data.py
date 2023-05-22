import pandas as pd
import hypernetx as hnx

##
def split_data(edges_ , w_):
    splited_edges =[]
    splited_weights =[]
    for i in range (len(edges_)):
        for j in range (len(edges_[i])):
            splited_edges.append([i, edges_[i][j]])
            splited_weights.append(w_[i])
    s_edges= pd.DataFrame(splited_edges)
    return(s_edges, splited_weights)
    
def create_hypergraph(edges_, w_):    
    s_edges, s_weights = split_data(edges_ , w_)
    H=hnx.Hypergraph(s_edges, weights = s_weights)
    for e in H.edges:
        H.edges[e].weight = w_[e]
    for i in H.nodes() :
        E_i=H.nodes.memberships[i]
        H.nodes[i].strength = 0
        for e in E_i :
            H.nodes[i].strength += H.edges[e].weight
    return(H)