import hypernetx as hnx
import numpy as np
from collections import Counter
from scipy.stats import entropy
import itertools
import sys 
sys.path.insert(1, '/home/azaiez/Documents/Cours/These/Politics and Associations/Programs')
from utils.general import part2dict


##Eigenvector centrality 
def set_functions(mode):
    if mode == 'linear' :
        def f(x):
            return(x)
        def g(x):
            return (x)
        def psi(x):
            return(x)
        def phi(x):
            return(x)
        return(f,g,psi,phi)
    if mode == 'log exp':
        def f(x):
            return(x)
        def g(x):
            return (x**(1/2))
        def psi(x):
            return(np.exp(x))
        def phi(x):
            return(np.log(x))
        return(f,g,psi,phi)
    if mode == 'max':
        alpha = 10
        def f(x):
            return(x)
        def g(x):
            return (x)
        def psi(x):
            return(x**(1/alpha))
        def phi(x):
            return(x**alpha)
        return(f,g,psi,phi)
        

def eigenvector (H, mode , indecies) :
    maxiter = 1000
    tol = 1e-5
    f,g,psi,phi = set_functions(mode)
    B, idx , column  = H.incidence_matrix(weights=False, index = True)
    n,m = np.shape(B)
    
    edge_weights = [H.edges[e].weight for e in H.edges()]
    nodes_weights = [ 1 for agent in H.nodes()]
    
    W = np.diag(edge_weights, k=0)
    N= np.diag(nodes_weights, k=0)
    
    #x0 = np.ones((n,1))
    #y0 = np.ones((m,1))
    x0 = np.random.rand(n,1)
    y0 = np.random.rand(m,1)
    
    for it in range(maxiter):
        
        u = np.sqrt(x0 * g(B @ W @ f(y0)))
        v = np.sqrt(y0 * psi( np.transpose(B) @ N @ np.nan_to_num(phi(x0))))
        
        x = u / np.linalg.norm(u)
        y = v / np.linalg.norm(v)


        if np.linalg.norm(x - x0) + np.linalg.norm( y - y0) < tol :
            print('under tolerance value satisfied')
            x = np.reshape(x, n)
            y = np.reshape(y,m)
            eigenvector_centrality = {idx[i] : x[i] for i in range(len(idx))}
            return(eigenvector_centrality)
            
        else :
            x0 = np.copy(x)
            y0 = np.copy(y)
        
    print('under tolerance value not satisfied')

    x = np.reshape(x, n)
    y = np.reshape(y,m)
    eigenvector_centrality = {idx[i] : x[i] for i in range(len(idx))}
    
    return([eigenvector_centrality[agent]  for agent in indicies])
    

## Core to Periphery Centrality    
def core_to_periphery(H,C , indicies):
    membership = part2dict(C)
    s={}
    std_c = {i : np.std( [H.nodes[j].strength for j in c]) for i, c in C.items() }   
    mean_c = {i : np.mean( [H.nodes[j].strength for j in c]) for i, c in C.items() }  
    for agent in H.nodes():
        if std_c[membership[agent]] != 0:
            s[agent] = (H.nodes[agent].strength - mean_c[membership[agent]])/ std_c[membership[agent]]

        else :
            s[agent] = 0
    return([s[agent] for agent in indicies])

##Diversity
def diversity(H, C , indicies):
    membership = part2dict(C)
    o= { agent : 0 for agent in H.nodes()}
    for e in H.edges():
        w_e = H.edges[e].weight
        d = H.size(e)      
        c= Counter([ membership[agent] for agent in H.edges[e] ])
        pk =[c[item]/d for item in c.keys()]
        h_e = entropy(pk)
        for agent in H.edges[e]:        
            o[agent] +=  w_e*  h_e
    return([o[agent] for agent in indicies])

##Political Participation
def Political_Body(orga_cat ,  individuals ) :
    pol = []
    for i , agent in enumerate(individuals.index):
        memberships = list( individuals.iloc[i]['Membership'].split("/"))
        orga_cat_memberships = [orga_cat[m] for m in memberships]
        if 'Political' in orga_cat_memberships :
            pol.append('Y')
        else :
            pol.append('N')
    individuals['Political Body'] = pol
    return(individuals)
    
def political_participation( H , individuals, indicies):
    pol = dict (individuals['Political Body'])
    p = { agent : 0 for agent in H.nodes()}
    for e in H.edges():
        w_e = H.edges[e].weight
        d = H.size(e)
        counter_e = Counter([ pol[agent] for agent in H.edges[e] ])
        for agent in H.edges[e] :
            p[agent] +=  w_e * counter_e['Y']/d
    return([p[agent] for agent in indicies])
    
    
def activity_diversity_pol(H,C,pol):
    h_e=[]
    p_e=[]
    membership = mod.part2dict(C)

    for e in H.edges():
        
        w_e = H.edges[e].weight
        d = H.size(e)
        c= Counter([ membership[agent] for agent in H.edges[e] ])
        pk =[c[item]/d for item in c.keys()]

        h_e.append( entropy(pk)*w_e)
        counter_e = Counter([ pol[agent] for agent in H.edges[e] ])
        p_e.append(counter_e['Y']/d *w_e)
    return(h_e, p_e)  