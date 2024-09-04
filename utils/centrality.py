''' Centrality measure algorithms '''
from collections import Counter
import pandas as pd
import numpy as np
from scipy.stats import entropy
from .general import part2dict

##Eigenvector centrality
def set_functions(mode):
    ''' Define f, g, psi and phi to run the eigenvector centrality for hypergraphs'''
    if mode == 'linear':
        def f(x):
            return x
        def g(x):
            return x
        def psi(x):
            return x
        def phi(x):
            return x
        return f,g,psi,phi
    elif mode == 'log exp':
        def f(x):
            return x
        def g(x):
            return x**(1/2)
        def psi(x):
            return np.exp(x)
        def phi(x):
            return np.log(x)
        return(f,g,psi,phi)
    elif mode == 'max':
        alpha = 10
        def f(x):
            return x**alpha
        def g(x):
            return x**(1/alpha)
        def psi(x):
            return x
        def phi(x):
            return x
        return f,g,psi,phi

def eigenvector(H, mode) :
    ''' Eigenvector centrality for hypergrapph cf Tudisco and Higham 2021'''
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
    for _ in range(maxiter):
        u = np.sqrt(x0 * g(B @ W @ f(y0)))
        v = np.sqrt(y0 * psi( np.transpose(B) @ N @ np.nan_to_num(phi(x0))))
        x = u / np.linalg.norm(u)
        y = v / np.linalg.norm(v)
        if np.linalg.norm(x - x0) + np.linalg.norm( y - y0) < tol :
            print('under tolerance value satisfied')
            break
        x0 = np.copy(x)
        y0 = np.copy(y)
    
    else: 
        print('under tolerance value not satisfied')
    x = np.reshape(x, n)
    y = np.reshape(y,m)
    cent =  dict(zip(idx, x))
    cent.update(dict(zip(column, y)))
    return  pd.DataFrame({f'EV_{mode}': cent.values() }, index = cent.keys())

## Core to Periphery Centrality
def cluster_z_score(H,C ):
    ''' Core periphery centrality measure'''
    membership = part2dict(C)
    s={}
    std_c = {i : np.std( [H.nodes[j].strength for j in c]) for i, c in C.items() }
    mean_c = {i : np.mean( [H.nodes[j].strength for j in c]) for i, c in C.items() }
    for agent in H.nodes():
        if std_c[membership[agent]] != 0:
            s[agent] = (H.nodes[agent].strength - mean_c[membership[agent]])/ std_c[membership[agent]]
        else :
            s[agent] = 0
    return pd.DataFrame({'Core_to_periphery': s.values()}, index = s.keys())

##Diversity
def diversity(H, C ):
    ''' Diversty centrality '''
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
        o[e] = w_e*h_e
    return pd.DataFrame({'Diversity': o.values()}, index = o.keys())



##Political Participation
def political_body(orga_cat ,  individuals ) :
    ''' Return the ensemble of members of the political body '''
    pol = []
    for i , agent in enumerate(individuals.index):
        memberships = list( individuals.iloc[i]['Membership'].split("/"))
        orga_cat_memberships = [orga_cat[m] for m in memberships]
        if 'Political' in orga_cat_memberships :
            pol.append('Y')
        else :
            pol.append('N')
    individuals['Political Body'] = pol
    return individuals

def political_participation( H , individuals):
    ''' Compute political participation of individuals'''

    pol = dict (individuals['Political Body'])
    p = { agent : 0 for agent in H.nodes()}
    for e in H.edges():
        w_e = H.edges[e].weight
        d = H.size(e)
        counter_e = Counter([ pol[agent] for agent in H.edges[e] ])
        for agent in H.edges[e] :
            p[agent] +=  w_e * counter_e['Y']/d
        p[e] = w_e* counter_e['Y']/d
    return pd.DataFrame({'Political participation': p.values()}, index = p.keys())

def activity_diversity_pol(H,C,pol):
    ''' Return the entropy and the politcal rate of an activity'''
    h_e=[]
    p_e=[]
    membership = part2dict(C)

    for e in H.edges():
        # Activity's entropy
        w_e = H.edges[e].weight
        d = H.size(e)
        c= Counter([ membership[agent] for agent in H.edges[e] ])
        pk =[c[item]/d for item in c.keys()]
        h_e.append( entropy(pk)*w_e)
        # Activity's political rate
        counter_e = Counter([ pol[agent] for agent in H.edges[e] ])
        p_e.append(w_e * counter_e['Y']/d )
    return h_e, p_e
    