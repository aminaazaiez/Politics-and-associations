''' Anlayse clusters' compisition '''
from collections import Counter
import itertools
import seaborn as sns
import numpy as np


## Cluster composition
def aggregated_membership_per_cluster(clusters, memberships, asso_cat):
    ''' clusters  = {i : cluster[i]}  clusters is dictionary where i in the index of the cluster and clusters[i] is a list of the agent in the cluster_i 
        FM  ={ agent : memberships of agent i } is dictionnary where the keys are the agents and the values are the list of categroy(formal membership) of agent 
        return list of counters, each counter corresponds to the number of categories in a cluster. If an agent is memeber of two educationnal associations, this will contribute 2 '''
    result=[]
    for cluster in clusters.values():
        count = Counter()
        for agent in cluster :
            count += Counter([asso_cat[ membership ]for membership in memberships[agent]])
        result.append(count)
    return (result)
    

    
## Similarity between agents
def vectorization_agents_orga(clusters ,FM, orgas):
    ''' Assign a vector for each agents. M_ij = 1 if agent i belong to asso j, 0 othewise '''
    M = []
    for c in clusters.values():
        M_c = np.zeros(shape = (len(c), len(orgas))) #  M_ij = 1 if agent i belong to asso j, 0 othewise
        for i , agent in enumerate(c) :
            memberships = FM[agent]
            for m in memberships:
                j=  orgas.index(m)
                M_c[i,j] =1
        M.append(M_c)
    return(M)
    
def vectorization_agents_cat(clusters ,FM, orga_cat_):
    ''' M_ij = 1 if agent i belong to orga category j, 0 othewise '''
    M = []
    categories = list(set(orga_cat_.values())) 
    for c in clusters.values():
        M_c = np.zeros(shape = (len(c), len(categories) ) )  #  M_ij = 1 if agent i belong to orga category j, 0 othewise
        for i , agent in enumerate(c) :
            memberships = FM[agent]
            for m in memberships:
                j = categories.index(orga_cat_[m] )
                M_c[i,j] =1
        M.append(M_c)
    return(M)
    

def inter_similarity(M):
    cos =[]
    M_couple = [(M[p[0]], M[p[1]]) for p in itertools.combinations(np.arange(len(M)), 2)] 
    for M_1 , M_2 in M_couple :
        for u in M_1 :
            for v in M_2 :
                cos.append( np.dot(u,v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return(cos)       
    
def intra_similarity(M):
    cos=[]
    for M_c in M :
        l = [(p[0], p[1]) for p in itertools.combinations(np.arange(len(M_c)), 2)] 
        for u , v in l :
            cos.append( np.dot(M_c[u],M_c[v]) / (np.linalg.norm(M_c[u]) * np.linalg.norm(M_c[v])))
    return(cos)
    
##Random categorization

def random_categorization(orga_cat):
    count = Counter(orga_cat.values())
    r_orga_cat ={}
    for orga in orga_cat.keys():
        remained_cat = [ cat for cat in count.keys() if count[cat]>0]
        cat = np.random.choice(remained_cat)
        r_orga_cat[orga] = cat
        count[cat] -=1
    return(r_orga_cat)