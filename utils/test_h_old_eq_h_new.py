


## test H_old == H_new
key_id =  pd.read_excel( '/home/azaiez/Documents/Cours/These/Politics and Associations/Entretiens/keys.ods',engine='odf', index_col = 'key')
sys.path.insert(1, '/home/azaiez/Documents/Cours/These/Politics and Associations/Draft/DraftCode')
import load_data as ld

data , Personnes, Assos = ld.load_data()
H_old = ld.create_hypergraph(data , Personnes)
H_new = create_hypergraph(edges, w)

individual_id = { name : key for name, key in zip (key_id.index, key_id['id'])} 

for e in H_old.edges():
    if set([individual_id [agent]  for agent in H_old.edges[e]]) != set([agent  for agent in H_new.edges[e]] ) or H_old.edges[e].weight != H_new.edges[e].weight :
        print(set([individual_id [agent]  for agent in H_old.edges[e]]) , set( [agent  for agent in H_new.edges[e]]))
        
        
        
