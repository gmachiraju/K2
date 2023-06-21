import numpy as np
import networkx as nx

def convert_arr2graph(Z):
    """
    Convert embedded image (Array of embeddings) to map graph
    """
    G = nx.Graph(origin=(None, None))
    origin_flag = False
    for (i,j), value in np.ndenumerate(Z[:,:,0]):
        if np.sum(Z[i,j,:]) == 0.0:
            continue
        else:
            if origin_flag == False:
                G['origin'] = (i,j)
                origin_flag = True
            G.add_node((i,j), pos=(i,j), emb=Z[i,j,:])
    

def process_image_dataset(arr_dir, save_dir):
    pass
