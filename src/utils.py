import numpy as np
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import os
CMAP="tab20"
custom_cmap = plt.get_cmap(CMAP)
custom_cmap.set_bad(color='white')

class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    Note: doesn't work for complex nested dictionaries
    adapted from: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def serialize(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)

def deserialize(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)

def convert_arr2graph(Z):
    """
    Convert embedded image (Array of embeddings) to map graph
    """    
    G = nx.Graph(origin=(None, None))
    h,w,d = Z.shape[0], Z.shape[1], Z.shape[2]
    G.graph.update({'array_size': (h,w)})
    G.graph.update({'d': d})

    origin_flag = False
    idx = 0 
    grid2lin_map = {}
    lin2grid_map = {}
    for (i,j), _ in np.ndenumerate(Z[:,:,0]):
        if np.sum(Z[i,j,:]) == 0.0:
            continue
        else:
            if origin_flag == False:
                G.graph.update({'origin': (idx)})
                origin_flag = True
            embed = Z[i,j,:].reshape(1, -1).astype('double')
            G.add_node(idx, pos=(i,j), emb=embed)
            lin2grid_map[idx] = (i,j)
            grid2lin_map[(i,j)] = idx
            idx += 1 # update linear counter
        
    nodelist = list(G.nodes())
    for node in nodelist:
        i,j = G.nodes[node]['pos']
        for di in range(i-1, i+2):
            for dj in range(j-1, j+2):
                if di == i and dj == j:
                    continue
                if di < 0 or di >= h or dj < 0 or dj >= w:
                    continue
                if (di,dj) not in grid2lin_map:
                    continue
                nhbr = grid2lin_map[(di,dj)]
                G.add_edge(node, nhbr, weight=1)

    G.graph.update({'pos_dict': lin2grid_map})
    return G

def construct_sprite(G, processor):
    """
    Takes a Map Graph G and constructs a sprite from it by applying an embedding quantizer
    AKA: "embedding quantization" for sprite construction
    """
    S = G.copy()
    S.graph.update({'d': 1})
    for node in S.nodes:
        embedding = S.nodes[node]['emb']
        motif_label = processor.quantizer.predict(embedding.reshape(1, -1).astype(float))[0]
        S.nodes[node]['emb'] = motif_label
    return S

def get_prospect_range(P):
    """
    Computes the centered range of values for a prospect graph to display
    ± maxmag is the range of values to display
    Input: 
        P: prospect graph
    Output: maxmag
    """
    if type(P) == nx.classes.graph.Graph:
        values = list(nx.get_node_attributes(P, 'emb').values())
        minP = np.min(values)
        maxP = np.max(values)
    elif type(P) == np.ndarray:
        minP = np.nanmin(P)
        maxP = np.nanmax(P)
    maxmag = np.max([np.abs(minP), np.abs(maxP)])
    return maxmag

def visualize_sprite(G, modality="graph", prospect_flag=False):
    # Visualize sprite
    plt.figure()
    colors = list(nx.get_node_attributes(G, 'emb').values())
    colors = [int(c) for c in colors]
    if type(colors[0]) != int:
        raise Exception("Error: Sprite detected as multi-channel when it should be single-channel and categorical. Please quantize the Datum's Map Graph.")

    our_cmap = custom_cmap
    if prospect_flag:
        our_cmap = plt.get_cmap("bwr")
        maxmag = get_prospect_range(G)
        
    # for visualization, we scale the positions
    shape = "o"
    if modality == "image":
        shape = "o"
        eps = 0.1
        spread_pos_dict = G.graph["pos_dict"]
        for k in spread_pos_dict.keys():
            spread_pos_dict[k] = (spread_pos_dict[k][1] * eps, -spread_pos_dict[k][0] * eps)

        pos = nx.spring_layout(G, pos=spread_pos_dict, fixed=spread_pos_dict.keys(), k=10, iterations=100)
    elif modality == "graph":
        pos = nx.kamada_kawai_layout(G)
    
    if prospect_flag:
        nx.draw(G, pos=pos, edge_color="gray", node_size=15, node_shape=shape, node_color=colors, cmap=our_cmap, vmin=-maxmag, vmax=maxmag)
    else:
        nx.draw(G, pos=pos, edge_color="gray", node_size=15, node_shape=shape, node_color=colors, cmap=our_cmap)
    plt.axis('off')
    plt.draw()

def convert_graph2arr(S):
    h,w = S.graph["array_size"]
    d = S.graph["d"]
    A = np.zeros((h,w,d)) 
    A[:] = np.nan

    for node in S.nodes():
        i,j = S.nodes[node]['pos']
        A[i,j,:] = S.nodes[node]['emb']
    return A

#==========================================
# utility functions for visualizing arrays
#==========================================
def quantize_Z(Z, kmeans_model, mode="dict"):
    h,w,d = Z.shape
    Z_viz = np.zeros((h, w, 1)) 
    Z_viz[:] = np.nan

    idx = 0
    # add motifs/clusters for image
    for i in range(h):
        for j in range(w):
            Zij = Z[i,j,:]
            if np.sum(Zij) != 0:
                if mode == "dict":
                    Zij = Zij.reshape(1, -1).astype('double')
                    cluster = kmeans_model.predict(Zij)[0]
                elif mode == "memmap":
                    Zij = Zij.reshape(1, -1).astype('double')
                    Zij = np.array(Zij, copy=True).astype('double')
                    cluster = kmeans_model.predict(Zij)[0]
                Z_viz[i,j,:] = cluster
                idx += 1
    return Z_viz

def visualize_Z(Z_path, kmeans_model, mode="dict"):
    Z = np.load(Z_path)
    Z_viz = quantize_Z(Z, kmeans_model, mode=mode)  
    visualize_quantizedZ(Z_viz)    

def visualize_quantizedZ(Z_viz, prospect_flag=False):
    plt.figure(figsize=(18, 12), dpi=100)
    plt.yticks([])
    plt.xticks([])
    plt.axis('off')
    if prospect_flag == True:
        maxmag = get_prospect_range(Z_viz)
        print(maxmag)
        plt.imshow(Z_viz, cmap=plt.get_cmap("bwr"), vmin=-maxmag, vmax=maxmag)
        plt.colorbar() 
    else:
        plt.imshow(Z_viz, cmap=custom_cmap)
    plt.show()
    
#===========================================
# utility functions for visualizing proteins
#===========================================

def load_structure(pdbc, pdb_dir='/scratch/users/aderry/pdb'):
    from collapse import process_pdb
    from atom3d.util.formats import df_to_bp
    
    pdb, chain = pdbc[:4], pdbc[-1]
    fname = os.path.join(pdb_dir, pdb[1:3], f'pdb{pdb}.ent.gz')
    df = process_pdb(fname, chain=chain, include_hets=False)
    return df_to_bp(df)
    
def visualize_protein_sprite(sprite):
    import nglview
    from matplotlib.colors import rgb2hex
    struct = load_structure(sprite.graph['id'])
    color_resids = list(zip([rgb2hex(custom_cmap(x)) for x in nx.get_node_attributes(sprite, 'emb').values()], [x[1:] for x in nx.get_node_attributes(sprite, 'resid').values()]))
    scheme = nglview.color._ColorScheme(color_resids, 'sprite')
    view = nglview.show_biopython(struct, default_representation=False)
    view.add_cartoon("protein", color=scheme)
    return view