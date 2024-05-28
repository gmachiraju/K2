import numpy as np
import networkx as nx
import pickle
import dill

from skimage.filters import threshold_otsu
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import os
import pdb
CMAP="tab20"
custom_cmap = plt.get_cmap(CMAP)
custom_cmap.set_bad(color='white')

# deals with more than 20 colors
CMAP2 = "Pastel2"
joint_cmap = colors.ListedColormap(cm.tab20.colors + cm.Pastel2.colors, name='tab40')
joint_cmap.set_bad(color='white')

#================================================================
# General functions for saving/loading data
#----------------------------------------------------------------
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
    
def serialize_model(obj, path):
    with open(path, 'wb') as fh:
        return dill.dump(obj, fh)

def deserialize_model(path):
    with open(path, 'rb') as fh:
        return dill.load(fh)
    
# sentence things
def process_sent(s):
    s = s.strip() # left and right stripping
    if s == "":                    # Don't change empty strings.
        return s
    if s[-1] in ["?", ".", "!"]:   # Don't change if already okay.
        return s
    return s + "."  

def process_sentences(sents):
    ret = []
    for s in sents:
        s = process_sent(s)
        if s != "":
            ret.append(s)
    return ret
    

#================================================================
# useful transformations
#----------------------------------------------------------------
def linearize_graph(G):
    """
    Converts graph to datum vector. Useful for prospect analysis.
    Inputs:
        P: networkx prospect graph
    """
    vec = {}
    for node in G.nodes:
        vec[node] = G.nodes[node]['emb'] # value of prospect graph
    sorted_keys = sorted(vec.keys()) # can inspect
    vec = np.array([vec[key] for key in sorted_keys]) # enforcing ordering of linear index
    return vec

def set_graph_emb(G, attr=None):
    """sets emb attribute for each node using another attribute"""
    Y = G.copy()
    nx.set_node_attributes(Y, nx.get_node_attributes(G, attr), 'emb')
    return Y

def expand_positive_nodes(G):
    """
    Expands positive nodes to include all neighbors
    Inputs:
        G: networkx graph
    Outputs:
        G: networkx graph with expanded positive nodes
    """
    G_expanded = G.copy()
    for node in G.nodes:
        if G.nodes[node]['gt'] == 1:
            for nhbr in G.neighbors(node):
                G_expanded.nodes[nhbr]['gt'] = 1
    return G_expanded

def flatten_graph_embs(G):
    """
    Flatten/squeeze graph embeddings.
    Inputs: 
        G: Graph with vector-valued node attributes
    Outputs:
        R: flattened version of G
    """
    R = G.copy()
    for node in G.nodes:
        e = G.nodes[node]['emb']
        flat_e = e[0][0]
        R.nodes[node]['emb'] = flat_e
    return R

def rescale_graph(G):
    """
    Min-max scaling to [0,1]. Helpful for importance values of K2 graph
    Inputs: 
        G: Graph with real-valued node attributes
    Outputs:
        R: rescaled version of G
    """
    R = G.copy()
    vals = {}
    for node in G.nodes:
        vals[node] = G.nodes[node]['emb']
    sorted_keys = sorted(vals.keys()) # can inspect
    val_vec = np.array([vals[key] for key in sorted_keys]) # enforcing ordering of linear index
    rescaled_vals = rescale_vec(val_vec)
    for idx, node in enumerate(R.nodes):
        R.nodes[node]['emb'] = rescaled_vals[idx]
    return R

def binarize_graph_otsu(G):
    # automatic binarization
    G = rescale_graph(G)
    G_vec = linearize_graph(G)
    thresh = threshold_otsu(G_vec)    
    B = binarize_graph(G, thresh)
    return B

def binarize_graph(G, thresh, conditional=None):
    """
    Binarize graph based on threshold
    Inputs:
        G: Graph with real-valued node attributes
        thresh: threshold value
    Outputs:
        B: binarized version of G
    """
    B = G.copy()
    vals = {}
    for node in G.nodes:
        vals[node] = G.nodes[node]['emb']
    sorted_keys = sorted(vals.keys()) # can inspect
    val_vec = np.array([vals[key] for key in sorted_keys]) # enforcing ordering of linear index
    binarized_vals = binarize_vec(val_vec, thresh, conditional=conditional)
    for idx, node in enumerate(B.nodes):
        B.nodes[node]['emb'] = binarized_vals[idx]
    return B

def rescale_vec(vec):
    """
    Min-max scaling to [0,1]. Helpful for importance values of K2 graph
    Inputs: 
        vec: vector of importance values
    """
    vec = vec.astype(float)
    denom = np.max(vec) - np.min(vec)
    if denom == 0: # all same value
        return np.ones(vec.shape) * 0.5
    return (vec - np.min(vec)) / denom

def binarize_vec(vec, thresh, conditional=None):
    if conditional == "<":
        return np.where(vec < thresh, 1, 0)
    return np.where(vec > thresh, 1, 0)

def compute_adaptive_thresh_graph(P):
    # Adaptive threshold like Borji et al / Achanta et al
    # t = (2 / (num_els)) * sum(P)
    val_summation = np.sum(list(nx.get_node_attributes(P, 'emb').values()))
    return (2 / P.number_of_nodes()) * val_summation

def compute_adaptive_thresh_vec(vec):
    # Adaptive threshold like Borji et al / Achanta et al
    # t = (2 / (num_els)) * sum(vec)
    val_summation = np.sum(vec)
    return (2 / len(vec)) * val_summation

#================================================================
# Data processing functions
#----------------------------------------------------------------
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

def convert_text2graph_gt(gt, G):
    """
    gt: ground truth array
    G: map graph
    """
    G_gt = G.copy()
    for node in G_gt.nodes():
        # print("hi")
        # i = G_gt.nodes[node]['pos']
        G_gt.nodes[node]['emb'] = int(gt[node])
    # pdb.set_trace()
    return G_gt

def convert_text2graph(Z):
    """
    Convert embedded text (Array of embeddings) to map graph
    """
    G = nx.Graph(origin=(None, None))
    l,d = Z.shape[0], Z.shape[1]
    G.graph.update({'array_size': (l)})
    G.graph.update({'d': d})

    origin_flag = False
    for i in range(l):
        if origin_flag == False:
            G.graph.update({'origin': (i)})
            origin_flag = True
        embed = Z[i,:].reshape(1, -1).astype('double')
        G.add_node(i, pos=i, emb=embed)
        
    nodelist = list(G.nodes())
    for node in nodelist:
        i = G.nodes[node]["pos"] # same as "node"
        for di in range(i-2, i+3):
            if di == i:
                continue
            if di < 0 or di >= l:
                continue
            G.add_edge(node, di, weight=1)

    G.graph.update({'pos_dict': None})
    return G

def convert_arr2graph_gt(gt, G):
    """
    gt: ground truth array
    G: map graph
    """
    G_gt = G.copy()
    for node in G_gt.nodes():
        i,j = G_gt.nodes[node]['pos']
        if gt is not None: # class-1
            G_gt.nodes[node]['emb'] = int(gt[i,j])
            # print("gt[i,j]:", gt[i,j])
        else:
            G_gt.nodes[node]['emb'] = 0 # class-0: ground truth is all zeros
    return G_gt

def convert_arr2graph_GfromGT(G_gt, arr):
    """
    gt: ground truth array
    """
    G = G_gt.copy()
    for node in G.nodes():
        i,j = G.nodes[node]['pos']
        if arr is not None: # class-1
            G.nodes[node]['emb'] = float(arr[i,j][0])
        else:
            G.nodes[node]['emb'] = 0 # class-0: ground truth is all zeros
    return G

def construct_sprite(G, processor, key_in="emb", key_out="emb"):
    """
    Takes a Map Graph G and constructs a sprite from it by applying an embedding quantizer
    AKA: "embedding quantization" for sprite construction
    """
    S = G.copy()
    S.graph.update({'d': 1})
    for node in S.nodes:
        embedding = S.nodes[node][key_in]
        if isinstance(embedding, np.ndarray):
            motif_label = processor.quantizer.predict(embedding.reshape(1, -1).astype(float))[0]
        else: # if not using FM embedding (e.g. amino acid baseline)
            motif_label = processor.quantizer.predict(embedding)
        S.nodes[node][key_out] = motif_label
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

def visualize_sprite(G, modality="graph", prospect_flag=False, gt_flag=False, checking_flag=False, labels=False, color_assign=None):
    # Visualize sprite
    plt.figure()
    colors = list(nx.get_node_attributes(G, 'emb').values())
    print(colors)
    if checking_flag == True:
        colors = [1 for c in colors]
    else:
        colors = [int(c) for c in colors]

    print("colors (min, max):", np.min(colors), np.max(colors))
    
    if type(colors[0]) != int:
        raise Exception("Error: Sprite detected as multi-channel when it should be single-channel and categorical. Please quantize the Datum's Map Graph.")

    our_cmap = joint_cmap # custom_cmap
    
    print(color_assign)
    if color_assign is not None:
        print("yupppp")
        our_cmap = matplotlib.colors.ListedColormap(color_assign, name='from_list')
        colors = [our_cmap([G.nodes[n]["emb"]]) for n in G.nodes]

    if prospect_flag:
        our_cmap = plt.get_cmap("bwr")
        maxmag = get_prospect_range(G)
    if gt_flag:
        our_cmap = plt.get_cmap("bone")
    if checking_flag:
        our_cmap = plt.get_cmap("jet")
        
    # for visualization, we scale the positions
    shape = "o"
    if modality == "image":
        shape = "o"
        eps = 0.1
        spread_pos_dict = G.graph["pos_dict"]
        for k in spread_pos_dict.keys():
            spread_pos_dict[k] = (spread_pos_dict[k][1] * eps, -spread_pos_dict[k][0] * eps)
        pos = nx.spring_layout(G, pos=spread_pos_dict, fixed=spread_pos_dict.keys(), k=10, iterations=100)
    elif modality in ["graph", "text"]:
        pos = nx.kamada_kawai_layout(G)
    
    if prospect_flag:
        if labels:
            nx.draw(G, pos=pos, edge_color="gray", node_size=15, node_shape=shape, node_color=colors, cmap=our_cmap, vmin=-maxmag, vmax=maxmag, with_labels=True, labels=nx.get_node_attributes(G, 'resid'), font_size=6)
        else:
            nx.draw(G, pos=pos, edge_color="gray", node_size=15, node_shape=shape, node_color=colors, cmap=our_cmap, vmin=-maxmag, vmax=maxmag, font_size=6)
    else:
        nx.draw(G, pos=pos, edge_color="gray", node_size=15, node_shape=shape, node_color=colors, cmap=our_cmap)
    plt.axis('off')
    plt.draw()
    
    
def visualize_cell_graph(G, key="cell_type", node_colors=None, prospect_flag=False):
    """Plot dot-line graph for the cellular graph
    Adapted from SPACE_GM codebase
    Args:
        G (nx.Graph): full cellular graph of the region
        node_colors (list): list of node colors. Defaults to None.
    """
    # Extract basic node attributes
    node_coords = [G.nodes[n]['center_coord'] for n in G.nodes]
    node_coords = np.stack(node_coords, 0)
    plt.figure()
    
    if prospect_flag == True:
        node_colors = [matplotlib.cm.get_cmap("bwr")([G.nodes[n][key]]) for n in G.nodes]
    else:
        if node_colors is None:
            unique_cell_types = sorted(set([G.nodes[n][key] for n in G.nodes]))
            cell_type_to_color = {ct: matplotlib.cm.get_cmap("tab20")(i % 20) for i, ct in enumerate(unique_cell_types)}
            node_colors = [cell_type_to_color[G.nodes[n][key]] for n in G.nodes]
            
    assert len(node_colors) == node_coords.shape[0]
    for (i, j, edge_type) in G.edges.data():
        xi, yi = G.nodes[i]['center_coord']
        xj, yj = G.nodes[j]['center_coord']
        if edge_type['edge_type'] == 'neighbor':
            plotting_kwargs = {"c": "k",
                               "linewidth": 1,
                               "linestyle": '-'}
        else:
            plotting_kwargs = {"c": (0.4, 0.4, 0.4, 1.0),
                               "linewidth": 0.3,
                               "linestyle": '--'}
        plt.plot([xi, xj], [yi, yj], zorder=1, **plotting_kwargs)

    plt.scatter(node_coords[:, 0],
                node_coords[:, 1],
                s=10,
                c=node_colors,
                linewidths=0.3,
                zorder=2)
    plt.xlim(0, node_coords[:, 0].max() * 1.01)
    plt.ylim(0, node_coords[:, 1].max() * 1.01)
    plt.axis('off')
    return


def convert_graph2arr(S):
    try:
        h,w = S.graph["array_size"]
        d = S.graph["d"]
        A = np.zeros((h,w,d)) 
        A[:] = np.nan
        for node in S.nodes():
            i,j = S.nodes[node]['pos']
            A[i,j,:] = S.nodes[node]['emb']
    except TypeError:
        w = len(S.nodes())
        h = 1
        d = S.graph["d"]
        A = np.zeros((h,w,d)) 
        A[:] = np.nan
        for node in S.nodes():
            A[:,node,:] = S.nodes[node]['emb']

    return A

def convert_GTgraph2arr(S):
    try:
        h,w = S.graph["array_size"]
        d = 1
        A = np.zeros((h,w,d)) 
        for node in S.nodes():
            i,j = S.nodes[node]['pos']
            A[i,j,:] = S.nodes[node]['emb']
    except TypeError:
        w = len(S.nodes())
        h = 1
        d = 1
        A = np.zeros((h,w,d)) 
        for node in S.nodes():
            A[:,node,:] = S.nodes[node]['emb']
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

def crop_Z(clinical_id, Z, crop_dict):
    i0, i1 = crop_dict[clinical_id][0]
    j0, j1 = crop_dict[clinical_id][1]
    print("crop coords:", crop_dict[clinical_id])
    Z_crop = Z[i0:i1, j0:j1, :]
    return Z_crop

def visualize_Z(Z_path, crop_dict, kmeans_model, mode="dict"):
    Z = np.load(Z_path)
    clinical_id = Z_path.split("/")[-1].split(".")[0].split("-")[1]
    Z = crop_Z(clinical_id, Z, crop_dict)
    Z_viz = quantize_Z(Z, kmeans_model, mode=mode)  
    visualize_quantizedZ(Z_viz)    

def visualize_quantizedZ(Z_viz, prospect_flag=False, colors=None):
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
        cmap = matplotlib.colors.ListedColormap(colors, name='from_list')
        plt.imshow(Z_viz, cmap=cmap)
        plt.colorbar() 
    plt.show()

def visualize_GTmap(arr, sprite_arr):
    if arr.shape[0] == 1:
        maxmag = get_prospect_range(arr)
        basemap = np.where(sprite_arr >= 0, -maxmag/8, 0) # light blue
        new_map = np.where(arr > 0, maxmag, basemap)
    else:
        maxmag = get_prospect_range(arr)
        basemap = np.where(sprite_arr >= 0, -maxmag/8, 0) # light blue
        new_map = np.where(arr > 0, maxmag, basemap)

    plt.figure(figsize=(18, 12), dpi=100)
    plt.yticks([])
    plt.xticks([])
    plt.axis('off')
    plt.imshow(new_map, cmap=plt.get_cmap("bwr"), vmin=-maxmag, vmax=maxmag)
    plt.colorbar() 
    plt.show()
    
#===========================================
# utility functions for visualizing proteins
#===========================================
from Bio.PDB import PDBList, PDBParser, Select, PDBIO
from Bio.PDB.Selection import unfold_entities
pdbl = PDBList()
class NonHetSelect(Select):
    def accept_residue(self, residue):
        return 1 if residue.id[0] == " " else 0

def load_structure(pdbc, pdb_dir='../data'):
    # from collapse import process_pdb
    # from atom3d.util.formats import df_to_bp
    
    pdb, chain = pdbc[:4], pdbc[-1]
    # fname = os.path.join(pdb_dir, f'{pdb}.pdb')
    # df = process_pdb(fname, chain=chain, include_hets=False)
    
    pdbl.retrieve_pdb_file(pdb, pdir='../data', file_format='pdb')
    bp = PDBParser().get_structure(pdbc, f'../data/pdb{pdb}.ent')
    io = PDBIO()
    io.set_structure(bp)
    io.save(f'../data/pdb{pdb}.ent', NonHetSelect())
    bp = PDBParser().get_structure(pdbc, f'../data/pdb{pdb}.ent')
    
    for c in unfold_entities(bp, "C"):
        if c.id == chain:
            return c
    # return bp
    
def visualize_protein_sprite(sprite, prospect_flag=False, gt_flag=False, colors=None):
    import nglview
    from matplotlib.colors import rgb2hex, Normalize
    
    if colors is not None:
        our_cmap = matplotlib.colors.ListedColormap(colors, name='from_list')
    else:
        our_cmap = joint_cmap
    if prospect_flag:
        our_cmap = plt.get_cmap("bwr")
        maxmag = get_prospect_range(sprite)
        norm = Normalize(vmin=-maxmag, vmax=maxmag)
    elif gt_flag:
        our_cmap = plt.get_cmap("bwr")
        norm = Normalize(vmin=-1, vmax=1)
    else:
        norm = lambda x: x
    
    struct = load_structure(sprite.graph['id'])
    color_resids = list(zip([rgb2hex(our_cmap(norm(x))) for x in list(nx.get_node_attributes(sprite, 'emb').values())], [x[1:] for x in list(nx.get_node_attributes(sprite, 'resid').values())]))
    scheme = nglview.color._ColorScheme(color_resids, 'sprite')
    view = nglview.show_biopython(struct, default_representation=False)
    view.add_cartoon(color=scheme)
    # view.add_ball_and_stick(color=scheme)
    # view.add_surface(color=scheme)
    view.center_view()
    return view

class AAQuantizer(object):
    def __init__(self):
        self.aa_to_label = lambda x: {
            'A': 0,
            'R': 1,
            'N': 2,
            'D': 3,
            'C': 4,
            'E': 5,
            'Q': 6,
            'G': 7,
            'H': 8,
            'I': 9,
            'L': 10,
            'K': 11,
            'M': 12,
            'F': 13,
            'P': 14,
            'S': 15,
            'T': 16,
            'W': 17,
            'Y': 18,
            'V': 19
        }.get(x, 20)
    
    def predict(self, resid):
        return self.aa_to_label(resid[0])

