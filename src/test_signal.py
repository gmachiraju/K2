import pdb
from utils import binarize_graph_otsu, binarize_graph_0, deserialize, serialize, visualize_cell_graph, deserialize_model
from utils import prospect_diffusive
import networkx as nx
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression

from evaluation import region_prevalence, mean_region_dispersion
from utils import visualize_cell_graph, construct_sprite

import argparse
import warnings
from time import sleep
from alive_progress import alive_bar

def get_dataset_threshold():
    # for dataset-wide otsu norm
    # raise NotImplementedError()
    pass

def get_binary_P(G_path, k2m, viz_flag=False, thresh_style="otsu", prospect_style="equal"):
    """
    Prospect and automatically returns binary graph. Uses Otsu thresholding as a binary thresholding method
    Input:
        G_path: str, path to serialized graph
        k2m: K2 Model object
    Output:
        B: binary graph
    """
    G = deserialize(G_path)
    
    if prospect_style == "equal":
        P = k2m.prospect(G)
    elif prospect_style == "decay":
        P = prospect_diffusive(k2m, G)
    else:
        raise ValueError("Invalid prospect style")
        
    if thresh_style == "otsu":
        B = binarize_graph_otsu(P)
    elif thresh_style == "sign":
        B = binarize_graph_0(P)
    else:
        raise ValueError("Invalid thresholding style")
    
    if viz_flag == True:
        visualize_cell_graph(P, key="emb", prospect_flag=True)
        visualize_cell_graph(B, key="emb", prospect_flag=True, binarized_flag=True)
    return B

def construct_synthetic_binary(G_path, ccs):
    G = deserialize(G_path)
    B = G.copy()
    for node in B.nodes:
        B.nodes[node]["emb"] = 0
    for cc in ccs:
        for node in cc:
            B.nodes[node]["emb"] = 1
    # visualize_cell_graph(B, key="emb", prospect_flag=True)
    return B

def get_label_from_dict(G_name, label_dict):
    G_id = int(G_name.split(".")[0].split("S")[1])
    return label_dict[G_id]

def class0_accuracy_fpr(B): 
    N = B.number_of_nodes()
    fp = np.sum(list(nx.get_node_attributes(B, "emb").values()))
    return fp / N

def class0_accuracy_tnr(B):
    N = B.number_of_nodes()
    fp = np.sum(list(nx.get_node_attributes(B, "emb").values()))
    return 1 - (fp / N)

def get_salient_subgraphs(B):
    B_drop = B.copy()
    B_drop.remove_nodes_from([n for n in B_drop.nodes if B_drop.nodes[n]["emb"] == 0])
    ccs = [c for c in nx.connected_components(B_drop)]
    return ccs

def get_pooled_embeds(G_path, ccs, pool_fn="max"):
    G = deserialize(G_path)
    cc_embeds = [np.array([G.nodes[n]["emb"] for n in cc]) for cc in ccs]
    if pool_fn == "max":
        cc_pooled = [np.max(cc, axis=0) for cc in cc_embeds]
    elif pool_fn == "mean":
        cc_pooled = [np.mean(cc, axis=0) for cc in cc_embeds]
    return cc_pooled

def construct_class_dataset(embed_dict_path, size_dict_path, cutoff=0):
    """
    takes list of vectors and pools them
    also finds range of subgraph sizes to sample from class-0 
    """
    embed_dict = deserialize(embed_dict_path)
    size_dict = deserialize(size_dict_path)
    
    all_embeds = []
    for example in embed_dict.keys():
        embed_list = embed_dict[example]
        if cutoff > 0:
            cc_sizes = size_dict[example]
            for i,s in enumerate(cc_sizes):
                if s > cutoff:
                    all_embeds.append(embed_list[i])
        else: 
            all_embeds.extend(embed_list)
        
    Xc = np.array(all_embeds)
    return Xc

def sample_region(G, s):
    cc = nx.generate_random_paths(G, sample_size=1, path_length=s) # newer versions may have seed
    cc = [random_path for random_path in cc]
    cc = list(set(cc[0])) # remove the duplicates
    return cc

def sample_regions_in_range(G_path, size_range, num_samples, cutoff):
    G = deserialize(G_path) 
    sizes = np.random.uniform(size_range[0], size_range[1], num_samples)
    ccs = []
    for s in sizes:
        cc = sample_region(G, int(s))
        # future make sure they aren't overlapping?
        ccs.append(cc)
    return ccs

def dummy_baseline(G_dir, label_dict, size_range, num_samples, dict_path): #, cutoff):
    cc_stat_dict = load_stat_dict(dict_path)
    model_str = "dummy_" + str(size_range[0]) + "-" + str(size_range[1]) + "_" + str(num_samples) #+ "_" + str(cutoff)
    print(model_str)
    if model_str in cc_stat_dict.keys():
        print("Skipping b/c already analyzed:", model_str)
        return
    
    rps, mrds = [], []
    embeds = []
    ys = []
    tnrs = []
    cc_sizes = []
    sal = []
    datum_y = []
    N = len(os.listdir(G_dir))
    
    # assuming notebook usage
    with alive_bar(N, force_tty=True) as bar:
        bar.text('both classes')
        for j,G in enumerate(os.listdir(G_dir)):
            y = get_label_from_dict(G, label_dict)
            G_path = os.path.join(G_dir, G)
            cc_list = sample_regions_in_range(G_path, size_range, num_samples, None)
            B = construct_synthetic_binary(G_path, cc_list)
            
            cc_sizes.append([len(c) for c in cc_list])
            cc_pooled = get_pooled_embeds(G_path, cc_list)
            sal.append(np.max(cc_pooled, axis=0))
            embeds.extend(cc_pooled)
            if y == 1:
                datum_y.append(1)
                rps.append(region_prevalence(B))
                mrds.append(mean_region_dispersion(B))
                ys.extend([1] * num_samples)
            else:
                datum_y.append(0)
                tnrs.append(1.0)
                ys.extend([0] * num_samples)
            bar()
                        
    X = np.array(embeds)
    y = np.array(ys)
    print("Training model to predict region class...")
    outs = test_of_signal(X,y)
    
    X = np.array(sal)
    y = np.array(datum_y)
    print("Training model to predict salience class...")
    outs_sal = test_of_signal(X,y)
    
    if size_range[1] > 1:
        # now do Test of Signal with region cutoffs
        X,y = filter_by_cutoff_sampling(cc_sizes, X, y, cutoff=1)
        print("Training model to predict region class (cutoff 1)...")
        outs_c1 = test_of_signal(X, y)
        
        X,y = filter_by_cutoff_sampling(cc_sizes, X, y, cutoff=10)
        print("Training model to predict region class (cutoff 10)...")
        outs_c10 = test_of_signal(X, y)
        
        X,y = filter_by_cutoff_sampling(cc_sizes, X, y, cutoff=25)
        print("Training model to predict region class (cutoff 25)...")
        outs_c25 = test_of_signal(X, y)
    else:
        outs_c1 = ("none")
        outs_c10 = ("none")
        outs_c25 = ("none")

    # load stats into dict
    cc_stat_dict[model_str] = {"prev": rps,
                            "disp": mrds,
                            "tnr": tnrs,
                            "auc_score_sal": outs_sal,
                            "auc_score": outs,
                            "auc_score_c1": outs_c1,
                            "auc_score_c10": outs_c10,
                            "auc_score_c25": outs_c25}
    
    serialize(cc_stat_dict, dict_path)
    print("Completed:", model_str)

def sample_class0_regions(G_path, size_dict_path, num_samples, viz_flag=False):
    all_sizes = []
    size_dict = deserialize(size_dict_path)
    for k in size_dict.keys():
        all_sizes.extend(size_dict[k])
    if viz_flag:
        plt.figure()
        plt.hist(all_sizes)
        plt.show()

    G = deserialize(G_path)    
    sizes = np.random.choice(all_sizes, size=num_samples)
    
    ccs = []
    for s in sizes:
        cc = sample_region(G, s)
        ccs.append(cc)
    return ccs

def filter_by_cutoff_sampling(cc_sizes, X, y, cutoff=1):
    Xc = []
    yc = []
    # flatten cc_sizes
    cc_sizes = [s for sublist in cc_sizes for s in sublist]
    
    for i,ex in enumerate(X):
        if cc_sizes[i] > cutoff:
            Xc.append(ex)
            yc.append(y[i])
    return np.array(Xc), np.array(yc)


def filter_by_cutoff(tmp_dir1, tmp_dir0, size_dict_path, cutoff=10):
    X1c = construct_class_dataset(tmp_dir1, size_dict_path, cutoff=cutoff)
    N = X1c.shape[0]
    y1c = np.ones(N)
    X0c = construct_class_dataset(tmp_dir0, size_dict_path, cutoff=cutoff)
    if X0c.size == 0:
        X0c = construct_class_dataset(tmp_dir0, size_dict_path)[:N,:] # no cutoff here since we could lose
    y0c = np.zeros(X0c.shape[0])

    Xc = np.vstack([X1c, X0c])
    yc = np.hstack([y1c, y0c])
    return Xc, yc


def test_of_signal(X,y):
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # clf = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    clf = LogisticRegression(random_state=0, max_iter=5000).fit(X, y)
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:,1] # class 1
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
    return metrics.auc(fpr, tpr), clf.score(X_test, y_test)

#=============================================
# main function and helper function
def load_stat_dict(dict_path):
    if os.path.isfile(dict_path):
        print("loading stat dict...")
        cc_stat_dict = deserialize(dict_path)
    else:
        print("creating stat dict...")
        cc_stat_dict = {}
        serialize(cc_stat_dict, dict_path)
    return cc_stat_dict
    
def load_size_dict(size_dict_path):
    if os.path.isfile(size_dict_path):
        print("loading size dict...")
        cc_sizes = deserialize(size_dict_path)
    else:
        print("creating size dict...")
        cc_sizes = {}
        serialize(cc_sizes, size_dict_path)
    return cc_sizes

def get_cache_path(G_name, tmp_dir):
    G_name = G_name.split(".")[0]
    G_curr = G_name + ".pkl"
    return G_curr, os.path.join(tmp_dir, G_curr)

def analyze_model(model, model_str, label_dict, G_dir, cache_dir, status_dict_path, debugging_flag=True, notebook_flag=None, ignore_flag=False):    
    model_name = model_str.split(".model")[0]
    print("On model:", model_name)
    size_dict_path = os.path.join(cache_dir, model_name + ".size_dict")
    stat_dict_path = os.path.join(cache_dir, model_name + ".stat_dict")
    embed0_dict_path = os.path.join(cache_dir, model_name + ".embed0_dict")
    embed1_dict_path = os.path.join(cache_dir, model_name + ".embed1_dict")
    # last two vars are for pooled embeds
    status_dict = deserialize(status_dict_path)
    
    num_total = len(os.listdir(G_dir))
    num_1s = int(np.sum([get_label_from_dict(G, label_dict) for G in os.listdir(G_dir)]))
    num_0s = num_total - num_1s

    # pdb.set_trace()
    # check if model has existing dicts
    if os.path.isfile(stat_dict_path):
        print("Found model cache, checking this model...")
        stat_dict = deserialize(stat_dict_path)
        # pdb.set_trace()
        if stat_dict != {}:
            if stat_dict == ["Limited detection"] or len(stat_dict["prev"]) == num_1s or status_dict[model_str] == 1:
                print("Model already analyzed, skipping...")
                status_dict[model_str] == 1
                serialize(status_dict, status_dict_path)
                return 1
            elif status_dict[model_str] == 0.5:
                if ignore_flag == True:
                    print("Model is currently being analyzed or is partially analyzed, skipping as requested...")
                    status_dict[model_str] == 0.5
                    serialize(status_dict, status_dict_path)
                    return 0.5
                else:
                    print("Model is currently being analyzed or is partially analyzed, completing it as requested...")
                    status_dict[model_str] == 0.5
                    serialize(status_dict, status_dict_path)
            else:
                print("Model has missing stats, re-analyzing...")
        else:
            print("Model has missing stats, re-analyzing...")
    else:
        print("New model to analyze! Creating model cache...")
    
    # iterate on dataset
    rps, mrds, tnrs = [], [], []
    size_dict, stat_dict, embed0_dict, embed1_dict = {}, {}, {}, {}
    
    sal_pooled_list = []
    nonsal_pooled_list = []
    
    # save copy of dicts so other workers can move on
    serialize(size_dict, size_dict_path)
    serialize(stat_dict, stat_dict_path)
    serialize(embed0_dict, embed0_dict_path)
    serialize(embed1_dict, embed1_dict_path)
    
    # starting analysis
    status_dict[model_str] == 0.5
    serialize(status_dict, status_dict_path)
    
    # first iterate through class-1
    # with alive_bar(num_1s, force_tty=notebook_flag) as bar:
    #     bar.text('Class-1')
    embed_log = []
    for j,G in enumerate(os.listdir(G_dir)):
        y = get_label_from_dict(G, label_dict)
        # embed_log = []
        if y == 1:
            G_name = G.split(".")[0]
            G_path = os.path.join(G_dir, G)
            B = get_binary_P(G_path, model, viz_flag=False)
            with warnings.catch_warnings(): # after 3.10, add: action="ignore"
                warnings.simplefilter("ignore") # needs updating after 3.10
                rps.append(region_prevalence(B))
                mrds.append(mean_region_dispersion(B))
                
            cc_list = get_salient_subgraphs(B)
            size_dict[G_name] = [len(c) for c in cc_list]    
            cc_pooled = get_pooled_embeds(G_path, cc_list)
            #=======================
            # ADD sal_pooled to get pooled for salient region (1) - add to a list
            sal_pooled_list.append(np.max(cc_pooled, axis=0))
            #=======================
            embed1_dict[G_name] = cc_pooled
            
            if len(embed_log) > 30 and np.sum(embed_log) == 0:
                print("Warning: breaking early b/c very few detected biomarkers in class-1 examples")
                stat_dict = ["Limited detection"]
                serialize(stat_dict, stat_dict_path)
                return 1
            else:
                valid_cc = [1 for el in cc_pooled if el is not None]
                valid_ex = np.sum(valid_cc)
                embed_log.append(valid_ex) # track what we find
                if debugging_flag == True:
                    break
        print("class-1 sample:", j)
                # bar()
    
    # save dicts
    serialize(size_dict, size_dict_path)
    serialize(embed1_dict, embed1_dict_path)
    
    X1 = construct_class_dataset(embed1_dict_path, size_dict_path)
    y1 = np.ones(X1.shape[0])
    num_embeds = X1.shape[0]
    print("num embeds for class-1:", num_embeds)
    num_samples = np.max([num_embeds // num_0s, 1])
    print("num sample per class-0 example:", num_samples)
    
    # now iterate through class-0 again for sampling
    # with alive_bar(num_0s, force_tty=notebook_flag) as bar:
    #     bar.text('Class-0')
    for j,G in enumerate(os.listdir(G_dir)):
        y = get_label_from_dict(G, label_dict)
        if y == 0:
            G_name = G.split(".")[0]
            G_path = os.path.join(G_dir, G)
            B = get_binary_P(G_path, model)
            tnrs.append(class0_accuracy_tnr(B))
            
            cc_list = sample_class0_regions(G_path, size_dict_path, num_samples)
            size_dict[G_name] = [len(c) for c in cc_list]
            cc_pooled = get_pooled_embeds(G_path, cc_list)
            nonsal_pooled_list.append(np.max(cc_pooled, axis=0))
            embed0_dict[G_name] = cc_pooled
            
            if debugging_flag == True:
                break
        print("class-0 sample:", j)
                # bar()
    
    serialize(size_dict, size_dict_path)
    serialize(embed0_dict, embed0_dict_path)
    X0 = construct_class_dataset(embed0_dict_path, size_dict_path)
    y0 = np.zeros(X0.shape[0])
    print("num embeds for class-0:", X0.shape[0])
    
    # combine datasets
    X = np.vstack([X1, X0])
    y = np.hstack([y1, y0])
    # print(y)
    print("Training model to predict region class...")
    print("shapes:", X.shape, y.shape)
    outs = test_of_signal(X, y)
    
    # ============TO ADD: pool per sample=======
    X1 = np.array(sal_pooled_list)
    X0 = np.array(nonsal_pooled_list)
    y1 = np.ones(X1.shape[0])
    y0 = np.zeros(X0.shape[0])
    X = np.vstack([X1, X0])
    y = np.hstack([y1, y0])
    # print(y)
    print("Training model to predict pooled salience class...")
    print("shapes:", X.shape, y.shape)
    outs_sal = test_of_signal(X, y)
    #=========================================
    
    # now do Test of Signal with region cutoffs
    X,y = filter_by_cutoff(embed1_dict_path, embed0_dict_path, size_dict_path, cutoff=1)
    print("Training model to predict region class (cutoff 1)...")
    print("shapes:", X.shape, y.shape)
    outs_c1 = test_of_signal(X, y)
    
    X,y = filter_by_cutoff(embed1_dict_path, embed0_dict_path, size_dict_path, cutoff=10)
    print("Training model to predict region class (cutoff 10)...")
    print("shapes:", X.shape, y.shape)
    outs_c10 = test_of_signal(X, y)
    
    X,y = filter_by_cutoff(embed1_dict_path, embed0_dict_path, size_dict_path, cutoff=25)
    print("Training model to predict region class (cutoff 25)...")
    print("shapes:", X.shape, y.shape)
    outs_c25 = test_of_signal(X, y)

    # load stats into dict
    stat_dict = {"prev": rps,
                "disp": mrds,
                "tnr": tnrs,
                "auc_score_sal": outs_sal,
                "auc_score": outs,
                "auc_score_c1": outs_c1,
                "auc_score_c10": outs_c10,
                "auc_score_c25": outs_c25}
    
    if debugging_flag == True:
        print("Debugging flag is on, breaking early")
        print(stat_dict)
        return 0
    
    serialize(stat_dict, stat_dict_path)
    print("Completed:", model_str)
    print()
    return 1 # successful exit code

    
    
def generate_hypotheses_for_model(model, label_dict, G_dir, save_dir, notebook_flag=None, thresh_style="otsu", prospect_style="equal"):    
    num_total = len(os.listdir(G_dir))
    print("Total:", num_total)
    print("skipping class-0 data")
    # first iterate through class-1
    with alive_bar(num_total, force_tty=notebook_flag) as bar:
        bar.text('Full dataset')
        for j,G in enumerate(os.listdir(G_dir)):
            y = get_label_from_dict(G, label_dict)
            if y == 0:
                bar()
                continue
            G_path = os.path.join(G_dir, G)
            B = get_binary_P(G_path, model, viz_flag=False, thresh_style=thresh_style, prospect_style=prospect_style)
            G_new = G.split(".")[0] + "_bin.obj"
            serialize(B, os.path.join(save_dir, G_new))
            bar()

def graph_hadamard(G1, G2, key1, key2):
    G = G1.copy()
    for n in G.nodes:
        G.nodes[n][key1] = np.multiply(G1.nodes[n][key1], G2.nodes[n][key2])
    return G

def graph_element_add(G, to_add, key):
    G_new = G.copy()
    for n in G_new.nodes:
        G_new.nodes[n][key] = G.nodes[n][key] + to_add
    return G_new

def graph_element_assign(G, if_value, then_assign, key):
    G_new = G.copy()
    for n in G_new.nodes:
        if G.nodes[n][key] == if_value:
            G_new.nodes[n][key] = then_assign
    return G_new

def graph_element_copy(G, old_key, new_key):
    """Copies an attribute stored in all nodes (using old_key) to new key (new_key)
    Args:
        G (nx graph): graph to modify
        old_key (str): key to pull from
        new_key (str): key to assign to

    Returns:
        nx graph
    """
    G_new = G.copy()
    for n in G_new.nodes:
        G_new.nodes[n][new_key] = G_new.nodes[n][old_key] 
    return G_new

def generate_join_hypotheses(G_dir1, G_dir2, save_dir, notebook_flag=None):
    num_total = len(os.listdir(G_dir1))
    print("Total:", num_total)
    
    with alive_bar(num_total, force_tty=notebook_flag) as bar:
        bar.text('Full dataset')
        for G1, G2 in zip(os.listdir(G_dir1), os.listdir(G_dir2)):
            G_new = G1.split(".")[0] + "_joint.obj"
            G1_path = os.path.join(G_dir1, G1)
            G2_path = os.path.join(G_dir2, G2)
            assert G1 == G2
            G1 = deserialize(G1_path)
            G2 = deserialize(G2_path)
            G = graph_hadamard(G1, G2, "emb", "emb")
            serialize(G, os.path.join(save_dir, G_new))
            bar()

def generate_join_concepts(G_dir1, G_dir2, label_dict, proc_path, save_dir, notebook_flag=None):
    # G1: og 
    # G2: joint binary
    num_total = len(os.listdir(G_dir1))
    print("Total:", num_total)
    print("skipping class-0...")
    
    proc = deserialize_model(proc_path)
    
    with alive_bar(num_total, force_tty=notebook_flag) as bar:
        bar.text('Full dataset')
        for G1 in os.listdir(G_dir1):
            y = get_label_from_dict(G1, label_dict)
            if y == 0:
                continue
            G_new = G1.split(".")[0] + "_joint_concept.obj"
            G1_path = os.path.join(G_dir1, G1)
            try:
                G2 = G1.split(".")[0] + "_bin_joint.obj"
                G2_path = os.path.join(G_dir2, G2)
                G2 = deserialize(G2_path)
            except FileNotFoundError:
                G2 = G1.split(".")[0] + "_bin.obj"
                G2_path = os.path.join(G_dir2, G2)
                G2 = deserialize(G2_path)
            
            G1 = deserialize(G1_path)
            S = construct_sprite(G1, proc, key_in="emb", key_out="concept")
            S = graph_element_copy(S, "concept", "salient")  
            
            # all used to say "concept" instead of "salient"
            S_shift1 = graph_element_add(S, 1, "salient")
            S_cut = graph_hadamard(S_shift1, G2, "salient", "emb")         
            S_drop = graph_element_assign(S_cut, 0, np.nan, "salient") # call this new attribute salient 
            S_shift2 = graph_element_add(S_drop, -1, "salient")
            
            # visualize_cell_graph(S_shift2, key="concept", edge_flag=True)
            # pdb.set_trace()
            serialize(S_shift2, os.path.join(save_dir, G_new))
            bar()
    
def graph2df(G, attributes):
    """
    Converts nodes of a networkx graph G and specified attributes to a pandas DataFrame.
    
    Parameters:
    G (networkx.Graph): The input graph.
    attributes (list): A list of node attributes to include as columns in the DataFrame.
    
    Returns:
    pd.DataFrame: A DataFrame containing the nodes and their specified attributes.
    """
    # Initialize an empty list to store node data
    data = []
    
    # Iterate over all nodes in the graph
    for node in G.nodes(data=True):
        # Extract node id and attributes
        node_id, node_attrs = node
        # Create a dictionary to hold the node data
        node_data = {'node': node_id}
        # Add requested attributes to the node data
        for attr in attributes:
            node_data[attr] = node_attrs.get(attr, None)
        # Append the node data to the list
        data.append(node_data)
    
    # Convert the list of node data to a pandas DataFrame
    df = pd.DataFrame(data)
    return df


#========================BASH SCRIPTING========================
def main():
    model_dir = "/scr/gmachi/prospection/K2/notebooks/spatial-bio/outputs/gridsearch_results/k2models"    
    label_path = "/scr/biggest/gmachi/datasets/celldive_lung/processed/label_dict.obj"
    label_dict = deserialize(label_path)
    G_dir = "/scr/biggest/gmachi/datasets/celldive_lung/for_ml/for_prospect/"
    cache_dir = "/scr/biggest/gmachi/datasets/celldive_lung/analysis_cache"
    status_dict_path = "/scr/gmachi/prospection/K2/notebooks/spatial-bio/status_dict.obj"
    status_dict = deserialize(status_dict_path)
        
    # change for testing ================
    k = 11 #[8,9,10,11,12,13,14,15,16,17,18,19,20]
    #====================================
    print("Starting analysis for k =", k)
    # count number valid out of 16 models
    all_models = [model_str for model_str in os.listdir(model_dir) if "k"+str(k) in model_str]
    N = len(all_models)
    print("Total models to analyze:", N)
    
    num_total = len(os.listdir(G_dir))
    num_1s = int(np.sum([get_label_from_dict(G, label_dict) for G in os.listdir(G_dir)]))
    num_0s = num_total - num_1s

    ran_set = []
    running_set = []
    to_run_set = []
    for file_str in os.listdir(cache_dir):
        if "k"+str(k) not in file_str:
            continue
        if "tau1.00" in file_str or "tau2.00" in file_str:
            print("Skipping b/c tau > 0:", model_str)
            continue
        if "stat_dict" not in file_str:
            continue
        try:
            rd = deserialize(os.path.join(cache_dir, file_str))
            if len(rd.keys()) == 8 or len(rd["prev"]) == num_1s:
                ran_set.append(file_str.split(".")[0])
            else:
                running_set.append(file_str.split(".")[0])
        except KeyError:
            to_run_set.append(file_str.split(".")[0])
            # print(rd)
            # print(os.path.join(cache_dir, file_str))
            
    print("Already analyzed:", len(ran_set), "out of", N)
    print("Currently running:", len(running_set), "out of", N)
    print("To analyze (partially complete):", len(to_run_set), "out of", N)
    print()
    # print("ran:", ran_set)
    # print("to run:", to_run_set)
    # xxxxx
    
    # start analysis
    for i,model_str in enumerate(os.listdir(model_dir)):
        if "k"+str(k) not in model_str:
            continue
        if "tau1.00" in model_str or "tau2.00" in model_str:
            print("Skipping b/c tau > 0:", model_str)
            continue
        if model_str.split(".")[0] in ran_set:
            print("Skipping b/c already analyzed:", model_str)
            continue
        
        print("On model:", i, "/", len(os.listdir(model_dir)), ":", model_str)
        model_path = os.path.join(model_dir, model_str)

        # check if valid model
        try:
            model = deserialize_model(model_path)
        except EOFError:
            print("Skipping b/c corrupted:", model_str)
            continue
        
        exit_code = analyze_model(model, model_str, label_dict, G_dir, cache_dir, status_dict_path, debugging_flag=False, notebook_flag=False, ignore_flag=False)
        status_dict[model_str] = exit_code
        serialize(status_dict, "status_dict.obj")

if __name__ == "__main__":
    main()
