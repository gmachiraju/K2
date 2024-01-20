import os
import numpy as np
import pandas as pd
# from tqdm.notebook import tqdm
import pdb 
import time
from copy import copy
import argparse
import networkx as nx
import utils

from utils import serialize, deserialize, serialize_model, deserialize_model
from utils import compute_adaptive_thresh_graph, compute_adaptive_thresh_vec
from utils import binarize_graph, binarize_vec, rescale_graph, rescale_vec, linearize_graph, set_graph_emb, AAQuantizer
from k2 import K2Processor, K2Model
from metrics import confusion, msd, auroc, auprc, ap
import metrics

def train_gridsearch(sweep_dict, save_dir, encoder_name, gt_dir, process_args, model_args):
    """
    master script to run grid search over hyperparameters (aka tuning)
    Inputs:
        sweep_dict: dictionary of hyperparameters to sweep over
        save_dir: directory to store results
        encoder_name: name of encoder
        gt_dir: directory of ground truth graphs
        process_args: dictionary of hyperparameters for processor
        model_args: dictionary of hyperparameters for model
    """
    results_dict, results_dir, results_cache_dir, proc_cache_dir, model_cache_dir, linearized_cache_dir = setup_gridsearch(save_dir, encoder_name)
    if "cutoff" in sweep_dict.keys():
        num_ht = len(sweep_dict["k"]) * len(sweep_dict["r"]) * len(sweep_dict["alpha"]) * len(sweep_dict["tau"]) * len(sweep_dict["cutoff"])
    else:
        num_ht = len(sweep_dict["k"]) * len(sweep_dict["r"]) * len(sweep_dict["alpha"]) * len(sweep_dict["tau"])
    num_lm = len(sweep_dict["k"]) * len(sweep_dict["r"]) * len(sweep_dict["lambda"])
    print("We have %d models to train..." % (num_ht +  num_lm))
    print("...and have %d models trained so far!" % len(os.listdir(model_cache_dir)))
    print("="*40)

    if "metal" in process_args.keys():
        metal = process_args["metal"]
    
    for cutoff in sweep_dict.get("cutoff", [np.nan]):
        if not np.isnan(cutoff):
            if encoder_name == 'AA':
                encoder_name = 'COLLAPSE'
            process_args["embeddings_path"] = f"../data/{encoder_name}_{metal}_{cutoff}_train_embeddings_2.pkl"
            model_args["train_graph_path"] = f"../data/{encoder_name}_{metal}_{cutoff}_train_graphs_2"
        for k in sweep_dict["k"]:
            proc, processor_name = fetch_processor(k, proc_cache_dir, process_args, cutoff=cutoff)
            # if process_args["embeddings_type"] == "memmap" and process_args["datatype"] == "histo":
            #     serialize(proc, os.path.join(proc_cache_dir, processor_name)) # special case 
            # else:
            # pdb.set_trace()
            serialize_model(proc, os.path.join(proc_cache_dir, processor_name))

            for r in sweep_dict["r"]:
                # Predictive: ElasticNet
                model_args["variant"] = "predictive"
                for lam in sweep_dict["lambda"]:
                    print("Gridsearch: currently on ElasticNet model with k=%d, r=%d, cutoff=%f, lam=%f" % (k, r, cutoff, lam))
                    model, model_str = fetch_model(proc, r, model_cache_dir, model_args, cutoff=cutoff, lam=lam)
                    if model_str in results_dict.keys():
                        continue # already trained and stored
                    gridsearch_iteration_wrapper(model, model_str, model_args, gt_dir, results_cache_dir, results_dict, results_dir, model_cache_dir, linearized_cache_dir)

                # Inferential: Hypothesis test
                model_args["variant"] = "inferential"
                for alpha in sweep_dict["alpha"]:
                    for tau in sweep_dict["tau"]:
                        print("Gridsearch: currently on hypothesis test with k=%d, r=%d, cutoff%f, alpha=%f, tau=%f" % (k, r, cutoff, alpha, tau))   
                        
                        model, model_str = fetch_model(proc, r, model_cache_dir, model_args, cutoff=cutoff, alpha=alpha, tau=tau)
                        if model_str in results_dict.keys():
                            continue # already trained and stored
                        gridsearch_iteration_wrapper(model, model_str, model_args, gt_dir, results_cache_dir, results_dict, results_dir, model_cache_dir, linearized_cache_dir)
                
def gridsearch_iteration_wrapper(model, model_str, model_args, gt_dir, results_cache_dir, results_dict, results_dir, model_cache_dir, linearized_cache_dir):
    start = time.time()
    model_results_dict, datum_linearized_dict = gridsearch_iteration(model, model_args, gt_dir)
    results_dict[model_str] = os.path.join(results_cache_dir, model_str) # save_path
    serialize(model_results_dict, os.path.join(results_cache_dir, model_str))
    serialize(results_dict, results_dir)
    serialize(datum_linearized_dict, os.path.join(linearized_cache_dir, model_str))
    serialize_model(model, os.path.join(model_cache_dir, model_str))
    print("Saved model/results!" + "\n" + "Time elapsed: %.2f seconds" % (time.time() - start) + "\n" + "-"*40)

# Helper functions
#=================
def gridsearch_iteration(model, model_args, gt_dir, thresh="all", arm="train"):
    """
    This function can be used as a part of a grid search or for a single model/threshold (testing)
    Inputs:
        model: trained K2Model object
    Outputs:
        model_results_dict: a dictionary of metrics
    """
    if thresh == "all":
        reg_thresholds = [np.round(el,1) for el in np.linspace(0,1,11)] # [0.1, 0.2, ..., 0.9, 1.0]
        idx_adaptive = len(reg_thresholds)
        thresholds = reg_thresholds + [np.nan, np.nan] # placeholders for adaptive
    else:
        thresholds = [thresh]
    
    model_results_dict = {} # returned object
    data_linearized_dict = {} # for IID eval
    G_files = os.listdir(model_args["train_graph_path"])
    # with tqdm(total=len(G_files), desc="Eval samples...") as pbar:
    for t, G_name in enumerate(G_files):
        data_results_dict = {}
        G = deserialize(os.path.join(model_args["train_graph_path"], G_name))
        if model_args["modality"] == "graph":
            Y = set_graph_emb(G, 'gt')
        else:
            if arm == "train":
                Y = deserialize(os.path.join(gt_dir, G_name + "_gt")) # groud truth
            elif arm == "test":
                Y = deserialize(os.path.join(gt_dir, G_name + "-graph")) # groud truth

        if model.processor.quantizer_type == "AA":
            G = set_graph_emb(G, 'resid')
        P = model.prospect(G)
        if thresh == "all":
            threshold_a = compute_adaptive_thresh_graph(P)
            thresholds[idx_adaptive] = (">", threshold_a) # adaptive forward
            thresholds[idx_adaptive + 1] = ("<", threshold_a) # adaptive backward

        # get label
        if model.train_label_dict and arm == "train": # loaded a dict
            y = model.train_label_dict[G_name]
        elif model.train_label_dict and arm == "test": # loaded a dict
            y = model_args["train_label_dict"][G_name]
        else: # internally stored in graph
            y = G.graph["label"]
        
        ## JUST FOR TRACKING GRAPHS WITH NO POSITIVES -- REMOVE LATER
        if (y == 1) and (linearize_graph(Y).sum() == 0):
            print('skipping ' + G.graph["id"] + ': no positive residues')
            continue
        
        if model.variant == "predictive":
            sprite = model.construct_sprite(G)
            g = model.embed_sprite(sprite)
            y_hat = model.classifier.predict_proba(np.expand_dims(g, axis=0))
        else:
            y_hat = np.nan # no prediction for inferential model

        # pdb.set_trace()
        dicts = eval_suite(G_name, P, Y, y, y_hat, thresholds)
        data_results_dict["thresh_msd"] = dicts[0]
        data_results_dict["thresh_cm"] = dicts[1]
        data_results_dict["cont"] = dicts[2]
        data_results_dict["pred"] = dicts[3]
        datum_linearized_dict = dicts[4]
        
        model_results_dict[G_name] = data_results_dict
        data_linearized_dict[G_name] = datum_linearized_dict
        # pdb.set_trace()
        # pbar.set_description('processed: %d' % (1 + t))
        # pbar.update(1)

    return model_results_dict, data_linearized_dict


def eval_baseline_explanations(P_path, Y_path, thresh="all", modality="image", label_dict=None):
    """
    Evaluates baseline models
    """
    if thresh == "all":
        reg_thresholds = [np.round(el,1) for el in np.linspace(0,1,11)] # [0.1, 0.2, ..., 0.9, 1.0]
        idx_adaptive = len(reg_thresholds)
        thresholds = reg_thresholds + [np.nan, np.nan] # placeholders for adaptive
    else:
        thresholds = [thresh]
    
    model_results_dict = {} # returned object
    data_linearized_dict = {} # for IID eval
    for t, P_name in enumerate(os.listdir(P_path)):
        data_results_dict = {}
        P = deserialize(os.path.join(P_path, P_name))
        if modality == "graph":
            pdb.set_trace()
        else:
            Y = deserialize(os.path.join(Y_path, P_name + "-graph")) # groud truth
        
        if thresh == "all":
            threshold_a = compute_adaptive_thresh_graph(P)
            thresholds[idx_adaptive] = (">", threshold_a) # adaptive forward
            thresholds[idx_adaptive + 1] = ("<", threshold_a) # adaptive backward

        # get label
        if modality == "graph": # not implemented
            pdb.set_trace()
        else:
            y = label_dict[P_name]
        y_hat = np.nan

        # pdb.set_trace()
        dicts = eval_suite(P_name, P, Y, y, y_hat, thresholds)
        data_results_dict["thresh_msd"] = dicts[0]
        data_results_dict["thresh_cm"] = dicts[1]
        data_results_dict["cont"] = dicts[2]
        data_results_dict["pred"] = dicts[3]
        datum_linearized_dict = dicts[4]
        
        model_results_dict[P_name] = data_results_dict
        data_linearized_dict[P_name] = datum_linearized_dict
        # pdb.set_trace()
        # pbar.set_description('processed: %d' % (1 + t))
        # pbar.update(1)

    return model_results_dict, data_linearized_dict


def eval_suite(G_name, P, Y, y, y_hat, thresholds):
    """
    Calls on metrics to evaluate a single graph datum
    Can use for any [0,1] heatmap: K2 prospect map, saliency, attention, probability
    """
    # datum_linearized_dict, datum_cont_dict, datum_pred_dict = {}, {}, {}
    P_scaled = rescale_graph(P) # first rescale to [0,1] 
    # Continuous & IID eval
    #----------------------
    P_vec = linearize_graph(P_scaled)
    if y == 1:
        # print(G_name)
        # print(P_vec)
        pass
    Y_vec = linearize_graph(Y)
    datum_linearized = (P_vec, Y_vec) # store for IID eval
    # _dict[(G_name, y)]

    # Continuous eval can only run on class-1 data
    datum_cont = {"auroc": np.nan, "auprc": np.nan, "ap": np.nan}
    if y == 1:
        # try:
        datum_cont = {"auroc": auroc(P_vec, Y_vec), "auprc": auprc(P_vec, Y_vec), "ap": ap(P_vec, Y_vec)}
        # except ValueError:
        #     pdb.set_trace()
        # _dict[(G_name, y)]

    # compute predictions from maps
    #------------------------------
    y_hat_map = few_hot_classification(P_vec, few=10)
    datum_pred = (y_hat, y_hat_map)
    # _dict[(G_name, y)]

    # multi-thresholding eval
    #------------------------
    datum_thresh_msd_dict, datum_thresh_cm_dict = {}, {}
    for thresh in thresholds:
        if type(thresh) == tuple:
            cond, t = thresh[0], thresh[1]
        else:
            cond, t = None, copy(thresh)
        P_bin = binarize_graph(P_scaled, t, conditional=cond) # binarize with threshold
        datum_thresh_msd_dict[thresh] = msd(P_bin)
        P_bin_vec = linearize_graph(P_bin) # linearize
        datum_thresh_cm_dict[thresh] = confusion(P_bin_vec, Y_vec)

    return datum_thresh_msd_dict, datum_thresh_cm_dict, datum_cont, datum_pred, datum_linearized

def setup_gridsearch(save_dir, encoder_name):
    """
    Builds directory structure for grid search. Checks if previous results exist.
    """
    # check if save_dir exists
    if not os.path.exists(save_dir):
        print("Requested save path not found: %s" % save_dir)
        print("Please create directory and try again.")
        exit()

    results_dir = os.path.join(save_dir, encoder_name + "-results_dict.obj")    
    if not os.path.exists(results_dir):
        print("No previous results found at: %s" % results_dir)
        print("Creating new results dictionary...")
        results_dict = {}
        serialize(results_dict, results_dir)
    else:
        print("Previous results found at: %s" % results_dir)
        results_dict = deserialize(results_dir)

    # make a directory for models
    results_cache_dir = os.path.join(save_dir, encoder_name + "-eval_results")
    proc_cache_dir = os.path.join(save_dir, encoder_name + "-fitted_k2_processors")
    model_cache_dir = os.path.join(save_dir, encoder_name + "-fitted_k2_models")
    linearized_cache_dir = os.path.join(save_dir, encoder_name + "-linearized_data")
    if not os.path.exists(results_cache_dir):
        print("Don't see previous folder for results.. Creating directory at: %s" % results_cache_dir)
        os.makedirs(results_cache_dir)
    if not os.path.exists(proc_cache_dir):
        print("Don't see previous folder for fitted K2 processors... Creating directory at: %s" % proc_cache_dir)
        os.makedirs(proc_cache_dir)
    if not os.path.exists(model_cache_dir):
        print("Don't see previous folder for fitted K2 models... Creating directory at: %s" % model_cache_dir)
        os.makedirs(model_cache_dir)
    if not os.path.exists(linearized_cache_dir):
        print("Don't see previous folder for linearized data files... Creating directory at: %s" % linearized_cache_dir)
        os.makedirs(linearized_cache_dir)
    return results_dict, results_dir, results_cache_dir, proc_cache_dir, model_cache_dir, linearized_cache_dir

def fetch_model(proc, r, model_cache_dir, model_args, cutoff=np.nan, alpha=np.nan, tau=np.nan, lam=np.nan):
    """
    Checks existence of model in cache, otherwise spawns and fits model
    Inputs:
        proc: K2Processor object
        r: receptive field
        model_cache_dir: directory to store fitted models
        model_args: dictionary of hyperparameters for model
        alpha, tau, lam: hyperparameters for inferential / predictive model
    Note: np.nan defaults allows us to overwrite and indicate which variant we are using
    """
    k = proc.k
    if not np.isnan(cutoff):
        model_name = "k%d_r%d_cutoff%.2f_alpha%.3f_tau%.2f_lam%.2f.model" % (k, r, cutoff, alpha, tau, lam)
    else:   
        model_name = "k%d_r%d_alpha%.3f_tau%.2f_lam%.2f.model" % (k, r, alpha, tau, lam)

    if model_name in os.listdir(model_cache_dir):
        print("Found fitted model for " + model_name)
        model = deserialize_model(os.path.join(model_cache_dir, model_name))
    else:
        print("Fitting model for " + model_name)
        model = spawn_model(proc, r, alpha, tau, lam, model_args)
        model.create_train_array()
        model.fit_kernel()
    return model, model_name

def fetch_processor(k, proc_cache_dir, process_args, cutoff=np.nan):
    """
    Checks existence of processor in cache, otherwise spawns and fits processor
    Inputs:
        k: chosen number of concepts
        proc_cache_dir: directory to store fitted processors
        process_args: dictionary of hyperparameters for processor
    """
    if not np.isnan(cutoff):
        processor_name = "k%d_cutoff%.2f.processor" % (k, cutoff)
    else:
        processor_name = "k%d.processor" % k    

    if processor_name in os.listdir(proc_cache_dir):
        print("Found fitted processor for k=%d, cutoff=%.2f" % (k, cutoff))
        proc = deserialize_model(os.path.join(proc_cache_dir, processor_name))
    else:
        print("Fitting processor for k=%d" % k)
        proc = spawn_processor(k, process_args)
        if process_args["quantizer_type"] == "AA":
            proc.quantizer = AAQuantizer()
        else:
            proc.fit_quantizer()
    return proc, processor_name

def spawn_processor(k, process_args):
    """
    uses hyperparameters to train/eval K2 processors
    """
    process_args["k"] = k
    processor = K2Processor(process_args)
    return processor

def spawn_model(processor, r, alpha, tau, lam, model_args):
    """
    uses hyperparameters to train/eval K2 models
    """
    model_args["processor"] = processor
    model_args["r"] = r
    model_args["hparams"] = {"alpha": alpha, "tau": tau, "lambda": lam}
    model = K2Model(model_args)
    return model

def few_hot_classification(P_probs, few=10):
    """
    This helps us make a datum-level prediction from a probability map. u
    Note: this is useful for inferential K2 models since they have no classifier abilities post-training
    """
    # works for vector version of probabilities
    relu_prob = P_probs[P_probs > 0.0]
    if len(relu_prob) == 0:
        return 0.0
    sorted_index_array = np.argsort(relu_prob)
    sorted_array = relu_prob[sorted_index_array]
    top_few = sorted_array[-few:]
    return np.mean(top_few)

 # Test eval
 # ===========   
def test_eval(model_str, threshold, test_metrics, model_cache_dir, processor_cache_dir, G_dir, gt_dir, label_dict=None, modality="image", arm="train"):
    """
    Test-set evaluation using top models extracted from training grid search
    SHOULD JUST CALL GRIDSEARCH_ITERATION
    """
    # get model params from model_str
    if modality == "image":
        k,r,alpha,tau,lam = extract_params(model_str)
        hparams = {"alpha": alpha, "tau": tau, "lambda": lam}
        processor = deserialize_model(os.path.join(processor_cache_dir, "k%d.processor" % k))
    elif modality == 'graph':
        k,r,cutoff,alpha,tau,lam = extract_params(model_str)
        hparams = {"alpha": alpha, "tau": tau, "lambda": lam}
        processor = deserialize_model(os.path.join(processor_cache_dir, "k%d_cutoff%.2f.processor" % (k, cutoff)))
        
    encoder = processor_cache_dir.split('/')[-1].split("-")[0]
    if modality == "image":
        encoder_terms = {"vit_iid": "ViT", "clip": "CLIP", "plip": "PLIP"}
        if encoder in encoder_terms.keys():
            encoder = encoder_terms[encoder]

    # load model from string
    model = deserialize_model(os.path.join(model_cache_dir, model_str))
    # pass to gridsearch iteration
    model_args = {"modality": modality,
                  "processor": processor,
                  "r": r,
                  "variant": None,
                  "hparams": hparams,
                  "train_graph_path": G_dir,
                  "train_label_dict": label_dict}
    
    model_results_dict, _ = gridsearch_iteration(model, model_args, gt_dir, thresh=threshold, arm=arm)    
    return get_test_metrics(model_results_dict, encoder, model_str, threshold, test_metrics)

def get_test_metrics(model_results_dict, encoder, model_str, threshold, test_metrics):
    # evaluate using test_metrics
    valid_conf_metrics = ["specificity", "precision", "fnr", "fdr", "recall", "accuracy", "balanced_acc", "correlation", "threat_score", "prevalence", "dice", "jaccard"]
    valid_cont_metrics = ["auroc", "auprc", "ap"]
    outdata = []
    for G_name, data in model_results_dict.items():
        # added for adaptive support
        if threshold == "<" or threshold == ">":
            adaptive = [key for key in data['thresh_cm'].keys() if type(key)==tuple]
            t = [key for key in adaptive if key[0]==threshold][0]
        else:
            t = threshold

        y = get_label(data)
        ravel = data['thresh_cm'][t]
        for metric_str in test_metrics:
            if metric_str == 'msd':
                val = data['thresh_msd'][t]
            elif metric_str in valid_conf_metrics:
                metric = check_eval_metric(metric_str, valid_conf_metrics)
                val = metric(ravel)
                if y == 1:
                    outdata.append([encoder, model_str, t, G_name, 'class-1', metric_str, val])
            elif metric_str in valid_cont_metrics:
                val = data['cont'][metric_str]
            outdata.append([encoder, model_str, t, G_name, 'all', metric_str, val])
    return pd.DataFrame(outdata, columns=['encoder', 'model', 'threshold', 'datum_id', 'regime', 'metric', 'value'])

def extract_params(model_str):
    model_params = model_str.rstrip('.model').split("_")
    if "cutoff" in model_str:
        k = int(model_params[0].split("k")[1])
        r = int(model_params[1].split("r")[1])
        cutoff = float(model_params[2].split("cutoff")[1])
        alpha = float(model_params[3].split("alpha")[1])
        tau = float(model_params[4].split("tau")[1])
        lam = float(model_params[5].split("lam")[1])
        return k,r,cutoff,alpha,tau,lam
    else:
        k = int(model_params[0].split("k")[1])
        r = int(model_params[1].split("r")[1])
        alpha = float(model_params[2].split("alpha")[1])
        tau = float(model_params[3].split("tau")[1])
        lam = float(model_params[4].split("lam")[1])
        return k,r,alpha,tau,lam

def check_eval_metric(metric_str, valid_metrics):
    if metric_str not in valid_metrics:
        print("Error. Requested metric not available for evaluation.")
        print("Please choose from: " + str(valid_metrics))
        exit()
    elif metric_str == "msd":
        return eval("metrics.identity")
    else:
        return eval("metrics." + metric_str)

def get_label(datum_results_dict):
    cont_scores = datum_results_dict["cont"]
    if np.isnan(cont_scores["auroc"]):
        return 0
    return 1

# Properties of embed/regions
#==============================
def num_cc(G):
    G_drop = G.copy()
    G_drop.remove_nodes_from([n for n in G_drop.nodes if G_drop.nodes[n]["emb"] == 0])
    size_cc = [len(c) for c in nx.connected_components(G_drop)]
    return len(size_cc)

def compute_test_ccs(gts_path):
    """
    num ccs
    """
    mcss = {}
    for gt_file in os.listdir(gts_path):
        gt_id = gt_file.split("-")[0:2]
        gt_id = '-'.join(gt_id)
        gt_path = os.path.join(gts_path, gt_file)
        gt = deserialize(gt_path)
        mcss[gt_id] = num_cc(gt)
    return mcss

def mean_cc_size(G):
    G_drop = G.copy()
    G_drop.remove_nodes_from([n for n in G_drop.nodes if G_drop.nodes[n]["emb"] == 0])
    size_cc = [len(c) for c in nx.connected_components(G_drop)]
    return np.sum(size_cc) / len(size_cc) # prevalence / num CC

def compute_test_mcs(gts_path):
    """
    mean component size
    """
    mcss = {}
    for gt_file in os.listdir(gts_path):
        gt_id = gt_file.split("-")[0:2]
        gt_id = '-'.join(gt_id)
        gt_path = os.path.join(gts_path, gt_file)
        gt = deserialize(gt_path)
        mcss[gt_id] = mean_cc_size(gt)
    return mcss    

def mean_region_dispersion(G):
    G_drop = G.copy()
    G_drop.remove_nodes_from([n for n in G_drop.nodes if G_drop.nodes[n]["emb"] == 0])
    size_cc = [len(c) for c in nx.connected_components(G_drop)]
    return len(size_cc) / np.mean(size_cc) # num ccs / mean size

def compute_test_mrds(gts_path):
    mrds = {}
    for gt_file in os.listdir(gts_path):
        gt_id = gt_file.split("-")[0:2]
        gt_id = '-'.join(gt_id)
        gt_path = os.path.join(gts_path, gt_file)
        gt = deserialize(gt_path)
        mrds[gt_id] = mean_region_dispersion(gt)
    return mrds

def region_prevalence(G):
    mask_vals = list(nx.get_node_attributes(G, "emb").values())
    return np.sum(mask_vals) / len(mask_vals)
    
def compute_test_rps(gts_path):
    rps = {}
    for gt_file in os.listdir(gts_path):
        gt_id = gt_file.split("-")[0:2]
        gt_id = '-'.join(gt_id)
        gt_path = os.path.join(gts_path, gt_file)
        gt = deserialize(gt_path)
        rps[gt_id] = region_prevalence(gt)
    return rps

# def compute_seg_per_config(encoder_alias, model_str, threshold, cache_dir, Gs_dir, gts_dir, label_dict_path):
#     """
#     Takes top model per encoder and compares segmentation auprc per test example
#     """
#     cache_dir = cache_dir + encoder_alias + "_gridsearch/"
#     Gs_dir = Gs_dir + encoder_alias
#     gts_dir = gts_dir + encoder_alias
#     label_dict = utils.deserialize(label_dict_path + encoder_alias + ".obj")

#     results_cache_dir = cache_dir + encoder_alias + "-eval_results"
#     model_cache_dir = cache_dir + encoder_alias + "-fitted_k2_models"
#     processor_cache_dir = cache_dir + encoder_alias + "-fitted_k2_processors"
#     linearized_cache_dir = cache_dir + encoder_alias + "-linearized_data"
#     test_metrics = ["auprc"]

#     df = test_eval(model_str, threshold, test_metrics, model_cache_dir, processor_cache_dir, Gs_dir, gt_dir=gts_dir, label_dict=label_dict, modality="image", arm="test")
#     return df

# def compute_seg_all_configs(encoder_top_models, cache_dir, Gs_dir, gts_dir, label_dict_path):
#     test_df = []
#     for encoder, (model_str, threshold) in encoder_top_models.items():
#         print(encoder)
#         if encoder == "ViT":
#             encoder_alias = "vit_iid"
#         else:
#             encoder_alias = encoder.lower()
#         df = compute_seg_per_config(encoder_alias, model_str, threshold, cache_dir, Gs_dir, gts_dir, label_dict_path)
#         test_df.append(df)
#     test_df = pd.concat(test_df)
#     return test_df


#========================BASH SCRIPTING========================
def main():
    from job_params import experiment_setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, help="directory to store results")
    parser.add_argument("--encoder_name", type=str, help="name of encoder")
    parser.add_argument("--gt_dir", type=str, help="directory of ground truth graphs")
    args = parser.parse_args()
    # Note: change params in script below if needed!
    sweep_dict, proc_args, model_args = experiment_setup(args.encoder_name)

    train_gridsearch(sweep_dict, args.save_dir, args.encoder_name, args.gt_dir, proc_args, model_args)

if __name__ == "__main__":
    main()
