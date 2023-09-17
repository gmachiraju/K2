import os
import numpy as np
from tqdm.notebook import tqdm
import pdb 
import time
from copy import copy

from utils import serialize, deserialize, serialize_model, deserialize_model
from utils import compute_adaptive_thresh_graph, compute_adaptive_thresh_vec
from utils import binarize_graph, binarize_vec, rescale_graph, rescale_vec, linearize_graph
from k2 import K2Processor, K2Model
from metrics import confusion, msd, auroc, auprc, ap

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
    num_ht = len(sweep_dict["k"]) * len(sweep_dict["r"]) * len(sweep_dict["alpha"]) * len(sweep_dict["tau"])
    num_lm = len(sweep_dict["k"]) * len(sweep_dict["r"]) * len(sweep_dict["lambda"])
    print("We have %d models to train..." % (num_ht +  num_lm))
    print("...and have %d models trained so far!" % len(results_dict.keys()))
    print("="*40)

    for k in sweep_dict["k"]:
        proc, processor_name = fetch_processor(k, proc_cache_dir, process_args)
        serialize_model(proc, os.path.join(proc_cache_dir, processor_name))

        for r in sweep_dict["r"]:
            # Predictive: ElasticNet
            model_args["variant"] = "predictive"
            for lam in sweep_dict["lambda"]:
                print("Gridsearch: currently on ElasticNet model with k=%d, r=%d, lam=%f" % (k, r, lam))
                model, model_str = fetch_model(proc, r, model_cache_dir, model_args, lam=lam)
                if model_str in results_dict.keys():
                    continue # already trained and stored
                gridsearch_iteration_wrapper(model, model_str, model_args, gt_dir, results_cache_dir, results_dict, results_dir, model_cache_dir, linearized_cache_dir)

            # Inferential: Hypothesis test
            model_args["variant"] = "inferential"
            for alpha in sweep_dict["alpha"]:
                for tau in sweep_dict["tau"]:
                    print("Gridsearch: currently on hypothesis test with k=%d, r=%d, alpha=%f, tau=%f" % (k, r, alpha, tau))   
                    model, model_str = fetch_model(proc, r, model_cache_dir, model_args, alpha=alpha, tau=tau)
                    if model_str in results_dict.keys():
                        continue # already trained and stored
                    gridsearch_iteration_wrapper(model, model_str, model_args, gt_dir, results_cache_dir, results_dict, results_dir, model_cache_dir, linearized_cache_dir)
                
def gridsearch_iteration_wrapper(model, model_str, model_args, gt_dir, results_cache_dir, results_dict, results_dir, model_cache_dir, linearized_cache_dir):
    model_results_dict, datum_linearized_dict = gridsearch_iteration(model, model_args, gt_dir)
    results_dict[model_str] = os.path.join(results_cache_dir, model_str) # save_path
    serialize(model_results_dict, os.path.join(results_cache_dir, model_str))
    serialize(results_dict, results_dir)
    serialize(datum_linearized_dict, os.path.join(linearized_cache_dir, model_str))
    serialize_model(model, os.path.join(model_cache_dir, model_str))
    print("Saved model/results!" + "\n" + "-"*40)

# Helper functions
#=================
def gridsearch_iteration(model, model_args, gt_dir, thresh="all"):
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
        Y = deserialize(os.path.join(gt_dir, G_name + "_gt")) # groud truth
        P = model.prospect(G)
        if thresh == "all":
            threshold_a = compute_adaptive_thresh_graph(P)
            thresholds[idx_adaptive] = (">", threshold_a) # adaptive forward
            thresholds[idx_adaptive + 1] = ("<", threshold_a) # adaptive backward

        # get label
        if model.train_label_dict: # loaded a dict
            y = model.train_label_dict[G_name]
        else: # internally stored in graph
            y = G.label
        
        if model.variant == "predictive":
            sprite = model.construct_sprite(G)
            g = model.embed_sprite(sprite)
            y_hat = model.classifier.predict_proba(np.expand_dims(g, axis=0))
        else:
            y_hat = np.nan # no prediction for inferential model

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
    Y_vec = linearize_graph(Y)
    datum_linearized = (P_vec, Y_vec) # store for IID eval
    # _dict[(G_name, y)]

    # Continuous eval can only run on class-1 data
    datum_cont = {"auroc": np.nan, "auprc": np.nan, "ap": np.nan}
    if y == 1:
        datum_cont = {"auroc": auroc(P_vec, Y_vec), "auprc": auprc(P_vec, Y_vec), "ap": ap(P_vec, Y_vec)}
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

def fetch_model(proc, r, model_cache_dir, model_args, alpha=np.nan, tau=np.nan, lam=np.nan):
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

def fetch_processor(k, proc_cache_dir, process_args):
    """
    Checks existence of processor in cache, otherwise spawns and fits processor
    Inputs:
        k: chosen number of concepts
        proc_cache_dir: directory to store fitted processors
        process_args: dictionary of hyperparameters for processor
    """
    processor_name = "k%d.processor" % k
    if processor_name in os.listdir(proc_cache_dir):
        print("Found fitted processor for k=%d" % k)
        proc = deserialize_model(os.path.join(proc_cache_dir, processor_name))
    else:
        print("Fitting processor for k=%d" % k)
        proc = spawn_processor(k, process_args)
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
#===========
def test_eval():
    """
    Test-set evaluation using top models extracted from training grid search
    SHOULD JUST CALL GRIDSEARCH_ITERATION
    """
    pass