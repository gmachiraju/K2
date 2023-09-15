import os
import numpy as np
from tqdm.notebook import tqdm
import pdb 

from utils import serialize, deserialize
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
    results_dict, results_dir, proc_cache_dir, model_cache_dir, linearized_cache_dir = setup_gridsearch(save_dir, encoder_name)

    for k in sweep_dict["k"]:
        proc = fetch_processor(k, proc_cache_dir, process_args)
        
        for r in sweep_dict["r"]:
            # Predictive: ElasticNet
            model_args["variant"] = "predictive"
            for lam in sweep_dict["lam"]:
                print("Training ElasticNet model with k=%d, r=%d, lam=%f" % (k, r, lam))
                model, model_str = fetch_model(proc, r, model_cache_dir, model_args, lam=lam)
                if model_str in results_dict.keys():
                    continue # already trained and stored
                model_results_dict, datum_linearized_dict = gridsearch_iteration(model, model_args, gt_dir)
                results_dict[model_str] = model_results_dict
                serialize(results_dict, results_dir)
                serialize(datum_linearized_dict, os.path.join(linearized_cache_dir, model_str))
                print("Saved model/results!")

            # Inferential: Hypothesis test
            model_args["variant"] = "inferential"
            for alpha in sweep_dict["alpha"]:
                for tau in sweep_dict["tau"]:
                    print("Running hypothesis test with k=%d, r=%d, alpha=%f, tau=%f" % (k, r, alpha, tau))   
                    model, model_str = fetch_model(proc, r, model_cache_dir, model_args, alpha=alpha, tau=tau)
                    if model_str in results_dict.keys():
                        continue # already trained and stored
                    model_results_dict, datum_linearized_dict = gridsearch_iteration(model, model_args, gt_dir)
                    results_dict[model_str] = model_results_dict
                    serialize(results_dict, results_dir)
                    serialize(datum_linearized_dict, os.path.join(linearized_cache_dir, model_str))
                    print("Saved model/results!")

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
        thresholds = list(np.linspace(0,1,11)) # [0.1, 0.2, ..., 0.9, 1.0]
    else:
        thresholds = [thresh]
    
    model_results_dict = {} # returned object
    G_files = os.listdir(model_args["train_graph_path"])
    with tqdm(total=len(G_files), desc="Eval samples...") as pbar:
        for t, G_name in enumerate(G_files):
            G = deserialize(os.path.join(model_args["train_graph_path"], G_name))
            Y = deserialize(os.path.join(gt_dir, G_name + "_gt")) # groud truth
            P = model.prospect(G)
            if thresh == "all":
                threshold_a = compute_adaptive_thresh_graph(P)
                thresholds.append((">", threshold_a)) # adaptive forward
                thresholds.append(("<", threshold_a)) # adaptive backward

            # get label
            if model.train_label_dict: # loaded a dict
                y = model.train_label_dict[G_name]
            else: # internally stored in graph
                y = G.label
            
            if model.variant == "predictive":
                sprite = model.construct_sprite(G)
                g = model.embed_sprite(sprite)
                y_hat = model.classifier.predict(g)
            else:
                y_hat = np.nan # no prediction for inferential model

            dicts = eval_suite(G_name, P, Y, y, y_hat, thresholds)
            model_results_dict["thresh_msd"] = dicts[0]
            model_results_dict["thresh_cm"] = dicts[1]
            model_results_dict["cont"] = dicts[2]
            model_results_dict["pred"] = dicts[3]
            datum_linearized_dict = dicts[4]
            pbar.set_description('processed: %d' % (1 + t))
            pbar.update(1)
    return model_results_dict, datum_linearized_dict

def eval_suite(G_name, P, Y, y, y_hat, thresholds):
    """
    Calls on metrics to evaluate a single graph datum
    Can use for any [0,1] heatmap: K2 prospect map, saliency, attention, probability
    """
    datum_linearized_dict, datum_cont_dict, datum_pred_dict = {}, {}, {}
    P_scaled = rescale_graph(P) # first rescale to [0,1] 
    # Continuous & IID eval
    #----------------------
    P_vec = linearize_graph(P_scaled)
    Y_vec = linearize_graph(Y)
    datum_linearized_dict[(G_name, y)] = (P_vec, Y_vec) # store for IID eval
    
    # Continuous eval can only run on class-1 data
    datum_cont_dict[(G_name, y)] = {"auroc": np.nan, "auprc": np.nan, "ap": np.nan}
    if y == 1:
        datum_cont_dict[(G_name, y)] = {"auroc": auroc(P_vec, Y_vec), "auprc": auprc(P_vec, Y_vec), "ap": ap(P_vec, Y_vec)}

    # compute predictions from maps
    #------------------------------
    y_hat_map = few_hot_classification(P_vec, few=10)
    datum_pred_dict[(G_name, y)] = (y_hat, y_hat_map)

    # multi-thresholding eval
    #------------------------
    datum_thresh_msd_dict, datum_thresh_cm_dict = {}, {}
    for thresh in thresholds:
        if type(thresh) == tuple:
            cond, thresh = thresh[0], thresh[1]
            P_bin = binarize_graph(P_scaled, thresh, conditional=cond) # binarize with threshold
        P_bin = binarize_graph(P_scaled, thresh)
        datum_thresh_msd_dict[(G_name, thresh)] = msd(P_bin)
        P_bin_vec = linearize_graph(P_bin) # linearize
        datum_thresh_cm_dict[(G_name, thresh)] = confusion(P_bin_vec, Y_vec)

    return datum_thresh_msd_dict, datum_thresh_cm_dict, datum_cont_dict, datum_pred_dict, datum_linearized_dict

def setup_gridsearch(save_dir, encoder_name):
    """
    Builds directory structure for grid search. Checks if previous results exist.
    """
    # check if save_dir exists
    results_dir = os.path.join(save_dir, encoder_name + "-results_dict.obj")
    if not os.path.exists(save_dir):
        print("Creating directory to store results: %s" % save_dir)
        os.makedirs(save_dir)
        results_dict = {}
        serialize(results_dict, results_dir)
    else:
        print("Directory found at: %s" % save_dir)
        results_dict = deserialize(results_dir)

    # make a directory for models
    proc_cache_dir = os.path.join(save_dir, encoder_name + "-fitted_k2_processors")
    model_cache_dir = os.path.join(save_dir, encoder_name + "-fitted_k2_models")
    linearized_cache_dir = os.path.join(save_dir, encoder_name + "-linearized_data")
    if not os.path.exists(model_cache_dir):
        print("Don't see previous folder for fitted K2 models... Creating directory at: %s" % model_cache_dir)
        os.makedirs(model_cache_dir)
    if not os.path.exists(proc_cache_dir):
        print("Don't see previous folder for fitted K2 processors... Creating directory at: %s" % proc_cache_dir)
        os.makedirs(proc_cache_dir)
    if not os.path.exists(linearized_cache_dir):
        print("Don't see previous folder for linearized data files... Creating directory at: %s" % linearized_cache_dir)
        os.makedirs(linearized_cache_dir)
    return results_dict, results_dir, proc_cache_dir, model_cache_dir, linearized_cache_dir

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
    model_name = "k%d_r%d_alpha%f_tau%f_lam%f.K2model" % (k, r, alpha, tau, lam)
    if model_name in os.listdir(model_cache_dir):
        print("Found fitted model for k=%d, r=%d, alpha=%f, tau=%f, lam=%f" % (k, r, alpha, tau, lam))
        model = deserialize(os.path.join(model_cache_dir, model_name))
    else:
        model = spawn_model(proc, r, alpha, tau, lam, model_args).create_train_array().fit_kernel()
        serialize(model, os.path.join(model_cache_dir, model_name))
    return model, model_name

def fetch_processor(k, proc_cache_dir, process_args):
    """
    Checks existence of processor in cache, otherwise spawns and fits processor
    Inputs:
        k: chosen number of concepts
        proc_cache_dir: directory to store fitted processors
        process_args: dictionary of hyperparameters for processor
    """
    processor_name = "k%d.K2processor" % k
    if processor_name in os.listdir(proc_cache_dir):
        print("Found fitted processor for k=%d" % k)
        proc = deserialize(os.path.join(proc_cache_dir, processor_name))
    else:
        proc = spawn_processor(k, process_args).fit_quantizer()
        serialize(proc, os.path.join(proc_cache_dir, processor_name))
    return proc

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
    model_args["hparams"] = {"alpha": alpha, "tau": tau, "lam": lam}
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