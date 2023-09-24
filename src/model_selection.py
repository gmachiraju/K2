import os
import numpy as np
import time 
from copy import copy
from tqdm import tqdm

import metrics
from evaluation import eval_suite
from utils import serialize, deserialize, expand_positive_nodes, linearize_graph, set_graph_emb, binarize_vec, compute_adaptive_thresh_vec

def top_model_confusion(metric_str, results_cache_dir, model_cache_dir, eval_class="both"):
    valid_metrics = ["msd", "specificity", "precision", "fnr", "fdr", "recall", "accuracy", "balanced_acc", "correlation", "threat_score", "prevalence", "dice", "jaccard"]
    metric = check_eval_metric(metric_str, valid_metrics)
    metric_dict = {}
    for model_str in os.listdir(model_cache_dir):
        model_results_dict = deserialize(os.path.join(results_cache_dir, model_str))

        setup_flag = False
        N = 0
        for graph_id in model_results_dict.keys():
            datum_results_dict = model_results_dict[graph_id]
            if eval_class in [0,1]:
                y = get_label(datum_results_dict)
                if y != eval_class:
                    continue
            if metric_str == "msd":
                datum_cms = datum_results_dict["thresh_msd"]
            else:
                datum_cms = datum_results_dict["thresh_cm"]
            
            # begin dictionary if not yet initialized
            if setup_flag == False:
                model_cms = {} 
                for thresh in datum_cms.keys():
                    if type(thresh) == tuple:
                        new_thresh = thresh[0]
                        model_cms[new_thresh] = 0.0
                    else:
                        model_cms[thresh] = 0.0
                setup_flag = True

            for thresh in datum_cms.keys():
                if type(thresh) == tuple:
                    new_thresh = thresh[0]
                    model_cms[new_thresh] += metric(datum_cms[thresh])
                else:
                    model_cms[thresh] += metric(datum_cms[thresh])
            N += 1
                
        # now average over all graphs
        for thresh in model_cms.keys():
            model_cms[thresh] /= N
        # get top score and threshold
        scores = [(thresh, model_cms[thresh]) for thresh in model_cms.keys()]
        max_score = max(scores, key=lambda item: item[1])
        min_score = min(scores, key=lambda item: item[1])
        stability = max_score[1] - min_score[1]
        metric_dict[model_str] = (max_score[0], max_score[1], stability)
    # get top model
    top_model_str = max(metric_dict, key=lambda item: metric_dict[item][1])
    (threshold, top_metric_score, stability) = metric_dict[top_model_str]
    return top_model_str, threshold, top_metric_score, stability
        
def top_model_continuous_avg(metric_str, results_cache_dir, model_cache_dir):
    """
    These metrics are the average of the continuous metrics over all data.
    Metrics include auroc, auprc, and ap. Only applicable to class-1 data.
    """
    valid_metrics = ["auroc", "auprc", "ap"]
    _ = check_eval_metric(metric_str, valid_metrics)
    metric_dict = {}
    for model_str in os.listdir(model_cache_dir):
        model_results_dict = deserialize(os.path.join(results_cache_dir, model_str))
        scores = []
        for graph_id in model_results_dict.keys():
            datum_results_dict = model_results_dict[graph_id]
            y = get_label(datum_results_dict)
            if y == 0:
                continue
            scores.append(datum_results_dict["cont"][metric_str])
        metric_dict[model_str] = np.mean(scores)
    # get top model
    top_model_str = max(metric_dict, key=lambda item: metric_dict[item])
    top_metric_score = metric_dict[top_model_str]
    return top_model_str, top_metric_score

def top_model_continuous_iid(metric_str, model_cache_dir, linearized_cache_dir):
    """
    These metrics compute single values for the entire unrolled dataset. Each element is treated IID.
    """
    valid_metrics = ["auroc", "auprc", "ap"]
    metric = check_eval_metric(metric_str, valid_metrics)
    metric_dict = {}
    for model_str in os.listdir(model_cache_dir):
        preds, gts = [], []
        lin_dict = deserialize(os.path.join(linearized_cache_dir, model_str))
        for graph_id in lin_dict.keys():
            P_vec, Y_vec = lin_dict[graph_id]
            preds.extend(P_vec)
            gts.extend(Y_vec)
        metric_dict[model_str] = metric(preds, gts)
    # get top model
    top_model_str = max(metric_dict, key=lambda item: metric_dict[item])
    top_metric_score = metric_dict[top_model_str]
    return top_model_str, top_metric_score

def top_model_preds(metric_str, results_cache_dir, model_cache_dir):
    """
    This metric computes the continuous metric for datum-level predicitions.

    """
    valid_metrics = ["auroc", "auprc", "ap"]
    metric = check_eval_metric(metric_str, valid_metrics)
    metric_dict = {}
    for model_str in os.listdir(model_cache_dir):
        model_results_dict = deserialize(os.path.join(results_cache_dir, model_str))
        ys, y_hats, y_map_hats = [], [], []
        for graph_id in model_results_dict.keys():
            datum_results_dict = model_results_dict[graph_id]
            y = get_label(datum_results_dict)
            datum_preds = datum_results_dict["pred"]
            ys.append(y)
            y_hats.append(datum_preds[0])
            y_map_hats.append(datum_preds[1])
        try:
            y_hat_metric = metric(y_hats, ys) # nans for hypothesis test
        except ValueError:
            y_hat_metric = 0.0
        y_map_hat_metric = metric(y_map_hats, ys)
        metric_dict[model_str] = np.max([y_hat_metric, y_map_hat_metric])
    # get top model
    top_model_str = max(metric_dict, key=lambda item: metric_dict[item])
    top_metric_score = metric_dict[top_model_str]
    return top_model_str, top_metric_score

def k_hop_precision(datatype, results_cache_dir, linearized_cache_dir, model_cache_dir, graph_base_dir, eval_class="both"):
    
    reg_thresholds = [np.round(el,1) for el in np.linspace(0,1,11)] # [0.1, 0.2, ..., 0.9, 1.0]
    idx_adaptive = len(reg_thresholds)
    thresholds = reg_thresholds + [np.nan, np.nan] # placeholders for adaptive
    
    metric_dict = {}
    for model_str in tqdm(os.listdir(model_cache_dir)):
        model_results_dict = deserialize(os.path.join(results_cache_dir, model_str))
        lin_results_dict = deserialize(os.path.join(linearized_cache_dir, model_str))
        
        setup_flag = False
        N = 0
        for graph_id in model_results_dict.keys():
            datum_results_dict = model_results_dict[graph_id]
            
            if eval_class in [0,1]:
                y = get_label(datum_results_dict)
                if y != eval_class:
                    continue
            pred, y = lin_results_dict[graph_id]
            if datatype == 'protein':
                cutoff = np.round(float(model_str.split("_")[2].replace('cutoff', '')), 1)
                graph_base_dir = graph_base_dir.replace('cutoff', str(cutoff))
            G = deserialize(os.path.join(graph_base_dir, graph_id))
            
            G = expand_positive_nodes(G)
            
            G = set_graph_emb(G, 'gt')
            
            Y_1hop = linearize_graph(G)
            
            
            threshold_a = compute_adaptive_thresh_vec(pred)
            thresholds[idx_adaptive] = (">", threshold_a) # adaptive forward
            thresholds[idx_adaptive + 1] = ("<", threshold_a) # adaptive backward
            
            precisions = {}
            for thresh in thresholds:
                if type(thresh) == tuple:
                    cond, t = thresh[0], thresh[1]
                else:
                    cond, t = None, copy(thresh)
                P_bin_vec = binarize_vec(pred, t, conditional=cond) # binarize with threshold
                ravel = metrics.confusion(P_bin_vec, Y_1hop)
                # ravel = confusion_matrix(P_bin_vec, Y_1hop)
                precision_1hop = metrics.precision(ravel)
                precisions[thresh] = precision_1hop
            
            if setup_flag == False:
                model_cms = {} 
                for thresh in precisions.keys():
                    if type(thresh) == tuple:
                        new_thresh = thresh[0]
                        model_cms[new_thresh] = 0.0
                    else:
                        model_cms[thresh] = 0.0
                setup_flag = True
                
            for thresh, prec in precisions.items():
                if type(thresh) == tuple:
                    new_thresh = thresh[0]
                    model_cms[new_thresh] += prec
                else:
                    model_cms[thresh] += prec
            N += 1
        
            # now average over all graphs
        for thresh in model_cms.keys():
            model_cms[thresh] /= N
        # get top score and threshold
        scores = [(thresh, model_cms[thresh]) for thresh in model_cms.keys()]
        metric_dict[model_str] = max(scores, key=lambda item: item[1])
    # get top model
    top_model_str = max(metric_dict, key=lambda item: metric_dict[item][1])
    (threshold, top_metric_score) = metric_dict[top_model_str]
    return top_model_str, threshold, top_metric_score

def get_label(datum_results_dict):
    cont_scores = datum_results_dict["cont"]
    if np.isnan(cont_scores["auroc"]):
        return 0
    return 1

def check_eval_metric(metric_str, valid_metrics):
    if metric_str not in valid_metrics:
        print("Error. Requested metric not available for evaluation.")
        print("Please choose from: " + str(valid_metrics))
        exit()
    elif metric_str == "msd":
        return eval("metrics.identity")
    else:
        return eval("metrics." + metric_str)
