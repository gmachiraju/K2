import numpy as np
import networkx as nx
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def confusion(a,b):
    """
    Computes a unraveled confusion matrix from two binary vectors.
    Inputs:
        a: iterable binary (predicted labels)
        b: iterable binary (true labels)
    Outputs: unraveled confusion matrix
    """
    cm = confusion_matrix(a, b, labels=[True, False])
    ravel = cm.ravel() # tn, tp, fn, fp
    return ravel

###########
# Metrics
###########
# Inputs:
#   a: iterable binary
#   b: iterable binary
#   prob: iterable of probabilities [0,1]
#   target: iterable of binary labels
###########

def specificity(ravel): # aka TNR
    tn, tp, fn, fp = ravel
    if tn + fp == 0: # nan
        return 1 
    return tn / (tn + fp)

def precision(ravel): # aka PPV
    tn, tp, fn, fp = ravel
    if (tp + fp) == 0: # nan
        return 0
    return tp / (tp + fp)

def fnr(ravel): # aka miss rate
    # False negative rate
    tn, tp, fn, fp = ravel
    if (tp + fn) == 0: # nan
        return 0
    return fn / (tp + fn)

def fdr(ravel):
    # False discovery rate, not to be confused with franklin delano roosevelt
    tn, tp, fn, fp = ravel
    if (tp + fp) == 0: # nan
        return 0
    return fp / (tp + fp)

def recall(ravel): # aka sensitivity, TPR
    tn, tp, fn, fp = ravel
    if tp + fn == 0: # nan
        return 1 
    return tp / (tp + fn)

def sensitivity(ravel): # aka recall, TPR
    return recall(ravel) # alias support

def accuracy(ravel):
    tn, tp, fn, fp = ravel
    return (tp + tn) / (tp + tn + fp + fn)

def balanced_acc(ravel, adjusted=False):
    # Balanced accuracy
    # https://github.com/scikit-learn/scikit-learn/blob/364c77e04/sklearn/metrics/_classification.py#L2111
    tn, tp, fn, fp = ravel
    C = np.array([[tp, fn], [fp, tn]])
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(C) / C.sum(axis=1)

    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score

def correlation(ravel):
    # Mathew's correlation coefficient (MCC)
    tn, tp, fn, fp = ravel
    C = np.array([[tp, fn], [fp, tn]])

    # https://github.com/scikit-learn/scikit-learn/blob/364c77e04/sklearn/metrics/_classification.py#L848
    t_sum = C.sum(axis=1, dtype=np.float64)
    p_sum = C.sum(axis=0, dtype=np.float64)
    n_correct = np.trace(C, dtype=np.float64)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples**2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples**2 - np.dot(t_sum, t_sum)
    if cov_ypyp * cov_ytyt == 0:
        return 0.0
    else:
        return cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)

def threat_score(ravel):
    tn, tp, fn, fp = ravel
    return tp / (tp + fn + fp)

def prevalence(ravel):
    tn, tp, fn, fp = ravel
    return (tp + fn) / (tp + tn + fp + fn)
    
# below metrics give NaN for class-0 data
def dice(ravel):
    tn, tp, fn, fp = ravel
    if ((2*tp) + fp + fn) == 0: # nan
        return 1 
    return (2*tp) / ((2*tp) + fp + fn)

def jaccard(ravel):
    tn, tp, fn, fp = ravel
    if (tp + fp + fn) == 0: # nan
        return 1 
    return tp / (tp + fp + fn)

# below are metrics that require probabilities and a mix of classes
def auroc(prob, target):
    # area under receiver operating characteristic curve
    return roc_auc_score(target, prob)
            
def auprc(prob, target):
    # area under precision-recall curve
    precision, recall, _ = precision_recall_curve(target, prob)
    return auc(recall, precision)

def ap(prob, target):
    # average precision
    return average_precision_score(target, prob)

# below used for predictions alone
def msd(P):
    """
    Mean salience dispersal 
    Inputs: 
        binarized prospect graph 
    """
    P_msd = P.copy()
    # 1. drop all nodes with value 0   
    value_dict = nx.get_node_attributes(P, 'emb')
    to_remove = [node for node,emb in value_dict.items() if emb == 0]
    P_msd.remove_nodes_from(to_remove)
    # 2. get connected components
    ccs = [P_msd.subgraph(c).copy() for c in nx.connected_components(P_msd)]
    # 3. go through each component and compute number of nodes
    cardinalities = []
    for C in ccs:
        cardinalities.append(C.number_of_nodes())
    if len(cardinalities) == 0:
        return 0
    return np.mean(cardinalities) / len(ccs)