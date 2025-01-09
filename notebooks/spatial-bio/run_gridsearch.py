import sys
sys.path.insert(1, '/scr/gmachi/prospection/K2/src')
from k2 import K2Processor, K2Model
from evaluation import fetch_processor, fetch_model
from utils import serialize_model, deserialize_model, serialize, deserialize
import numpy as np
import os

proc_cache_dir = "/scr/gmachi/prospection/K2/notebooks/spatial-bio/outputs/gridsearch_results_final/k2processors"
model_cache_dir = "/scr/gmachi/prospection/K2/notebooks/spatial-bio/outputs/gridsearch_results_final/k2models"
embeds_path = "/scr/biggest/gmachi/datasets/celldive_lung/embed_sample_final.obj"
G_dir = "/scr/biggest/gmachi/datasets/celldive_lung/for_ml/for_prospect_final/"
label_path = "/scr/biggest/gmachi/datasets/celldive_lung/processed/label_dict.obj"
label_dict = deserialize(label_path)

proc_args = {"datatype":"cells",
        "k":20,
        "quantizer_type": "kmeans",
        "embeddings_path": embeds_path,
        "embeddings_type": "multidict",
        "mapping_path": None,
        "sample_size": 10160,
        "sample_scheme": "random",
        "dataset_path": None,
        "verbosity": "full",
        "so_dict_path": None,
        "mapping_path": label_path,
        "marker_flag": "labels"}

model_args = {"modality": "cells",
        "processor": None,
        "r":1,
        "variant": "inferential",
        "hparams": None,
        "train_graph_path": G_dir,
        "train_label_dict": label_dict}

ks = [20] #[8,9,10,11,12,13,14,15,16,17,18,19,20] 
# old run: [8,9,10,11,12,13,14,15,16,17,18,19,20]
rs = [1,2,3,4,5,6,7,8] # [1,2,3,4,5,6,7,8]
alphas = [0.05, 1e10] # [1e10] # 0.01, 0.025, 0.05
taus = [0, 1] # [0] #used to also have [1,2]
lambdas = [0.5] # keep as elastic


def main():    
    num_ht = len(ks) * len(rs) * len(alphas) * len(taus)
    num_lm = len(ks) * len(rs) * len(lambdas)
    print("We have %d models to train..." % (num_ht +  num_lm))
    print("...and have %d models trained so far!" % len(os.listdir(model_cache_dir)))
    print("="*40)
    
    for k in ks:
        proc, processor_name = fetch_processor(k, proc_cache_dir, proc_args)
        serialize_model(proc, os.path.join(proc_cache_dir, processor_name))
    
        for r in rs:
            model_args["variant"] = "predictive"
            for l in lambdas:
                print("Gridsearch: currently fitting predictive model with k=%d, r=%d, lam=%f" % (k, r, l))
                model, model_name = fetch_model(proc, r, model_cache_dir, model_args, alpha=np.nan, tau=np.nan, lam=l)
                serialize_model(model, os.path.join(model_cache_dir, model_name))
    
            model_args["variant"] = "inferential"
            for a in alphas:
                for t in taus:
                    print("Gridsearch: currently fitting inferential model with k=%d, r=%d, alpha=%f, tau=%f" % (k, r, a, t))   
                    model, model_name = fetch_model(proc, r, model_cache_dir, model_args, alpha=a, tau=t, lam=np.nan)
                    serialize_model(model, os.path.join(model_cache_dir, model_name))
            
if __name__=='__main__':
    main()