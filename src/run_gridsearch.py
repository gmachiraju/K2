import sys
from evaluation import train_gridsearch

if __name__=='__main__':
    encoder = sys.argv[1]
    metal = sys.argv[2]
    
    if encoder == "AA":
        ks = [21]
    else:
        ks = [15, 20, 25, 30]
    rs = [1,2,4]
    alphas = [0.01, 0.1, 0.5]
    taus = [1, 2, 4]
    lambdas = [0.5] # keep as elastic
    cutoffs = [4.0, 6.0, 8.0] # edge cutoff for protein graph

    proc_args = {"datatype": "protein",
            "k": None,
            "metal": metal,
            "quantizer_type": "AA" if encoder == "AA" else "kmeans",
            "embeddings_path": None,
            "embeddings_type": "dict",
            "mapping_path": None,
            "sample_size": None,
            "sample_scheme": None,
            "dataset_path": None,
            "verbosity": "low", # change this to low!
            "so_dict_path": None}

    hparams = {"alpha": None, "tau": None, "lambda": None}
    model_args = {"modality":"graph",
            "processor": None,
            "r": None,
            "variant": "inferential",
            "hparams": hparams,
            "train_graph_path": None,
            "train_label_dict": None}
    
    sweep_dict = {"k": ks, "r": rs, "alpha": alphas, "tau": taus, "lambda": lambdas, "cutoff": cutoffs}
    save_dir = f"../data/{encoder}_{metal}_gridsearch_results"
    gt_dir = None

    train_gridsearch(sweep_dict, save_dir, encoder, gt_dir, proc_args, model_args)