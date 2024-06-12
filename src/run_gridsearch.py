import sys
from evaluation import train_gridsearch

if __name__=='__main__':
    encoder = sys.argv[1]
    dataset = sys.argv[2]
    
    if encoder == "AA":
        ks = [21]
    else:
        ks = [15, 20, 25, 30]
    rs = [0,1,2,4]
    alphas = [1.0]#[0.0001, 0.001, 0.01]
    taus = [0, 1, 2, 4]
    lambdas = [] # keep as elastic
    cutoffs = [8.0] # edge cutoff for protein graph

    proc_args = {"datatype": "protein",
            "k": None,
            "dataset": dataset,
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
    save_dir = f"../data/{encoder}_{dataset}_gridsearch_results"
    gt_dir = None

    train_gridsearch(sweep_dict, save_dir, encoder, gt_dir, proc_args, model_args)
