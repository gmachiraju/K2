from evaluation import train_gridsearch

if __name__=='__main__':
    encoder = 'ESM'
    metal = 'CA'

    ks = [10,15,20]
    rs = [0,1,2]
    alphas = [0.001, 0.01, 0.1]
    taus = [0, 1, 2]
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
    save_dir = f"../data/{encoder}_gridsearch_results"
    gt_dir = None

    train_gridsearch(sweep_dict, save_dir, encoder, gt_dir, proc_args, model_args)