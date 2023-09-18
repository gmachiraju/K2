import utils

def experiment_setup(encoder_name):
    if encoder_name == "tile2vec":
        embed_dict_path = "/home/lofi/lofi/src/outputs_tile2vec/train_sampled_inference_z_embeds.obj"
    elif encoder_name == "vit_iid":
        embed_dict_path = "/home/lofi/lofi/src/outputs_vit/train_vit_iid_sampled_inference_z_embeds.obj"
    elif encoder_name == "clip": 
        embed_dict_path = "/home/lofi/lofi/src/outputs_clip/train_clip_sampled_inference_z_embeds.obj"
    elif encoder_name == "plip":
        embed_dict_path = "/home/lofi/lofi/src/outputs_plip/train_plip_sampled_inference_z_embeds.obj"

    if encoder_name == "tile2vec":
        G_dir = "/home/data/tinycam/train/clean_Gs_tile2vec"
    elif encoder_name == "vit_iid":
        G_dir = "/home/data/tinycam/train/clean_Gs_vit_iid"
    elif encoder_name == "clip":
        G_dir = "/home/data/tinycam/train/clean_Gs_clip"
    elif encoder_name == "plip":
        G_dir = "/home/data/tinycam/train/clean_Gs_plip"

    # hparams
    ks = [10,15,20,25,30]
    rs = [0,1,2,4,8]
    alphas = [0.01, 0.025, 0.05, 1e10]
    taus = [0,1,2]
    lambdas = [0.5]
    sweep_dict = {"k": ks, "r": rs, "alpha": alphas, "tau": taus, "lambda": lambdas}

    proc_args = {"datatype": "histo",
            "k": None,
            "quantizer_type": "kmeans",
            "embeddings_path": embed_dict_path,
            "embeddings_type": "dict",
            "mapping_path": None,
            "sample_size": 4440,
            "sample_scheme": "random",
            "dataset_path": "/home/data/tinycam/train/train.hdf5",
            "verbosity": "low", # change this to low!
            "so_dict_path": "/home/lofi/lofi/src/outputs/train_so_dict.obj"}

    hparams = {"alpha": None, "tau": None, "lambda": None}
    label_dict = utils.deserialize("/home/k2/K2/src/outputs/refined_train_labeldict.obj")

    model_args = {"modality":"image",
            "processor": None,
            "r": None,
            "variant": None,
            "hparams": hparams,
            "train_graph_path": G_dir,
            "train_label_dict": label_dict}
    
    return sweep_dict, proc_args, model_args
