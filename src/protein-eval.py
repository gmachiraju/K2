from evaluation import test_eval, extract_params, get_test_metrics
import utils
import pandas as pd
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")
import pdb

#=================
modality = "graph"
#=================

if modality == "graph":
    metal = 'ZN'

conf_metrics = ["msd", "specificity", "precision", "fnr", "fdr", "recall", "accuracy", "balanced_acc", "correlation", "threat_score", "prevalence", "dice", "jaccard"]
cont_metrics = ["auroc", "auprc", "ap"]
test_metrics = conf_metrics + cont_metrics

# from model selection
if modality == "graph":
    encoder_list = ['COLLAPSE', 'ESM', 'AA']
    
    encoder_top_models = { \
    'COLLAPSE': ('k25_r4_cutoff8.00_alpha1.0000_tau4.00_lamnan.model', 0.95), \
    # 'COLLAPSE': ('k15_r1_cutoff8.00_alpha0.0100_tau0.00_lamnan.model', 0.8), \
    'ESM': ('k20_r1_cutoff8.00_alpha1.0000_tau0.00_lamnan.model', 0.8), \
    'AA': ('k21_r1_cutoff8.00_alpha0.010_tau1.00_lamnan.model', 0.5)}

baseline_top_models = \
    {'COLLAPSE': ('COLLAPSE-ZN-8.0-0.0001-100', 0.7), \
    'ESM': ('ESM-ZN-6.0-0.001-500', 0.4), \
    'AA': ('AA-ZN-6.0-0.001-200', 0.5)}
    
elif modality == "image":
    encoder_list = ["tile2vec", "ViT", "CLIP", "PLIP"]
    encoder_top_models = \
        {'tile2vec': ('k20_r8_alpha10000000000.000_tau0.00_lamnan.model', 0.4), \
        'ViT':       ('k20_r2_alpha0.050_tau0.00_lamnan.model', 0.5), \
        'CLIP':      ('k30_r2_alpha10000000000.000_tau2.00_lamnan.model', 0.1), \
        'PLIP':      ('k15_r1_alpha0.010_tau2.00_lamnan.model', 0.5)}
    baseline_top_models = \
        {'ViT': [('attn_vit_iid', ">"), ("probs_vit_iid", ">")], \
        'CLIP': [('probs_clip', ">")], \
        'PLIP': [('probs_plip', ">")]}
    
#--------------------------------------------------
# Above is hard-coded based on model selection
#--------------------------------------------------


# baseline results on test
#=========================
base_df = []
if modality == "graph":
    for encoder in encoder_list:
        best_model, best_thresh = baseline_top_models[encoder]
        results_dict = utils.deserialize(f'../data/baselines/{encoder}_test_results.pkl')
        df = get_test_metrics(results_dict, encoder, best_model, best_thresh, test_metrics) 
        base_df.append(df)
    base_df = pd.concat(base_df)
    
elif modality == "image":
    for encoder in ["ViT", "CLIP", "PLIP"]:
        for model_thresh_pair in baseline_top_models[encoder]:
            best_model, best_thresh = model_thresh_pair
            print(best_model)
            results_dict = utils.deserialize("/home/k2/K2/src/outputs/baselines/results_dict-" + best_model +".obj")
            df = get_test_metrics(results_dict, encoder, best_model, best_thresh, test_metrics)
            # add a "method" column to the df (K2, Attn, Prob)
            df["method"] = best_model.split("_")[0].title()
            base_df.append(df)
    base_df = pd.concat(base_df)

# K2 results on test
#===================
test_df = []
if modality == "graph":
    for encoder, (model_str, threshold) in encoder_top_models.items():
        results_cache_dir = f"../data/{encoder}_{metal}_gridsearch_results/{encoder}-eval_results"
        model_cache_dir = f"../data/{encoder}_{metal}_gridsearch_results/{encoder}-fitted_k2_models"
        processor_cache_dir = f"../data/{encoder}_{metal}_gridsearch_results/{encoder}-fitted_k2_processors"
        linearized_cache_dir = f"../data/{encoder}_{metal}_gridsearch_results/{encoder}-linearized_data"

        _,_,cutoff,_,_,_ = extract_params(model_str)

        if encoder == 'AA':
            g_encoder = 'COLLAPSE'
        else:
            g_encoder = encoder

        G_dir = f"../data/{g_encoder}_{metal}_{cutoff}_test_graphs_2"
        df = test_eval(model_str, threshold, test_metrics, model_cache_dir, processor_cache_dir, G_dir, gt_dir=None, label_dict=None, modality="graph")
        test_df.append(df)
    test_df = pd.concat(test_df)

if modality == "image":
    for encoder, (model_str, threshold) in encoder_top_models.items():
        if encoder == "ViT":
            encoder_alias = "vit_iid"
        else:
            encoder_alias = encoder.lower()
        cache_dir = "/home/k2/K2/src/outputs/" + encoder_alias + "_gridsearch/"
        results_cache_dir = cache_dir + encoder_alias + "-eval_results"
        model_cache_dir = cache_dir + encoder_alias + "-fitted_k2_models"
        processor_cache_dir = cache_dir + encoder_alias + "-fitted_k2_processors"
        linearized_cache_dir = cache_dir + encoder_alias + "-linearized_data"

        G_dir = "/home/data/tinycam/test/clean_Gs_" + encoder_alias
        gt_dir = "/home/data/tinycam/test/gt_graphs_" + encoder_alias
        label_dict = utils.deserialize("/home/k2/K2/src/outputs/refined_test_labeldict-" + encoder_alias + ".obj")

        df = test_eval(model_str, threshold, test_metrics, model_cache_dir, processor_cache_dir, G_dir, gt_dir=gt_dir, label_dict=label_dict, modality="image", arm="test")
        df['method'] = 'K2'
        test_df.append(df)
    test_df = pd.concat(test_df)


# combine all results together
#==============================
combined_df = pd.concat([test_df, base_df])
if modality == "graph": 
    combined_df['method'] = ['K2']*len(test_df) + ['GAT+GNNExplainer']*len(base_df)

mean_df = combined_df.groupby(['encoder', 'method','regime', 'metric'])['value'].mean().reset_index()
sem_df = combined_df.groupby(['encoder', 'method','regime', 'metric'])['value'].sem().reset_index()

mean_pvt = mean_df.pivot(index=['encoder', 'method', 'regime'], columns='metric', values='value')
mean_pvt = mean_pvt[test_metrics]

sem_pvt = sem_df.pivot(index=['encoder', 'method', 'regime'], columns='metric', values='value')
sem_pvt = sem_pvt[test_metrics]

if modality == "graph":
    # mean_pvt.to_csv(f'../data/results/all_test_results_mean.csv')
    # sem_pvt.to_csv(f'../data/results/all_test_results_sem.csv')
    mean_pvt.to_csv(f'../data/results/all_test_results_mean.csv')
    sem_pvt.to_csv(f'../data/results/all_test_results_sem.csv')
    combined_df.to_csv(f'../data/results/all_test_results_points.csv')
elif modality == "image":
    mean_pvt.to_csv('/home/k2/K2/src/outputs/k2-test/all_test_results_mean.csv')
    sem_pvt.to_csv('/home/k2/K2/src/outputs/k2-test/all_test_results_sem.csv')
    combined_df.to_csv('/home/k2/K2/src/outputs/k2-test/all_test_results_points.csv') # graph-level results

