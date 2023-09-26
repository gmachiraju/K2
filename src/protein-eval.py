from evaluation import test_eval, extract_params, get_test_metrics
import utils
import pandas as pd
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

encoders = ['COLLAPSE', 'ESM', 'AA']
metal = 'ZN'

conf_metrics = ["msd", "specificity", "precision", "fnr", "fdr", "recall", "accuracy", "balanced_acc", "correlation", "threat_score", "prevalence", "dice", "jaccard"]
cont_metrics = ["auroc", "auprc", "ap"]
test_metrics = conf_metrics + cont_metrics

encoder_top_models = \
    {'COLLAPSE': ('k20_r0_cutoff4.00_alpha0.001_tau0.00_lamnan.model', 0.95), \
    'ESM': ('k30_r1_cutoff4.00_alpha0.500_tau1.00_lamnan.model', 0.0), \
    'AA': ('k21_r1_cutoff8.00_alpha0.100_tau1.00_lamnan.model', 0.5)}

baseline_top_models = \
    {'COLLAPSE': ('COLLAPSE-ZN-8.0-0.0005', 0.7), \
    'ESM': ('ESM-ZN-8.0-0.0005', 0.4), \
    'AA': ('AA-ZN-8.0-0.0005', 0.6)}

base_df = []
for encoder in ['COLLAPSE', 'ESM', 'AA']:
    best_model, best_thresh = baseline_top_models[encoder]
    results_dict = utils.deserialize(f'../data/baselines/{encoder}_test_results.pkl')
    df = get_test_metrics(results_dict, encoder, best_model, best_thresh, test_metrics)
    base_df.append(df)
base_df = pd.concat(base_df)

test_df = []
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

    G_dir = f"../data/{g_encoder}_{metal}_{cutoff}_test_graphs"
    
    df = test_eval(model_str, threshold, test_metrics, model_cache_dir, processor_cache_dir, G_dir, gt_dir=None, label_dict=None, modality="graph")
    test_df.append(df)
test_df = pd.concat(test_df)

combined_df = pd.concat([test_df, base_df])
combined_df['method'] = ['K2']*len(test_df) + ['GAT+GNNExplainer']*len(base_df)

mean_df = combined_df.groupby(['encoder', 'method','regime', 'metric'])['value'].mean().reset_index()
sem_df = combined_df.groupby(['encoder', 'method','regime', 'metric'])['value'].sem().reset_index()

mean_pvt = mean_df.pivot(index=['encoder', 'method', 'regime'], columns='metric', values='value')
mean_pvt = mean_pvt[test_metrics]

sem_pvt = sem_df.pivot(index=['encoder', 'method', 'regime'], columns='metric', values='value')
sem_pvt = sem_pvt[test_metrics]

mean_pvt.to_csv(f'../data/results/all_test_results_mean_2.csv')
sem_pvt.to_csv(f'../data/results/all_test_results_sem_2.csv')
