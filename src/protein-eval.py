from model_selection import top_model_confusion, top_model_preds, top_model_continuous_avg, top_model_continuous_iid
from utils import serialize, deserialize, serialize_model, deserialize_model
import pandas as pd

encoders = ['COLLAPSE', 'ESM', 'AA']
metal = 'CA'

conf_metrics = ["msd", "specificity", "precision", "fnr", "fdr", "recall", "accuracy", "balanced_acc", "correlation", "threat_score", "prevalence", "dice", "jaccard"]
cont_metrics = ["auroc", "auprc", "ap"]

confusion_all = []
confusion_class1 = []
datum_level = []
continuous_avg = []
continuous_iid = []

confusion_all_models = {}
confusion_class1_models = {}
datum_level_models = {}
continuous_avg_models = {}
continuous_iid_models = {}

for encoder in encoders:
    print('on encoder', encoder)
    results_cache_dir = f"../data/{encoder}_{metal}_gridsearch_results/{encoder}-eval_results"
    model_cache_dir = f"../data/{encoder}_{metal}_gridsearch_results/{encoder}-fitted_k2_models"
    linearized_cache_dir = f"../data/{encoder}_{metal}_gridsearch_results/{encoder}-linearized_data"
    
    print('Confusion metrics (all)')
    row = []
    for metric_str in conf_metrics:
        best_model, best_thresh, best_score, stability = top_model_confusion(metric_str,results_cache_dir, model_cache_dir)
        row.append(best_score)
        confusion_all_models[metric_str] = (best_model, best_thresh)
    confusion_all.append(row)

    print('Confusion metrics (class-1)')
    row = []
    for metric_str in conf_metrics:
        best_model, best_thresh, best_score, stability = top_model_confusion(metric_str,results_cache_dir, model_cache_dir, eval_class=1)
        row.append(best_score)
        confusion_class1_models[metric_str] = (best_model, best_thresh)
    confusion_class1.append(row)

    print('Datum-level metrics')
    row = []
    for metric_str in cont_metrics:
        best_model, best_score = top_model_preds(metric_str, results_cache_dir, model_cache_dir)
        row.append(best_score)
    datum_level.append(row)

    print('Continuous metrics (avg)')
    row = []
    for metric_str in cont_metrics:
        best_model, best_score = top_model_continuous_avg(metric_str, results_cache_dir, model_cache_dir)
        row.append(best_score)
        continuous_avg_models[metric_str] = best_model
    continuous_avg.append(row)

    print('Continuous metrics (iid)')
    row = []
    for metric_str in cont_metrics:
        best_model, best_score = top_model_continuous_iid(metric_str, model_cache_dir, linearized_cache_dir)
        row.append(best_score)
        continuous_iid_models[metric_str] = best_model
    continuous_iid.append(row)
    
    serialize(confusion_all_models, f"../data/results/{encoder}_{metal}_confusion_all_train_models.pkl")
    serialize(confusion_class1_models, f"../data/results/{encoder}_{metal}_confusion_class1_train_models.pkl")
    serialize(datum_level_models, f"../data/results/{encoder}_{metal}_datum_level_train_models.pkl")
    serialize(continuous_avg_models, f"../data/results/{encoder}_{metal}_continuous_avg_train_models.pkl")
    serialize(continuous_iid_models, f"../data/results/{encoder}_{metal}_continuous_iid_train_models.pkl")

pd.DataFrame(confusion_all, index=encoders, columns=conf_metrics).to_csv(f'../data/results/{metal}_confusion_all_train_scores.csv')
pd.DataFrame(confusion_class1, index=encoders, columns=conf_metrics).to_csv(f'../data/results/{metal}_confusion_class1_train_scores.csv')
pd.DataFrame(datum_level, index=encoders, columns=cont_metrics).to_csv(f'../data/results/{metal}_datum_level_train_scores.csv')
pd.DataFrame(continuous_avg, index=encoders, columns=cont_metrics).to_csv(f'../data/results/{metal}_continuous_avg_train_scores.csv')
pd.DataFrame(continuous_iid, index=encoders, columns=cont_metrics).to_csv(f'../data/results/{metal}_continuous_iid_train_scores.csv')
