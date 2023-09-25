import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import argparse
from tqdm import tqdm
from utils import serialize, deserialize, set_graph_emb, AAQuantizer
from gvp.atom3d import BaseModel, BaseTransform
from collapse.data import process_pdb
from process_protein_data import create_splits
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils.convert import from_networkx, to_networkx
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig, ExplainerConfig, ThresholdConfig
from torch_geometric.explain.config import ModelMode
import networkx as nx
import metrics
import utils
from evaluation import setup_gridsearch

# import torch_geometric
# print(torch_geometric.__version__)

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        # self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)


    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = global_mean_pool(x, batch)
        return self.fc(x).view(-1)

def aa_quantize(S, quantizer):
    for node in S.nodes:
        embedding = S.nodes[node]['emb']
        motif_label = quantizer.predict(embedding)
        aa_onehot = np.zeros(21)
        aa_onehot[motif_label] = 1
        S.nodes[node]['emb'] = aa_onehot.astype('float32')
    return S

class ProteinData(Dataset):
    
    def __init__(self, data_path, encoder):
        self.path = data_path
        self.encoder = encoder
        self.names = os.listdir(self.path)
        self.quantizer = AAQuantizer()
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        g_nx = deserialize(os.path.join(self.path, self.names[idx]))
        if self.encoder == 'AA':
            g_nx = set_graph_emb(g_nx, 'resid')
            g_nx = aa_quantize(g_nx, self.quantizer)
        
        graph = from_networkx(g_nx)
        graph.x = graph.emb
        return graph, float(graph.label)

def eval(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch, y in dataloader:
            batch = batch.to(device)
            y_true.extend(y.tolist())
            y = y.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            # loss = criterion(out, y)
            y_pred.extend(torch.sigmoid(out).cpu().tolist())
    return y_true, y_pred
    
def gridsearch_iteration(loader, model_config):
    model_results_dict = {} # returned object
    thresholds = [np.round(el,1) for el in np.linspace(0,1,11)]
    
    for g, y in tqdm(loader):
        # g = g.to(device)
        # y = y.to(device)
        G_name = g.id[0]
        
        out = model(g.x, g.edge_index, g.batch)
        y_hat = torch.sigmoid(out).detach().cpu().squeeze().numpy()
        
        datum_thresh_msd_dict, datum_thresh_cm_dict = {}, {}
        fprs = []
        tprs = []
        precs = []
        recalls = []
        for t in thresholds:
            data_results_dict = {}
            threshold_config = ThresholdConfig(threshold_type='hard', value=t)
            explainer = Explainer(model, algorithm=GNNExplainer(), model_config=model_config, explanation_type='phenomenon', node_mask_type='object', threshold_config=threshold_config)
            
            explanation = explainer(x=g.x, edge_index=g.edge_index, target=y.long())
            # print(explanation)
            y_pred_bin = explanation.node_mask.cpu().squeeze().numpy()
            y_true = g.gt.cpu().numpy()
        
            # y_pred_bin = utils.binarize_vec(y_pred, t, conditional=False) # binarize with threshold
            g_nx = to_networkx(g, to_undirected=True)
            nx.set_node_attributes(g_nx, dict(zip(range(len(y_pred_bin)), y_pred_bin)), 'emb')
            
            datum_thresh_msd_dict[t] = metrics.msd(g_nx)
            datum_thresh_cm_dict[t] = metrics.confusion(y_pred_bin, y_true)
            
            ravel = metrics.confusion(y_pred_bin, y_true)
            tprs.append(metrics.sensitivity(ravel))
            fprs.append(1 - metrics.specificity(ravel))
            precs.append(metrics.precision(ravel))
            recalls.append(metrics.recall(ravel))
        
        datum_cont = {"auroc": np.nan, "auprc": np.nan, "ap": np.nan}
        if y.squeeze().item() == 1:
            try:
                datum_cont = {"auroc": metrics.auc(fprs, tprs), "auprc": metrics.auc(recalls, precs), "ap": np.nan}
            except ValueError:
                print('ValueError on example', G_name)
                continue
        
        data_results_dict["thresh_msd"] = datum_thresh_msd_dict
        data_results_dict["thresh_cm"] = datum_thresh_cm_dict
        data_results_dict["pred"] = (y_hat, None)
        data_results_dict["cont"] = datum_cont
    
        model_results_dict[G_name] = data_results_dict
    return model_results_dict
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='../data')
    parser.add_argument('--metal', type=str, default='ZN')
    parser.add_argument('--run_name', type=str, default='baseline-GAT')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = ModelConfig(mode=ModelMode.binary_classification, task_level='graph', return_type='raw')
    
    best_params = {}
    for encoder in ['COLLAPSE', 'ESM', 'AA']:
        # save_dir = os.path.join(args.base_dir, f'{encoder}_{args.metal}_{args.run_name}_gridsearch_results')
        # results_dict, results_dir, results_cache_dir, proc_cache_dir, model_cache_dir, linearized_cache_dir = setup_gridsearch(save_dir, encoder)
        best_auprc = 0
        best_auroc = 0
        for cutoff in [6.0, 8.0]:
            for lr in [1e-5, 0.0001, 0.0005]:
                # print('on', encoder, args.metal, cutoff, lr)
                run_name = args.run_name + f'-{encoder}-{args.metal}-{cutoff:.1f}-{lr}'
                if encoder == 'COLLAPSE':
                    encoder_name = 'COLLAPSE'
                    in_dim = 512
                elif encoder == 'ESM':
                    encoder_name = 'ESM'
                    in_dim = 1280
                elif encoder == 'AA':
                    encoder_name = 'COLLAPSE'
                    in_dim = 21

                model_str = f'{encoder}-{args.metal}-{cutoff:.1f}-{lr}'
                # train_graph_path = f"{args.base_dir}/{encoder_name}_{args.metal}_{cutoff:.1f}_train_graphs"
                # train_dataset = ProteinData(train_graph_path, encoder)
                # train_loader = DataLoader(train_dataset, batch_size=1)
                test_graph_path = f"{args.base_dir}/{encoder_name}_{args.metal}_{cutoff:.1f}_test_graphs"
                test_dataset = ProteinData(test_graph_path, encoder)
                test_loader = DataLoader(test_dataset, batch_size=64)
    
                model = GAT(in_dim, 100).to(device)
    
                model.load_state_dict(torch.load(f'../data/baselines/{run_name}-best.pt'))
                model = model.to(device)
                model.eval()
                
                y_true, y_pred = eval(model, test_loader, device)
                test_auroc = metrics.auroc(y_pred, y_true)
                test_auprc = metrics.auprc(y_pred, y_true)
                if test_auprc > best_auprc:
                    best_auprc = test_auprc
                    best_auroc = test_auroc
                    best_params[encoder] = (cutoff, lr)
        print(encoder)
        print('\tBest Test AUROC:', best_auroc)
        print('\tBest Test AUPRC:', best_auprc)
        print('\tBest params:', best_params[encoder])
                
    for encoder in ['COLLAPSE', 'ESM', 'AA']:
        print(f'Evaluating {encoder} on test set')
        test_loader = DataLoader(test_dataset, batch_size=1)
        
        model = GAT(in_dim, 100).to('cpu')
        model.load_state_dict(torch.load(f'../data/baselines/{run_name}-best.pt'))
        model.eval()
        
        model_results_dict = gridsearch_iteration(test_loader, model_config)
        # results_dict[model_str] = os.path.join(results_cache_dir, model_str) # save_path
        serialize(model_results_dict, os.path.join('../data/baselines', f'{encoder}_test_results.pkl'))
        # serialize(model, os.path.join(model_cache_dir, model_str))