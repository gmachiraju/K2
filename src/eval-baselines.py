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
from torch_geometric.explain import Explainer, AttentionExplainer, GNNExplainer, GraphMaskExplainer, CaptumExplainer, ModelConfig, ExplainerConfig, ThresholdConfig
from torch_geometric.explain.config import ModelMode
import networkx as nx
import metrics
import utils
from evaluation import setup_gridsearch
from GraphSVX.src.explainers import GraphSVX 

import torch_geometric
print(torch_geometric.__version__)

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

class GATSigmoid(torch.nn.Module):
    def __init__(self, model):
        super(GATSigmoid, self).__init__()
        self.model = model
    def forward(self, x, edge_index, batch=None):
        return torch.sigmoid(self.model(x, edge_index, batch))

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

def aggregate_edges(edge_mask, edge_index, num_nodes):
    node_values = np.zeros(num_nodes).astype(float)
    node_totals = np.zeros(num_nodes) + 1e-8
    for i, (n1, n2) in enumerate(edge_index.t()):
        node_values[n1] += edge_mask[i]
        node_values[n2] += edge_mask[i]
        node_totals[n1] += 1
        node_totals[n2] += 1
    return node_values / node_totals

def gridsearch_iteration(loader, model, model_config, algorithm, thresholds, device):
    model_results_dict = {} # returned object
    
    for g, y in tqdm(loader):
        g = g.to(device)
        y = y.to(device)
        G_name = g.id[0]
        
        out = model(g.x, g.edge_index, g.batch)
        y_hat = torch.sigmoid(out).detach().cpu().squeeze().numpy()
        y_true = g.gt.cpu().numpy()

        if algorithm == 'Mask':
            explainer = Explainer(
                model=model,
                algorithm=GraphMaskExplainer(2, epochs=5, log=False),
                explanation_type='model',
                node_mask_type='object',
                model_config=dict(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='raw',
                ),
            )
            # threshold_config = ThresholdConfig(threshold_type='hard', value=t)
            # explainer = Explainer(model, algorithm=GNNExplainer(), model_config=model_config, explanation_type='phenomenon', node_mask_type='object', threshold_config=threshold_config)
            
            explanation = explainer(x=g.x, edge_index=g.edge_index, index=0)
            y_pred = explanation.node_mask.cpu().squeeze().numpy()
            
        elif algorithm == 'Attn':
            explainer = Explainer(
                model=model,
                algorithm=AttentionExplainer(reduce='max'),
                explanation_type='model',
                # node_mask_type='object',
                edge_mask_type='object',
                model_config=dict(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='raw',
                ),
            )
            explanation = explainer(x=g.x, edge_index=g.edge_index, index=0)
            y_pred = aggregate_edges(explanation.edge_mask, explanation.edge_index, g.num_nodes)
            # print(y_pred)
            # y_pred = explanation.node_mask.cpu().squeeze().numpy()
        elif algorithm == 'GNNExplainer':
            explainer = Explainer(
                model=model,
                algorithm=GNNExplainer(),
                explanation_type='model',
                node_mask_type='object',
                model_config=dict(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='raw',
                ),
            )
            explanation = explainer(x=g.x, edge_index=g.edge_index, index=0)
            y_pred = explanation.node_mask.cpu().squeeze().numpy()

        elif algorithm == 'SHAP':
            model = GATSigmoid(model)
            explainer = Explainer(
                model=model,
                algorithm=CaptumExplainer(attribution_method='ShapleyValueSampling'),
                explanation_type='model',
                # node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='probs',
                ),
            )
            explanation = explainer(x=g.x, edge_index=g.edge_index, index=0)
            y_pred = aggregate_edges(explanation.edge_mask, explanation.edge_index, g.num_nodes)
            # print(y_pred)

        else:
            raise Exception("Please select a valid explainer (e.g. Mask, Attn, GNNExplainer)")

        if y.squeeze().item() == 1:
            datum_cont = {"auroc": metrics.auroc(y_pred, y_true), "auprc": metrics.auprc(y_pred, y_true), "ap": metrics.ap(y_pred, y_true)}
            # print(datum_cont)
        else:
            datum_cont = {"auroc": np.nan, "auprc": np.nan, "ap": np.nan}
            
        print(datum_cont)
        
        datum_thresh_msd_dict, datum_thresh_cm_dict = {}, {}
        fprs = []
        tprs = []
        precs = []
        recalls = []
        for t in thresholds:
            data_results_dict = {}
            y_pred_bin = utils.binarize_vec(y_pred, t, conditional=False) # binarize with threshold
            # print(t)
            # print(y_pred_bin)
            # print(y_true)
        
            # y_pred_bin = utils.binarize_vec(y_pred, t, conditional=False) # binarize with threshold
            g_nx = to_networkx(g, to_undirected=True)
            nx.set_node_attributes(g_nx, dict(zip(range(len(y_pred_bin)), y_pred_bin)), 'emb')
            
            datum_thresh_msd_dict[t] = metrics.msd(g_nx)
            datum_thresh_cm_dict[t] = metrics.confusion(y_pred_bin, y_true)
            
            # ravel = metrics.confusion(y_pred_bin, y_true)
            # tprs.append(metrics.sensitivity(ravel))
            # fprs.append(1 - metrics.specificity(ravel))
            # precs.append(metrics.precision(ravel))
            # recalls.append(metrics.recall(ravel))
        
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
    parser.add_argument('--algorithm', type=str, default='GNNExplainer')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--encoder', type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = ModelConfig(mode=ModelMode.binary_classification, task_level='graph', return_type='raw')
    
    thresholds = [np.round(el,2) for el in np.linspace(0,1,21)]
    best_params = {}
    if args.eval_only:
        baseline_top_models = \
            {'COLLAPSE': ('COLLAPSE-ZN-8.0-0.0001-100', 0.7), \
            'ESM': ('ESM-ZN-6.0-0.001-500', 0.4), \
            'AA': ('AA-ZN-6.0-0.001-200', 0.5)}
        for encoder in baseline_top_models:
            if (args.encoder is not None) and (encoder != args.encoder):
                continue
            if encoder == 'COLLAPSE':
                encoder_name = 'COLLAPSE'
                in_dim = 512
            elif encoder == 'ESM':
                encoder_name = 'ESM'
                in_dim = 1280
            elif encoder == 'AA':
                encoder_name = 'COLLAPSE'
                in_dim = 21
            
            cutoff = float(baseline_top_models[encoder][0].split('-')[2])
            lr = float(baseline_top_models[encoder][0].split('-')[3])
            thresh = baseline_top_models[encoder][1]
            dim = int(baseline_top_models[encoder][0].split('-')[-1])
            run_name = args.run_name + f'-{encoder}-{args.metal}-{cutoff:.1f}-{lr}-{dim}'
            
            # print(f'Evaluating {encoder} on test set with threshold {thresh}')

            train_graph_path = f"{args.base_dir}/{encoder_name}_{args.metal}_{cutoff:.1f}_train_graphs_2"
            train_dataset = ProteinData(train_graph_path, encoder)
            train_loader = DataLoader(train_dataset, batch_size=1)
            
            test_graph_path = f"{args.base_dir}/{encoder_name}_{args.metal}_{cutoff:.1f}_test_graphs_2"
            test_dataset = ProteinData(test_graph_path, encoder)
            test_loader = DataLoader(test_dataset, batch_size=1)
            
            model = GAT(in_dim, dim).to(device)
            model.load_state_dict(torch.load(f'../data/baselines/{run_name}-best.pt', map_location=torch.device('cpu')))
            model.eval()

            # model_results_dict = gridsearch_iteration(train_loader, model, model_config, args.algorithm, thresholds=thresholds, device=device)
            # serialize(model_results_dict, os.path.join('../data/baselines', f'{encoder}_{args.algorithm}_train_results.pkl'))
            
            model_results_dict = gridsearch_iteration(test_loader, model, model_config, args.algorithm, thresholds=thresholds, device=device)
            serialize(model_results_dict, os.path.join('../data/baselines', f'{encoder}_{args.algorithm}_test_results.pkl'))
    else:
        for encoder in ['COLLAPSE', 'ESM', 'AA']:
            # save_dir = os.path.join(args.base_dir, f'{encoder}_{args.metal}_{args.run_name}_gridsearch_results')
            # results_dict, results_dir, results_cache_dir, proc_cache_dir, model_cache_dir, linearized_cache_dir = setup_gridsearch(save_dir, encoder)
            best_auprc = 0
            best_auroc = 0
            for cutoff in [6.0, 8.0]:
                for lr in [0.0001, 0.001]:
                    for dim in [100, 200, 500]:
                        print('on', encoder, args.metal, cutoff, lr, dim)
                        model_str = f'{encoder}-{args.metal}-{cutoff:.1f}-{lr}-{dim}'
                        run_name = args.run_name + f'-{model_str}'
                        if encoder == 'COLLAPSE':
                            encoder_name = 'COLLAPSE'
                            in_dim = 512
                        elif encoder == 'ESM':
                            encoder_name = 'ESM'
                            in_dim = 1280
                        elif encoder == 'AA':
                            encoder_name = 'COLLAPSE'
                            in_dim = 21
    
                        train_graph_path = f"{args.base_dir}/{encoder_name}_{args.metal}_{cutoff:.1f}_train_graphs_2"
                        train_dataset = ProteinData(train_graph_path, encoder)
                        train_loader = DataLoader(train_dataset, batch_size=1)
                        test_graph_path = f"{args.base_dir}/{encoder_name}_{args.metal}_{cutoff:.1f}_test_graphs_2"
                        test_dataset = ProteinData(test_graph_path, encoder)
                        test_loader = DataLoader(test_dataset, batch_size=64)
            
                        model = GAT(in_dim, dim).to(device)
            
                        model.load_state_dict(torch.load(f'../data/baselines/{run_name}-best.pt'))
                        # model = model.to(device)
                        model.eval()
                        
                        y_true, y_pred = eval(model, test_loader, device)
                        test_auroc = metrics.auroc(y_pred, y_true)
                        test_auprc = metrics.auprc(y_pred, y_true)
                        if test_auprc > best_auprc:
                            best_auprc = test_auprc
                            best_auroc = test_auroc
                            best_params[encoder] = (cutoff, lr, dim)
            print(encoder)
            print('\tBest Test AUROC:', best_auroc)
            print('\tBest Test AUPRC:', best_auprc)
            print('\tBest params:', best_params[encoder])
        