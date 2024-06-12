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
from torch_geometric.utils.convert import from_networkx
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig, ExplainerConfig, ThresholdConfig
from torch_geometric.explain.config import ModelMode
import wandb
import metrics

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
    
def train_loop(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    for batch, y in tqdm(dataloader, desc='Epoch ' + str(epoch+1)):
        batch = batch.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

def eval(model, dataloader, criterion, device):
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch, y in dataloader:
            batch = batch.to(device)
            y_true.extend(y.tolist())
            y = y.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, y)
            y_pred.extend(torch.sigmoid(out).cpu().tolist())
            losses.append(loss.item())
    return np.mean(losses), y_true, y_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='../data')
    parser.add_argument('--metal', type=str, default='CA')
    parser.add_argument('--encoder', type=str, default='COLLAPSE')
    parser.add_argument('--cutoff', type=float, default=8.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--run_name', type=str, default='baseline-GAT')
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()
    
    args.run_name = args.run_name + f'-{args.encoder}-{args.metal}-{args.cutoff:.1f}-{args.lr}-{args.dim}'
    # wandb.init(project="K2", name=args.run_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.encoder == 'COLLAPSE':
        encoder_name = 'COLLAPSE'
        in_dim = 512
    elif args.encoder == 'ESM':
        encoder_name = 'ESM'
        in_dim = 1280
    elif args.encoder == 'AA':
        encoder_name = 'COLLAPSE'
        in_dim = 21
    
    train_graph_path = f"{args.base_dir}/{encoder_name}_{args.metal}_{args.cutoff:.1f}_train_graphs_2"
    test_graph_path = f"{args.base_dir}/{encoder_name}_{args.metal}_{args.cutoff:.1f}_test_graphs_2"
    train_dataset = ProteinData(train_graph_path, args.encoder)
    test_dataset = ProteinData(test_graph_path, args.encoder)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = GAT(in_dim, args.dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # wandb.watch(model, log='gradients', log_freq=100)
    
    if not args.eval_only:
        best_loss = np.inf
        for epoch in range(args.epochs):
            train_loop(model, train_loader, optimizer, criterion, device, epoch)
            train_loss, _, _ = eval(model, train_loader, criterion, device)
            test_loss, _, _ = eval(model, test_loader, criterion, device)
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), f'../data/baselines/{args.run_name}-best.pt')
            # wandb.log({'train_loss': train_loss, 'Test loss': test_loss})
            print(f'Train loss: {train_loss} | Test loss: {test_loss}')
    
    quit()
    # model.load_state_dict(torch.load(f'../data/baselines/{args.run_name}-best.pt'))
    # model = model.to(device)
    # model.eval()
    
    # test_loss, y_true, y_pred = eval(model, test_loader, criterion, device)
    # test_auroc = metrics.auroc(y_pred, y_true)
    # test_auprc = metrics.auprc(y_pred, y_true)
    # print('Test AUROC:', test_auroc)
    # print('Test AUPRC:', test_auprc)
    
    # train_loader = DataLoader(train_dataset, batch_size=1)
    
    # model_config = ModelConfig(mode=ModelMode.binary_classification, task_level='graph', return_type='raw')
    
    # thresholds = [np.round(el,1) for el in np.linspace(0,1,11)]
    # for t in thresholds:
    #     threshold_config = ThresholdConfig(threshold_type='hard', value=t)
    #     explainer = Explainer(model, algorithm=GNNExplainer(), model_config=model_config, explanation_type='phenomenon', node_mask_type='object', threshold_config=threshold_config)
    #     precs = []
    #     for g, y in train_loader:
    #         g = g.to(device)
    #         y = y.to(device)
    #         if y.squeeze() == 0:
    #             continue
    #         explanation = explainer(x=g.x, edge_index=g.edge_index, target=y.long())
    #         # print(explanation)
    #         y_pred = explanation.node_mask.cpu().squeeze().numpy()
    #         y_true = g.gt.cpu().numpy()
    #         ravel = metrics.confusion(y_pred, y_true)
    #         # ravel = confusion_matrix(P_bin_vec, Y_1hop)
    #         precision = metrics.precision(ravel)
    #         precs.append(precision)
    #         break
    #     print(t, np.mean(precision))