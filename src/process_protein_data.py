"""
Embed protein data, create graphs, and save to disk.

This script creates two types of data: embeddings and graphs.
- Embeddings are saved as a dictionary with keys of the form {label}_{is_keyres}_{pdb_chain}_{resid}. Values are COLLAPSE embeddings of shape (1, 512).
- Graphs are saved as nx graphs in separate train and test directories. Graphs are saved as {pdb_chain}.pkl. 
    Graphs have the following global attributes:
        - id: pdb_chain
        - label: global label (0 or 1)
    Nodes have the following attributes:
        - resid: residue id
        - emb: COLLAPSE embedding of shape (1, 512)
    Edges are nearest neighbors of each node within distance of 5 Angstrom, equally weighted.

"""


import os
import argparse
import numpy as np
import pandas as pd
from utils import serialize, deserialize
from sklearn.model_selection import train_test_split
from atom3d.datasets import load_dataset
from collapse import process_pdb, embed_protein, initialize_model, atom_info
from torch_geometric.nn import radius_graph
import networkx as nx
import torch
from scipy.sparse import csr_array
from tqdm import tqdm

def create_splits(database, metal):
    db = database[metal]
    db_pos, db_neg = db['pos'], db['neg']
    
    num_pos_ec = db_pos['EC_NUMBER'].unique()
    train_ec, test_ec = train_test_split(num_pos_ec, test_size=0.2, random_state=77)
    train_df = db_pos[db_pos['EC_NUMBER'].isin(train_ec)]
    test_df = db_pos[db_pos['EC_NUMBER'].isin(test_ec)]
    train_keyres = dict(train_df.groupby('pdb_chain')['interactions'].apply(lambda x: [i for i in sum(x, []) if i.split('_')[0] in atom_info.aa]))
    test_keyres = dict(test_df.groupby('pdb_chain')['interactions'].apply(lambda x: [i for i in sum(x, []) if i.split('_')[0] in atom_info.aa]))
    
    train_neg_pdb, test_neg_pdb = train_test_split(db_neg['pdb_chain'].unique(), test_size=0.2, random_state=77)
    return train_keyres, test_keyres, train_neg_pdb, test_neg_pdb

def compute_adjacency(df, resids, r):
    df = df[df.name == 'CA']
    df['resid'] = df['resname'].apply(lambda x: atom_info.aa_to_letter(x)) + df['residue'].astype(str) + df['insertion_code'].astype(str).str.strip()
    df = df[df.resid.isin(resids)]
    edge_index = radius_graph(torch.tensor(df[['x', 'y', 'z']].to_numpy()), r=r)
    return edge_index.numpy()

def embed_pdb(pdb_chain, pdb_dir, model, device, r):
    pdb, chain = pdb_chain[:4], pdb_chain[-1]
    atom_df = process_pdb(os.path.join(pdb_dir, pdb[1:3], f'pdb{pdb}.ent.gz'), chain=chain, include_hets=False)
    try:
        outdata = embed_protein(atom_df.copy(), model, device=device, include_hets=False, env_radius=10.0)
    except Exception as e:
        print(f'{pdbc} failed with exception {e}')
        return
    if outdata is None:
        return
    outdata['adj'] = compute_adjacency(atom_df.copy(), outdata['resids'], r)
    return outdata

def dict2graph(emb_data):
    adj = csr_array((np.ones(emb_data['adj'].shape[1]), (emb_data['adj'][0], emb_data['adj'][1])), shape=(len(emb_data['resids']), len(emb_data['resids'])))
    G = nx.from_scipy_sparse_array(adj)
    G.graph.update({'d': emb_data['embeddings'].shape[1]})
    
    node_attr = {i: {'resid': emb_data['resids'][i], 'emb': emb_data['embeddings'][i]} for i in range(len(emb_data['resids']))}
    nx.set_node_attributes(G, node_attr)
    return G

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../data/metal_database_balanced.pkl')
    parser.add_argument('--pdb_dir', type=str, default='/scratch/users/aderry/pdb')
    parser.add_argument('--metal', type=str, default='K')
    parser.add_argument('--nn_radius', type=float, default=8.0)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(device=device)
    
    database = deserialize(args.dataset)
    train_keyres, test_keyres, train_neg, test_neg = create_splits(database, args.metal)
    
    train_embed_dict = {}
    train_graph_dir = '../data/train_graphs'
    os.makedirs(train_graph_dir, exist_ok=True)
    
    for pdbc, res in tqdm(train_keyres.items(), 'train positive'):
        emb_data = embed_pdb(pdbc, args.pdb_dir, model, device, args.nn_radius)
        if emb_data is None:
            continue
        labels = np.zeros(len(emb_data['resids']))
        pos_resids = [atom_info.aa_to_letter(i.split('_')[0]) + i.split('_')[1] for i in res]
        labels[np.isin(emb_data['resids'], pos_resids)] = 1
        
        G = dict2graph(emb_data)
        G.graph.update({'id': pdbc, 'label': 1})
        serialize(G, os.path.join(train_graph_dir, f'{pdbc}.pkl'))
        
        for i in range(len(emb_data['embeddings'])):
            key = f"1_{int(labels[i])}_{pdbc}_{emb_data['resids'][i]}"
            train_embed_dict[key] = emb_data['embeddings'][i]
    
    for pdbc in tqdm(train_neg, 'train negative'):
        emb_data = embed_pdb(pdbc, args.pdb_dir, model, device, args.nn_radius)
        if emb_data is None:
            continue
        
        G = dict2graph(emb_data)
        G.graph.update({'id': pdbc, 'label': 1})
        serialize(G, os.path.join(train_graph_dir, f'{pdbc}.pkl'))
        
        for i in range(len(emb_data['embeddings'])):
            key = f"0_0_{pdbc}_{emb_data['resids'][i]}"
            train_embed_dict[key] = emb_data['embeddings'][i]
    
    serialize(train_embed_dict, f'../data/{args.metal}_train_embeddings.pkl')
    
    test_embed_dict = {}
    test_graph_dir = '../data/test_graphs'
    os.makedirs(test_graph_dir, exist_ok=True)
    
    for pdbc, res in tqdm(test_keyres.items(), 'test positive'):
        emb_data = embed_pdb(pdbc, args.pdb_dir, model, device, args.nn_radius)
        if emb_data is None:
            continue
        labels = np.zeros(len(emb_data['resids']))
        pos_resids = [atom_info.aa_to_letter(i.split('_')[0]) + i.split('_')[1] for i in res]
        labels[np.isin(emb_data['resids'], pos_resids)] = 1
        
        G = dict2graph(emb_data)
        G.graph.update({'id': pdbc, 'label': 0})
        serialize(G, os.path.join(test_graph_dir, f'{pdbc}.pkl'))

        for i in range(len(emb_data['embeddings'])):
            key = f"1_{int(labels[i])}_{pdbc}_{emb_data['resids'][i]}"
            test_embed_dict[key] = emb_data['embeddings'][i]
    
    for pdbc in tqdm(test_neg, 'test negative'):
        emb_data = embed_pdb(pdbc, args.pdb_dir, model, device, args.nn_radius)
        if emb_data is None:
            continue
        
        G = dict2graph(emb_data)
        G.graph.update({'id': pdbc, 'label': 0})
        serialize(G, os.path.join(test_graph_dir, f'{pdbc}.pkl'))
        
        for i in range(len(emb_data['embeddings'])):
            key = f"0_0_{pdbc}_{emb_data['resids'][i]}"
            test_embed_dict[key] = emb_data['embeddings'][i].astype('float')
    
    serialize(test_embed_dict, f'../data/{args.metal}_test_embeddings.pkl')
    