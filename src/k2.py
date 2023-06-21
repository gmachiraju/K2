import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx

CMAP="tab20"
import utils


class K2Processor():
    """
    Fits clustering models over sampled embeddings, transforming embedded data into sprites
    ——
    k: number of partitions for the embedding space
    quantizer_type: type of clustering model to use for the embedding space (e.g. k-means)
    embeddings_path: path to embeddings to use for the processor
        If None, then the processor will sample embeddings from the dataset
    """
    def __init__(self, args):
        self.k = args.k
        self.quantizer_type = args.quantizer_type
        self.embeddings_path = args.embeddings_path

        self.sample_size = args.sample_size
        self.sample_scheme = args.sample_scheme
        self.dataset_path = args.dataset_path

        if self.embeddings_path is None:
            print("No embeddings path provided, sampling from dataset")
            # determine sample size PER datum based on: sample_size/dataset size
            N = len(os.listdir(self.dataset_path)) 
            if N < self.sample_size:
                self.to_sample = self.sample_size // N 
            else:
                self.to_sample = 1 # and then take sample of that
        
        if self.embeddings_path is None:
            self.embeddings_path = self.sample_embeddings()

        self.quantizer = self.fit_quantizer()  
        self.description = self.quantizer_type + "-" + str(self.k) + "-" + self.sampling_scheme + "-" + str(self.sample_size)

    def sample_embeddings(self):
        """
        Samples embeddings from the dataset. Uses Numpy memmap for efficient embedding storage.
        Serilizes the embeddings to a .npy file and returns the path to the file.
        """
        pass
    
    def fit_quantizer(self):
        """
        Fits quantizer to use for the embedding space (e.g. sklearn k-means model)
        """
        embeddings = np.load(self.embeddings_path)
        if self.quantizer_type == "kmeans":
            model = KMeans(n_clusters=self.k)
        else:
            raise NotImplementedError
        return model.fit(embeddings)


class K2Model():
    """
    processor: K2Processor object
    r: context window size
    variant: "predictive" K2 models (e.g. LASSO) vs "inferential" K2 models (e.g. differential expression)
    hparams: hyperparameters for the variants (e.g. regularization strength)
        alpha: significance level for INFERENTIAL variants
        tau: log2 fold change threshold for INFERENTIAL variants
        lambda: regularization strength for PREDICTIVE variants
    """
    def __init__(self, args):
        self.processor = args.processor
        self.variant = args.variant
        self.hparams = args.hparams
        if self.processor == None:
            raise Exception("Error: K2 Processor is not provided.")

        self.k = self.preprocessor.k
        # now construct data structures
        self.motif_graph = self.instantiate_motif_graph() # K_k
        self.description = self.variant + "-" + self.quantizer + "-" + str(self.k) + "-" + str(self.hparams)
        
        self.class_graphs = [] # where we load mean graphs per class

    def instantiate_motif_graph(self):
        # Make motif graph
        G = nx.complete_graph(self.k)
        [G.add_edge(i,i) for i in range(self.k)]
        # zero weights
        for node in G.nodes:
            G.nodes[node]['n_weight'] = 0.0
        for edge in G.edges:
            G.edges[edge]['e_weight'] = 0.0
        return G
    
    def visualize_motif_graph(self):
        # Visualize motif graph
        pos = nx.circular_layout(self.motif_graph)
        colors = [node for node in list(self.motif_graph.nodes())]
        plt.figure()

        n_weights = nx.get_node_attributes(self.motif_graph, 'n_weight').values()
        n_size = []
        for nw in n_weights:
            ns = int(np.max([1, nw]))
            ns = int(np.min([ns, 10]))
            n_size.append(ns)
        nx.draw_networkx_nodes(self.motif_graph, pos=pos, linewidths=n_size, node_color=colors, cmap=CMAP, edgecolors='black')

        e_weights = nx.get_edge_attributes(self.motif_graph, 'e_weight').values()
        e_thickness = []
        for ew in e_weights:
            et = int(np.max([1, ew]))
            et = int(np.min([et, 10]))
            e_thickness.append(et)
        nx.draw_networkx_edges(self.motif_graph, pos=pos, width=e_thickness, alpha=0.5)
        plt.draw()

    def construct_sprite(self, G):
        """
        Takes a Map Graph G and constructs a sprite from it by applying an embedding quantizer
        """
        for node in G.nodes:
            embedding =  G.nodes[node]['embedding']
            # reassign embedding attribute as a motif pseudo-label
            G.nodes[node]['embedding'] = self.processor.quantizer.predict(embedding)
        return G

    def visualize_sprite(self, G):
        # Visualize sprite
        pos = nx.planar_layout(G)
        nx.draw(self.motif_graph, pos=pos)
        plt.draw()
        #TO-ADD: add colors to nodes; should match colors in motif graph

    def hash_datum(self, G):
        origin = G['origin']
        # ITERATE through nodes in order starting from origin. Row-wise then column-wise?
        # but this coord system doesn't exist in proteins, so we need to figure out a new way to iterate from origin
        
        # subgraph = nx.ego_graph(G,node,radius=k) 
        # then neighbors are nodes of the subgraph: 
        # neighbors = list(subgraph.nodes())

    def motif_graph_hash(self, data_path):    
        for datum_path in os.listdir(data_path):
            G = utils.deserialize
            data_hashed = self.hash_datum(data_path)

        if self.variant == "predictive":
            pass
        elif self.variant == "inferential":
            pass


    def construct_prospect_graph(self, G):
        pass

    def visualize_prospect_graph(self, G):
        pass

    def convert_prospect_to_map(self, G):
        pass
    

def evaluation():
    pass






