import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx

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
    variant: "predictive" K2 models (e.g. LASSO) vs "inferential" K2 models (e.g. differential expression)
    hparams: hyperparameters for the model (e.g. regularization strength)
        alpha: significance level for the model
        tau: log2 fold change threshold for the model
        lambda: regularization strength for the model
    """
    def __init__(self, args):
        # arguments 
        self.processor = args.processor
        self.variant = args.variant
        self.hparams = args.hparams

        self.k = self.preprocessor.k
        # now construct data structures
        self.motif_graph = self.instantiate_motif_graph() # K_k
        self.description = self.variant + "-" + self.quantizer + "-" + str(self.k) + "-" + str(self.hparams)
        
    def instantiate_motif_graph(self):
        # Make motif graph
        G = nx.complete_graph(self.k)
        [G.add_edge(i,i) for i in range(self.k)]
        # TO-DO: add attribute values
        return G
    
    def visualize_motif_graph(self):
        # Visualize motif graph
        pos = nx.circular_layout(self.motif_graph)
        nx.draw(self.motif_graph, pos=pos)
        plt.draw()
        #TO-ADD: add colors to nodes -- try adding labels as attributes as well

    def visualize_sprite(self, G):
        # Visualize sprite
        pos = nx.planar_layout(G)
        nx.draw(self.motif_graph, pos=pos)
        plt.draw()
        #TO-ADD: add colors to nodes; should match colors in motif graph

    def predictive_hashing(self, data_path):
        pass

    def inferential_hashing(self, data_path):
        pass

    def motif_graph_hash(self, data_path):
        if self.variant == "predictive":
            pass
        elif self.variant == "inferential":
            pass
        pass
    









