import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import utils



class K2Preprocessor(args):
    # fits k-means models over embeddings 
    def __init__(self, args):
        self.k = args.k
        self.quantizer_type = args.quantizer_type

        # construct
        self.quantizer = self.fit_quantizer()


class K2(args):
    """
    k: number of partitions for the embedding space
    quantizer: fitted quantizer to use for the embedding space (e.g. sklearn k-means model)
    variant: "predictive" K2 models (e.g. LASSO) vs "inferential" K2 models (e.g. differential expression)
    hparams: hyperparameters for the model (e.g. regularization strength)
        alpha: significance level for the model
        tau: log2 fold change threshold for the model
        lambda: regularization strength for the model
    """
    def __init__(self, args):
        # arguments 
        self.k = args.k
        self.preprocessor = args.preprocessor
        self.variant = args.variant
        self.hparams = args.hparams

        # construct
        self.motif_graph = self.instantiate_motif_graph() # K_k
        self.description = self.variant + "-" + self.quantizer + "-" + str(self.k) + "-" + str(self.hparams)
        
    def instantiate_motif_graph(self):
        # Make motif graph
        G = nx.complete_graph(self.k)
        [G.add_edge(i,i) for i in range(self.k)]
        return G
    
    def visualize_motif_graph(self):
        # Visualize motif graph
        pos = nx.circular_layout(self.motif_graph)
        nx.draw(self.motif_graph, pos=pos)
        plt.draw()


    def visualize_sprite(self, G):
        

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
    









