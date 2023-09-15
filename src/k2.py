import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
import pandas as pd
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm
from time import sleep
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import scipy

import utils
import pdb
EPS = 1e-10 # numerical stability
CMAP="tab20"
custom_cmap = plt.get_cmap(CMAP)
custom_cmap.set_bad(color='white')

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
        args = utils.dotdict(args)
        self.datatype = args.datatype
        self.k = args.k
        self.quantizer_type = args.quantizer_type
        self.embeddings_path = args.embeddings_path
        self.embeddings_type = args.embeddings_type
        self.mapping_path = args.mapping_path # only used if embeddings are stored as a memmap
        self.sample_size = args.sample_size
        self.sample_scheme = args.sample_scheme
        self.dataset_path = args.dataset_path
        self.verbosity = args.verbosity
        self.so_dict_path = args.so_dict_path

        if self.dataset_path is None:
            raise Exception("Error: dataset path is not provided.")

        if self.embeddings_path is None:
            print("No embeddings path provided, sampling from dataset")
            # determine sample size PER datum based on: sample_size/dataset size
            N = len(os.listdir(self.dataset_path)) 
            if N < self.sample_size:
                self.to_sample = self.sample_size // N 
            else:
                self.to_sample = 1 # and then take sample of that
        
            # run sampling
            self.embeddings_path = self.sample_embeddings()
        else:
            print("Embeddings path provided, loading embeddings...")

        self.motif_graph = self.instantiate_motif_graph() # K_k
        self.quantizer = None
        self.description = self.quantizer_type + "-" + str(self.k) + "-" + self.sample_scheme + "-" + str(self.sample_size)

    def sample_embeddings(self):
        """
        Samples embeddings from the dataset. Uses Numpy memmap for efficient embedding storage.
        Serilizes the embeddings to a .npy file and returns the path to the file.
        """
        raise NotImplementedError
    
    def split_embeddings(self, embeddings=None, labels=None):
        if embeddings is None:
            embeddings = self.embedding_array.copy()
        if labels is None:
            labels = self.quantizer.labels_.copy()
            
        embedded_o = self.embedding_array[0:self.idx_o,:]
        labels_o = labels[0:self.idx_o]
        embedded_x = self.embedding_array[self.idx_o:self.idx_x,:]
        labels_x = labels[self.idx_o:self.idx_x]
        if self.idx_X == 0:
            embedded_X = self.embedding_array[self.idx_x:,:]
            labels_X = labels[self.idx_x:]
        else:
            embedded_X = self.embedding_array[self.idx_x:self.idx_X,:]
            labels_X = labels[self.idx_x:self.idx_X]
        return embedded_o, labels_o, embedded_x, labels_x, embedded_X, labels_X
    
    def fit_quantizer(self):
        """
        Fits quantizer to use for the embedding space (e.g. sklearn k-means model)
        """
        if self.quantizer_type == "kmeans":
            print("Chosen KMeans model for quantization...")
            model = KMeans(n_clusters=self.k, random_state=0)
        else:
            raise NotImplementedError
        
        self.embedding_array = self.load_embeddings()
        self.quantizer = model.fit(self.embedding_array)

    def load_embeddings(self):
        """
        Prepares embeddings for quantizer fitting
        """
        if self.embeddings_type == "dict":
            embed_dict = utils.deserialize(self.embeddings_path)
            id_list = list(embed_dict.keys())
            # for k in embed_dict.keys():
            #     id_list.append(k)
            mapping_dict = {} # dummy
        elif self.embeddings_type == "memmap":
            embed_dict = np.memmap(self.embeddings_path, dtype='float32', mode='r', shape=(self.sample_size,1024)) # not an actual dict as you can see
            mapping_dict = utils.deserialize(self.mapping_path)
            id_list = mapping_dict.keys()

        self.embed_dict = embed_dict
        self.mapping_dict = mapping_dict
        self.id_list = id_list
        array = self.partition_data()
        return array

    def partition_data(self):
        """
        This helper function helps us partition data into classes (and within class-1, identify salient objects).
        Also helps for visualization downstream.
        """
        # create df and partition by marker
        if self.datatype == 'histo':
            ms = self.get_plot_markers()
        elif self.datatype == 'protein':
            ms = self.get_residue_labels()
        ms_dict = dict(zip(self.id_list, ms))
        embed_df = pd.DataFrame.from_dict(ms_dict, orient='index', columns=["marker"])
        o_df = embed_df.loc[embed_df['marker'] == "o"] # 0-class
        x_df = embed_df.loc[embed_df['marker'] == "x"] # 1-class, non-salient
        X_df = embed_df.loc[embed_df['marker'] == "X"] # 1-class, salient

        # separate arrays and indices
        array_o, self.idx_o = self.partition_by_marker(o_df) # idx_o is end index of 0 class
        array_x, self.idx_x = self.partition_by_marker(x_df)
        array_X, self.idx_X = self.partition_by_marker(X_df)

        # update indices of arrays
        self.idx_x += self.idx_o # end index of x class
        self.idx_X += self.idx_x # end index of X class 

        # concatenate arrays
        arrays = [array_o, array_x, array_X]
        array = np.vstack(arrays).astype("double")
        if self.verbosity == "full":
            print("total embeds:", array.shape[0])
            print("collapsing from dim", array.shape[1], "--> 2")
        return array # (num_embeds, embed_dim)

    def get_plot_markers(self):
        """
        Gives us a way to visualize different chunks
        """
        sal_counter = 0
        so_dict = self.so_dict_path
        if so_dict is not {}:
            so_dict = utils.deserialize(so_dict)            

        ms = []
        for id_val in self.id_list:
            pieces = id_val.split("_")
            lab = pieces[0]
            id_num = lab+"_"+pieces[1]
            coordi, coordj = int(pieces[2]), int(pieces[3])
            val = 0
            if (so_dict is not {}) and (id_num in so_dict.keys()): # catching if train has no annotated salient objects
                val = so_dict[id_num][(coordi,coordj)]
            if val == 0 and lab == "normal":
                m = "o"
            elif val == 0 and lab == "tumor":
                m = "x"
            elif val == 1 and lab == "tumor":
                sal_counter += 1
                m = "X"
            ms.append(m) # this is the case when so_dict is not passed in

        print("sampled", str(sal_counter), "known salient objects!")
        return ms
    
    def get_residue_labels(self):
        """
        Protein-specific labels
        """
        sal_counter = 0      

        ms = []
        for id_val in self.id_list:
            pieces = id_val.split("_")
            lab = int(pieces[0])
            val = int(pieces[1])
            
            if val == 0 and lab == 0:
                m = "o"
            elif val == 0 and lab == 1:
                m = "x"
            elif val == 1 and lab == 1:
                sal_counter += 1
                m = "X"
            ms.append(m) # this is the case when so_dict is not passed in

        print("sampled", str(sal_counter), "known salient objects!")
        return ms
    
    def partition_by_marker(self, mark_df):
        embeds_list_mark = []
        for k in list(mark_df.index):
            if self.embeddings_type == "dict":
                v = self.embed_dict[k]
            elif self.embeddings_type == "memmap":
                pos = self.mapping_dict[k]
                v = self.embed_dict[pos,:]
            embeds_list_mark.append(v)
        array_mark = np.vstack(embeds_list_mark)
        idx_mark = len(embeds_list_mark)
        return array_mark, idx_mark
        
    def visualize_quantizer(self, subsample=None):
        """visualize quantizer labels (clusters) in tSNE reduced-dim space

        :param subsample: if provided, subsample negative elements to this number, defaults to None
        :type subsample: int, optional

        """
        if self.quantizer is None:
            raise Exception("Error: quantizer is not fitted.")

        if subsample:
            if self.verbosity == "full":
                print(f'subsampling elements to {subsample*100} %')
            embedded_o, labels_o, embedded_x, labels_x, embedded_X, labels_X = self.split_embeddings()
            print(f'num o: {int(embedded_o.shape[0] * subsample)}, num x: {int(embedded_x.shape[0] * subsample)}, num X: {int(embedded_X.shape[0] * subsample)}')
            sample_o = np.random.choice(embedded_o.shape[0], int(embedded_o.shape[0] * subsample), replace=False)
            embedded_o = embedded_o[sample_o,:]
            labels_o = labels_o[sample_o]
            sample_x = np.random.choice(embedded_x.shape[0], int(embedded_x.shape[0] * subsample), replace=False)
            embedded_x = embedded_x[sample_x,:]
            labels_x = labels_x[sample_x]
            sample_X = np.random.choice(embedded_X.shape[0], int(embedded_X.shape[0] * subsample), replace=False)
            embedded_X = embedded_X[sample_X,:]
            labels_X = labels_X[sample_X]
            embedding_array = np.vstack([embedded_o, embedded_x, embedded_X])
            cluster_labs = np.hstack([labels_o, labels_x, labels_X])
        else:
            embedding_array = self.embedding_array.copy()
            cluster_labs = self.quantizer.labels_.copy()

        # tsne - color by source
        for perplexity in [5,10,20]:
            tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity).fit_transform(embedding_array)
            if self.verbosity == "full":
                fig, ax1 = plt.subplots(1, 1)
                fig.suptitle('Sampled embeddings for cluster assignment')
                ax1.set_xlabel("tSNE-0")
                ax1.set_ylabel("tSNE-1")

            if self.verbosity == "full":
                ax1.set_xlabel("tSNE-0")
                ax1.set_title("t-SNE with K="+str(self.k)+" clusters (perplexity="+str(perplexity)+")")
                
                ax1.scatter(tsne[:len(embedded_o),0], tsne[:len(embedded_o),1], c=labels_o, alpha=0.3, s=5, marker="o", cmap="Dark2")
                ax1.scatter(tsne[len(embedded_o):len(embedded_o)+len(embedded_x),0], tsne[len(embedded_o):len(embedded_o)+len(embedded_x),1], c=labels_x, alpha=0.3, s=30, marker="x", cmap="Dark2")
                ax1.scatter(tsne[len(embedded_o)+len(embedded_x):,0], tsne[len(embedded_o)+len(embedded_x):,1], c=labels_X, alpha=0.6, s=300, edgecolors="k", marker="X", cmap="Dark2") 
                
        unique, counts = np.unique(cluster_labs, return_counts=True)
        if self.verbosity == "full":
            plt.figure()
            plt.title("Cluster bar chart")
            plt.bar(unique, height=counts)

    def instantiate_motif_graph(self):
        """
        Make motif graph K_k
        """
        Kk = nx.complete_graph(self.k)
        [Kk.add_edge(i,i) for i in range(self.k)] # self edges
        # initialize zero weights
        for node in Kk.nodes:
            Kk.nodes[node]['n_weight'] = 0.0
        for edge in Kk.edges:
            Kk.edges[edge]['e_weight'] = 0.0
        return Kk
    

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
        args = utils.dotdict(args)
        self.modality = args.modality # graph, image 
        self.processor = args.processor
        self.r = args.r
        self.variant = args.variant
        self.hparams = args.hparams
        self.train_graph_path = args.train_graph_path

        # allows for labels to be stored in Graphs or loaded as a dictionary
        if "train_label_dict" in args.keys():
            self.train_label_dict = args.train_label_dict
        else:
            self.train_label_dict = None

        if self.processor == None:
            raise Exception("Error: K2 Processor is not provided.")
        self.k = self.processor.k
        self.quantizer = self.processor.quantizer
        
        # now construct data structures
        self.Kk_unweight = self.processor.motif_graph
        self.Kk_pruning() # only runs if r=0
        self.description = self.variant + "-" + self.processor.quantizer_type + "-" + str(self.k) + "-" + str(self.hparams)
        self.class_graphs = [] # where we load mean graphs per class

    def Kk_pruning(self):
        if self.r == 0: # if r=0, then we need to prune to just nodes
            # prune edges, including self-edges
            for edge in self.Kk_unweight.edges:
                self.Kk_unweight.remove_edge(edge[0], edge[1])

            print("Note: r=0, so pruned Kk to k nodes only")    
            print(self.Kk_unweight)

        # create a sorted ordering for hashmap keys
        key_list = [node for node in self.Kk_unweight.nodes] + [edge for edge in self.Kk_unweight.edges]
        self.hash_keys = self.sort_keys(key_list)    

    def create_train_array(self):
        """
        Preprocesses map graphs as training vectors for K2
        """
        X,y = [],[]
        G_files = os.listdir(self.train_graph_path)
        with tqdm(total=len(G_files), desc="Creating K2 training array...") as pbar:
            for t in range(len(G_files)):
                G_file = G_files[t]
                # load map graph data
                G = utils.deserialize(os.path.join(self.train_graph_path, G_file))
                
                # load in labels
                if self.train_label_dict is None:
                    y.append(G.graph['label'])
                else:
                    y.append(self.train_label_dict[G_file])

                # quantize and embed sprite
                sprite = self.construct_sprite(G)
                g = self.embed_sprite(sprite)
                # store g-vector in array for training
                X.append(g)
                # y.append(self.train_label_dict[G_file])
                pbar.set_description('processed: %d' % (1 + t))
                pbar.update(1)

        self.training_data = np.vstack(X)
        n,p = self.training_data.shape
        self.labels = np.array(y)
        print("Complete! Created a training array for few-shot classification...")
        print("Number of training examples:", n)
        print("Number of Kk features:", p)

    def fit_kernel(self, normalize_flag=True, alpha=None, tau=None):
        """
        Main method for training K2 model
        Inputs:
            normalize_flag: a boolean T/F value to toggle TF-IDF normalization
            alpha: optionally override alpha hyperparameter attribute for fitting
            tau: optionally override tau hyperparameter attribute for fitting
        """
        # tfidf scaling
        if normalize_flag == True:
            print("Normalizing training data with TF-IDF...")
            self.tfidf()
        if alpha is not None:
            print(f'updating alpha to {alpha}')
            self.hparams['alpha'] = alpha
        if tau is not None:
            print(f'updating tau to {tau}')
            self.hparams['tau'] = tau

        if self.variant == "predictive":
            self.train_predictive_k2()
        elif self.variant == "inferential":
            self.train_inferential_k2()

        # create kernel for prospection
        self.w_hmap = dict(zip(self.hash_keys, self.B)) # hashmap kernel
        self.w_hgraph = self.Kk_unweight.copy() # hashgraph version of kernel
        for node in self.w_hgraph.nodes:
            self.w_hgraph.nodes[node]['n_weight'] = self.w_hmap[node]
        for edge in self.w_hgraph.edges:
            self.w_hgraph.edges[edge]['e_weight'] = self.w_hmap[edge]

    def train_predictive_k2(self, scaling_flag=True):
        """
        Predictive K2 model
        Inputs:
            scaling_flag: a boolean T/F value to toggle additional standard scaling
        """
        l = self.hparams["lambda"]
        print("Fitting ElasticNet with l1 ratio: "+str(l)+"...")
        model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=l, random_state=0, max_iter=3000)
        X = self.training_data

        if scaling_flag == True:
            print("performing standard scaling beforehand...")
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        y = self.labels.astype(int)
        model = model.fit(X, y)
        self.classifier = model # storing classifier for test set classification
        self.B = model.coef_[0] # importance weights

    def train_inferential_k2(self):
        """
        Inferential K2 model
        """
        alpha = self.hparams["alpha"]
        tau = self.hparams["tau"]
        print("Differential Expression with alpha,tau: "+str(alpha)+","+str(tau))
        
        # split data into classes
        X = self.training_data
        y = self.labels.astype(int)
        X0 = X[y==0,:]
        X1 = X[y==1,:]
        # compute mean vectors
        mu0 = np.mean(X0, axis=0)
        mu1 = np.mean(X1, axis=0)
        log2fc = np.log2((mu1 + EPS)/(mu0 + EPS))

        # compute p-values between classes
        pvals = []
        p = X0.shape[1]
        for idx in range(p):
            _, pval = scipy.stats.mannwhitneyu(X1[:,idx], X0[:,idx])
            pvals.append(pval / p) # bonferroni correction
        
        # building a mask to "regularize"/squash features to zero
        sig_mask = []
        for idx,l in enumerate(log2fc):
            pval = pvals[idx]
            if (pval > alpha) or (np.abs(l) < tau):
                sig_mask.append(0)
            else:
                sig_mask.append(1)

        self.B = log2fc * np.array(sig_mask)
        self.classifier = None # no classification capability
    
    def prospect(self, G):
        """
        Predicts class-differential nodes in a Map Graph
        Performs prospection: nonlinear convolution with fitted B
        Inputs:
            G: networkx map graph 
        """
        # construct sprite via quantization
        S = self.construct_sprite(G)
        P = S.copy()
        # reset P's nodes to 0 
        for node in P.nodes:
            P.nodes[node]['emb'] = 0.0
        
        # for n in self.w_hgraph.nodes:
        #     print(n, self.w_hgraph.nodes[n]['n_weight'])
        # for e in self.w_hgraph.edges:
        #     print(e, self.w_hgraph.edges[e]['e_weight'])
            
        # nonlinear convolve with hashmap
        for node in S.nodes:
            node_motif = S.nodes[node]['emb']
            subgraph = nx.ego_graph(S, node, radius=self.r) 
            neighbors = list(subgraph.nodes())
            nhbr_motifs = [S.nodes[n]['emb'] for n in neighbors]
            unique, counts = np.unique(nhbr_motifs, return_counts=True)

            # load node presence in weighted graph
            P.nodes[node]['emb'] += self.w_hgraph.nodes[node_motif]["n_weight"]
            # print(node_motif, self.w_hgraph.nodes[node_motif]["n_weight"])
            # load in edge presence in weighted graph (skipgrams)
            for idx, u in enumerate(unique):
                P.nodes[node]['emb'] += (self.w_hgraph.edges[(node_motif,u)]["e_weight"] * counts[idx])
                # print((node_motif,u), self.w_hgraph.edges[(node_motif,u)]["e_weight"])
        
        # return prospect map P w/ class-differential nodes
        return P

    #==================
    # Helper functions
    #==================
    def tfidf(self):
        """
        TF-IDF normalization, in-place
        """
        X = self.training_data
        n = X.shape[0]
        doc_freq = np.count_nonzero(X, axis=0)
        idf = np.log(n / (1+doc_freq))
        tf = X / np.sum(X, axis=1).reshape(-1,1)
        X = tf * idf
        self.training_data = X

    def embed_sprite(self, S):
        """
        Embeds a graph into the embedding space
        Input:
            S: sprite graph
        Output:
            g: vector representation of sprite
        """
        # compute vector representation for sprite
        Kk_weight = self.motif_graph_weight(S)
        g = self.convert_motif2vec(Kk_weight)
        return g

    def motif_graph_weight(self, S):
        """
        Scan through sprite nodes and their neighborhoods to get a weighted motif graph
        Input:
            S: sprite graph
        Output: 
            Kk_weight: weighted motif graph
        """
        Kk_weight = self.Kk_unweight.copy()
        # Iterate through nodes in natural ordering
        for node in S.nodes:
            node_motif = S.nodes[node]['emb']
            subgraph = nx.ego_graph(S, node, radius=self.r) 
            neighbors = list(subgraph.nodes())
            nhbr_motifs = [S.nodes[n]['emb'] for n in neighbors]
            unique, counts = np.unique(nhbr_motifs, return_counts=True)
            
            # load node presence in weighted graph
            Kk_weight.nodes[node_motif]['n_weight'] += 1
            # load in edge presence in weighted graph (skipgrams)
            for idx, u in enumerate(unique):
                Kk_weight.edges[(node_motif,u)]['e_weight'] += counts[idx]
        return Kk_weight

    def convert_motif2vec(self, Kk):
        """
        Converts motif graph to datum vector.
        Inputs:
            Kk: networkx motif graph
        """
        vec = {}
        for node in Kk.nodes:
            vec[node] = int(Kk.nodes[node]['n_weight'])
        for edge in Kk.edges:
            vec[edge] = int(Kk.edges[edge]['e_weight'])

        vec = np.array([vec[key] for key in self.hash_keys])
        return vec

    def visualize_motif_graph(self, G=None, labels=False):
        """
        Inputs:
            G: networkx map graph
        """
        if G is None:
            print("No G provided, showing model-wide kernel hash-graph")
            G = self.w_hgraph
            logged = lambda x: np.log2(x)
            
            print("Displaying motif graph with log2 scaling")
        else:
            # get sample-specific motif graph from map graph 
            S = self.construct_sprite(G)
            G = self.motif_graph_weight(S)
            logged = lambda x: np.log10(x)
            print("Displaying motif graph with log10 scaling")

        pos = nx.circular_layout(G)
        colors = [node for node in list(G.nodes())]
        plt.figure()

        n_weights = nx.get_node_attributes(G, 'n_weight').values()
        n_size = []
        for nw in n_weights:
            ns = int(np.max([1, nw]))
            # ns = int(np.min([ns, 10])) # cap thickness to 10
            n_size.append(logged(ns))
        nx.draw_networkx_nodes(G, pos=pos, linewidths=n_size, node_color=colors, cmap=CMAP, edgecolors='black')
        if labels:
            nx.draw_networkx_labels(G, pos)

        e_weights = list(nx.get_edge_attributes(G, 'e_weight').values())
        e_sign = np.sign(list(nx.get_edge_attributes(self.w_hgraph, 'e_weight').values()))
        e_cmap = {-1.: "blue", 1.: "red", 0.: "black"}
        e_colors = [e_cmap[sign] for sign in e_sign]
        max_wt = np.max(e_weights)
        # print(nx.get_edge_attributes(G, 'e_weight'))
        e_thickness = []
        for ew in e_weights:
            # et = int(np.max([1, ew]))
            # et = int(np.min([et, 10])) # cap thickness to 10
            # e_thickness.append(logged(et))
            e_thickness.append(ew / max_wt)
        nx.draw_networkx_edges(G, pos=pos, width=e_thickness, alpha=0.5, edge_color=e_colors)
        plt.draw()
    
    def sort_keys(self, key_list):
        """
        Helper function to take a list of keys (either single nodes or edges) and sort. 
        For edges, we sort by 0th element and then1st element in tuple.
        """
        # collect all nodes and edges
        nodes, edges = [], []
        for k in key_list:
            if type(k) == int:
                nodes.append(k)
            elif type(k) == tuple:
                edges.append(k)
        # sort nodes and edges, combine
        nodes.sort()
        edges.sort(key=lambda tup: (tup[0],tup[1]))
        sorted = nodes + edges
        return sorted
    
    def visualize_prospect_graph(self, P):
        """
        Inputs:
            P: prospect graph
        """
        utils.visualize_sprite(P, self.modality, prospect_flag=True)
    
    def visualize_prospect_map(self, P):
        prospect_map = utils.convert_graph2arr(P)
        utils.visualize_quantizedZ(prospect_map, prospect_flag=True)
    
    def construct_sprite(self, G):
        """
        Takes a Map Graph G and constructs a sprite from it by applying an embedding quantizer
        AKA: "embedding quantization" for sprite construction
        """
        G = utils.construct_sprite(G, self.processor)
        return G

    def visualize_sprite(self, G):
        utils.visualize_sprite(G, self.modality)

