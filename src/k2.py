import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import seaborn as sns

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

# deals with more than 20 colors
CMAP2 = "Pastel2"
joint_cmap = colors.ListedColormap(cm.tab20.colors + cm.Pastel2.colors, name='tab40')
joint_cmap.set_bad(color='white')


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
        self.fitted_flag = False
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

        if self.embeddings_path is None:
            print("No embeddings path provided, sampling from dataset")
            if self.dataset_path is None:
                raise Exception("Error: dataset path is not provided.")
            # determine sample size PER datum based on: sample_size/dataset size
            N = len(os.listdir(self.dataset_path)) 
            if N < self.sample_size:
                self.to_sample = self.sample_size // N 
            else:
                self.to_sample = 1 # and then take sample of that
        
            # run sampling
            self.embeddings_path = self.sample_embeddings()
            self.description = self.quantizer_type + "-" + str(self.k) + "-" + self.sample_scheme + "-" + str(self.sample_size)
        else:
            print("Embeddings path provided, loading embeddings...")
            self.description = self.quantizer_type + "-" + str(self.k)

        self.motif_graph = self.instantiate_motif_graph() # K_k
        self.quantizer = None

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
            model = KMeans(n_clusters=self.k, random_state=0, n_init=10) # 10 is default
        else:
            raise NotImplementedError
        
        self.embedding_array = self.load_embeddings()
        self.quantizer = model.fit(self.embedding_array)
        self.fitted_flag = True

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
        elif self.embeddings_type == "multidict":
            embed_dict = utils.deserialize(self.embeddings_path)
            id_list = list(embed_dict.keys())
            mapping_dict = utils.deserialize(self.mapping_path)
        elif self.embeddings_type == "memmap":
            embed_dict = np.memmap(self.embeddings_path, dtype='float32', mode='r', shape=(self.sample_size,1024)) # not an actual dict as you can see
            mapping_dict = utils.deserialize(self.mapping_path)
            id_list = mapping_dict.keys()

        self.embed_dict = embed_dict
        self.mapping_dict = mapping_dict
        self.id_list = id_list
        array = self.partition_data()
        # reassign embed_dict bc not needed anymore
        self.embed_dict = None # causes pickling issues with memmap
        return array

    def partition_data(self):
        """
        This helper function helps us partition data into classes (and within class-1, identify salient objects).
        Also helps for visualization downstream.
        """
        # create df and partition by marker
        if self.datatype == 'histo':
            ms = self.get_plot_markers()
        elif self.datatype == "text":
            ms = self.get_text_labels()
        elif self.datatype == 'protein':
            ms = self.get_graph_labels()
        
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

    def get_text_labels(self):
        """
        For use with text data. 
        Returns:
            ms: list of markers
        """
        sal_counter = 0
        so_dict = self.so_dict_path
        if so_dict is not {}:
            so_dict = utils.deserialize(so_dict) 
        mapping_dict = self.mapping_dict # ID to doc label

        ms = []
        for id_val in self.id_list:
            doc_id = int(id_val.split("_")[0])
            doc_lab = int(mapping_dict[doc_id] > 0) # converting counts of target to 0,1
            el = int(so_dict[id_val])
            if (doc_lab == 1) and (el == 1):
                m = "X"
                sal_counter += 1
            elif (doc_lab == 1) and (el == 0):
                m = "x"
            elif (doc_lab == 0) and (el == 0):
                m = "o"
            # print(doc_lab, el)
            ms.append(m)
        print("sampled", str(sal_counter), "known salient objects!")
        return ms

    def get_plot_markers(self):
        """
        Gives us a way to visualize different chunks -- only used for histopath
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
    
    def get_graph_labels(self):
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
            if self.embeddings_type in ["dict", "multidict"]:
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
            if not subsample:
                embedded_o = tsne[0:self.idx_o,:]
                embedded_x = tsne[self.idx_o:self.idx_x,:]
                if self.idx_X == 0:
                    embedded_X = tsne[self.idx_x:,:]
                else:
                    embedded_X = tsne[self.idx_x:self.idx_X,:]
                labels_o = cluster_labs[0:self.idx_o]
                labels_x = cluster_labs[self.idx_o:self.idx_x]
                if self.idx_X == 0:
                    labels_X = cluster_labs[self.idx_x:]
                else:
                    labels_X = cluster_labs[self.idx_x:self.idx_X]
            
            if self.verbosity == "full":
                fig, ax1 = plt.subplots(1, 1)
                fig.suptitle('Sampled embeddings for cluster assignment')
                ax1.set_xlabel("tSNE-0")
                ax1.set_ylabel("tSNE-1")
                ax1.set_title("t-SNE with K="+str(self.k)+" clusters (perplexity="+str(perplexity)+")")
                
                ax1.scatter(tsne[:len(embedded_o),0], tsne[:len(embedded_o),1], c=labels_o, alpha=0.3, s=5, marker="o", cmap=joint_cmap) # used to be Dark2
                ax1.scatter(tsne[len(embedded_o):len(embedded_o)+len(embedded_x),0], tsne[len(embedded_o):len(embedded_o)+len(embedded_x),1], c=labels_x, alpha=0.3, s=30, marker="x", cmap=joint_cmap)
                ax1.scatter(tsne[len(embedded_o)+len(embedded_x):,0], tsne[len(embedded_o)+len(embedded_x):,1], c=labels_X, alpha=0.6, s=300, edgecolors="k", marker="X", cmap=joint_cmap) 
                
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
        self.fitted_flag = False
        self.modality = args.modality # graph, image 
        self.processor = args.processor
        self.r = args.r
        self.variant = args.variant
        self.hparams = args.hparams
        self.train_graph_path = args.train_graph_path
        # add verbosity flag!

        if "verbosity_flag" not in args.keys():
            self.verbosity_flag = "low"

        # allows for labels to be stored in Graphs or loaded as a dictionary
        if "train_label_dict" in args.keys():
            self.train_label_dict = args.train_label_dict
        else:
            self.train_label_dict = None

        if self.processor == None or "processor" not in args.keys():
            raise Exception("Error: K2 Processor is not provided.")
        self.k = self.processor.k
        self.quantizer = self.processor.quantizer
        
        # now construct data structures
        self.Kk_unweight = self.processor.motif_graph.copy()
        self.Kk_pruning() # only runs if r=0
        self.description = self.variant + "-" + self.processor.quantizer_type + "-" + str(self.k) + "-" + str(self.hparams)
        self.class_graphs = [] # where we load mean graphs per class

    def Kk_pruning(self):
        if self.r == 0: # if r=0, then we need to prune to just nodes
            # prune edges, including self-edges
            for edge in self.Kk_unweight.edges:
                self.Kk_unweight.remove_edge(edge[0], edge[1])
            # print("Note: r=0, so pruned Kk to k nodes only")    
            # print(self.Kk_unweight)

        # create a sorted ordering for hashmap keys
        key_list = [node for node in self.Kk_unweight.nodes] + [edge for edge in self.Kk_unweight.edges]
        self.hash_keys = self.sort_keys(key_list)    

    def create_train_array(self):
        """
        Preprocesses map graphs as training vectors for K2
        """
        X,y = [],[]
        G_files = os.listdir(self.train_graph_path)
        # with tqdm(total=len(G_files), desc="Creating K2 training array...") as pbar:
        for t, G_file in enumerate(G_files):
            # load map graph data
            G = utils.deserialize(os.path.join(self.train_graph_path, G_file))
            if self.processor.quantizer_type == 'AA':
                G = utils.set_graph_emb(G, 'resid')
            # load in labels
            if self.train_label_dict is None:
                y.append(G.graph['label'])
            else:
                if self.modality == "text":
                    G_id = int(G_file.split("_")[1])
                    y.append(self.train_label_dict[G_id])
                else:
                    y.append(self.train_label_dict[G_file])

            # quantize and embed sprite
            sprite = self.construct_sprite(G)
            g = self.embed_sprite(sprite)
            # store g-vector in array for training
            X.append(g)
            # y.append(self.train_label_dict[G_file])
            # pbar.set_description('processed: %d' % (1 + t))
            # pbar.update(1)

        self.training_data = np.vstack(X)
        n,p = self.training_data.shape
        self.labels = np.array(y)
        if self.verbosity_flag == "full":
            print("Complete! Created a training array for few-shot classification...")
            print("Number of training examples:", n)
            print("Number of Kk features:", p)

    def fit_kernel(self, normalize_flag=True, alpha=None, tau=None, lam=None):
        """
        Main method for training K2 model
        Inputs:
            normalize_flag: a boolean T/F value to toggle TF-IDF normalization
            alpha: optionally override alpha hyperparameter attribute for fitting
            tau: optionally override tau hyperparameter attribute for fitting
        """
        # tfidf scaling
        if normalize_flag == True:
            if self.verbosity_flag == "full":
                print("Normalizing training data with TF-IDF...")
            self.tfidf()
        if alpha is not None:
            print(f'updating alpha to {alpha}')
            self.hparams['alpha'] = alpha
        if tau is not None:
            print(f'updating tau to {tau}')
            self.hparams['tau'] = tau
        if lam is not None:
            print(f'updating tau to {lam}')
            self.hparams['lambda'] = lam

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
            if self.verbosity_flag == "full":
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
        # print(np.min(log2fc), np.max(log2fc))
        # print(np.round(log2fc, 2))
        print(np.where(np.abs(log2fc) > tau))

        # compute p-values between classes
        pvals = []
        p = X0.shape[1]
        for idx in range(p):
            _, pval = scipy.stats.mannwhitneyu(X1[:,idx], X0[:,idx])
            pvals.append(pval * p) # bonferroni correction
        # print(np.max(pvals), np.min(pvals))
        # print(np.max(np.array(pvals)*p), np.min(np.array(pvals)*p))
        # pvals = scipy.stats.false_discovery_control(pvals, method='bh')
        
        # print(np.max(pvals), np.min(pvals))
        # print(np.round(pvals, 4))
        print(np.where(np.array(pvals) < alpha))
        
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
            # load node presence in weighted graph
            node_motif = S.nodes[node]['emb']
            P.nodes[node]['emb'] += self.w_hgraph.nodes[node_motif]["n_weight"]

            if self.r > 0:
                # load in edge presence in weighted graph (skipgrams)
                subgraph = nx.ego_graph(S, node, radius=self.r) 
                neighbors = list(subgraph.nodes())
                nhbr_motifs = [S.nodes[n]['emb'] for n in neighbors]
                unique, counts = np.unique(nhbr_motifs, return_counts=True) 
                # print(node_motif, self.w_hgraph.nodes[node_motif]["n_weight"])
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
        # Iterate through nodes in natural ordering -- or should we iterate by linear index? 
        # # K2 should be deterministic though
        for node in S.nodes:
            # load node presence in weighted graph
            node_motif = S.nodes[node]['emb']
            Kk_weight.nodes[node_motif]['n_weight'] += 1
            # load edge presence in weighted graph (skipgrams)
            if self.r > 0:
                subgraph = nx.ego_graph(S, node, radius=self.r) 
                neighbors = list(subgraph.nodes())
                nhbr_motifs = [S.nodes[n]['emb'] for n in neighbors]
                unique, counts = np.unique(nhbr_motifs, return_counts=True)
                for idx, u in enumerate(unique):
                    # try:
                    Kk_weight.edges[(node_motif,u)]['e_weight'] += counts[idx]
                    # except KeyError:
                    #     pdb.set_trace()
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

    def visualize_motif_graph(self, G=None, labels=False, style="graph"):
        """
        Inputs:
            G: networkx map graph
        """
        model_flag = False
        if G is None:
            print("No G provided, showing model-wide kernel hash-graph")
            G = self.w_hgraph
            logged = lambda x: np.log2(x)
            model_flag = True
            print("Displaying motif graph with log2 scaling")
        else:
            # get sample-specific motif graph from map graph 
            S = self.construct_sprite(G)
            G = self.motif_graph_weight(S)
            logged = lambda x: np.log10(x)
            print("Displaying motif graph with log10 scaling")

        if style == "graph":
            self.plot_graph_view(G, logged, labels, model_flag)
        elif style == "matrix":
            self.plot_matrix_view(G, logged, labels, model_flag)


    def plot_graph_view(self, G, logged, labels, model_flag):
        pos = nx.circular_layout(G)
        colors = [node for node in list(G.nodes())]
        # print(colors)
        plt.figure()

        n_weights = nx.get_node_attributes(G, 'n_weight').values()
        n_size = []
        for nw in n_weights:
            ns = int(np.max([1, nw]))
            # ns = int(np.min([ns, 10])) # cap thickness to 10
            n_size.append(logged(ns))
        
        # old command isn't explicit about color:
        # nx.draw_networkx_nodes(G, pos=pos, linewidths=n_size, node_color=colors, cmap=joint_cmap, edgecolors='black') #cmap used to be CMAP
        
        # new command is explicit with color order
        k = len(list(G.nodes))
        color_assignments = list(joint_cmap(range(k))) 
        nx.draw_networkx_nodes(G, pos=pos, linewidths=n_size, node_color=color_assignments, edgecolors='black') #cmap used to be CMAP
        if labels or (k > 20):
            nx.draw_networkx_labels(G, pos)

        e_weights = list(nx.get_edge_attributes(G, 'e_weight').values())
        e_sign = np.sign(list(nx.get_edge_attributes(self.w_hgraph, 'e_weight').values()))
        e_cmap = {-1.: "blue", 1.: "red", 0.: "black"}
        e_colors = [e_cmap[sign] for sign in e_sign]
        max_wt = np.max(e_weights)

        e_thickness = []
        for ew in e_weights:
            # et = int(np.max([1, ew]))
            # et = int(np.min([et, 10])) # cap thickness to 10
            # e_thickness.append(logged(et))
            e_thickness.append(ew / max_wt)
        if model_flag == False:
            e_colors = ["black" for el in e_colors] # keep bacl lines for datum motif graph
        nx.draw_networkx_edges(G, pos=pos, width=e_thickness, alpha=0.5, edge_color=e_colors)
        plt.axis('off')
        plt.draw()

    def plot_matrix_view(self, G, logged, labels, model_flag):
        """
        adapted from:
        - https://stackoverflow.com/questions/65810567/aligning-subplots-with-a-pyplot-barplot-and-seaborn-heatmap
        - https://stackoverflow.com/questions/33379261/how-can-i-have-a-bar-next-to-python-seaborn-heatmap-which-shows-the-summation-of 
        - https://stackoverflow.com/questions/40489821/how-to-write-text-above-the-bars-on-a-bar-plot-python
        """
        k = len(list(G.nodes))
        M = np.zeros((k,k))
        for i in range(k):
            for j in range(k):
                M[i,j] = G[i][j]["e_weight"]
        bars = pd.Series(list(nx.get_node_attributes(G, 'n_weight').values()))
        # print(bars)

        fig = plt.figure(figsize=(20,20))
        ax1 = plt.subplot2grid((20,20), (1,0), colspan=19, rowspan=19)
        ax2 = plt.subplot2grid((20,20), (0,0), colspan=19, rowspan=1)
        ax3 = plt.subplot2grid((20,20), (1,19), colspan=1, rowspan=19)

        mask = np.zeros_like(M)
        mask[np.tril_indices_from(mask,k=-1)] = True
        if model_flag == True:
            sns.heatmap(M, ax=ax1, annot=False, cmap="bwr", mask=mask, linecolor='b', cbar = False)
        else:
            sns.heatmap(M, ax=ax1, annot=False, cmap="binary", mask=mask, linecolor='b', cbar = False)
        ax1.xaxis.tick_bottom()
        
        # print(list(joint_cmap(range(k))))
        # pdb.set_trace()
        # sns.barplot(bars.transpose(), ax=ax2, palette=list(joint_cmap(range(k))))
        # sns.barplot(bars,             ax=ax3, palette=list(joint_cmap(range(k))))

        x_tick_pos = [i for i in range(k)]
        signs = np.sign(bars)
        
        bar_top = ax2.bar(x=x_tick_pos, height=np.abs(bars), color=list(joint_cmap(range(k))), align="center")
        for i,rect in enumerate(bar_top):
            h = rect.get_height()
            if h > 0:
                if signs[i] > 0 and model_flag == True:
                    ax2.text(x_tick_pos[i], h, f'{signs[i] * h:.2f}', color="red", ha='center', va='bottom')
                elif signs[0] < 0 and model_flag == True:
                    ax2.text(x_tick_pos[i], h, f'{signs[i] * h:.2f}', color="blue", ha='center', va='bottom')
                else:
                    ax2.text(x_tick_pos[i], h, f'{signs[i] * h:.2f}', color="black", ha='center', va='bottom')

        ax2.set_xticks(list(range(k)))
        ax2.set_xlim(x_tick_pos[0] - 0.5, x_tick_pos[-1] + 0.5)
        ax2.spines[['right', 'top']].set_visible(False)

        bar_right = ax3.barh(y=x_tick_pos, width=np.abs(bars), color=list(joint_cmap(range(k))), align="center")
        for i,rect in enumerate(bar_right):
            w = rect.get_width()
            h = rect.get_height()
            x,y = rect.get_x(), rect.get_y()
            if w > 0:
                if signs[i] > 0 and model_flag == True:
                    ax3.text(x+w, y+h/2, f'{signs[i] * w:.2f}', color="red", ha='left', va='center')
                elif signs[i] < 0 and model_flag == True:
                    ax3.text(x+w, y+h/2, f'{signs[i] * w:.2f}', color="blue", ha='left', va='center')
                else:
                    ax3.text(x+w, y+h/2, f'{signs[i] * w:.2f}', color="black", ha='left', va='center')


        ax3.set_yticks(list(range(k)))
        ax3.set_ylim(x_tick_pos[0] - 0.5, x_tick_pos[-1] + 0.5)
        ax3.invert_yaxis()
        ax3.spines[['right', 'top']].set_visible(False)

        plt.tight_layout()

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
    
    def visualize_prospect_graph(self, P, labels=False):
        """
        Inputs:
            P: prospect graph
        """
        utils.visualize_sprite(P, self.modality, prospect_flag=True, labels=labels)
    
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

