from typing import Union, List
import transformers
from transformers import pipeline
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, ZeroShotClassificationPipeline
import shap
from lime.lime_text import LimeTextExplainer
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy
from utils import process_sentences
from utils import serialize, deserialize
from utils import convert_text2graph
import random
import networkx as nx

import pdb

from time import process_time 
from time import perf_counter
import json
import os

device = torch.device("cuda")

hypothesis = "this sentence is about genetics"
labels_bin = ["genetics", "other"]
labels = ['genetics', 'other', 'classification', 'treatment', 'symptom', 'screening', 'prognosis', 'tomography', 'mechanism', 'pathophysiology', 'epidemiology', 'geography', 'medication', 'fauna', 'surgery', 'prevention', 'infection', 'culture', 'research', 'history', 'risk', 'cause', 'complication', 'pathology', 'management', 'diagnosis', 'etymology']
weights = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# hypothesis = "this sentence is about politics"
# labels = ["politics", "other", "ice cream", "cars", "world", "nature", "cats", "genetics"]
# labels_bin = ["politics", "other"]
# weights = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Adapted from: https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
def deberta_zsc(sentence, mode="bin"):
    classifier = pipeline("zero-shot-classification", model=weights, device=device)
    if mode == "bin":
        outs = classifier(sentence, labels_bin)
    elif mode == "many":
        outs = classifier(sentence, labels)
    return outs["scores"][0]

#------------------------------------------------------
# shared function
def deberta_attn_NLI(sentence, viz_flag=False):
    tokenizer = AutoTokenizer.from_pretrained(weights)
    model = AutoModelForSequenceClassification.from_pretrained(weights, output_attentions=True)
    model.to(device)

    input = tokenizer(sentence, hypothesis, truncation=True, return_tensors="pt")
    output = model(input["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: float(pred) for pred, name in zip(prediction, label_names)}
    
    tokens = tokenizer.convert_ids_to_tokens(input["input_ids"][0])  # Convert input ids to token strings
    A = output["attentions"][-1]
    maxA = torch.max(A,dim=1)[0].squeeze().cpu().detach().numpy()

    # now get the upper right block using hypothesis indices
    h_start = tokens.index("[SEP]") + 1
    h_end = -1
    evidence = maxA[1:h_start-1, h_start:h_end]
    evidence_score = np.max(evidence)

    if viz_flag == True:
        new_maxA = np.copy(maxA)
        new_maxA[1:h_start-1, h_start:h_end] = 1
        plt.figure()
        plt.imshow(new_maxA)
        plt.xticks(list(range(len(tokens))), tokens, rotation=90)
        plt.yticks(list(range(len(tokens))), tokens)
        plt.colorbar()
        plt.show()
    return evidence_score, prediction["entailment"]


def deberta_NLI(sentence):
    tokenizer = AutoTokenizer.from_pretrained(weights)
    model = AutoModelForSequenceClassification.from_pretrained(weights, output_attentions=True)
    model.to(device)

    input = tokenizer(sentence, hypothesis, truncation=True, return_tensors="pt")
    output = model(input["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: float(pred) for pred, name in zip(prediction, label_names)}
    return prediction["entailment"]

def deberta_attn(sentence, viz_flag=False):
    tokenizer = AutoTokenizer.from_pretrained(weights)
    model = AutoModelForSequenceClassification.from_pretrained(weights, output_attentions=True)
    model.to(device)

    input = tokenizer(sentence, hypothesis, truncation=True, return_tensors="pt")
    output = model(input["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}

    tokens = tokenizer.convert_ids_to_tokens(input["input_ids"][0])  # Convert input ids to token strings
    A = output["attentions"][-1]
    maxA = torch.max(A,dim=1)[0].squeeze().cpu().detach().numpy()

    # now get the upper right block using hypothesis indices
    h_start = tokens.index("[SEP]") + 1
    h_end = -1
    evidence = maxA[1:h_start-1, h_start:h_end]
    evidence_score = np.max(evidence)

    if viz_flag == True:
        new_maxA = np.copy(maxA)
        new_maxA[1:h_start-1, h_start:h_end] = 1
        plt.figure()
        plt.imshow(new_maxA)
        plt.xticks(list(range(len(tokens))), tokens, rotation=90)
        plt.yticks(list(range(len(tokens))), tokens)
        plt.colorbar()
        plt.show()

    return evidence_score
#----------------------------------------------



# Too slow
#----------

# Adpted from: https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/text_entailment/Textual%20Entailment%20Explanation%20Demo.html
# wrapper function for NLI model for SHAP
# takes in masked string which is in the form: premise <separator token(s)> hypothesis
def shap_wrapper_nli(x):
    model = AutoModelForSequenceClassification.from_pretrained(weights)
    tokenizer = AutoTokenizer.from_pretrained(weights)
    outputs = []
    for _x in x:
        encoding = torch.tensor([tokenizer.encode(_x)])
        output = model(encoding)[0].detach().cpu().numpy()
        outputs.append(output[0])
    outputs = np.array(outputs)
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = scipy.special.logit(scores)
    return val

# Adpted from: https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/text_entailment/Textual%20Entailment%20Explanation%20Demo.html
def deberta_shapNLI(sentence):
    model = AutoModelForSequenceClassification.from_pretrained(weights)
    tokenizer = AutoTokenizer.from_pretrained(weights)

    nli_labels = ["contradiction", "neutral", "entailment"]
    # explainer = shap.Explainer(shap_wrapper_nli, tokenizer, output_names=nli_labels)
    explainer = shap.PartitionExplainer(shap_wrapper_nli, tokenizer, masker=shap.maskers.Text, output_names=nli_labels)
    # ignore the start and end tokens, since tokenizer will naturally add them    
    encoded = tokenizer(sentence, hypothesis)["input_ids"][1:-1]  
    decoded = tokenizer.decode(encoded)
    shap_values = explainer([decoded])  # wrap input in list
    print(shap_values)
    return


# adapted from: https://stackoverflow.com/questions/69628487/how-to-get-shap-values-for-huggingface-transformer-model-prediction-zero-shot-c
# Create your own pipeline that only requires the text parameter 
# for the __call__ method and provides a method to set the labels
class MyZeroShotClassificationPipeline(ZeroShotClassificationPipeline):
    # Overwrite the __call__ method
    def __call__(self, *args):
      o = super().__call__(args[0], self.workaround_labels)[0]
      return [[{"label":x[0], "score": x[1]}  for x in zip(o["labels"], o["scores"])]]

    def set_labels_workaround(self, labels: Union[str,List[str]]):
      self.workaround_labels = labels

# adapted from: https://stackoverflow.com/questions/69628487/how-to-get-shap-values-for-huggingface-transformer-model-prediction-zero-shot-c
def deberta_shapZSC(sentence, viz_flag=False):
    model = AutoModelForSequenceClassification.from_pretrained(weights)
    tokenizer = AutoTokenizer.from_pretrained(weights)
    
    # only supports binary clf bc otherwise too costly
    labs = labels_bin

    # In the following, we address issue 2.
    model.config.label2id.update({v:k for k,v in enumerate(labs)})
    model.config.id2label.update({k:v for k,v in enumerate(labs)})

    pipe = MyZeroShotClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=device)
    pipe.set_labels_workaround(labs)
    
    # explainer = shap.Explainer(pipe)
    # masker=shap.maskers.Text
    explainer = shap.Explainer(pipe)
    if len(sentence) == 1:
        shap_values = explainer(sentence)
    else:
        shap_values = explainer(sentence)
    # word_scores = shap_grid[:, 0] # target lab is in 0th column
    
    # batching enabled
    if len(shap_values) == 1:
        shap_grid = shap_values.values.squeeze()
        mav = np.mean(np.abs(shap_grid), 1) # mean abs value
        mav = mav[1:-1] # cut first and last tokens
        ret = [[np.max(mav)]] #used to be np.max(word_scores)
    else:
        ret = []
        for sv in shap_values:
            shap_grid = sv.values.squeeze()
            mav = np.mean(np.abs(shap_grid), 1)
            mav = mav[1:-1]
            ret.append([np.max(mav)])
    
    if viz_flag == True:
        print(shap_values)
        plt.figure()
        plt.imshow(shap_grid)
        tokens = list(shap_values.data[0])
        plt.yticks(list(range(len(tokens))), tokens)
        plt.show()

    return ret 


def deberta_shapZSC_fullcontext(sentence, viz_flag=False):
    model = AutoModelForSequenceClassification.from_pretrained(weights)
    tokenizer = AutoTokenizer.from_pretrained(weights)
    
    # only supports binary clf bc otherwise too costly
    labs = labels_bin

    # In the following, we address issue 2.
    model.config.label2id.update({v:k for k,v in enumerate(labs)})
    model.config.id2label.update({k:v for k,v in enumerate(labs)})

    pipe = MyZeroShotClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=device)
    pipe.set_labels_workaround(labs)
    
    # explainer = shap.Explainer(pipe)
    # masker=shap.maskers.Text
    explainer = shap.Explainer(pipe)
    if len(sentence) == 1:
        shap_values = explainer(sentence)
    else:
        shap_values = explainer(sentence)
    # word_scores = shap_grid[:, 0] # target lab is in 0th column
    
    ret = shap_values.values.squeeze()
    
    if viz_flag == True:
        print(shap_values)
        plt.figure()
        plt.imshow(shap_grid)
        tokens = list(shap_values.data[0])
        plt.yticks(list(range(len(tokens))), tokens)
        plt.show()

    return ret 




class MyZeroShotClassificationPipeline2(ZeroShotClassificationPipeline):
    # Overwrite the __call__ method
    def __call__(self, *args):
      o = super().__call__(args[0], self.workaround_labels)[0]
      pdb.set_trace()
      return np.array([o["scores"]])

    def set_labels_workaround(self, labels: Union[str,List[str]]):
      self.workaround_labels = labels

# adapted from: https://stackoverflow.com/questions/69628487/how-to-get-shap-values-for-huggingface-transformer-model-prediction-zero-shot-c
def deberta_limeZSC(sentence, viz_flag=False):
    model = AutoModelForSequenceClassification.from_pretrained(weights)
    tokenizer = AutoTokenizer.from_pretrained(weights)
    
    # only supports binary clf bc otherwise too costly
    labs = labels_bin

    # In the following, we address issue 2.
    model.config.label2id.update({v:k for k,v in enumerate(labs)})
    model.config.id2label.update({k:v for k,v in enumerate(labs)})

    pipe = MyZeroShotClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=device)
    pipe.set_labels_workaround(labs)

    explainer = LimeTextExplainer(class_names=["1","0"])
    if len(sentence) == 1:
        lime_values = explainer.explain_instance([sentence], pipe, num_features=10, num_samples=200)
    else:
        # print("Not parallelizable! Exiting...")
        lime_values = explainer.explain_instance(sentence, pipe, num_features=10, num_samples=200)
    # word_scores = shap_grid[:, 0] # target lab is in 0th column
    pdb.set_trace()
    print(lime_values.as_list())

    # batching enabled
    if len(shap_values) == 1:
        shap_grid = shap_values.values.squeeze()
        mav = np.mean(np.abs(shap_grid), 1) # mean abs value
        mav = mav[1:-1] # cut first and last tokens
        ret = [[np.max(mav)]] #used to be np.max(word_scores)
    else:
        ret = []
        for sv in shap_values:
            shap_grid = sv.values.squeeze()
            mav = np.mean(np.abs(shap_grid), 1)
            mav = mav[1:-1]
            ret.append([np.max(mav)])
    
    if viz_flag == True:
        print(shap_values)
        plt.figure()
        plt.imshow(shap_grid)
        tokens = list(shap_values.data[0])
        plt.yticks(list(range(len(tokens))), tokens)
        plt.show()

    return ret 

# adapted from notebook: Doc-shap.ipynb
def chunker(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def process_Zs_scores_idxs(test_data, idxs, score_fn, save_path_Gs, batch_size=1, kill_after=1e10):
    dt_cpu_data, dt_cpu_batches, dt_cpu_tokens = [], [], []
    dt_wall_data, dt_wall_batches, dt_wall_tokens = [], [], []
    finished = 0
    ran_idxs = []
    
    for i in range(len(test_data)):
        if finished == kill_after:
            return ran_idxs, dt_cpu_tokens, dt_wall_tokens, dt_cpu_batches, dt_wall_batches, dt_cpu_data, dt_wall_data

        if i not in idxs:
            continue
        print("="*100)
        print("Working on index:", i)
        print("="*100)

        G_file = "doc_"+str(i) + "_graph.obj"
        save_path_G = os.path.join(save_path_Gs, G_file)
        if os.path.isfile(save_path_G) == True:
            print("skipping sample bc already created:", G_file)
            continue

        text = test_data[i]["text"]
        annots = test_data[i]["annotations"]

        scores = []
        doc_sents = []
        for annot in annots:
            begin = int(annot["begin"])
            idx = [begin, begin+int(annot["length"])]
            chunk = text[idx[0]:idx[1]]
            sents = [s for s in chunk.split(".")]
            sents = process_sentences(sents)
            doc_sents.extend(sents)

        #===============================
        start_cpu_datum = process_time()
        start_wall_datum = perf_counter()
        dt_cpu_batch, dt_cpu_token = [], []
        dt_wall_batch, dt_wall_token = [], []
        for batch in chunker(doc_sents, batch_size):
            #========================
            start_cpu_batch = process_time()  
            start_wall_batch = perf_counter()
            es = score_fn(batch)
            stop_cpu_batch = process_time() 
            stop_wall_batch = perf_counter()
            #========================
            tdb_cpu = float(stop_cpu_batch - start_cpu_batch) # batch
            tdb_wall = float(stop_wall_batch - start_wall_batch) # batch
            print("dt cpu -> batch:", tdb_cpu)
            print("dt wall -> batch:", tdb_wall)
            dt_cpu_batch.append(tdb_cpu)
            dt_wall_batch.append(tdb_wall)

            tdt_cpu = float(tdb_cpu / batch_size) # token
            tdt_wall = float(tdb_wall / batch_size) # token
            print("dt cpu -> token:", tdt_cpu)
            print("dt wall -> token:", tdt_wall)
            dt_cpu_token.append(tdt_cpu)
            dt_wall_token.append(tdt_wall)

            scores.extend(es)

        stop_cpu_datum = process_time()  
        stop_wall_datum = perf_counter() 
        #==============================
        tdd_cpu = float(stop_cpu_datum - start_cpu_datum) # datum
        tdd_wall = float(stop_wall_datum - start_wall_datum) # datum
        print("dt cpu -> datum:", tdd_cpu)
        print("dt wall -> datum:", tdd_wall)
        dt_cpu_data.append(tdd_cpu)
        dt_wall_data.append(tdd_wall)
        
        dt_cpu_tokens.append(dt_cpu_token)
        dt_wall_tokens.append(dt_wall_token)
        dt_cpu_batches.append(dt_cpu_batch)
        dt_wall_batches.append(dt_wall_batch)
        
        # save
        Z = np.array(scores)
        G = convert_text2graph(Z) # save as graph
        serialize(G, save_path_G)
        finished += 1
        print("finished:", finished)
        ran_idxs.append(i)

    return ran_idxs, dt_cpu_tokens, dt_wall_tokens, dt_cpu_batches, dt_wall_batches, dt_cpu_data, dt_wall_data


# running shap ZSC
def main():

    test = "/dfs/scratch1/gmachi/datasets/WikiSection/wikisection_en_disease_test.json"
    with open(test, 'r') as f:
        test_data = json.load(f)
    save_path = "/dfs/scratch1/gmachi/datasets/wikisection_processed/"
    save_path_shap = os.path.join(save_path, "shap_Gs")

    # change per node
    # idxs = list(range(21,200))
    # idxs = list(range(201,376))
    # idxs = list(range(377,400))
    # idxs = list(range(401,500))
    # idxs = list(range(203,220))
    # idxs = list(range(451,501))
    idxs = list(range(701,718))

    outs = process_Zs_scores_idxs(test_data, idxs, deberta_shapZSC, save_path_shap)
    save_path = "/dfs/scratch1/gmachi/k2/K2/src/outputs/shapley_wallclock"
    filename = '-'.join(str(x) for x in outs[0]) + "_wallclock.obj"
    save_path = os.path.join(save_path, filename)
    serialize(outs, save_path)

if __name__=='__main__':
    main()