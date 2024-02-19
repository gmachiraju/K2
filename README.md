# Prospector Heads
![Prospector_pipeline](assets/prospect.png "Prospector head")

## Important links:  
- [arxiv](coming soon!)
- [xeets](coming soon!)   
- [blog](coming soon!)

## Intro
Welcome to the K2 repo! Here, we share our implementation of prospector heads (aka "prospectors"). This repo is nicknamed "K2" for readability purposes and references a key operator in prospector module called \***k2conv**\* (see our arXiv paper above!). As a work in progress, this repo will continue to get weekly updates.  

At a high level, prospectors offer feature attribution capabilties to virtually any encoder while being:
- Computationally efficient (sub-quadratic)
- Data-efficient (through parameter efficiency)
- Performant at feature localization
- Modality-generalizable
- Interpretable... and more!

## Usage
The core functionality of prospectors follows a similar API to `scikit-learn` package. As detailed in our arXiv preprint, prospectors contain two trainable layers, which we train sequentially in this implementation. Prospectors can be used with 3 simple steps:

1. Convert data to `networkx` graph objects, where each graph node is loaded with a token embedding (e.g. see `Doc-Step1-Embed.ipynb` for text encoders). Note: connectivity and resolution is defined by the user.
2. Construct a `K2Processor` object and then fit layer (I)'s quantizer via `.fit_quantizer()` command (e.g. see `Histo-Step2-VizSetup.ipynb`). Note: this can assume a random sample of token embeddings
3. Construct a `K2Model` object and then fit layer (II)'s convolutional kernel via the `.create_train_array()` and `.fit_kernel()` commands (e.g. see `Doc-Step2-VizSetup.ipynb`)

The IPython notebooks herein also give examples on how to visualize:
- Data sprites (false color representations of data colored by concepts)
- Concept monogram and skip-bigram frequencies (per datum) as fully connected graphs
- Prospector convolutional kernels as fully connected graphs
- Prospect maps as outputs for feature attribution

## Dependencies
Prospectors' dependecies are very light, only requiring the following popular/maintained packages:
- `os`
- `numpy`
- `pandas`
- `networkx`
- `scikit-learn`
- `pickle`
- `dill`
This work was originally implemented in Python 3.10. 