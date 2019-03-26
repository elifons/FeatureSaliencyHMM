# FeatureSaliency Hidden Markov Model

This repository presents a partial implementation of the feature saliency HMM algorithm as proposed by Adams et al in the paper [Feature Selection for Hidden Markov Models and Hidden Semi-Markov Models](https://ieeexplore.ieee.org/document/7450620).  
The implementation is a modification of the GaussianHMM class of [hmmlearn](https://hmmlearn.readthedocs.io/).

### Requirements:

Install dependencies via pip install -r requirements.txt

* hmmlearn==0.2.0
* scikit-learn==0.19.1

### Quick Start:
The notebook FSHMM_example.ipynb has a short example on how to use the library and shows a simple test case.

### Resources:
* [Feature Selection for Hidden Markov Models and Hidden Semi-Markov Models](https://ieeexplore.ieee.org/document/7450620): Original paper where FSHMM is presented. 
* [A novel dynamic asset allocation system using Feature Saliency Hidden Markov models for smart beta investing](https://arxiv.org/abs/1902.10849): Paper that uses FSHMM for regime identification in financial markets. 
* [hmmlearn](https://hmmlearn.readthedocs.io/): python library for HMMs.
* [Simultaneous Feature Selection and Parameter Estimation for Hidden Markov Models](https://libraetd.lib.virginia.edu/public_view/x059c7639): PhD thesis with detailed derivation of the algorithm.

