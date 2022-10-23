## Kannada MNIST Kaggle Challenge

This repository contains code and data for the Kaggle Kannada MNIST Challenge.

To reproduce the results below, clone the repo, create the environment using `conda env create -f py39-torch.yml`

Configs for models based on three different architectures are in the config folder - a CNN, a ResNet and a model based on Inception blocks.

The model that gave my best score (98.4) was the InceptionNet, to reproduce this run python `main.py -c configs/Inception_config.py`

Default will train for 5 epochs, use the `-n` flag to set the number of epochs.

Training should be compatible with GPU, I don't have one on my laptop so copied it to a Kaggle notebook with GPU support. 

Download the data from the Kaggle Kannada MNIST competitions page and store it as follows: `data -> train -> train.csv`, `data -> test -> test.csv` and `data -> hard_test -> Dig-MNIST.csv`