## Kannada MNIST Kaggle Challenge

This repository contains code and data for the Kaggle Kannada MNIST Challenge.

To reproduce the results below, clone the repo, create the environment using `conda env create -f py39-torch.yml`

Configs for models based on three different architectures are in the config folder - a CNN, a ResNet and a model based on Inception blocks.

The model that gave my best score (98.3) on the Kaggle leaderboard was the InceptionNet, to reproduce this run python `main.py -c configs/Inception_config.py`