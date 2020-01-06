# Code

This folder contains the code for representation learning in reinforcement learning (RL) environments.
The parameters can be found in the `config.py` file.
The `bash` scripts can be used to run the program locally or remotely via 
[slurm](https://slurm.schedmd.com/documentation.html):

+ `_run` to run the algorithm with current configurations.
+ `_play` to activate play mode for displaying the performance in real time without learning for the current 
configurations.
+ `_play_learn` to activate the play mode with learning.
+ `_run_remote` to run the algorithm with `slurm`. 

# Utilities

In the following we briefly explain the utility of the various subdirectories.

## analysis

This directory contains the main tool for analyzing the results and models.
Here, we generated some of the figures used in the `README` and in the paper.

## data

This directory contains the training data used for prediction in `selection` mode.

## load_models

This directory contains the trained models. We use this directory to load models
from.

## networks

This directory contains the `pytorch` modules used in this project.

## results_log

This directory contains the results files which are updated during training.

## saved_models

This directory contains trained models. We use this directory to save models 
during training.
