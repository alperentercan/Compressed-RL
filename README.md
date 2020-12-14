# Compressed-RL

This repository contains the necessary file to combine tf-agents, tfmot, and keras.

# Important Files

* `layers.py` contains `HashedNet` and `EfficientHashedNet` implementations.
* `distillation.py` contains my Policy Distillation implementation.
* `tfagents_wrappers.py` contains a QNetWrapper that can be used to do use a custom network as Q-Network in TF-agents.

# References
* `train_utils.py` and `compression_utils.py` uses some code from TF-agents and TF Model Optimization library tutorials.
* `train_scripts.py` is loosely adapted from TF-agents tutorial.
* `HashedNet` implementation is based on this [PyTorch Implementation](https://github.com/jfainberg/hashed_net)
