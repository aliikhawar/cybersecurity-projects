"""
                                                  GPU-Accelerated Deep Residual Intrusion Detection System (CICIDS2017)

                                                                      src/final_ids/final_ids.py

"""

from pathlib import Path    # for file and folder locations
import gc  # memory management
import time  # measuring time
import numpy as np  # for numerical calculations
import pandas as pd # for loading organizing and analyzing datasets
import torch # for deep learning and GPU acceleration
import torch.nn as nn # for building neural networks. it is a pytorch module containing all the building blocks for neural networks


# Listing folder paths from src/final_ids/final_ids.py to gpu-ml-ids/

PROJECT_DIR = Path(__file__).resolve().parents[2] # project directory
CICIDS_DIR = PROJECT_DIR / "data" / "CICIDS2017" # CICIDS2017 dataset directory
MODEL_DIR = PROJECT_DIR / "models" # models directory
DOCS_DIR = PROJECT_DIR / "docs" # documentation directory

# Training hyperparameters which control how Deep Residual Neural Networks learn from the CICIDS2017 dataset.
BATCH_SIZE = 4096 # number of samples the model processes sequentially 
EPOCHS = 12 # number to times the model goes through the entire training dataset
LEARNING_RATE = 1e-3 # controls the step the model takes when adjusting internal parameters during training. This number equals to 0.001
SEED = 42 # A random number to insert in parameters to ensure reproducibility of results.


CLASS_NAMES = [ "BENIGN", "Botnet" , "DoS/DDoS" , "Infiltration", "PortScan", "Web Attack" ] # class names for the CICIDS2017 dataset

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)} # mapping class names to indices
