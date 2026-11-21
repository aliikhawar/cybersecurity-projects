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

def map_cicids_label(raw_label: str) -> int :
    """" Convert CICIDS2017 raw labels to integer indices. """
    lbl = str(raw_label).strip().lower() # convert the raw label to lowercase and remove whitespace
    if lbl == "benign":
        return CLASS_TO_IDX["BENIGN"]
    if "dos" in lbl or "ddos" in lbl or "heartbleed" in lbl:
        return CLASS_TO_IDX["DoS/DDoS"]
    if "portscan" in lbl:
        return CLASS_TO_IDX["PortScan"]
    if "patator" in lbl:
        return CLASS_TO_IDX["BruteFroce"]
    if "web attack" in lbl:
        return CLASS_TO_IDX["Web Attack"]
    if "bot" in lbl:
        return CLASS_TO_IDX["Botnet"]
    if "infiltration" in lbl:
        return CLASS_TO_IDX["Infiltration"]
    return CLASS_TO_IDX["BENIGN"]  # default to BENIGN if label is unrecognized
        return CLASS_TO_IDX["Dos/DDoS"]

if __name__ == "__main__":
    # Example usage of the map_cicids_label function
    print(f"CUDA ready: {torch.cuda.is_available()} | Dataset folder exists: {CICIDS_DIR.exists()}")
    print("Sample label check -> 'DDoS' maps to class ID:", map_cicids_label("DDoS"))