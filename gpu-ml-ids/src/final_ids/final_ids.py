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
from sklearn.model_selection import train_test_split  # splits data into train and test sets
from sklearn.preprocessing import StandardScaler  # normalizes features to mean=0, std=1


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
        

if __name__ == "__main__":
    # Example usage of the map_cicids_label function
    print(f"CUDA ready: {torch.cuda.is_available()} | Dataset folder exists: {CICIDS_DIR.exists()}")
    print("Sample label check -> 'DDoS' maps to class ID:", map_cicids_label("DDoS"))

def load_csv_chunks(data_dir: Path):
    """Stream-load all CICIDS2017 CSVs into clean, compact float32 NumPy chunks."""
    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"[INFO] Found {len(csv_files)} CSV files in {data_dir}")
    X_chunks, y_chunks = [], []
    feature_cols = None

    for fpath in csv_files:
        print(f"  Loading & cleaning {fpath.name} ...")
        df = pd.read_csv(fpath, encoding="latin1", low_memory=True)
        df.columns = df.columns.str.strip()  # remove leading spaces in headers

        label_col = next(c for c in df.columns if c.lower() == "label")
        y_chunk = df[label_col].map(map_cicids_label).to_numpy(dtype=np.int64)
        df.drop(columns=[label_col], inplace=True)

        # Lock the exact same numeric feature columns across all 8 files
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Convert to float32 immediately (uses half the RAM of float64)
        X_chunk = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        del df

        # Keep only rows with finite numbers (drops NaN and +/- Infinity)
        valid_mask = np.isfinite(X_chunk).all(axis=1)
        X_chunks.append(X_chunk[valid_mask])
        y_chunks.append(y_chunk[valid_mask])
        del X_chunk, y_chunk
        gc.collect()

    return X_chunks, y_chunks


def prepare_datasets(data_dir: Path):
    """Combine cleaned CSV chunks, run stratified 80/20 split, scale, and return PyTorch tensors."""
    X_chunks, y_chunks = load_csv_chunks(data_dir)

    # Stack all 8 daily chunks into one contiguous float32 array
    X_all = np.vstack(X_chunks)
    y_all = np.concatenate(y_chunks)
    del X_chunks, y_chunks
    gc.collect()

    print(f"[INFO] Combined cleaned shape: {X_all.shape} (~{X_all.nbytes / 1e6:.0f} MB in float32)")

    # Stratified 80/20 split keeps rare attack ratios identical in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
    )
    del X_all, y_all
    gc.collect()

    # Fit scaler ONLY on training data to prevent data leakage into the test set
    scaler = StandardScaler(copy=False)
    X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test = scaler.transform(X_test).astype(np.float32, copy=False)

    print(f"[INFO] Split ready -> Train: {X_train.shape} | Test: {X_test.shape}\n")
    return (
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )