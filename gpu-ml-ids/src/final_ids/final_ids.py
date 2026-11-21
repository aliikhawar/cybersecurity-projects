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
import matplotlib.pyplot as plt  # for generating GPU vs CPU speedup charts


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



class ResidualBlock(nn.module):
    """A single residual block for the deep residual neural network."""
    def __init__(self, width: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))  # skip connection adds input x to the block output   



class DeepResIDSNet(nn.Module):
    """~1.2M-parameter Deep Residual Network for 7-class CICIDS2017 intrusion detection."""

    def __init__(self, input_dim: int, num_classes: int, width: int = 512):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.res1 = ResidualBlock(width, dropout=0.2)
        self.res2 = ResidualBlock(width, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(width, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h)



def sync(device: torch.device):
    """Wait for all queued CUDA GPU kernels to finish before reading the timer."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_and_benchmark(
    device_name: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
):
    """Train and evaluate DeepResIDSNet on 'cuda' or 'cpu' with exact hardware timing."""
    torch.manual_seed(SEED)
    device = torch.device(device_name)
    num_samples, input_dim = X_train.shape

    model = DeepResIDSNet(input_dim, len(CLASS_NAMES), width=512).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    print("=======================================================")
    print(f" Running DeepResIDSNet ({param_count:,} params) on {device_name.upper()}")
    if device.type == "cuda":
        print(f" Hardware: {torch.cuda.get_device_name(0)}")
    print(f" Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS}")
    print("=======================================================")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Keep full float32 tensors on the target device (~700 MB fits easily in 6 GB VRAM)
    X_tr_dev = X_train.to(device)
    y_tr_dev = y_train.to(device)
    X_te_dev = X_test.to(device)

    # Warm-up pass so CUDA context initialization isn't counted in epoch 1
    if device.type == "cuda":
        _ = model(X_tr_dev[:BATCH_SIZE])
        sync(device)

    # 1. Timed Training Loop
    model.train()
    sync(device)
    t_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.perf_counter()
        perm = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0

        for start in range(0, num_samples, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            bx, by = X_tr_dev[idx], y_tr_dev[idx]

            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bx.size(0)

        sync(device)
        ep_sec = time.perf_counter() - ep_start
        elapsed = time.perf_counter() - t_start
        print(f"  epoch {epoch:2d}/{EPOCHS} | loss {epoch_loss / num_samples:.4f} | epoch: {ep_sec:5.2f}s | elapsed: {elapsed:6.1f}s")

    sync(device)
    train_seconds = time.perf_counter() - t_start

    # 2. Timed Inference (Intrusion Detection on unseen test flows)
    model.eval()
    preds_list = []
    sync(device)
    i_start = time.perf_counter()

    with torch.no_grad():
        for start in range(0, len(X_te_dev), BATCH_SIZE * 2):
            logits = model(X_te_dev[start : start + BATCH_SIZE * 2])
            preds_list.append(torch.argmax(logits, dim=1))

    sync(device)
    infer_seconds = time.perf_counter() - i_start

    y_pred = torch.cat(preds_list).cpu()
    overall_acc = float((y_pred == y_test).float().mean().item()) * 100.0
    throughput = len(y_test) / max(infer_seconds, 1e-6)

    print(f"  -> Train time     : {train_seconds:.2f} s ({train_seconds / 60:.2f} min)")
    print(f"  -> Inference time : {infer_seconds:.4f} s ({throughput:,.0f} flows/sec)")
    print(f"  -> Test accuracy  : {overall_acc:.2f}%")
    for idx, cname in enumerate(CLASS_NAMES):
        mask = y_test == idx
        cnt = int(mask.sum().item())
        if cnt > 0:
            c_acc = float((y_pred[mask] == y_test[mask]).float().mean().item()) * 100.0
            print(f"       {cname:<12s}: {c_acc:6.2f}% ({cnt:,} rows)")
    print()

    del X_tr_dev, y_tr_dev, X_te_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model,
        "params": param_count,
        "train_seconds": train_seconds,
        "infer_seconds": infer_seconds,
        "throughput": throughput,
        "accuracy": overall_acc,
    }

def save_chart(gpu_res: dict, cpu_res: dict, out_path: Path):
    """Save a side-by-side GPU vs. CPU training time and inference throughput chart."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))

    devices = ["GPU\n(GTX 1660 Super)", "CPU\n(Ryzen 5 3600)"]

    # Left subplot: Training time (lower is better)
    bars1 = axes[0].bar(devices, [gpu_res["train_seconds"], cpu_res["train_seconds"]], color=["#10B981", "#64748B"], width=0.48)
    axes[0].set_title("DeepResIDSNet Training Time (Lower is Better)")
    axes[0].set_ylabel("Seconds (s)")
    for b in bars1:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, h * 1.02, f"{h:.1f}s", ha="center", fontweight="bold")

    # Right subplot: Inference throughput (higher is better)
    bars2 = axes[1].bar(devices, [gpu_res["throughput"] / 1e6, cpu_res["throughput"] / 1e6], color=["#2563EB", "#64748B"], width=0.48)
    axes[1].set_title("Inference Throughput (Millions of Flows/sec)")
    axes[1].set_ylabel("Million Flows / sec")
    for b in bars2:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, h * 1.02, f"{h:.2f}M/s", ha="center", fontweight="bold")

    speedup = cpu_res["train_seconds"] / gpu_res["train_seconds"]
    fig.suptitle(f"CICIDS2017 DeepResIDSNet ({gpu_res['params']:,} Params) — {speedup:.2f}x GPU Speedup", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved benchmark chart to {out_path}")


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    X_train, y_train, X_test, y_test = prepare_datasets(CICIDS_DIR)

    gpu_res = train_and_benchmark("cuda", X_train, y_train, X_test, y_test) if torch.cuda.is_available() else None
    cpu_res = train_and_benchmark("cpu", X_train, y_train, X_test, y_test)

    if gpu_res is not None:
        train_speedup = cpu_res["train_seconds"] / gpu_res["train_seconds"]
        infer_speedup = cpu_res["infer_seconds"] / gpu_res["infer_seconds"]
        total_min = (gpu_res["train_seconds"] + cpu_res["train_seconds"]) / 60.0

        print("=======================================================")
        print("   CICIDS2017 DeepResIDSNet BENCHMARK (GPU vs. CPU)    ")
        print("=======================================================")
        print(f"  Model Size     : {gpu_res['params']:,} parameters")
        print(f"  GPU Training   : {gpu_res['train_seconds']:.2f} s | Accuracy: {gpu_res['accuracy']:.2f}%")
        print(f"  CPU Training   : {cpu_res['train_seconds']:.2f} s | Accuracy: {cpu_res['accuracy']:.2f}%")
        print(f"  Total Run Time : {total_min:.2f} minutes")
        print(f"  Train Speedup  : {train_speedup:.2f}x faster on GPU")
        print(f"  Infer Speedup  : {infer_speedup:.2f}x faster on GPU")
        print("=======================================================")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "ids_cicids_deep_gpu.pth"
        torch.save(gpu_res["model"].state_dict(), model_path)
        print(f"[INFO] Saved trained GPU model to {model_path}")

        save_chart(gpu_res, cpu_res, DOCS_DIR / "cicids2017_deep_benchmark.png")


if __name__ == "__main__":
    main()