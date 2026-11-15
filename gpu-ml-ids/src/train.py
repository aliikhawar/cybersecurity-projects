"""
Train the intrusion detection network and compare GPU speed against CPU.

Uses the official NSL-KDD train and test files. The same small network is
trained once on CUDA and once on the CPU. One warmup pass is not timed, so
the comparison measures training rather than CUDA startup.
"""

import time

import numpy as np
import torch
import torch.nn as nn

from model import IDSNet
from preprocess import CLASS_NAMES, check_gpu, load_nsl_kdd

EPOCHS = 10
BATCH_SIZE = 4096
LEARNING_RATE = 0.001
SEED = 42


def class_weights(y_train, device):
    """Give rare attack classes a stronger voice without letting them take over."""
    counts = np.bincount(y_train, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = np.sqrt(counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1.0)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, optimizer, loss_fn, features, labels, batch_size):
    model.train()
    order = torch.randperm(features.shape[0], device=features.device)
    features = features[order]
    labels = labels[order]
    total_loss = 0.0
    steps = 0
    for start in range(0, features.shape[0], batch_size):
        batch_x = features[start:start + batch_size]
        batch_y = labels[start:start + batch_size]
        optimizer.zero_grad()
        loss = loss_fn(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / steps


def evaluate(model, x_test, y_test, device):
    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(x_test, dtype=torch.float32, device=device))
        predictions = scores.argmax(dim=1).cpu().numpy()
    accuracy = float((predictions == y_test).mean())
    print(f"  test accuracy: {accuracy:.4f}")
    for index, name in enumerate(CLASS_NAMES):
        mask = y_test == index
        if mask.sum() == 0:
            continue
        class_accuracy = float((predictions[mask] == index).mean())
        print(f"    {name}: {class_accuracy:.4f} ({int(mask.sum())} rows)")
    return accuracy


def train_and_evaluate(x_train, y_train, x_test, y_test, device):
    """Train a fresh model on one device and return seconds plus test accuracy."""
    torch.manual_seed(SEED)
    model = IDSNet(x_train.shape[1], num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, device))

    features = torch.tensor(x_train, dtype=torch.float32, device=device)
    labels = torch.tensor(y_train, dtype=torch.long, device=device)

    run_epoch(model, optimizer, loss_fn, features, labels, BATCH_SIZE)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        loss = run_epoch(model, optimizer, loss_fn, features, labels, BATCH_SIZE)
        print(f"  epoch {epoch}/{EPOCHS}  loss {loss:.4f}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    accuracy = evaluate(model, x_test, y_test, device)
    return elapsed, accuracy


def main():
    check_gpu()
    x_train, x_test, y_train, y_test = load_nsl_kdd()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Activate the ids-gpu environment and try again.")

    print("\nTraining on GPU")
    gpu_seconds, gpu_accuracy = train_and_evaluate(
        x_train, y_train, x_test, y_test, torch.device("cuda")
    )

    print("\nTraining on CPU")
    cpu_seconds, cpu_accuracy = train_and_evaluate(
        x_train, y_train, x_test, y_test, torch.device("cpu")
    )

    print("\nSpeed comparison")
    print(f"  GPU time: {gpu_seconds:.2f} s   accuracy: {gpu_accuracy:.4f}")
    print(f"  CPU time: {cpu_seconds:.2f} s   accuracy: {cpu_accuracy:.4f}")
    print(f"  Speedup:  {cpu_seconds / gpu_seconds:.2f}x")


if __name__ == "__main__":
    main()
