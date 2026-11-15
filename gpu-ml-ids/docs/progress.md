# GPU-ML-IDS progress

Last updated: 14 November 2026.

This note is for the team. It describes what is already in the project, how to run it, and what is still left.

## Goal

Build an intrusion detection system that classifies network traffic with a small PyTorch neural network. Train it on a GPU with CUDA, then train the same model on the CPU and report the speedup.

## What is done

- The project folder, README, and `.gitignore` are in the repo.
- PyTorch with CUDA is set up in a conda environment named `ids-gpu`. On the machine used for the first run, PyTorch saw an NVIDIA GeForce GTX 1660 SUPER.
- `src/model.py` defines `IDSNet`, a small fully connected network: 41 inputs, then 64 units, 32 units, and one score per class. The outputs are logits for `CrossEntropyLoss`.
- The dataset in use is NSL-KDD (`KDDTrain+.txt` and `KDDTest+.txt`). These files are not in Git. `.gitignore` ignores `data/*.txt`, `data/*.csv`, and `data/*.zip`.
- `src/preprocess.py` loads those two files. Each row has 41 features, an attack name, and a difficulty score. The difficulty score is dropped. The attack name is the label. Text features (`protocol_type`, `service`, `flag`) are turned into integers using the training set. Features are scaled with `StandardScaler` fit on the training set only.
- Attack names are grouped into five classes: `normal`, `dos`, `probe`, `r2l`, `u2r`. Probe covers port scans. R2L covers remote attacks such as password guessing.
- `src/train.py` trains that network for 10 epochs on the GPU and again on the CPU. One warmup pass is not included in the timer, so the comparison is training time rather than CUDA startup. Rare classes get a larger loss weight so DoS and normal traffic do not hide them completely.

## How to run it

From the `gpu-ml-ids` folder, with the `ids-gpu` environment active:

```powershell
conda activate ids-gpu
python src/train.py
```

If `conda` is not on the PATH in PowerShell:

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\activate.ps1"
conda activate ids-gpu
python src/train.py
```

The plain `python` command outside that environment does not have PyTorch installed.

The dataset is downloaded separately. The University of New Brunswick page no longer hosts NSL-KDD. A public mirror of the same files is https://github.com/jmnwong/NSL-KDD-Dataset. Save `KDDTrain+.txt` and `KDDTest+.txt` into `gpu-ml-ids/data/`. Do not commit them.

From `gpu-ml-ids` in PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path data
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain+.txt" -OutFile "data\KDDTrain+.txt"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest+.txt" -OutFile "data\KDDTest+.txt"
```

If the dataset is used in a report, cite M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set,” Proceedings of the 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.

## First training result

Run on the GTX 1660 SUPER, 10 timed epochs, batch size 4096. GPU and CPU accuracies differ slightly because each run shuffles the training rows on its own device.

| Device | Time | Test accuracy |
| --- | --- | --- |
| GPU | 0.57 s | 0.7552 |
| CPU | 1.26 s | 0.7590 |
| Speedup | 2.21x | |

GPU accuracy by class on the official NSL-KDD test set:

| Class | Accuracy | Test rows |
| --- | --- | --- |
| normal | 0.9696 | 9711 |
| dos | 0.7389 | 7458 |
| probe | 0.7233 | 2421 |
| r2l | 0.1105 | 2887 |
| u2r | 0.4179 | 67 |

Training counts were normal 67343, dos 45927, probe 11656, r2l 995, u2r 52. R2L stays weak because the training file has few of those rows, and several R2L names appear only in the test file.

## Files

| File | Role |
| --- | --- |
| `src/model.py` | The neural network |
| `src/preprocess.py` | Load NSL-KDD and prepare arrays |
| `src/train.py` | Train on GPU and CPU, then print speedup |
| `data/` | Local dataset only. Not committed |
| `docs/progress.md` | This note |

## Not done yet

- No saved model file yet.
- No plots, confusion matrix, or written report of the speedup.
- R2L detection is still poor.
- CICIDS has not been tried. NSL-KDD is the dataset for this stage because it is small enough to train on a laptop GPU.
