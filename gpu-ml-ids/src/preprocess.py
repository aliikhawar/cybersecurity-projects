"""
Preprocess the NSL-KDD dataset for the intrusion detection system.

Each row is 41 traffic features, an attack name, and a difficulty score.
This file drops the difficulty score, groups attack names into five classes
(normal, dos, probe, r2l, u2r), turns text features into numbers, and scales
the features using the training set only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Column 41 is the attack name. Column 42 is a difficulty score, not the label.
LABEL_COLUMN = 41
DIFFICULTY_COLUMN = 42
TEXT_COLUMNS = [1, 2, 3]  # protocol_type, service, flag

CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]

ATTACK_GROUPS = {
    "normal": "normal",
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "mailbomb": "dos",
    "apache2": "dos",
    "processtable": "dos",
    "udpstorm": "dos",
    "satan": "probe",
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "mscan": "probe",
    "saint": "probe",
    "guess_passwd": "r2l",
    "ftp_write": "r2l",
    "imap": "r2l",
    "phf": "r2l",
    "multihop": "r2l",
    "warezmaster": "r2l",
    "warezclient": "r2l",
    "spy": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "snmpguess": "r2l",
    "snmpgetattack": "r2l",
    "httptunnel": "r2l",
    "sendmail": "r2l",
    "named": "r2l",
    "worm": "r2l",
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "rootkit": "u2r",
    "perl": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
    "ps": "u2r",
}


def check_gpu():
    """Print whether PyTorch can see the CUDA GPU."""
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")


def _read_nsl_kdd(path):
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 43:
        raise ValueError(f"{path} should have 43 columns, found {df.shape[1]}")
    return df


def _encode_text_columns(train_df, test_df):
    """Replace text features with integers learned from the training set."""
    for column in TEXT_COLUMNS:
        known = {
            value: index
            for index, value in enumerate(train_df[column].astype(str).unique())
        }
        unknown = len(known)
        train_df[column] = train_df[column].astype(str).map(known).astype(np.int64)
        test_df[column] = (
            test_df[column].astype(str).map(known).fillna(unknown).astype(np.int64)
        )


def _labels_to_groups(labels):
    groups = []
    for label in labels.astype(str):
        group = ATTACK_GROUPS.get(label)
        if group is None:
            raise ValueError(f"Unknown NSL-KDD label: {label}")
        groups.append(CLASS_NAMES.index(group))
    return np.asarray(groups, dtype=np.int64)


def load_nsl_kdd(train_path=None, test_path=None):
    """Load KDDTrain+.txt and KDDTest+.txt and return arrays ready for PyTorch."""
    if train_path is None:
        train_path = PROJECT_DIR / "data" / "KDDTrain+.txt"
    if test_path is None:
        test_path = PROJECT_DIR / "data" / "KDDTest+.txt"

    print(f"Loading {train_path}")
    train_df = _read_nsl_kdd(train_path)
    print(f"Loading {test_path}")
    test_df = _read_nsl_kdd(test_path)

    y_train = _labels_to_groups(train_df[LABEL_COLUMN])
    y_test = _labels_to_groups(test_df[LABEL_COLUMN])

    feature_columns = [col for col in train_df.columns if col != LABEL_COLUMN and col != DIFFICULTY_COLUMN]
    train_x = train_df[feature_columns].copy()
    test_x = test_df[feature_columns].copy()
    _encode_text_columns(train_x, test_x)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_x.to_numpy(dtype=np.float64)).astype(np.float32)
    x_test = scaler.transform(test_x.to_numpy(dtype=np.float64)).astype(np.float32)

    print("Train shape:", x_train.shape)
    print("Test shape:", x_test.shape)
    for index, name in enumerate(CLASS_NAMES):
        print(f"  {name}: train {(y_train == index).sum()} | test {(y_test == index).sum()}")

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    check_gpu()
    load_nsl_kdd()
