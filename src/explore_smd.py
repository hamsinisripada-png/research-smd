
from pathlib import Path

import pandas as pd
import numpy as np

# Locate the dataset folders
data_dir = Path(__file__).resolve().parent

train_path = data_dir / "train" / "machine-1-1.txt"
test_path = data_dir / "test" / "machine-1-1.txt"
label_path = data_dir / "test_label" / "machine-1-1.txt"

# Load the data
train = pd.read_csv(train_path, header=None)
test = pd.read_csv(test_path, header=None)
labels = np.loadtxt(label_path, delimiter=",")

# Display basic information
print("TRAINING DATA SHAPE:", train.shape)
print("TEST DATA SHAPE:", test.shape)
print("NUMBER OF LABELS:", len(labels))

print("\nFIRST FIVE TRAINING ROWS:")
print(train.head())

print("\nUNIQUE LABEL VALUES:")
print(np.unique(labels))

print("\nNORMAL TIME POINTS:", np.sum(labels == 0))
print("ANOMALOUS TIME POINTS:", np.sum(labels == 1))

# Verify that every test row has a corresponding label
assert len(test) == len(labels), (
    "Mismatch: test rows and labels have different lengths!"
)

print("\nSUCCESS: Test rows and labels match.")
