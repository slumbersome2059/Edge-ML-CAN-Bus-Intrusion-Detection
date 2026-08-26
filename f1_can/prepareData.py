import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from . import FEATURE_COLUMNS, SEQUENCE_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# PHASE 1: DATA PREPROCESSING & WINDOWING
# ==========================================


WINDOW_SIZE = 20  # 20 timesteps at 10 Hz = 2.0 seconds of history
STRIDE = 1        # Step size for sliding window
BATCH_SIZE = 64


def prepare_datasets(csv_path: str):
    df = pd.read_csv(csv_path)

    # 1. Segment-based dataset split (70% Train, 15% Val, 15% Test)
    # Splitting by segment_id avoids temporal correlation leakage between splits
    if "segment_id" in df.columns:
        segments = df["segment_id"].unique()
        np.random.shuffle(segments)

        n_train = int(len(segments) * 0.70)
        n_val = int(len(segments) * 0.15)

        train_segs = segments[:n_train]
        val_segs = segments[n_train : n_train + n_val]
        test_segs = segments[n_train + n_val :]

        train_df = df[df["segment_id"].isin(train_segs)].copy()
        val_df = df[df["segment_id"].isin(val_segs)].copy()
        test_df = df[df["segment_id"].isin(test_segs)].copy()
    else:
        # Fallback for simple single-sequence CSVs
        n = len(df)
        train_df = df.iloc[: int(n * 0.7)].copy()
        val_df = df.iloc[int(n * 0.7) : int(n * 0.85)].copy()
        test_df = df.iloc[int(n * 0.85) :].copy()

    # 2. Feature Scaling (Fit ONLY on training data to prevent leakage)
    scaler = StandardScaler()
    scaler.fit(train_df[*FEATURE_COLUMNS])#this calcualates mean and SD, later used in transform to scale things
    #The scaling/transformation does is z = x - \mu/\sigma, the z score stuff
    #It is really important you fit on training data, fitting on the other data means 
    #you gain info about something that is meant to be unknown(test and val are unseen data)

    for dframe in [train_df, val_df, test_df]:
        dframe[*FEATURE_COLUMNS] = scaler.transform(dframe[*FEATURE_COLUMNS])#perform the z score transformation

    # 3. Sliding Window Extraction (Respecting Segment Boundaries)
    def extract_windows(dframe):
        windows = []
        if "segment_id" in dframe.columns:
            for _, group in dframe.groupby("segment_id"):
                data = group[*FEATURE_COLUMNS].to_numpy()#returns numpy array with each row as list in the array
                for start in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
                    windows.append(data[start : start + WINDOW_SIZE])
        else:
            data = dframe[*FEATURE_COLUMNS].values
            for start in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
                windows.append(data[start : start + WINDOW_SIZE])
        return np.array(windows, dtype=np.float32)

    X_train = extract_windows(train_df)
    X_val = extract_windows(val_df)
    X_test = extract_windows(test_df)

    # Convert to PyTorch Conv1D shape: (Batch, Channels/Features, Window_Size)
    X_train_t = torch.tensor(X_train).transpose(1, 2)
    X_val_t = torch.tensor(X_val).transpose(1, 2)
    X_test_t = torch.tensor(X_test).transpose(1, 2)

    train_loader = DataLoader(
        TensorDataset(X_train_t), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t), batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler, X_val_t, X_test_t