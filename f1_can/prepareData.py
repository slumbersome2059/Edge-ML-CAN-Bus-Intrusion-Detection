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
    # 2. Feature Scaling (Fit ONLY on training data to prevent leakage)
    scaler = StandardScaler()
    print(train_df)
    scaler.fit(train_df[[*FEATURE_COLUMNS]])#this calcualates mean and SD, later used in transform to scale things
    #The scaling/transformation does is z = x - \mu/\sigma, the z score stuff
    #It is really important you fit on training data, fitting on the other data means 
    #you gain info about something that is meant to be unknown(test and val are unseen data)

    for dframe in [train_df, val_df, test_df]:
        dframe[[*FEATURE_COLUMNS]] = scaler.transform(dframe[[*FEATURE_COLUMNS]])#perform the z score transformation

    # 3. Sliding Window Extraction (Respecting Segment Boundaries)
    def extract_windows(dframe):
        windows = []
        for _, group in dframe.groupby("segment_id"):
            data = group[[*FEATURE_COLUMNS]].to_numpy()#returns numpy array with each row as list in the array
            for start in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
                windows.append(data[start : start + WINDOW_SIZE])#windows is a 3d array
        return np.array(windows, dtype=np.float32)

    X_train = extract_windows(train_df)
    X_val = extract_windows(val_df)
    X_test = extract_windows(test_df)

    # Currently the shape is (1, Window_Size, Channels/Features)
    # Convert to PyTorch Conv1D shape: (Batch, Channels/Features, Window_Size)
    X_train_t = torch.tensor(X_train).transpose(1, 2)
    X_val_t = torch.tensor(X_val).transpose(1, 2)
    
    # A Dataset is a way to store samples and if you need to you can store labels associated with tensors 
    # so "pos" might be associated with "good review"
    # Below we don't have any labels and just store the sample
    # A Dataset is something that represents where data is stored(it could be in 
    # some file, some nparray), the class requires only a getItem(int idx) method 
    # which should give you the (sample, label) or just sample if label is not there

    train_loader = DataLoader(
        TensorDataset(X_train_t), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t), batch_size=BATCH_SIZE, shuffle=False
    )
    
    # DataLoader is an iterable and when do next on it you end up getting a 
    # (batch_size, ...) tensor for (N, ...) shaped Dataset
    """
    - We normally pass data in batches of batch_size during training(from this we determine 
    the change in weights and biases) rather than using the whole set of data to produce 
     one change in weights and biases which means we can get many changes when we iterate 
     through one set of data
    - Everytime we iterate through data we select new batch to use to determine change 
    in weights and biases and eventually you will exhaust all the data(at that point you finish one epoch)
    - For the next epoch, you should shuffle the batches, using DataLoader has functionality to do this
    """
    # the shuffle is saying reshuffle at the end of every epoch

    return train_loader, val_loader, scaler, X_val_t, X_test

def inject_fault(raw_window: np.ndarray, fault_type: str, rng: np.random.Generator) -> np.ndarray:
    """Injects CAN bus sensor faults into raw unscaled window (WINDOW_SIZE, n_features)."""
    result = raw_window.copy()
    randSize = rng.choice([2,3,4,5])
    start = rng.integers(0, WINDOW_SIZE - randSize)  # Fault begins partway through window

    #Initially the code didn't use strings for columns and used hardcoded index values so I changed this
    if fault_type == "rpm_spike":
        result[start:start + randSize, "RPM"] = np.clip(
            result[start:start + randSize, "RPM"] * rng.uniform(1.45, 1.9), 0, 16000
        )
    elif fault_type == "speed_offset":
        result[start:start + randSize, "Speed"] = np.clip(
            result[start:start + randSize, "Speed"] + rng.choice((-1, 1)) * rng.uniform(55, 90), 0, 500
        )
    elif fault_type == "throttle_stuck":
        result[start:start + randSize, "Throttle"] = rng.choice((0, 100))
    else:  # gear manipulation
        result[start:start + randSize, "nGear"] = np.clip(
            result[start:start + randSize, "nGear"] + rng.choice((-3, -2, 2, 3)), 0, 8
        )

    return result


def generate_evaluation_dataset(
    raw_test_windows: np.ndarray, scaler, anomaly_ratio: float = 0.5, seed: int = 42
):
    """
    Generates balanced test windows with equal mix of clean data and 4 injected fault types.
    Normalizes data using the pre-fitted scaler.
    """
    rng = np.random.default_rng(seed)
    fault_types = ["rpm_spike", "speed_offset", "throttle_stuck", "gear"]
    
    processed_windows = []
    labels = []  # 0: Normal, 1: Anomaly
    fault_tags = []

    for window in raw_test_windows:
        is_anomaly = rng.random() < anomaly_ratio
        
        if is_anomaly:
            fault = rng.choice(fault_types)
            modified_window = inject_fault(window, fault, rng)
            label = 1
        else:
            fault = "clean"
            modified_window = window.copy()
            label = 0

        # Scale features using fitted scaler parameters
        scaled_window = scaler.transform(modified_window)
        processed_windows.append(scaled_window)
        labels.append(label)
        fault_tags.append(fault)

    X_test_scaled = np.array(processed_windows, dtype=np.float32)
    # Transpose to PyTorch Conv1D layout: (Batch, Channels/Features, Window_Size)
    X_test_tensor = torch.tensor(X_test_scaled).transpose(1, 2)
    y_test = np.array(labels, dtype=int)

    return X_test_tensor, y_test, fault_tags
