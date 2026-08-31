from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from Autoencoder import compute_reconstruction_errors

def evaluate_anomaly_detector(
    model: nn.Module,
    test_tensor: torch.Tensor,
    y_true: np.ndarray,
    threshold: float,
    fault_tags: list,
    device: str = "cpu"
):
    """Evaluates reconstruction loss against detection threshold and logs metrics."""
    model.eval()
    model.to(device)
    
    mse_errors = compute_reconstruction_errors(model, test_tensor)

    # Binary prediction based on threshold
    y_pred = (mse_errors > threshold).astype(int)#generates a vector

    # Calculate overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()#This line and the line above is how its done on g2g
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0#formula checked from wikipedia

    print("==================================================")
    print("      PHASE 4: ANOMALY DETECTION EVALUATION       ")
    print("==================================================")
    print(f"Detection Threshold: {threshold:.6f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"True Positives (Detected Attacks): {tp} | False Positives: {fp}")
    print(f"True Negatives (Clean Windows):    {tn} | False Negatives: {fn}\n")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4%}\n")

    # Breakdown detection rate per fault type
    """
    Add later
    print("--- Recall Breakdown by Injection Type ---")
        df_eval = pd.DataFrame({"fault": fault_tags, "true": y_true, "pred": y_pred})
        for fault in ["rpm_spike", "speed_offset", "throttle_stuck", "gear"]:
            sub = df_eval[df_eval["fault"] == fault]
            if len(sub) > 0:
                det_rate = (sub["pred"] == 1).mean()
                print(f"{fault:<15}: {det_rate:.2%} detected ({sub['pred'].sum()}/{len(sub)})")
    """
    
    return mse_errors, y_pred