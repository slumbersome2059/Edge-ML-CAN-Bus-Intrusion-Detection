"""Copy into the autoencoder clone and run with its Hydra environment."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate

from tsa import AutoEncForecast, train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def per_window_mse(loader, model: torch.nn.Module) -> np.ndarray:
    """Return one next-timestep MSE score per sequence, preserving order."""
    scores = []
    model.eval()
    with torch.no_grad():
        for features, history, target in loader:
            prediction = model(features.to(DEVICE), history.to(DEVICE))
            scores.extend(((prediction - target.to(DEVICE)) ** 2).mean(dim=1).cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float64)


def metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    labels = labels.astype(bool)
    predicted = predicted.astype(bool)
    tp = int(np.sum(labels & predicted))
    fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted))
    tn = int(np.sum(~labels & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "clean_false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


@hydra.main(version_base=None, config_path="configs", config_name="f1_reconstruction")
def run(cfg) -> None:
    dataset = instantiate(cfg.data)
    train_loader, validation_loader, feature_count = dataset.get_loaders()
    model = AutoEncForecast(cfg.training, input_size=feature_count).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    train(train_loader, validation_loader, model, criterion, optimizer, cfg, dataset)

    validation_errors = per_window_mse(validation_loader, model)
    threshold = float(np.percentile(validation_errors, cfg.general.threshold_percentile))
    test_loader, labels, fault_types = dataset.external_loader(cfg.general.external_test_path)
    test_errors = per_window_mse(test_loader, model)
    result = {
        "threshold": threshold,
        "threshold_percentile": float(cfg.general.threshold_percentile),
        "validation_examples": len(validation_errors),
        "test_examples": len(test_errors),
        "metrics": metrics(labels, test_errors > threshold),
        "fault_counts": {str(name): int(np.sum(fault_types == name)) for name in np.unique(fault_types)},
    }
    output_dir = Path(cfg.general.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "anomaly_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output_dir / "external_scores.npz", mse=test_errors, labels=labels, fault_types=fault_types)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
