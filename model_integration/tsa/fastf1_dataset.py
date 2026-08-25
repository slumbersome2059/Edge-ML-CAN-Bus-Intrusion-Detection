"""Copy into the autoencoder repository as ``tsa/fastf1_dataset.py``.

This keeps its scaler behavior but consumes fixed, group-safe sequence archives
instead of re-windowing a concatenated CSV across race-driver boundaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class FastF1WindowDataset:
    """The target project's data-loader contract for pre-windowed FastF1 data."""

    def __init__(self, data_path: str, batch_size: int, validation_fraction: float = 0.2):
        self.data_path = Path(to_absolute_path(data_path))
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.preprocessor = StandardScaler()
        self._fitted = False

    @staticmethod
    def _load(path: Path) -> dict[str, np.ndarray]:
        archive = np.load(path, allow_pickle=False)
        required = {"features", "history", "targets"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return {key: archive[key].astype(np.float32) for key in required}

    def _transform(self, arrays: dict[str, np.ndarray], *, fit: bool) -> dict[str, np.ndarray]:
        features = arrays["features"]
        if features.ndim != 3 or features.shape[2] != 4:
            raise ValueError("features must have shape [examples, sequence_length, 4]")
        if fit:
            self.preprocessor.fit(features.reshape(-1, features.shape[-1]))
            self._fitted = True
        if not self._fitted:
            raise RuntimeError("fit the normal training split before transforming data")
        return {
            key: self.preprocessor.transform(value.reshape(-1, value.shape[-1])).reshape(value.shape).astype(np.float32)
            for key, value in arrays.items()
        }

    @staticmethod
    def _loader(arrays: dict[str, np.ndarray], batch_size: int) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(arrays["features"]), torch.from_numpy(arrays["history"]),
                                torch.from_numpy(arrays["targets"]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    def get_loaders(self):
        arrays = self._load(self.data_path)
        split = int(len(arrays["features"]) * (1 - self.validation_fraction))
        if split == 0 or split == len(arrays["features"]):
            raise ValueError("archive must contain examples on both sides of the validation split")
        train = {key: value[:split] for key, value in arrays.items()}
        validation = {key: value[split:] for key, value in arrays.items()}
        train = self._transform(train, fit=True)
        validation = self._transform(validation, fit=False)
        return self._loader(train, self.batch_size), self._loader(validation, self.batch_size), 4

    def external_loader(self, path: str) -> tuple[DataLoader, np.ndarray, np.ndarray]:
        """Transform external data only with the already-fitted normal scaler."""
        archive_path = Path(to_absolute_path(path))
        raw = np.load(archive_path, allow_pickle=False)
        arrays = {key: raw[key].astype(np.float32) for key in ("features", "history", "targets")}
        labels = raw["labels"].astype(np.int8) if "labels" in raw.files else np.zeros(len(arrays["features"]), dtype=np.int8)
        faults = raw["fault_types"] if "fault_types" in raw.files else np.full(len(labels), "clean")
        return self._loader(self._transform(arrays, fit=False), self.batch_size), labels, faults
