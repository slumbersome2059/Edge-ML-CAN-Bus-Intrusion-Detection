"""Create a leakage-safe normal/anomalous dataset from extracted telemetry."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import json
import random

import numpy as np

from . import FEATURE_COLUMNS, SEQUENCE_LENGTH
from .faults import inject_balanced_faults


@dataclass(frozen=True)
class Example:
    """One target-model prediction sample sourced from one telemetry segment."""

    segment_id: str
    start_index: int
    features: np.ndarray
    history: np.ndarray
    target: np.ndarray

    @property
    def replay(self) -> np.ndarray:
        return np.vstack((self.features, self.target))


def make_examples(segment_id: str, values: np.ndarray, seq_length: int = SEQUENCE_LENGTH) -> list[Example]:
    """Build model-compatible examples: previous history, input, and next target."""
    if values.ndim != 2 or values.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("values must have shape [rows, 4]")
    examples = []
    for index in range(1, len(values) - seq_length):
        examples.append(Example(segment_id, index, values[index:index + seq_length],
                                values[index - 1:index + seq_length - 1], values[index + seq_length]))
    return examples


def _round_robin(groups: dict[str, list[Example]], count: int) -> list[Example]:
    queues = deque((key, deque(value)) for key, value in sorted(groups.items()) if value)
    selected = []
    while queues and len(selected) < count:
        key, examples = queues.popleft()
        selected.append(examples.popleft())
        if examples:
            queues.append((key, examples))
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} examples are available; need {count}")
    return selected


def _split_groups(groups: dict[str, list[Example]], test_group_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    keys = sorted(key for key, value in groups.items() if value)
    if len(keys) < 2:
        raise ValueError("at least two race-driver segments are required for a group holdout")
    rng = random.Random(seed)
    rng.shuffle(keys)
    test_count = max(1, round(len(keys) * test_group_fraction))
    test_keys = set(keys[:test_count])
    return set(keys).difference(test_keys), test_keys


def _archive_examples(examples: list[Example]) -> dict[str, np.ndarray]:
    return {
        "features": np.stack([example.features for example in examples]).astype(np.float32),
        "history": np.stack([example.history for example in examples]).astype(np.float32),
        "targets": np.stack([example.target for example in examples]).astype(np.float32),
        "replay": np.stack([example.replay for example in examples]).astype(np.float32),
        "segment_ids": np.asarray([example.segment_id for example in examples]),
        "start_indices": np.asarray([example.start_index for example in examples], dtype=np.int64),
    }


def build_archives(raw_csv: Path, output_dir: Path, *, train_count: int = 9000, test_count: int = 1000,
                   fault_count: int = 500, seed: int = 42, seq_length: int = SEQUENCE_LENGTH) -> dict[str, int]:
    """Build raw, unscaled normal and held-out anomaly archives.

    The selected test race-driver groups never contribute training examples. The
    model adapter performs its own internal 80/20 split and fitting of the
    target repository's StandardScaler after this function has completed.
    """
    import pandas as pd

    if test_count != 1000 or train_count != 9000 or fault_count != 500:
        raise ValueError("this plan requires exactly 9000 training, 1000 test, and 500 faulted examples")
    frame = pd.read_csv(raw_csv)
    required = {"segment_id", *FEATURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"raw CSV missing columns: {sorted(missing)}")
    groups: dict[str, list[Example]] = {}
    for segment_id, segment in frame.groupby("segment_id", sort=True):
        groups[str(segment_id)] = make_examples(str(segment_id), segment.loc[:, FEATURE_COLUMNS].to_numpy(np.float32), seq_length)
    train_groups, test_groups = _split_groups(groups, 0.10, seed)
    train_examples = _round_robin({key: groups[key] for key in train_groups}, train_count)
    test_examples = _round_robin({key: groups[key] for key in test_groups}, test_count)
    clean_test = _archive_examples(test_examples)
    faulted_replay, fault_data = inject_balanced_faults(clean_test["replay"], fault_count, seed + 1)
    # Model inputs are rows 0..9 and the next target is row 10. History begins
    # one timestep earlier, so only its last row overlaps a faulted input row.
    test = dict(clean_test)
    test["features"] = faulted_replay[:, :seq_length]
    test["history"] = np.array(clean_test["history"], copy=True)
    test["history"][:, -1] = faulted_replay[:, seq_length - 2]
    test["targets"] = faulted_replay[:, seq_length]
    test["replay"] = faulted_replay
    test["labels"] = fault_data[:, 0].astype(np.int8)
    test["fault_types"] = fault_data[:, 1]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "train_windows.npz", **_archive_examples(train_examples))
    np.savez_compressed(output_dir / "test_clean_windows.npz", **clean_test)
    np.savez_compressed(output_dir / "test_windows.npz", **test)
    manifest = {
        "feature_columns": list(FEATURE_COLUMNS), "sequence_length": seq_length, "train_examples": train_count,
        "test_examples": test_count, "faulted_test_examples": fault_count, "seed": seed,
        "train_segments": sorted(train_groups), "test_segments": sorted(test_groups), "scaled_by_builder": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"train": train_count, "test": test_count, "faulted": int(test["labels"].sum())}
