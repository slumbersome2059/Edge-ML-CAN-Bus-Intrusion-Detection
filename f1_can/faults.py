"""Deterministic faults applied to held-out telemetry sequences only."""

from __future__ import annotations

from collections.abc import Sequence
import random

import numpy as np

from . import FEATURE_COLUMNS

FAULT_TYPES = ("rpm_spike", "speed_offset", "throttle_stuck", "gear_spoof")


def inject_fault(sequence: np.ndarray, fault_type: str, rng: random.Random) -> np.ndarray:
    """Return a copy with a fault over the final three samples.

    Altering the target timestep guarantees the target autoencoder evaluates an
    actual anomaly instead of merely receiving a labelled normal sequence.
    """
    if sequence.ndim != 2 or sequence.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("sequence must have shape [timesteps, 4]")
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"unknown fault type: {fault_type}")
    result = np.array(sequence, dtype=np.float32, copy=True)
    start = max(0, len(result) - 3)
    rpm, speed, throttle, gear = range(4)
    if fault_type == "rpm_spike":
        result[start:, rpm] = np.clip(result[start:, rpm] * rng.uniform(1.45, 1.9), 0, 16000)
    elif fault_type == "speed_offset":
        result[start:, speed] = np.clip(result[start:, speed] + rng.choice((-1, 1)) * rng.uniform(55, 90), 0, 500)
    elif fault_type == "throttle_stuck":
        result[start:, throttle] = rng.choice((0, 100))
    else:
        result[start:, gear] = np.clip(result[start:, gear] + rng.choice((-3, -2, 2, 3)), 0, 8)
    return result


def inject_balanced_faults(sequences: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fault exactly ``count`` input sequences using an evenly cycled fault mix."""
    if not 0 <= count <= len(sequences):
        raise ValueError("count must be between zero and the number of sequences")
    rng = random.Random(seed)
    selected = list(range(len(sequences)))
    rng.shuffle(selected)
    selected = selected[:count]
    result = np.array(sequences, dtype=np.float32, copy=True)
    labels = np.zeros(len(sequences), dtype=np.int8)
    fault_names = np.full(len(sequences), "clean", dtype="U16")
    for ordinal, index in enumerate(selected):
        fault_type = FAULT_TYPES[ordinal % len(FAULT_TYPES)]
        result[index] = inject_fault(result[index], fault_type, rng)
        labels[index] = 1
        fault_names[index] = fault_type
    return result, np.stack((labels, fault_names), axis=1)
