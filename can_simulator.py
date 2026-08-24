"""Shared utilities for the synthetic CAN telemetry simulator.

The identifiers and payload format in this module are deliberately synthetic;
they are a stable test convention, not a vehicle DBC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

RPM_ID = 0x0C0
SPEED_ID = 0x0D0
STEERING_ID = 0x0E0
BRAKE_ID = 0x0F0

SIGNAL_SPECS = {
    "engine_rpm": (RPM_ID, 0.25, 0.0, 8000.0),
    "vehicle_speed_kph": (SPEED_ID, 0.01, 0.0, 260.0),
    "steering_angle_deg": (STEERING_ID, 0.1, -540.0, 540.0),
    "brake_pedal_pct": (BRAKE_ID, 0.1, 0.0, 100.0),
}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def make_frame(timestamp_ms: int, signal: str, value: float, *, source: str = "normal") -> dict[str, Any]:
    """Build a four-byte little-endian, unsigned CAN-style signal frame."""
    arbitration_id, scale, minimum, maximum = SIGNAL_SPECS[signal]
    bounded_value = clamp(value, minimum, maximum)
    raw = int(round((bounded_value - minimum) / scale))
    return {
        "timestamp_ms": int(timestamp_ms),
        "arbitration_id": f"0x{arbitration_id:03X}",
        "dlc": 4,
        "data_hex": raw.to_bytes(4, byteorder="little", signed=False).hex().upper(),
        "signal": signal,
        "value": round(bounded_value, 3),
        "source": source,
    }


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
