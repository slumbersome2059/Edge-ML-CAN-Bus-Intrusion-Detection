"""Fixed, synthetic CAN frame encoding for the four telemetry signals.

The layout is not a vehicle DBC. Each standard-ID ``0x100`` frame is eight
little-endian bytes: RPM (uint16, 1 RPM/bit), speed (uint16, 0.01 km/h/bit),
throttle (uint8, 1 %/bit), gear (uint8), rolling counter (uint8), reserved.
"""

from __future__ import annotations

import struct
from typing import Mapping

CAN_ID = 0x100
CAN_DLC = 8


def _bounded_int(value: float, lower: int, upper: int, name: str) -> int:
    integer = int(round(float(value)))
    if not lower <= integer <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {value}")
    return integer


def encode_frame(sample: Mapping[str, float | int], counter: int = 0) -> bytes:
    """Encode one raw-unscaled telemetry sample into the synthetic payload."""
    rpm = _bounded_int(sample["RPM"], 0, 65535, "RPM")
    speed = _bounded_int(float(sample["Speed"]) * 100, 0, 65535, "Speed")
    throttle = _bounded_int(sample["Throttle"], 0, 100, "Throttle")
    gear = _bounded_int(sample["nGear"], 0, 255, "nGear")
    return struct.pack("<HHBBBB", rpm, speed, throttle, gear, counter % 256, 0)


def decode_frame(payload: bytes) -> dict[str, float | int]:
    """Decode and validate an eight-byte payload made by :func:`encode_frame`."""
    if len(payload) != CAN_DLC:
        raise ValueError(f"CAN payload must be {CAN_DLC} bytes, got {len(payload)}")
    rpm, speed_raw, throttle, gear, counter, reserved = struct.unpack("<HHBBBB", payload)
    if reserved != 0:
        raise ValueError("reserved CAN byte must be zero")
    return {
        "RPM": rpm,
        "Speed": speed_raw / 100.0,
        "Throttle": throttle,
        "nGear": gear,
        "counter": counter,
    }
