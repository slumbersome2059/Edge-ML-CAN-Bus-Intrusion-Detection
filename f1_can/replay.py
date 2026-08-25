"""Replay prepared raw telemetry sequences over SocketCAN or as JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import FEATURE_COLUMNS
from .can_frames import CAN_ID, decode_frame, encode_frame


def rows_from_archive(path: Path, sequence_index: int) -> np.ndarray:
    archive = np.load(path, allow_pickle=False)
    if not 0 <= sequence_index < len(archive["replay"]):
        raise IndexError(f"sequence_index must be between 0 and {len(archive['replay']) - 1}")
    return archive["replay"][sequence_index]


def replay(rows: np.ndarray, *, channel: str, rate_hz: float, dry_run: bool, output: Path | None = None) -> None:
    """Send raw rows at a fixed rate, or write the exact frames as JSONL."""
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    handle = output.open("w", encoding="utf-8") if output else None
    bus = None
    if not dry_run:
        try:
            import can
        except ImportError as exc:  # pragma: no cover - depends on deployment
            raise RuntimeError("install python-can to transmit on SocketCAN") from exc
        bus = can.Bus(interface="socketcan", channel=channel)
    try:
        for counter, row in enumerate(rows):
            sample = dict(zip(FEATURE_COLUMNS, row.tolist(), strict=True))
            payload = encode_frame(sample, counter)
            record = {"arbitration_id": f"0x{CAN_ID:03X}", "data_hex": payload.hex().upper(), "decoded": decode_frame(payload)}
            if handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            if bus:
                bus.send(can.Message(arbitration_id=CAN_ID, data=payload, is_extended_id=False))
            if counter + 1 < len(rows):
                time.sleep(1 / rate_hz)
    finally:
        if handle:
            handle.close()
        if bus:
            bus.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="test_windows.npz from the builder")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--channel", default="vcan0")
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, help="optional JSONL log of payloads")
    args = parser.parse_args()
    replay(rows_from_archive(args.archive, args.sequence_index), channel=args.channel, rate_hz=args.rate_hz,
           dry_run=args.dry_run, output=args.output)


if __name__ == "__main__":
    main()
