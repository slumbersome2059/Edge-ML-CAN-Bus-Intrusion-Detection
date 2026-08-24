"""Agent 3: merge normal and forged frames into a chronological stream."""

from __future__ import annotations

import argparse
from pathlib import Path

from can_simulator import read_jsonl, write_jsonl


def merge_frames(normal_frames: list[dict], injected_frames: list[dict]) -> list[dict]:
    # Normal frames precede forged frames at identical timestamps, so baseline
    # traffic is consistently observed before an injected duplicate CAN ID.
    merged = normal_frames + injected_frames
    merged.sort(key=lambda frame: (frame["timestamp_ms"], 0 if frame.get("source") == "normal" else 1, frame["arbitration_id"]))
    # Source and event identifiers are simulation metadata; a detector receives
    # only the CAN-style telemetry schema. Ground truth remains separate.
    return [{key: value for key, value in frame.items() if key not in {"source", "event_id"}} for frame in merged]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-input", type=Path, required=True)
    parser.add_argument("--injected-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, merge_frames(read_jsonl(args.normal_input), read_jsonl(args.injected_input)))


if __name__ == "__main__":
    main()
