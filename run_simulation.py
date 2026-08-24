"""Run all three synthetic CAN simulator agents and write one dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregator import merge_frames
from can_simulator import read_jsonl, write_jsonl
from fault_injector import inject_faults
from normal_driving_simulator import generate_normal_frames


def run(output_dir: Path, duration: float, sample_rate: float, seed: int, injection_rate: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    normal = generate_normal_frames(duration, sample_rate, seed)
    injected, truth = inject_faults(normal, injection_rate, seed + 1)
    write_jsonl(output_dir / "normal_frames.jsonl", normal)
    write_jsonl(output_dir / "injected_frames.jsonl", injected)
    write_jsonl(output_dir / "ground_truth.jsonl", truth)
    write_jsonl(output_dir / "stream.jsonl", merge_frames(normal, injected))
    metadata = {
        "duration_seconds": duration,
        "sample_rate_hz": sample_rate,
        "seed": seed,
        "injection_seed": seed + 1,
        "injection_rate": injection_rate,
        "normal_frame_count": len(normal),
        "injected_frame_count": len(injected),
        "attack_event_count": len(truth),
        "format": "synthetic CAN-style JSON Lines v1",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_output"))
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--sample-rate", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--injection-rate", type=float, default=0.003)
    args = parser.parse_args()
    run(args.output_dir, args.duration, args.sample_rate, args.seed, args.injection_rate)


if __name__ == "__main__":
    main()
