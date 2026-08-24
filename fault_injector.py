"""Agent 2: inject forged brake-at-speed and RPM-spike CAN frames."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from can_simulator import make_frame, read_jsonl, write_jsonl


def inject_faults(normal_frames: list[dict], injection_rate: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Return forged frames and their separate event-level ground truth.

    ``injection_rate`` is the probability of beginning an attack at each eligible
    10 Hz sample. Active attacks last 2--4 samples and are not allowed to overlap.
    """
    if not 0.0 <= injection_rate <= 1.0:
        raise ValueError("injection_rate must be between 0 and 1")
    rng = random.Random(seed)
    samples: dict[int, dict[str, dict]] = defaultdict(dict)
    for frame in normal_frames:
        samples[frame["timestamp_ms"]][frame["signal"]] = frame

    injected: list[dict] = []
    truth: list[dict] = []
    event_number = 0
    active_until = -1
    for timestamp in sorted(samples):
        signals = samples[timestamp]
        if timestamp <= active_until or rng.random() >= injection_rate:
            continue
        speed = signals.get("vehicle_speed_kph", {}).get("value", 0.0)
        rpm = signals.get("engine_rpm", {}).get("value", 0.0)
        eligible = []
        if speed > 40.0:
            eligible.append("brake_at_speed")
        if rpm > 900.0:
            eligible.append("rpm_spike")
        if not eligible:
            continue

        attack_type = rng.choice(eligible)
        burst_length = rng.randint(2, 4)
        selected_timestamps = [t for t in sorted(samples) if timestamp <= t][:burst_length]
        if not selected_timestamps:
            continue
        event_number += 1
        event_id = f"attack-{event_number:05d}"
        frame_refs: list[dict] = []
        for attack_timestamp in selected_timestamps:
            current = samples[attack_timestamp]
            if attack_type == "brake_at_speed":
                forged_value = rng.uniform(80.0, 100.0)
                frame = make_frame(attack_timestamp, "brake_pedal_pct", forged_value, source="injected")
            else:
                base_rpm = current["engine_rpm"]["value"]
                forged_value = min(8000.0, base_rpm * rng.uniform(1.8, 2.5))
                frame = make_frame(attack_timestamp, "engine_rpm", forged_value, source="injected")
            frame["event_id"] = event_id
            injected.append(frame)
            frame_refs.append({"timestamp_ms": attack_timestamp, "arbitration_id": frame["arbitration_id"], "value": frame["value"]})
        truth.append(
            {
                "event_id": event_id,
                "attack_type": attack_type,
                "start_timestamp_ms": timestamp,
                "duration_frames": len(frame_refs),
                "context": {"speed_kph": speed, "rpm": rpm},
                "injected_frames": frame_refs,
            }
        )
        active_until = selected_timestamps[-1]
    return injected, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Normal-frame JSONL path")
    parser.add_argument("--injected-output", type=Path, required=True)
    parser.add_argument("--truth-output", type=Path, required=True)
    parser.add_argument("--injection-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    injected, truth = inject_faults(read_jsonl(args.input), args.injection_rate, args.seed)
    args.injected_output.parent.mkdir(parents=True, exist_ok=True)
    args.truth_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.injected_output, injected)
    write_jsonl(args.truth_output, truth)


if __name__ == "__main__":
    main()
