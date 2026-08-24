"""Agent 1: generate correlated, normal synthetic automotive telemetry."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from can_simulator import clamp, make_frame, write_jsonl


def generate_normal_frames(duration_seconds: float, sample_rate_hz: float, seed: int) -> list[dict]:
    if duration_seconds <= 0 or sample_rate_hz <= 0:
        raise ValueError("duration_seconds and sample_rate_hz must be positive")

    rng = random.Random(seed)
    step_seconds = 1.0 / sample_rate_hz
    samples = int(round(duration_seconds * sample_rate_hz))
    speed = 0.0
    steering = 0.0
    throttle = 0.22
    brake = 0.0
    frames: list[dict] = []

    for index in range(samples):
        timestamp_ms = int(round(index * 1000.0 / sample_rate_hz))
        # Change driver intent gradually; braking becomes more likely at speed.
        throttle = clamp(throttle + rng.uniform(-0.07, 0.07), 0.05, 0.78)
        braking_target = 0.0
        if speed > 8 and rng.random() < 0.035:
            braking_target = rng.uniform(12.0, 52.0)
        brake = clamp(brake * 0.52 + braking_target * 0.48 + rng.uniform(-1.0, 1.0), 0.0, 70.0)
        acceleration = (throttle * 3.6) - (brake * 0.095) - (speed * 0.012)
        speed = clamp(speed + acceleration * step_seconds + rng.gauss(0.0, 0.08), 0.0, 160.0)
        steering = clamp(steering * 0.78 + rng.gauss(0.0, 4.5), -120.0, 120.0)
        # A simplified gearing relationship with idle RPM and modest transient noise.
        rpm = clamp(780.0 + speed * 42.0 + throttle * 850.0 - brake * 4.0 + rng.gauss(0.0, 35.0), 700.0, 7200.0)

        frames.extend(
            (
                make_frame(timestamp_ms, "engine_rpm", rpm),
                make_frame(timestamp_ms, "vehicle_speed_kph", speed),
                make_frame(timestamp_ms, "steering_angle_deg", steering),
                make_frame(timestamp_ms, "brake_pedal_pct", brake),
            )
        )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Normal-frame JSONL path")
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--sample-rate", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, generate_normal_frames(args.duration, args.sample_rate, args.seed))


if __name__ == "__main__":
    main()
