"""Standard-library regression tests for the synthetic CAN simulator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aggregator import merge_frames
from can_simulator import SIGNAL_SPECS
from fault_injector import inject_faults
from normal_driving_simulator import generate_normal_frames
from run_simulation import run


class SimulatorTests(unittest.TestCase):
    def test_normal_frames_are_deterministic_and_bounded(self) -> None:
        first = generate_normal_frames(12, 10, 7)
        self.assertEqual(first, generate_normal_frames(12, 10, 7))
        self.assertEqual(len(first), 12 * 10 * 4)
        for frame in first:
            _, _, lower, upper = SIGNAL_SPECS[frame["signal"]]
            self.assertGreaterEqual(frame["value"], lower)
            self.assertLessEqual(frame["value"], upper)
            self.assertEqual(len(bytes.fromhex(frame["data_hex"])), frame["dlc"])

        speeds = [frame["value"] for frame in first if frame["signal"] == "vehicle_speed_kph"]
        rpms = [frame["value"] for frame in first if frame["signal"] == "engine_rpm"]
        mean_speed, mean_rpm = sum(speeds) / len(speeds), sum(rpms) / len(rpms)
        covariance = sum((speed - mean_speed) * (rpm - mean_rpm) for speed, rpm in zip(speeds, rpms))
        self.assertGreater(covariance, 0)

    def test_faults_and_merge_are_valid(self) -> None:
        normal = generate_normal_frames(90, 10, 3)
        injected, truth = inject_faults(normal, 1.0, 9)
        self.assertTrue(injected)
        self.assertTrue(truth)
        self.assertEqual({event["attack_type"] for event in truth}, {"brake_at_speed", "rpm_spike"})
        merged = merge_frames(normal, injected)
        self.assertEqual(len(merged), len(normal) + len(injected))
        self.assertEqual([frame["timestamp_ms"] for frame in merged], sorted(frame["timestamp_ms"] for frame in merged))
        self.assertTrue(all("source" not in frame and "event_id" not in frame for frame in merged))

    def test_runner_writes_label_free_stream_and_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "without-attacks"
            metadata = run(output, 10, 10, 5, 0.0)
            self.assertEqual(metadata["injected_frame_count"], 0)
            stream = [json.loads(line) for line in (output / "stream.jsonl").read_text().splitlines()]
            self.assertTrue(stream)
            self.assertTrue(all("event_id" not in frame and "source" not in frame for frame in stream))
            self.assertEqual((output / "ground_truth.jsonl").read_text(), "")

    def test_runner_outputs_are_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first, second = root / "first", root / "second"
            run(first, 30, 10, 42, 0.02)
            run(second, 30, 10, 42, 0.02)
            for filename in ("normal_frames.jsonl", "injected_frames.jsonl", "stream.jsonl", "ground_truth.jsonl", "metadata.json"):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())


if __name__ == "__main__":
    unittest.main()
