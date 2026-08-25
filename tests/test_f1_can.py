from __future__ import annotations

import random
import unittest

import numpy as np

from f1_can.can_frames import CAN_DLC, decode_frame, encode_frame
from f1_can.dataset import _split_groups, make_examples
from f1_can.faults import FAULT_TYPES, inject_balanced_faults, inject_fault


class CanFrameTests(unittest.TestCase):
    def test_frame_round_trip(self) -> None:
        payload = encode_frame({"RPM": 12345, "Speed": 281.23, "Throttle": 76, "nGear": 7}, counter=260)
        self.assertEqual(len(payload), CAN_DLC)
        self.assertEqual(decode_frame(payload), {"RPM": 12345, "Speed": 281.23, "Throttle": 76, "nGear": 7, "counter": 4})

    def test_frame_rejects_invalid_signal(self) -> None:
        with self.assertRaises(ValueError):
            encode_frame({"RPM": -1, "Speed": 0, "Throttle": 0, "nGear": 0})


class DatasetTests(unittest.TestCase):
    def test_examples_match_autoencoder_tensor_contract(self) -> None:
        rows = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
        examples = make_examples("2024-R01-1", rows, seq_length=10)
        self.assertEqual(len(examples), 9)
        self.assertEqual(examples[0].features.shape, (10, 4))
        self.assertTrue(np.array_equal(examples[0].history, rows[:10]))
        self.assertTrue(np.array_equal(examples[0].features, rows[1:11]))
        self.assertTrue(np.array_equal(examples[0].target, rows[11]))

    def test_group_split_is_deterministic_and_disjoint(self) -> None:
        group = make_examples("x", np.ones((15, 4), dtype=np.float32))
        groups = {f"segment-{index}": group for index in range(10)}
        train, test = _split_groups(groups, 0.1, 42)
        self.assertEqual(len(test), 1)
        self.assertFalse(train.intersection(test))
        self.assertEqual(train.union(test), set(groups))


class FaultTests(unittest.TestCase):
    def test_each_fault_changes_final_target_region(self) -> None:
        sequence = np.tile(np.array([6000, 180, 50, 5], dtype=np.float32), (11, 1))
        for fault in FAULT_TYPES:
            changed = inject_fault(sequence, fault, random.Random(5))
            self.assertFalse(np.array_equal(changed[-1], sequence[-1]), fault)
            self.assertTrue(np.array_equal(changed[:8], sequence[:8]), fault)

    def test_balanced_fault_selection_is_reproducible(self) -> None:
        sequences = np.tile(np.array([6000, 180, 50, 5], dtype=np.float32), (12, 11, 1))
        first, first_labels = inject_balanced_faults(sequences, 8, 7)
        second, second_labels = inject_balanced_faults(sequences, 8, 7)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(first_labels, second_labels))
        self.assertEqual(int(first_labels[:, 0].astype(np.int8).sum()), 8)
        self.assertEqual(set(first_labels[first_labels[:, 0] == "1", 1]), set(FAULT_TYPES))


if __name__ == "__main__":
    unittest.main()
