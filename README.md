# FastF1 CAN anomaly dataset tools

This repository builds a reproducible Formula 1 telemetry dataset for a
next-timestep LSTM autoencoder and replays the resulting raw telemetry as
synthetic SocketCAN traffic. It uses FastF1 2024 Grand Prix race car data for
all available drivers and keeps only `RPM`, `Speed`, `Throttle`, and `nGear`.
The CAN layout is deliberately synthetic, not a vehicle-specific DBC.

## Build the dataset

Install the dataset and replay dependencies, then download/process the full
2024 season. FastF1's cache makes repeat runs reuse session downloads.

```bash
python3 -m pip install -r requirements.txt
python3 build_fastf1_dataset.py --output-dir data/fastf1_2024
```

For an extraction smoke test, use `--max-sessions 2`; it will not contain
enough group-diverse telemetry to guarantee the final 10,000 examples.

The completed output directory contains raw, unscaled data:

- `fastf1_telemetry.csv`: 10 Hz source records with race/driver provenance.
- `train_windows.npz`: 9,000 normal examples for the autoencoder.
- `test_windows.npz`: 1,000 external examples: 500 clean and 500 faulted.
- `test_clean_windows.npz`: the unmodified held-out comparison set.
- `manifest.json`: feature order, split groups, counts, and seed.

Every archive contains `features` and `history` shaped `[examples, 10, 4]`,
next-step `targets` shaped `[examples, 4]`, and raw `replay` rows shaped
`[examples, 11, 4]`. No scaling occurs here.

## Autoencoder integration

Clone and install the supplied
[time-series-autoencoder](https://github.com/JulesBelveze/time-series-autoencoder)
repository. Copy `model_integration/tsa/fastf1_dataset.py` into its `tsa/`
directory, copy the config into `configs/`, and copy `run_f1_anomaly.py` to its
repository root. Replace both `/replace/with/...` values in the config with
absolute archive paths, then run:

```bash
python3 run_f1_anomaly.py
```

The adapter performs the model repository's training-only standardization and
its internal 80/20 normal train/validation split. The script derives a 99th
percentile next-step MSE threshold from clean validation windows, then reports
false positives for the 500 clean external windows and detection metrics for
the 500 faulted windows.

## CAN replay

Create a `vcan0` interface on the Raspberry Pi, then replay a selected
held-out sequence. Use dry run first to write the exact frames without sending.

```bash
python3 -m f1_can.replay --archive data/fastf1_2024/test_windows.npz --sequence-index 0 --dry-run --output frames.jsonl
python3 -m f1_can.replay --archive data/fastf1_2024/test_windows.npz --sequence-index 0 --channel vcan0 --rate-hz 10
```

Frames use standard ID `0x100`: RPM `uint16`, speed `uint16` at 0.01 km/h,
throttle `uint8`, gear `uint8`, counter `uint8`, and one zero reserved byte.

## Tests

```bash
python3 -m unittest discover -v
```
