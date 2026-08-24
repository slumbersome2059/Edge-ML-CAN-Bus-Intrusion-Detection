# Synthetic CAN Telemetry Simulator

This dependency-free test-data source produces plausible, correlated driving
telemetry and optional forged CAN-style frames. It is intentionally synthetic:
the arbitration IDs and payload encoding are stable test conventions, **not** a
vehicle-specific DBC or hardware interface.

## Quick start

```bash
python3 run_simulation.py --output-dir simulation_output
```

The defaults generate 300 seconds at 10 Hz with deterministic seed `42`.
Adjust `--duration`, `--sample-rate`, `--seed`, and `--injection-rate` as
needed. For example, a short test run with frequent attacks is:

```bash
python3 run_simulation.py --output-dir demo --duration 30 --injection-rate 0.02
```

The roles can also be run independently:

```bash
python3 normal_driving_simulator.py --output normal.jsonl
python3 fault_injector.py --input normal.jsonl --injected-output injected.jsonl --truth-output truth.jsonl
python3 aggregator.py --normal-input normal.jsonl --injected-input injected.jsonl --output stream.jsonl
```

## Outputs

- `normal_frames.jsonl`: baseline traffic from Agent 1.
- `injected_frames.jsonl`: only the forged traffic from Agent 2.
- `stream.jsonl`: timestamp-ordered merged detector input from Agent 3.
- `ground_truth.jsonl`: one labelled attack event per line, kept separate from telemetry.
- `metadata.json`: resolved configuration, seeds, and frame/event counts.

Each merged-stream telemetry line contains `timestamp_ms`, `arbitration_id`,
`dlc`, `data_hex`, `signal`, and `value`. The intermediate normal/injected
files additionally contain an internal `source` field. Payloads are four-byte,
little-endian unsigned integers. Values use these fixed synthetic conventions:

| Signal | CAN ID | Unit | Scale |
| --- | --- | --- | --- |
| `engine_rpm` | `0x0C0` | RPM | 0.25 RPM/bit |
| `vehicle_speed_kph` | `0x0D0` | km/h | 0.01 km/h/bit |
| `steering_angle_deg` | `0x0E0` | degrees | 0.1 degree/bit (offset +540) |
| `brake_pedal_pct` | `0x0F0` | percent | 0.1 percent/bit |

Injected attacks are either 80--100% brake commands while normal speed exceeds
40 km/h, or RPM values 1.8--2.5 times normal (capped at 8,000 RPM). They arrive
in bursts of two to four frames.

## Tests

```bash
python3 -m unittest -v
```
