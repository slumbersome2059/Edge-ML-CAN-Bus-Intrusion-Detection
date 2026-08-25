
## Summary

Do not pre-scale the exported data. The 90% normal-only dataset will be passed raw to the target repository, which performs its own training-only standardization and internal 80/20 train/validation split. The remaining 10% will be an external anomaly-detection test set: 500 clean sequences and 500 faulted sequences.


## Implementation changes

- Extract 2024 Grand Prix race telemetry for all drivers using FastF1; retain only `RPM`, `Speed`, `Throttle`, and `nGear`, resampled to 10 Hz and kept separate by race-driver segment.
    
- Build exactly 10,000 overlapping `[10,4]` windows without crossing segment boundaries:
    
    - 9,000 clean normal windows exported as the raw model-input CSV/NPZ dataset.
    - 1,000 held-out windows from distinct race-driver segments: 500 unchanged clean examples and 500 seeded mixed-fault examples.
- Inject a balanced mix of RPM spike, speed freeze/offset, throttle-stuck, and gear-spoof faults only into the external held-out set. Preserve clean originals, binary labels, fault type, and source metadata separately from model input.
    
- Add a Hydra config and minimal loader/evaluation extension for the target project:
    
    - Train on only the 9,000 clean windows.
    - Allow the repository’s existing loader to perform its normal 80/20 chronological split and `StandardScaler` fitting within that 90% dataset.
    - Reuse that fitted preprocessor for external clean/faulted test telemetry; do not refit or scale from the holdout data.
    - Calculate per-window MSE reconstruction/prediction errors on the internal clean validation split and set an anomaly threshold to its 99th percentile.
- Add a SocketCAN sender/decoder:
    
    - Emit one standard 8-byte frame per telemetry timestep on ID `0x100`.
    - Pack RPM, speed, throttle, gear, and a rolling counter using documented fixed scales.
    - Support `vcan0`, dry-run/log export, configurable replay rate, and replay of either clean or faulted held-out sequences without exposing labels in the payload.

## Test plan

- Confirm 10,000 valid windows, `[10,4]` feature shape/order, and no race-driver overlap between normal and held-out partitions.
- Confirm the model receives raw values and only its existing training-side scaler is used.
- Confirm exactly 500 clean and 500 anomalous external test sequences, deterministic faults, and correct labels/metadata.
- Confirm threshold derives solely from internal clean validation errors; verify clean holdout false-positive rate and fault detection precision, recall, F1, and confusion matrix.
- Round-trip test CAN payload encoding/decoding and SocketCAN dry-run ordering.

## Assumptions

- The target project’s internal 80/20 split serves as normal-model validation, while the external 10% is reserved solely for anomaly evaluation.
- The 99th-percentile clean-validation MSE is the initial unsupervised threshold; it can later be tuned from the reported clean-holdout false-positive rate.