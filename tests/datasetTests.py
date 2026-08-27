import io
import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch, MagicMock

# Import your functions and constants from the source modules
from f1_can.prepareData import prepare_datasets
from f1_can.telemetry import _resample_car_data
import f1_can


# =====================================================================
# 1. FIXTURES & MOCKS
# =====================================================================

@pytest.fixture
def dummy_feature_cols(monkeypatch):
    """Mock FEATURE_COLUMNS so tests do not depend on external global configuration."""
    cols = ("RPM", "Speed", "Throttle", "nGear", "DeltaTime")
    monkeypatch.setattr(f1_can, "FEATURE_COLUMNS", cols)
    return cols


@pytest.fixture
def raw_car_data():
    """Generates synthetic FastF1 car telemetry DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="100ms")
    return pd.DataFrame({
        "Date": dates,
        "RPM": [10000 + i * 100 for i in range(10)],
        "Speed": [200 + i for i in range(10)],
        "Throttle": [50.0, 60.0, 105.0, -5.0, 70.0, 80.0, 90.0, 100.0, 0.0, 50.0],  # Includes invalid values
        "nGear": [1, 2, 3, -1, 4, 5, 6, 7, 8, 8],                                 # Includes invalid values
    })


@pytest.fixture
def synthetic_telemetry_csv(tmp_path, dummy_feature_cols):
    """Creates a temporary valid CSV file for dataset preparation tests."""
    csv_file = tmp_path / "test_telemetry.csv"
    
    # Build 10 segments to allow split partitioning (70/15/15)
    records = []
    dates = pd.date_range("2024-01-01", periods=30, freq="100ms")
    
    for seg_idx in range(10):
        seg_id = f"2024-R01-driver_{seg_idx}"
        for d in dates:
            records.append({
                "RPM": np.random.randint(8000, 12000),
                "Speed": np.random.uniform(150, 300),
                "Throttle": np.random.uniform(0, 100),
                "nGear": np.random.randint(1, 8),
                "DeltaTime": np.random.rand()/2,
                "segment_id": seg_id,
                "year": 2024,
                "round": 1,
                "event": "Test GP",
                "driver": f"driver_{seg_idx}"
            })
            
    df = pd.DataFrame(records)
    df.to_csv(csv_file, index=False)
    return csv_file


# =====================================================================
# 2. UNIT TESTS: _resample_car_data
# =====================================================================

class TestResampleCarData:
    
    def test_missing_required_columns_raises_value_error(self, raw_car_data):
        """Should raise ValueError when required feature columns are missing from input."""
        df_missing = raw_car_data.drop(columns=["RPM"])
        with pytest.raises(ValueError, match="FastF1 car data is missing columns"):
            _resample_car_data(df_missing)

    def test_throttle_and_gear_filtering(self, raw_car_data, dummy_feature_cols):
        """Ensures out-of-range Throttle (>100 or <0) and nGear (<0) are dropped."""
        resampled = _resample_car_data(raw_car_data)
        
        # Verify bounded conditions
        assert (resampled["Throttle"] <= 100).all()
        assert (resampled["Throttle"] >= 0).all()
        assert (resampled["nGear"] >= 0).all()
        assert len(resampled) < len(raw_car_data)

    def test_delta_time_calculation_and_first_row_drop(self, raw_car_data):
        """Validates that DeltaTime is computed accurately and first row with NaN is dropped."""
        
        raw_car_data.loc[3, "Throttle"] = 50
        raw_car_data.loc[2, "Throttle"] = 50
        raw_car_data.loc[3, "nGear"] = 2
        resampled = _resample_car_data(raw_car_data)
        
        
        assert "DeltaTime" in resampled.columns
        print(resampled)
        assert not resampled["DeltaTime"].isna().any()
        # Expect ~0.1s diff between consecutive 100ms timestamps
        np.testing.assert_allclose(resampled["DeltaTime"].values, 0.1, rtol=1e-3)

    def test_empty_dataframe_handling(self):
        """Checks graceful handling when receiving an empty DataFrame."""
        empty_df = pd.DataFrame(columns=["Date", "RPM", "Speed", "Throttle", "nGear"])
        result = _resample_car_data(empty_df)
        assert result.empty



# =====================================================================
# 3. UNIT TESTS: prepare_datasets
# =====================================================================

class TestPrepareDatasets:

    def test_dataset_split_and_dataloader_shapes(self, synthetic_telemetry_csv):
        """Verifies DataLoader creation, batch shapes, and tensor transpositions."""
        train_loader, val_loader, test_loader, scaler, X_val_t, X_test_t, X_train_t = prepare_datasets(
            str(synthetic_telemetry_csv)
        )

        # 1. Output types check
        assert isinstance(X_train_t, torch.Tensor)
        assert isinstance(X_val_t, torch.Tensor)
        assert isinstance(X_test_t, torch.Tensor)

        # 2. Shape verification: Conv1D requires (Batch, Features/Channels, Window_Size)
        num_features = 5
        window_size = 20
        assert X_train_t.shape[1] == num_features
        assert X_train_t.shape[2] == window_size


        # 3. DataLoader batch sampling test
        batch = next(iter(train_loader))[0]
        assert batch.shape[1] == num_features
        assert batch.shape[2] == window_size