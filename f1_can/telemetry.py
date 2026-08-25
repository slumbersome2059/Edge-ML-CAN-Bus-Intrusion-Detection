"""FastF1 extraction and 10 Hz per-driver race telemetry resampling."""

from __future__ import annotations

from pathlib import Path

from . import FEATURE_COLUMNS


def _resample_car_data(car_data, *, sample_rate_hz: int):
    """Return valid car data resampled without combining separate drivers."""
    import pandas as pd

    required = ["Date", "SessionTime", *FEATURE_COLUMNS]
    missing = set(required).difference(car_data.columns)
    if missing:
        raise ValueError(f"FastF1 car data is missing columns: {sorted(missing)}")
    frame = car_data.loc[:, required].copy().dropna(subset=["Date", *FEATURE_COLUMNS])
    frame = frame[(frame["Throttle"] <= 100) & (frame["Throttle"] >= 0) & (frame["nGear"] >= 0)]
    frame = frame.drop_duplicates(subset="Date").sort_values("Date").set_index("Date")
    if frame.empty:
        return frame.reset_index()
    cadence = pd.Timedelta(seconds=1 / sample_rate_hz)
    signals = frame.loc[:, list(FEATURE_COLUMNS)].resample(cadence).asfreq()
    signals[["RPM", "Speed", "Throttle"]] = signals[["RPM", "Speed", "Throttle"]].interpolate(
        method="time", limit_direction="both"
    )
    signals["nGear"] = signals["nGear"].ffill().bfill()
    session_time = frame["SessionTime"].resample(cadence).asfreq().interpolate(method="time", limit_direction="both")
    return signals.assign(SessionTime=session_time).dropna().reset_index()


def extract_2024_races(output: Path, cache_dir: Path, *, year: int = 2024, sample_rate_hz: int = 10,
                       max_sessions: int | None = None) -> int:
    """Download completed race telemetry for every listed driver and write CSV.

    FastF1 data is cached locally, so rerunning safely reuses downloaded session
    data. Individual unavailable sessions or drivers are skipped with a message.
    """
    try:
        import fastf1
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("install requirements.txt before extracting FastF1 telemetry") from exc
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    cache_dir.mkdir(parents=True, exist_ok=True)
    #I don't think caching is really necessary because we create the data only once
    #we should never be making repeated calls
    fastf1.Cache.enable_cache(str(cache_dir))
    schedule = fastf1.get_event_schedule(year, include_testing=False)#include_testing is testing sessions of races
    #EventSchedule object, you can access info about different events from here
    rows = []
    completed = 0#number of completed sessions
    for _, event in schedule.iterrows():
        try:
            round_number = int(event["RoundNumber"])
        except (KeyError, TypeError, ValueError):
            continue
        if round_number <= 0:
            continue
        if max_sessions is not None and completed >= max_sessions:
            break
        try:
            session = fastf1.get_session(year, round_number, "R")# There are many sessions in one round(practice, qualifying) -> R is for race, main race(final one after all the qualifiers)
            #round is basically like a grandprix
            session.load(telemetry=True, weather=False, messages=False)
            #before you access any data from session so you do something like session.laps or something you need to do session.load()
        except Exception as exc:  # FastF1 availability varies by session
            print(f"Skipping round {round_number}: {exc}")
            continue
        completed += 1
        for driver in session.drivers:
            try:
                laps = session.laps.pick_drivers(driver)
                segment = _resample_car_data(laps.get_car_data(), sample_rate_hz=sample_rate_hz)
            except Exception as exc:
                print(f"Skipping {event['EventName']} driver {driver}: {exc}")
                continue
            if len(segment) < 12:
                continue
            segment_id = f"{year}-R{round_number:02d}-{driver}"
            segment["segment_id"] = segment_id
            segment["year"] = year
            segment["round"] = round_number
            segment["event"] = event["EventName"]
            segment["driver"] = str(driver)
            rows.append(segment)
    if not rows:
        raise RuntimeError("no valid FastF1 telemetry was extracted")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(output, index=False)
    return len(result)
