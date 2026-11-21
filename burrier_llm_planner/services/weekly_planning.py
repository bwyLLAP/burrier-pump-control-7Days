"""Monday forecast snapshots and local-time weekly planning inputs."""
from __future__ import annotations

from datetime import date, timezone, timedelta
import hashlib
import io

import numpy as np
import pandas as pd

from .pd7day_price import PD7DayPriceWindow, parse_pd7day_csv, parse_pd7day_zip

NEM_ZONE = timezone(timedelta(hours=10))
LOCAL_ZONE = "Australia/Sydney"
PRICE_FLOOR_AUD_PER_MWH = -100.0
PRICE_CAP_AUD_PER_MWH = 200.0


def containing_monday(selected_date: date) -> date:
    return selected_date - timedelta(days=selected_date.weekday())


def cap_weekly_prices(frame: pd.DataFrame) -> pd.DataFrame:
    capped = frame.copy()
    capped["Price"] = pd.to_numeric(capped["Price"], errors="raise").clip(
        PRICE_FLOOR_AUD_PER_MWH, PRICE_CAP_AUD_PER_MWH
    )
    return capped


def week_intervals(monday: date) -> pd.DatetimeIndex:
    if monday.weekday() != 0:
        raise ValueError("Select a Monday as the week start.")
    start = pd.Timestamp(monday).tz_localize(LOCAL_ZONE)
    end = pd.Timestamp(monday + timedelta(days=7)).tz_localize(LOCAL_ZONE)
    return pd.date_range(start, end, freq="30min", inclusive="left")


def select_week(window: PD7DayPriceWindow, monday: date, *, rolling: bool = False) -> pd.DataFrame:
    """AEMO interval timestamps denote interval ends in fixed AEST."""
    run = pd.Timestamp(window.run_datetime)
    run = run.tz_localize(NEM_ZONE) if run.tzinfo is None else run.tz_convert(NEM_ZONE)
    expected = week_intervals(monday)
    local_run = run.tz_convert(LOCAL_ZONE)
    if rolling:
        if local_run.date() != monday:
            raise ValueError("A rolling forecast must have been issued on the selected Monday.")
    elif not expected[0] - pd.Timedelta(days=1) <= local_run <= expected[0]:
        raise ValueError(
            "The forecast must have been issued during the day before the selected "
            "calendar week begins."
        )
    raw = window.frame.copy()
    index = pd.DatetimeIndex(raw["DateTime"])
    index = index.tz_localize(NEM_ZONE) if index.tz is None else index.tz_convert(NEM_ZONE)
    starts = (index - pd.Timedelta(minutes=30)).tz_convert(LOCAL_ZONE)
    if rolling:
        available = starts[(starts.date == monday) & (starts >= local_run)]
        if len(available) == 0:
            raise ValueError("The forecast has no usable Monday intervals after publication.")
        # A late Monday publication shortens the usable calendar week. Never roll
        # the missing Monday morning intervals into the following Monday.
        expected = expected[expected >= available.min()]
    if starts.has_duplicates:
        raise ValueError("The forecast contains duplicate intervals.")
    series = pd.Series(raw["Price"].to_numpy(float), index=starts)
    aligned = series.reindex(expected)
    missing = aligned[aligned.isna()]
    if len(missing):
        raise ValueError(
            f"Forecast does not cover the requested seven-day horizon: {len(missing)} "
            f"missing half-hours; first missing {missing.index[0]:%a %d %b %H:%M %Z}. "
            "Upload a complete Monday snapshot. Missing prices are not filled."
        )
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("Forecast prices must be finite.")
    return cap_weekly_prices(
        pd.DataFrame({"DateTime": expected, "Price": aligned.to_numpy()})
    )


def read_snapshot(content: bytes, filename: str) -> PD7DayPriceWindow:
    if content[:2] == b"PK":
        return parse_pd7day_zip(content, source_name=filename)
    if b"PRICESOLUTION" in content:
        return parse_pd7day_csv(content, source_name=filename)
    frame = pd.read_csv(io.BytesIO(content))
    required = {"DateTime", "Price", "ForecastRun"}
    if not required.issubset(frame.columns):
        raise ValueError("Snapshot CSV requires DateTime (interval end, AEST), Price, ForecastRun.")
    if frame["ForecastRun"].isna().any() or frame["ForecastRun"].nunique() != 1:
        raise ValueError("Snapshot must contain exactly one forecast run.")
    frame["DateTime"] = pd.to_datetime(frame["DateTime"], errors="raise")
    frame["Price"] = pd.to_numeric(frame["Price"], errors="raise")
    return PD7DayPriceWindow(frame, pd.Timestamp(frame["ForecastRun"].iloc[0]), filename,
                            pd.Timestamp.now(tz="UTC").to_pydatetime())


def snapshot_csv(window: PD7DayPriceWindow) -> bytes:
    frame = window.frame.loc[:, ["DateTime", "Price"]].copy()
    frame["ForecastRun"] = str(window.run_datetime)
    return frame.to_csv(index=False).encode("utf-8-sig")


def fingerprint(window, monday, parameters) -> str:
    return hashlib.sha256(snapshot_csv(window) + str((monday, parameters)).encode()).hexdigest()
