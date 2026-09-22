from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


LOCAL_ZONE = ZoneInfo("Australia/Sydney")


@dataclass(frozen=True)
class ActualPumpRun:
    """An operator-entered pump run that has already occurred."""

    start: pd.Timestamp
    end: pd.Timestamp
    cost_override_aud: float | None = None


@dataclass(frozen=True)
class ActualRunSummary:
    rows: pd.DataFrame
    achieved_volume_ml: float
    actual_cost_aud: float


def replacement_dates_for_runs(
    runs: list[ActualPumpRun] | tuple[ActualPumpRun, ...],
) -> set[date]:
    """Return local calendar dates touched by explicitly entered actual runs."""

    dates: set[date] = set()
    for run in runs:
        start = _as_timestamp(run.start, "Actual run start").tz_convert(LOCAL_ZONE)
        end = _as_timestamp(run.end, "Actual run end").tz_convert(LOCAL_ZONE)
        if end <= start:
            raise ValueError("Actual run end must be after start.")
        last_instant = end - pd.Timedelta(nanoseconds=1)
        dates.update(
            stamp.date()
            for stamp in pd.date_range(start.normalize(), last_instant.normalize(), freq="D")
        )
    return dates


def planned_volume_for_dates(
    timestamps: pd.DatetimeIndex,
    interval_volume_ml,
    replaced_dates: set[date] | list[date] | tuple[date, ...],
) -> float:
    """Sum baseline volume only on dates explicitly replaced by operator actuals."""

    timestamps = pd.DatetimeIndex(timestamps)
    volumes = np.asarray(interval_volume_ml, dtype=float)
    if len(timestamps) != len(volumes):
        raise ValueError("Timestamps and interval volumes must have the same length.")
    if timestamps.tz is None:
        local_times = timestamps.tz_localize(LOCAL_ZONE)
    else:
        local_times = timestamps.tz_convert(LOCAL_ZONE)
    selected = set(replaced_dates)
    mask = np.asarray([stamp.date() in selected for stamp in local_times])
    return float(volumes[mask].sum())


def _as_timestamp(value: pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must include a timezone.")
    return timestamp


def _estimated_cost(
    run: ActualPumpRun,
    *,
    power_kw: float,
    price_frame: pd.DataFrame,
) -> float:
    if not {"DateTime", "Price"}.issubset(price_frame.columns):
        raise ValueError("Price data must contain DateTime and Price columns.")
    prices = price_frame[["DateTime", "Price"]].copy()
    prices["DateTime"] = pd.to_datetime(prices["DateTime"])
    prices = prices.sort_values("DateTime")
    if prices.empty:
        raise ValueError("Price data does not cover the actual pump run.")

    covered_seconds = 0.0
    cost = 0.0
    for row in prices.itertuples(index=False):
        interval_start = pd.Timestamp(row.DateTime)
        interval_end = interval_start + pd.Timedelta(minutes=30)
        overlap_start = max(run.start, interval_start)
        overlap_end = min(run.end, interval_end)
        overlap_seconds = max(0.0, (overlap_end - overlap_start).total_seconds())
        if overlap_seconds:
            price = float(row.Price)
            if not np.isfinite(price):
                raise ValueError("Price data must be finite.")
            hours = overlap_seconds / 3600.0
            cost += hours * power_kw / 1000.0 * price
            covered_seconds += overlap_seconds

    duration_seconds = (run.end - run.start).total_seconds()
    if abs(covered_seconds - duration_seconds) > 1e-6:
        raise ValueError("Price data does not cover the actual pump run.")
    return cost


def summarise_actual_runs(
    runs: list[ActualPumpRun] | tuple[ActualPumpRun, ...],
    *,
    horizon_start: pd.Timestamp,
    horizon_end: pd.Timestamp,
    now: pd.Timestamp,
    flow_lps: float,
    power_kw: float,
    price_frame: pd.DataFrame,
) -> ActualRunSummary:
    """Validate operator entries and calculate achieved water and incurred cost."""

    horizon_start = _as_timestamp(horizon_start, "Horizon start")
    horizon_end = _as_timestamp(horizon_end, "Horizon end")
    now = _as_timestamp(now, "Current time")
    if not np.isfinite([flow_lps, power_kw]).all() or flow_lps <= 0 or power_kw <= 0:
        raise ValueError("Flow and pump power must be finite and greater than zero.")

    ordered = sorted(runs, key=lambda item: pd.Timestamp(item.start))
    normalised: list[ActualPumpRun] = []
    previous_end: pd.Timestamp | None = None
    for item in ordered:
        start = _as_timestamp(item.start, "Actual run start")
        end = _as_timestamp(item.end, "Actual run end")
        run = ActualPumpRun(start=start, end=end, cost_override_aud=item.cost_override_aud)
        if end <= start:
            raise ValueError("Actual run end must be after start.")
        if start < horizon_start or end > horizon_end:
            raise ValueError("Actual run must be within the planning horizon.")
        if end > now:
            raise ValueError("Actual run cannot extend into the future.")
        if previous_end is not None and start < previous_end:
            raise ValueError("Actual pump runs cannot overlap.")
        if item.cost_override_aud is not None:
            override = float(item.cost_override_aud)
            if not np.isfinite(override) or override < 0:
                raise ValueError("Actual cost override must be finite and non-negative.")
        previous_end = end
        normalised.append(run)

    rows: list[dict[str, object]] = []
    for run in normalised:
        duration_hours = (run.end - run.start).total_seconds() / 3600.0
        volume_ml = duration_hours * 3600.0 * flow_lps / 1_000_000.0
        if run.cost_override_aud is None:
            cost = _estimated_cost(run, power_kw=power_kw, price_frame=price_frame)
            basis = "Forecast estimate"
        else:
            cost = float(run.cost_override_aud)
            basis = "Operator override"
        rows.append({
            "Start": run.start,
            "End": run.end,
            "Duration (h)": duration_hours,
            "Volume (ML)": volume_ml,
            "Cost (AUD)": cost,
            "Cost basis": basis,
        })

    columns = ["Start", "End", "Duration (h)", "Volume (ML)", "Cost (AUD)", "Cost basis"]
    frame = pd.DataFrame(rows, columns=columns)
    return ActualRunSummary(
        rows=frame,
        achieved_volume_ml=float(frame["Volume (ML)"].sum()) if not frame.empty else 0.0,
        actual_cost_aud=float(frame["Cost (AUD)"].sum()) if not frame.empty else 0.0,
    )


def build_execution_lock(
    timestamps: pd.DatetimeIndex,
    runs: list[ActualPumpRun] | tuple[ActualPumpRun, ...],
    now: pd.Timestamp,
) -> tuple[pd.Series, int]:
    """Lock elapsed half-hour intervals and return the first plannable index."""

    timestamps = pd.DatetimeIndex(timestamps)
    now = _as_timestamp(now, "Current time")
    boundary = now.ceil("30min")
    start_index = int(timestamps.searchsorted(boundary, side="left"))
    lock = pd.Series(np.nan, index=range(len(timestamps)), dtype=float)
    lock.iloc[:start_index] = 0.0
    for index in range(start_index):
        interval_start = timestamps[index]
        interval_end = interval_start + pd.Timedelta(minutes=30)
        if any(run.start < interval_end and run.end > interval_start for run in runs):
            lock.iloc[index] = 1.0
    return lock, start_index
