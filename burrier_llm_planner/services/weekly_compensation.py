"""Pure business logic for selectable actual-versus-plan compensation windows."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd


ZONE = "Australia/Sydney"


@dataclass(frozen=True)
class CompensationDaySummary:
    day: date
    available_hours: float
    mean_price_aud_per_mwh: float | None
    low_price_window_hours: float
    low_price_start: pd.Timestamp | None
    low_price_end: pd.Timestamp | None


@dataclass(frozen=True)
class CompensationPlan:
    target_steps: int
    base_steps: int
    adjustment_steps: int
    minimum_steps_by_date: dict[date, int]
    maximum_steps_by_date: dict[date, int]


@dataclass(frozen=True)
class CompensationHorizon:
    timestamps: pd.DatetimeIndex
    allocation_prices: np.ndarray
    operating_prices: np.ndarray


@dataclass(frozen=True)
class PreparedCompensation:
    horizon: CompensationHorizon
    base_pump_on: np.ndarray
    plan: CompensationPlan


@dataclass(frozen=True)
class RollingBaseDefinition:
    """Constraints for extending the current-week plan through the rolling horizon."""

    fixed_pump_on: np.ndarray
    target_steps: int
    current_week_steps: int
    tail_target_steps: int
    minimum_steps_by_date: dict[date, int]


def _local_index(values) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    return index.tz_localize(ZONE) if index.tz is None else index.tz_convert(ZONE)


def _longest_low_price_window(
    timestamps: pd.DatetimeIndex, prices: np.ndarray,
) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    if len(timestamps) == 0:
        return 0.0, None, None
    lower = float(np.min(prices))
    upper = float(np.max(prices))
    cutoff = lower + 0.25 * (upper - lower)
    low = prices <= cutoff + 1e-9
    best_start = best_end = None
    run_start = None
    for index, is_low in enumerate(np.r_[low, False]):
        if is_low and run_start is None:
            run_start = index
        elif not is_low and run_start is not None:
            contiguous = all(
                timestamps[offset] - timestamps[offset - 1] == pd.Timedelta(minutes=30)
                for offset in range(run_start + 1, index)
            )
            if contiguous and (best_start is None or index - run_start > best_end - best_start):
                best_start, best_end = run_start, index
            run_start = None
    if best_start is None:
        return 0.0, None, None
    return (
        (best_end - best_start) * 0.5,
        timestamps[best_start],
        timestamps[best_end - 1] + pd.Timedelta(minutes=30),
    )


def compensation_day_summaries(
    price_frame: pd.DataFrame, now: pd.Timestamp, *, day_count: int = 7,
) -> list[CompensationDaySummary]:
    """Summarise D+1 to D+7 prices for operator compensation-date selection."""

    current = pd.Timestamp(now)
    current = current.tz_localize(ZONE) if current.tzinfo is None else current.tz_convert(ZONE)
    frame = price_frame.loc[:, ["DateTime", "Price"]].copy()
    frame["DateTime"] = _local_index(frame["DateTime"])
    frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
    frame = frame.dropna(subset=["Price"]).sort_values("DateTime")
    summaries: list[CompensationDaySummary] = []
    for offset in range(1, day_count + 1):
        day = current.date() + timedelta(days=offset)
        selected = frame[np.asarray([stamp.date() == day for stamp in frame["DateTime"]])]
        times = pd.DatetimeIndex(selected["DateTime"])
        prices = selected["Price"].to_numpy(float)
        low_hours, low_start, low_end = _longest_low_price_window(times, prices)
        summaries.append(CompensationDaySummary(
            day=day,
            available_hours=len(selected) * 0.5,
            mean_price_aud_per_mwh=(float(np.mean(prices)) if len(prices) else None),
            low_price_window_hours=low_hours,
            low_price_start=low_start,
            low_price_end=low_end,
        ))
    return summaries


def build_compensation_plan(
    *, timestamps, base_pump_on, selected_dates: set[date],
    variance_ml: float, interval_volume_ml: float,
) -> CompensationPlan:
    """Keep base daily volume fixed except where the operator permits correction."""

    index = _local_index(timestamps)
    states = np.rint(np.asarray(base_pump_on, dtype=float)).astype(int)
    if len(index) == 0 or len(states) != len(index):
        raise ValueError("Compensation requires matching non-empty timestamps and base states.")
    if interval_volume_ml <= 0 or not np.isfinite([variance_ml, interval_volume_ml]).all():
        raise ValueError("Compensation volumes must be finite and interval volume must be positive.")
    days = list(pd.unique(index.date))
    capacities = {day: int(np.sum(index.date == day)) for day in days}
    base = {day: int(states[np.asarray(index.date == day)].sum()) for day in days}
    base_steps = int(states.sum())
    desired_volume = max(0.0, base_steps * interval_volume_ml - float(variance_ml))
    target_steps = int(np.ceil(desired_volume / interval_volume_ml - 1e-9))
    adjustment_steps = target_steps - base_steps
    minimum: dict[date, int] = {}
    maximum: dict[date, int] = {}
    for day in days:
        base_day = base[day]
        if day not in selected_dates or adjustment_steps == 0:
            minimum[day] = maximum[day] = base_day
        elif adjustment_steps > 0:
            minimum[day], maximum[day] = base_day, capacities[day]
        else:
            minimum[day], maximum[day] = 0, base_day
    if adjustment_steps > 0:
        add_capacity = sum(maximum[day] - base[day] for day in days)
        if add_capacity < adjustment_steps:
            raise ValueError("Selected compensation dates cannot absorb the pumping deficit.")
    elif adjustment_steps < 0:
        removal_capacity = sum(base[day] - minimum[day] for day in days)
        if removal_capacity < -adjustment_steps:
            raise ValueError("Selected compensation dates cannot absorb the pumping surplus.")
    return CompensationPlan(
        target_steps=target_steps,
        base_steps=base_steps,
        adjustment_steps=adjustment_steps,
        minimum_steps_by_date=minimum,
        maximum_steps_by_date=maximum,
    )


def build_compensation_horizon(
    display_prices: pd.DataFrame,
    allocation_prices: pd.DataFrame,
    now: pd.Timestamp,
    *,
    day_count: int = 7,
) -> CompensationHorizon:
    """Return the continuous available horizon through D+7.

    Seven-day prices allocate volume.  Only D+1 receives the latest short-term
    price overlay when operating windows are refined.
    """

    current = pd.Timestamp(now)
    current = current.tz_localize(ZONE) if current.tzinfo is None else current.tz_convert(ZONE)
    start = current if current == current.floor("30min") else current.ceil("30min")
    requested_end = current.normalize() + pd.Timedelta(days=day_count + 1)

    display = display_prices.loc[:, ["DateTime", "Price"]].copy()
    display["DateTime"] = _local_index(display["DateTime"])
    display["Price"] = pd.to_numeric(display["Price"], errors="coerce")
    display = display.drop_duplicates("DateTime", keep="last").set_index("DateTime").sort_index()
    available = display.loc[(display.index >= start) & (display.index < requested_end)]
    if available.empty:
        raise ValueError("No future prices are available for compensation planning.")
    end = min(requested_end, available.index.max() + pd.Timedelta(minutes=30))
    expected = pd.date_range(start, end, freq="30min", inclusive="left")
    displayed = display.reindex(expected)
    missing_positions = np.flatnonzero(displayed["Price"].isna().to_numpy())
    if len(missing_positions):
        expected = expected[:int(missing_positions[0])]
        displayed = displayed.iloc[:int(missing_positions[0])]
    if len(expected) == 0:
        raise ValueError("No continuous future prices are available for compensation planning.")

    allocation = allocation_prices.loc[:, ["DateTime", "Price"]].copy()
    allocation["DateTime"] = _local_index(allocation["DateTime"])
    allocation["Price"] = pd.to_numeric(allocation["Price"], errors="coerce")
    allocation = allocation.drop_duplicates("DateTime", keep="last").set_index("DateTime").sort_index()
    allocated = allocation.reindex(expected)["Price"].fillna(displayed["Price"])
    if allocated.isna().any():
        raise ValueError("The allocation forecast has an unresolved price gap.")
    operating = allocated.to_numpy(float).copy()
    tomorrow = current.date() + timedelta(days=1)
    tomorrow_mask = np.asarray(expected.date == tomorrow)
    operating[tomorrow_mask] = displayed.loc[tomorrow_mask, "Price"].to_numpy(float)
    return CompensationHorizon(
        timestamps=expected,
        allocation_prices=allocated.to_numpy(float),
        operating_prices=operating,
    )


def base_states_for_horizon(
    timestamps,
    *,
    original_times,
    original_pump_on,
) -> np.ndarray:
    """Project the frozen Original schedule onto a longer horizon; later dates are zero."""

    horizon = _local_index(timestamps)
    original_index = _local_index(original_times)
    states = np.rint(np.asarray(original_pump_on, dtype=float)).astype(int)
    if len(original_index) != len(states):
        raise ValueError("Original schedule timestamps and states must have equal length.")
    lookup = pd.Series(states, index=original_index)
    return lookup.reindex(horizon, fill_value=0).to_numpy(int)


def build_rolling_base_definition(
    *, timestamps, current_week_times, current_week_pump_on,
    current_week_end: pd.Timestamp, weekly_target_ml: float,
    interval_volume_ml: float, minimum_daily_run_hours: float,
) -> RollingBaseDefinition:
    """Lock this week's remaining plan and assign the balance to the horizon tail.

    The combined rolling base target is one weekly target.  Therefore the tail
    receives the weekly target minus the already scheduled current-week volume.
    """

    horizon = _local_index(timestamps)
    if len(horizon) == 0:
        raise ValueError("The rolling base horizon cannot be empty.")
    values = np.asarray(
        [weekly_target_ml, interval_volume_ml, minimum_daily_run_hours], dtype=float,
    )
    if not np.isfinite(values).all() or weekly_target_ml < 0 or interval_volume_ml <= 0:
        raise ValueError("Rolling base targets must be finite and non-negative.")
    week_end = pd.Timestamp(current_week_end)
    week_end = (
        week_end.tz_localize(ZONE) if week_end.tzinfo is None
        else week_end.tz_convert(ZONE)
    )
    projected = base_states_for_horizon(
        horizon,
        original_times=current_week_times,
        original_pump_on=current_week_pump_on,
    )
    current_mask = np.asarray(horizon < week_end)
    fixed = np.full(len(horizon), np.nan)
    fixed[current_mask] = projected[current_mask]
    current_steps = int(projected[current_mask].sum())
    requested_steps = int(np.ceil(float(weekly_target_ml) / interval_volume_ml - 1e-9))
    target_steps = max(current_steps, requested_steps)
    tail_steps = target_steps - current_steps
    tail_capacity = int((~current_mask).sum())
    if tail_steps > tail_capacity:
        raise ValueError("The rolling-horizon tail cannot absorb the weekly target.")

    minimum_steps: dict[date, int] = {}
    if minimum_daily_run_hours > 0:
        daily_steps = int(np.ceil(minimum_daily_run_hours / 0.5 - 1e-9))
        tail_dates = np.asarray(horizon.date)
        for day in pd.unique(tail_dates[~current_mask]):
            available = int(np.sum((tail_dates == day) & ~current_mask))
            minimum_steps[day] = int(np.ceil(daily_steps * min(1.0, available / 48)))
        if sum(minimum_steps.values()) > tail_steps:
            raise ValueError(
                "The rolling-horizon tail target is below the selected daily minimums."
            )
    return RollingBaseDefinition(
        fixed_pump_on=fixed,
        target_steps=target_steps,
        current_week_steps=current_steps,
        tail_target_steps=tail_steps,
        minimum_steps_by_date=minimum_steps,
    )


def fixed_states_outside_selected_dates(
    *, timestamps, base_pump_on, selected_dates: set[date],
) -> np.ndarray:
    """Lock every interval except dates explicitly selected by the operator."""

    index = _local_index(timestamps)
    states = np.rint(np.asarray(base_pump_on, dtype=float)).astype(float)
    if len(index) != len(states):
        raise ValueError("Adjustment timestamps and base states must have equal length.")
    fixed = states.copy()
    unlocked = np.asarray([stamp.date() in selected_dates for stamp in index])
    fixed[unlocked] = np.nan
    return fixed


def prepare_compensation(
    *, display_prices: pd.DataFrame, allocation_prices: pd.DataFrame,
    now: pd.Timestamp, original_times, original_pump_on,
    selected_dates: set[date], variance_ml: float, interval_volume_ml: float,
) -> PreparedCompensation:
    horizon = build_compensation_horizon(display_prices, allocation_prices, now)
    base = base_states_for_horizon(
        horizon.timestamps,
        original_times=original_times,
        original_pump_on=original_pump_on,
    )
    plan = build_compensation_plan(
        timestamps=horizon.timestamps,
        base_pump_on=base,
        selected_dates=selected_dates,
        variance_ml=variance_ml,
        interval_volume_ml=interval_volume_ml,
    )
    return PreparedCompensation(horizon=horizon, base_pump_on=base, plan=plan)
