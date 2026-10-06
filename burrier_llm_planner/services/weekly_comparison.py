"""Frozen weekly schedule references and daily operating-window comparisons."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from burrier_llm_planner.optimization.aemo_weekly import (
    AEMOWeeklyInputs,
    run_aemo_weekly_optimization,
)
from burrier_llm_planner.services.weekly_actuals import ActualPumpRun


ZONE = "Australia/Sydney"


@dataclass(frozen=True)
class FrozenSchedule:
    monday: date
    source_name: str
    issued_at: pd.Timestamp
    weekly_target_ml: float
    parameters: dict[str, float]
    frame: pd.DataFrame


@dataclass(frozen=True)
class DailyWindowComparison:
    day: date
    timestamps: pd.DatetimeIndex
    comparison_label: str
    ideal_label: str
    price_source: str
    window_match_percent: float | None
    comparison_hours: float
    reference_hours: float
    ideal_hours: float
    comparison_volume_ml: float
    reference_volume_ml: float
    ideal_volume_ml: float
    volume_delta_ml: float
    comparison_cost_per_ml: float | None
    ideal_cost_per_ml: float | None
    cost_per_ml_delta: float | None
    hours_moved: float
    comparison_cost_aud: float
    ideal_cost_aud: float
    cost_delta_aud: float
    hours_delta: float
    potential_saving_aud: float
    potential_saving_percent: float | None
    comparison_states: np.ndarray
    ideal_states: np.ndarray


def _reference_paths(root: Path, monday: date) -> tuple[Path, Path]:
    directory = Path(root) / monday.isoformat()
    return directory / "metadata.json", directory / "schedule.csv"


def save_frozen_schedule(root: Path, schedule: FrozenSchedule) -> Path:
    """Persist once. A released schedule is never overwritten by later runs."""

    metadata_path, frame_path = _reference_paths(root, schedule.monday)
    if metadata_path.is_file() and frame_path.is_file():
        return metadata_path.parent
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.frame.to_csv(frame_path, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    metadata = {
        "monday": schedule.monday.isoformat(),
        "source_name": schedule.source_name,
        "issued_at": pd.Timestamp(schedule.issued_at).isoformat(),
        "weekly_target_ml": float(schedule.weekly_target_ml),
        "parameters": {key: float(value) for key, value in schedule.parameters.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path.parent


def frozen_schedule_from_result(
    *, monday: date, source_name: str, issued_at: pd.Timestamp,
    weekly_target_ml: float, parameters: dict[str, float], result,
) -> FrozenSchedule:
    return FrozenSchedule(
        monday=monday,
        source_name=source_name,
        issued_at=pd.Timestamp(issued_at),
        weekly_target_ml=float(weekly_target_ml),
        parameters=dict(parameters),
        frame=pd.DataFrame({
            "DateTime": pd.DatetimeIndex(result.timestamps),
            "Price": np.asarray(result.price_aud_per_mwh, dtype=float),
            "PumpOn": np.asarray(result.pump_on, dtype=float),
            "IntervalVolumeML": np.asarray(result.interval_volume_ml, dtype=float),
        }),
    )


def load_frozen_schedule(root: Path, monday: date) -> FrozenSchedule | None:
    metadata_path, frame_path = _reference_paths(root, monday)
    if not metadata_path.is_file() or not frame_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(frame_path)
    frame["DateTime"] = pd.to_datetime(frame["DateTime"], utc=True).dt.tz_convert(ZONE)
    for column in ("Price", "PumpOn", "IntervalVolumeML"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return FrozenSchedule(
        monday=date.fromisoformat(metadata["monday"]),
        source_name=str(metadata["source_name"]),
        issued_at=pd.Timestamp(metadata["issued_at"]),
        weekly_target_ml=float(metadata["weekly_target_ml"]),
        parameters={key: float(value) for key, value in metadata["parameters"].items()},
        frame=frame,
    )


def window_match_percent(first, second) -> float | None:
    first_on = np.asarray(first, dtype=float) > 0.5
    second_on = np.asarray(second, dtype=float) > 0.5
    if len(first_on) != len(second_on):
        raise ValueError("Window comparison requires equal-length schedules.")
    union = np.logical_or(first_on, second_on).sum()
    if union == 0:
        return None
    return float(np.logical_and(first_on, second_on).sum() / union * 100.0)


def schedule_runs_for_day(schedule: FrozenSchedule, day: date) -> list[ActualPumpRun]:
    frame = schedule.frame.copy()
    times = pd.DatetimeIndex(frame["DateTime"])
    local_times = times.tz_localize(ZONE) if times.tz is None else times.tz_convert(ZONE)
    selected = frame.loc[np.asarray(local_times.date) == day].copy()
    if selected.empty:
        return []
    selected_times = pd.DatetimeIndex(selected["DateTime"])
    states = selected["PumpOn"].to_numpy(float) > 0.5
    changes = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
    runs: list[ActualPumpRun] = []
    for first, stop in zip(changes[:-1], changes[1:]):
        if not states[first]:
            continue
        end = (
            selected_times[stop] if stop < len(selected_times)
            else selected_times[-1] + pd.Timedelta(minutes=30)
        )
        runs.append(ActualPumpRun(start=selected_times[first], end=end))
    return runs


def states_for_runs(
    timestamps: pd.DatetimeIndex,
    runs: list[ActualPumpRun] | tuple[ActualPumpRun, ...],
) -> np.ndarray:
    states = np.zeros(len(timestamps), dtype=float)
    for index, interval_start in enumerate(pd.DatetimeIndex(timestamps)):
        interval_end = interval_start + pd.Timedelta(minutes=30)
        if any(run.start < interval_end and run.end > interval_start for run in runs):
            states[index] = 1.0
    return states


def _pump_cost(states: np.ndarray, prices: np.ndarray, power_kw: float) -> float:
    return float(np.sum(states * prices * power_kw / 1000.0 * 0.5))


def compare_daily_windows(
    *,
    day: date,
    timestamps: pd.DatetimeIndex,
    prices,
    comparison_states,
    ideal_target_states=None,
    fixed_ideal_states=None,
    comparison_label: str,
    ideal_label: str = "Price ideal",
    price_source: str,
    flow_lps: float,
    power_kw: float,
    minimum_continuous_run_hours: float,
    minimum_off_hours: float,
    minimum_start_interval_hours: float,
    comparison_cost_override_aud: float | None = None,
) -> DailyWindowComparison:
    timestamps = pd.DatetimeIndex(timestamps)
    prices = np.asarray(prices, dtype=float)
    comparison_states = (np.asarray(comparison_states, dtype=float) > 0.5).astype(float)
    target_states = (
        comparison_states.copy() if ideal_target_states is None
        else (np.asarray(ideal_target_states, dtype=float) > 0.5).astype(float)
    )
    if not (len(timestamps) == len(prices) == len(comparison_states) == len(target_states)):
        raise ValueError("Daily timestamps, prices, and schedule must have equal lengths.")
    comparison_hours = float(comparison_states.sum() * 0.5)
    reference_hours = float(target_states.sum() * 0.5)
    interval_volume_ml = flow_lps * 0.5 * 3600.0 / 1_000_000.0
    comparison_volume_ml = float(comparison_states.sum() * interval_volume_ml)
    reference_volume_ml = float(target_states.sum() * interval_volume_ml)
    if fixed_ideal_states is not None:
        ideal_states = (np.asarray(fixed_ideal_states, dtype=float) > 0.5).astype(float)
        if len(ideal_states) != len(timestamps):
            raise ValueError("Fixed ideal schedule must match daily timestamps.")
    elif target_states.sum() == 0:
        ideal_states = np.zeros(len(timestamps), dtype=float)
    else:
        result = run_aemo_weekly_optimization(AEMOWeeklyInputs(
            timestamps=timestamps,
            price_aud_per_mwh=prices,
            weekly_target_ml=reference_volume_ml,
            flow_lps=flow_lps,
            max_power_kw=power_kw,
            standby_power_kw=0.0,
            minimum_continuous_run_hours=min(
                minimum_continuous_run_hours, reference_hours,
            ),
            minimum_off_hours=minimum_off_hours,
            minimum_start_interval_hours=minimum_start_interval_hours,
            minimum_daily_run_hours=0.0,
            constrain_total_volume=True,
            prorate_partial_days=True,
        ))
        ideal_states = (np.asarray(result.pump_on) > 0.5).astype(float)
    ideal_hours = float(ideal_states.sum() * 0.5)
    ideal_volume_ml = float(ideal_states.sum() * interval_volume_ml)
    comparison_cost = (
        _pump_cost(comparison_states, prices, power_kw)
        if comparison_cost_override_aud is None else float(comparison_cost_override_aud)
    )
    ideal_cost = _pump_cost(ideal_states, prices, power_kw)
    comparison_cost_per_ml = (
        comparison_cost / comparison_volume_ml if comparison_volume_ml > 0 else None
    )
    ideal_cost_per_ml = ideal_cost / ideal_volume_ml if ideal_volume_ml > 0 else None
    cost_per_ml_delta = (
        comparison_cost_per_ml - ideal_cost_per_ml
        if comparison_cost_per_ml is not None and ideal_cost_per_ml is not None
        else None
    )
    saving = comparison_cost - ideal_cost
    saving_percent = (
        float(saving / abs(comparison_cost) * 100.0)
        if abs(comparison_cost) >= 1.0 else None
    )
    return DailyWindowComparison(
        day=day, timestamps=timestamps,
        comparison_label=comparison_label,
        ideal_label=ideal_label,
        price_source=price_source,
        window_match_percent=window_match_percent(comparison_states, ideal_states),
        comparison_hours=comparison_hours,
        reference_hours=reference_hours,
        ideal_hours=ideal_hours,
        comparison_volume_ml=comparison_volume_ml,
        reference_volume_ml=reference_volume_ml,
        ideal_volume_ml=ideal_volume_ml,
        volume_delta_ml=comparison_volume_ml - ideal_volume_ml,
        comparison_cost_per_ml=comparison_cost_per_ml,
        ideal_cost_per_ml=ideal_cost_per_ml,
        cost_per_ml_delta=cost_per_ml_delta,
        hours_moved=float(np.logical_xor(
            comparison_states > 0.5, ideal_states > 0.5,
        ).sum() * 0.5),
        comparison_cost_aud=comparison_cost,
        ideal_cost_aud=ideal_cost,
        cost_delta_aud=comparison_cost - ideal_cost,
        hours_delta=comparison_hours - ideal_hours,
        potential_saving_aud=saving,
        potential_saving_percent=saving_percent,
        comparison_states=comparison_states,
        ideal_states=ideal_states,
    )
