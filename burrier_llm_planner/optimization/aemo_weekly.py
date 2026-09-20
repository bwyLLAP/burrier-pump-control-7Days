from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aemo_daily import AEMODailyOptimizerError


@dataclass(frozen=True)
class AEMOWeeklyInputs:
    timestamps: pd.DatetimeIndex
    price_aud_per_mwh: np.ndarray
    weekly_target_ml: float = 280.0
    flow_lps: float = 1050.0
    max_power_kw: float = 1950.0
    standby_power_kw: float = 12.0
    minimum_continuous_run_hours: float = 4.0
    minimum_off_hours: float = 2.0
    minimum_start_interval_hours: float = 6.0
    minimum_daily_run_hours: float = 6.0
    initial_pump_on: bool = False
    elapsed_state_steps: int = 8
    constrain_total_volume: bool = False
    prorate_partial_days: bool = False
    fixed_pump_on: np.ndarray | None = None
    optimisation_start_index: int = 0


@dataclass(frozen=True)
class AEMOWeeklyResult:
    status: str
    solver_name: str
    objective_aud: float
    solver_runtime_s: float
    timestamps: pd.DatetimeIndex
    price_aud_per_mwh: np.ndarray
    pump_on: np.ndarray
    pump_start: np.ndarray
    power_kw: np.ndarray
    interval_volume_ml: np.ndarray
    interval_cost_aud: np.ndarray

    @property
    def pumped_volume_ml(self) -> float:
        return float(self.interval_volume_ml.sum())

    @property
    def pump_hours(self) -> float:
        return float(self.pump_on.sum() * 0.5)

    @property
    def start_count(self) -> int:
        return int(self.pump_start.sum())


def run_aemo_weekly_optimization(inputs: AEMOWeeklyInputs) -> AEMOWeeklyResult:
    import pulp

    timestamps = pd.DatetimeIndex(inputs.timestamps)
    prices = np.asarray(inputs.price_aud_per_mwh, dtype=float)
    interval_count = len(timestamps)
    if interval_count == 0 or len(prices) != interval_count:
        raise ValueError("Weekly optimisation requires matching non-empty prices and timestamps.")
    if inputs.prorate_partial_days:
        if interval_count > 338:
            raise ValueError("Weekly optimisation cannot extend beyond one calendar week.")
    else:
        valid_counts = (334, 336, 338) if inputs.constrain_total_volume else (336,)
        if interval_count not in valid_counts:
            raise ValueError("Weekly optimisation requires a complete calendar week.")
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("Weekly timestamps must be unique and ordered.")
    if not pd.Series(timestamps).diff().dropna().eq(pd.Timedelta(minutes=30)).all():
        raise ValueError("Weekly optimisation requires continuous 30-minute prices.")
    parameters = [inputs.weekly_target_ml, inputs.flow_lps, inputs.max_power_kw,
                  inputs.standby_power_kw, inputs.minimum_continuous_run_hours,
                  inputs.minimum_off_hours, inputs.minimum_start_interval_hours,
                  inputs.minimum_daily_run_hours,
                  inputs.elapsed_state_steps]
    if not np.isfinite(parameters).all() or min(parameters) < 0:
        raise ValueError("Pump parameters must be finite and non-negative.")
    if inputs.flow_lps <= 0 or inputs.max_power_kw <= 0:
        raise ValueError("Flow and pump power must be greater than zero.")
    if not np.isfinite(prices).all() or inputs.weekly_target_ml < 0:
        raise ValueError("Weekly prices and target must be finite and non-negative.")

    start_index = int(inputs.optimisation_start_index)
    if start_index < 0 or start_index > len(timestamps):
        raise ValueError("Optimisation start index is outside the weekly horizon.")
    if inputs.fixed_pump_on is None:
        fixed = np.full(len(timestamps), np.nan)
    else:
        fixed = np.asarray(inputs.fixed_pump_on, dtype=float)
        if len(fixed) != len(timestamps):
            raise ValueError("Fixed pump state must match the weekly horizon.")
        finite_fixed = fixed[np.isfinite(fixed)]
        if not np.isin(finite_fixed, [0.0, 1.0]).all():
            raise ValueError("Fixed pump states must be zero, one, or missing.")
    if start_index and not np.isfinite(fixed[:start_index]).all():
        raise ValueError("Every elapsed interval must have a fixed pump state.")

    interval_volume_ml = inputs.flow_lps * 0.5 * 3600 / 1_000_000
    min_on_steps = int(np.ceil(inputs.minimum_continuous_run_hours / 0.5))
    min_off_steps = int(np.ceil(inputs.minimum_off_hours / 0.5))
    min_start_steps = int(np.ceil(inputs.minimum_start_interval_hours / 0.5))
    min_daily_steps = int(np.ceil(inputs.minimum_daily_run_hours / 0.5))
    dates = pd.Series(timestamps.date)
    day_indices = {
        day: np.flatnonzero(dates.to_numpy() == day).tolist()
        for day in dates.unique()
    }
    restricted = {
        i for i, timestamp in enumerate(timestamps)
        if timestamp.weekday() < 5 and 16 <= timestamp.hour < 20
    }
    future_steps = list(range(start_index, len(timestamps)))
    available = sum(i not in restricted for i in future_steps)
    if available * interval_volume_ml + 1e-9 < inputs.weekly_target_ml:
        raise AEMODailyOptimizerError("Weekly pumping target is infeasible.")

    model = pulp.LpProblem("AEMO_Weekly_Pump_Scheduler", pulp.LpMinimize)
    steps = range(len(timestamps))
    pump = pulp.LpVariable.dicts("pump_on", steps, cat="Binary")
    start = pulp.LpVariable.dicts("pump_start", steps, cat="Binary")
    model += pulp.lpSum(
        prices[i] * (inputs.max_power_kw * pump[i] + inputs.standby_power_kw * (1 - pump[i])) / 1000 * 0.5
        for i in future_steps
    )

    for i, value in enumerate(fixed):
        if np.isfinite(value):
            model += pump[i] == int(value)

    prior_on = int(inputs.initial_pump_on)
    model += start[0] >= pump[0] - prior_on
    model += start[0] <= pump[0]
    model += start[0] <= 1 - prior_on
    for i in range(1, len(timestamps)):
        model += start[i] >= pump[i] - pump[i - 1]
        model += start[i] <= pump[i]
        model += start[i] <= 1 - pump[i - 1]
    for i in restricted:
        if i < start_index:
            continue
        model += pump[i] == 0

    for i in future_steps:
        history_start = max(0, i - min_off_steps)
        previous_on = pulp.lpSum(pump[j] for j in range(history_start, i))
        if inputs.initial_pump_on and i < min_off_steps:
            previous_on += min_off_steps - i
        model += previous_on <= min_off_steps * (1 - start[i])
    if min_start_steps > 0:
        for i in future_steps:
            first_start = max(0, i - min_start_steps + 1)
            model += pulp.lpSum(start[j] for j in range(first_start, i + 1)) <= 1
    if start_index == 0 and not inputs.initial_pump_on:
        remaining_initial_off = max(
            0, min_off_steps - int(inputs.elapsed_state_steps)
        )
        for i in range(min(remaining_initial_off, len(timestamps))):
            model += start[i] == 0

    if start_index == 0 and inputs.initial_pump_on:
        remaining = max(0, min_on_steps - int(inputs.elapsed_state_steps))
        for i in range(remaining):
            model += pump[i] == 1
    for i in future_steps:
        if i + min_on_steps <= len(timestamps):
            model += pulp.lpSum(pump[i + offset] for offset in range(min_on_steps)) >= min_on_steps * start[i]
        else:
            model += start[i] == 0
    for indices in day_indices.values():
        if not any(i >= start_index for i in indices):
            continue
        required_daily = min_daily_steps
        if inputs.prorate_partial_days:
            required_daily = int(np.ceil(min_daily_steps * min(1.0, len(indices) / 48)))
        model += pulp.lpSum(pump[i] for i in indices) >= required_daily
    model += pulp.lpSum(pump[i] * interval_volume_ml for i in future_steps) >= inputs.weekly_target_ml
    if inputs.constrain_total_volume:
        target_steps = int(np.ceil(inputs.weekly_target_ml / interval_volume_ml - 1e-9))
        model += pulp.lpSum(pump[i] for i in future_steps) == target_steps

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120, gapRel=0.001)
    if not solver.available():
        raise AEMODailyOptimizerError("The open-source CBC solver is not available.")
    started = time.perf_counter()
    model.solve(solver)
    runtime = time.perf_counter() - started
    status = pulp.LpStatus.get(model.status, "UNKNOWN").upper()
    if status != "OPTIMAL":
        raise AEMODailyOptimizerError(f"Weekly optimisation was not feasible ({status}).")
    if model.sol_status != pulp.LpSolutionOptimal:
        raise AEMODailyOptimizerError("Solver stopped before confirming an optimal weekly schedule.")

    pump_on = np.array([float(pulp.value(pump[i])) for i in steps])
    pump_start = np.array([float(pulp.value(start[i])) for i in steps])
    pump_start[:start_index] = 0.0
    power_kw = np.where(pump_on > 0.5, inputs.max_power_kw, inputs.standby_power_kw)
    volumes = pump_on * interval_volume_ml
    costs = prices * power_kw / 1000 * 0.5
    volumes[:start_index] = 0.0
    costs[:start_index] = 0.0
    return AEMOWeeklyResult(
        status=status, solver_name="CBC", objective_aud=float(costs.sum()),
        solver_runtime_s=float(runtime), timestamps=timestamps,
        price_aud_per_mwh=prices, pump_on=pump_on, pump_start=pump_start,
        power_kw=power_kw, interval_volume_ml=volumes, interval_cost_aud=costs,
    )
