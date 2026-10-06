from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


class AEMODailyOptimizerError(RuntimeError):
    """Raised when the deterministic legacy AEMO scheduler cannot produce a plan."""


@dataclass(frozen=True)
class AEMODailyInputs:
    timestamps: pd.DatetimeIndex
    price_aud_per_mwh: np.ndarray
    daily_target_ml: float = 40.0
    flow_lps: float = 1050.0
    max_power_kw: float = 1950.0
    standby_power_kw: float = 0.0
    minimum_continuous_run_hours: float = 4.0
    minimum_daily_run_hours: float = 6.0
    initial_pump_on: bool = False
    elapsed_state_steps: int = 0


@dataclass(frozen=True)
class PumpPeriod:
    start: pd.Timestamp
    stop: pd.Timestamp
    start_price_aud_per_mwh: float
    stop_price_aud_per_mwh: float
    volume_ml: float


@dataclass(frozen=True)
class AEMODailyResult:
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
    periods: tuple[PumpPeriod, ...]
    initial_pump_on: bool = False
    initial_state_steps: int = 0
    final_pump_on: bool = False
    terminal_state_steps: int = 0
    remaining_minimum_on_steps: int = 0

    @property
    def pump_hours(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        hours = (self.timestamps[1] - self.timestamps[0]).total_seconds() / 3600
        return float(self.pump_on.sum() * hours)

    @property
    def pumped_volume_ml(self) -> float:
        return float(self.interval_volume_ml.sum())

    @property
    def start_count(self) -> int:
        return int(self.pump_start.sum())


def _validate(inputs: AEMODailyInputs) -> tuple[pd.DatetimeIndex, np.ndarray, float]:
    timestamps = pd.DatetimeIndex(inputs.timestamps)
    prices = np.asarray(inputs.price_aud_per_mwh, dtype=float)
    if len(timestamps) != len(prices) or len(timestamps) < 2:
        raise ValueError("AEMO daily prices and timestamps must have equal length.")
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("AEMO daily timestamps must be unique and ordered.")
    deltas = pd.Series(timestamps).diff().dropna()
    if not deltas.eq(pd.Timedelta(minutes=30)).all():
        raise ValueError("AEMO daily optimisation requires continuous 30-minute prices.")
    if not np.isfinite(prices).all():
        raise ValueError("AEMO daily prices must be finite.")
    numeric = {
        "daily target": inputs.daily_target_ml,
        "flow rate": inputs.flow_lps,
        "maximum power": inputs.max_power_kw,
        "standby power": inputs.standby_power_kw,
        "minimum continuous runtime": inputs.minimum_continuous_run_hours,
        "minimum daily runtime": inputs.minimum_daily_run_hours,
    }
    if any(not np.isfinite(value) or value < 0 for value in numeric.values()):
        raise ValueError("AEMO daily parameters must be finite and non-negative.")
    if inputs.flow_lps <= 0 or inputs.max_power_kw <= 0:
        raise ValueError("Flow rate and maximum power must be positive.")
    if inputs.standby_power_kw > inputs.max_power_kw:
        raise ValueError("Standby power cannot exceed maximum power.")
    if inputs.minimum_continuous_run_hours > 24 or inputs.minimum_daily_run_hours > 24:
        raise ValueError("Runtime limits cannot exceed 24 hours.")
    return timestamps, prices, 0.5


def run_aemo_daily_optimization(inputs: AEMODailyInputs) -> AEMODailyResult:
    """Run the legacy price-driven binary pump scheduler with open-source CBC."""
    try:
        import pulp
    except ImportError as exc:
        raise AEMODailyOptimizerError(
            "PuLP and CBC are required for AEMO daily optimisation."
        ) from exc

    timestamps, prices, step_hours = _validate(inputs)
    interval_volume_ml = inputs.flow_lps * step_hours * 3600 / 1_000_000
    min_on_steps = int(np.ceil(inputs.minimum_continuous_run_hours / step_hours))
    min_daily_steps = int(np.ceil(inputs.minimum_daily_run_hours / step_hours))
    dates = pd.Series(timestamps.date)
    day_indices = {
        day: np.flatnonzero(dates.to_numpy() == day).tolist()
        for day in dates.unique()
    }
    restricted = {
        index
        for index, timestamp in enumerate(timestamps)
        if timestamp.weekday() < 5 and 16 <= timestamp.hour < 20
    }
    for day, indices in day_indices.items():
        available = [index for index in indices if index not in restricted]
        if len(available) < min_daily_steps:
            raise AEMODailyOptimizerError(
                f"Minimum daily runtime is infeasible for {day}."
            )
        if len(available) * interval_volume_ml + 1e-9 < inputs.daily_target_ml:
            raise AEMODailyOptimizerError(
                f"Daily pumping target is infeasible for {day}."
            )

    model = pulp.LpProblem("AEMO_Daily_Pump_Scheduler", pulp.LpMinimize)
    steps = range(len(timestamps))
    pump = pulp.LpVariable.dicts("pump_on", steps, cat="Binary")
    start = pulp.LpVariable.dicts("pump_start", steps, cat="Binary")
    model += pulp.lpSum(
        prices[index]
        * (
            inputs.max_power_kw * pump[index]
            + inputs.standby_power_kw * (1 - pump[index])
        )
        / 1000
        * step_hours
        for index in steps
    )

    prior_on = int(inputs.initial_pump_on)
    model += start[0] >= pump[0] - prior_on, "Start_Lower_0"
    model += start[0] <= pump[0], "Start_Upper_On_0"
    model += start[0] <= 1 - prior_on, "Start_Upper_Prior_0"
    for index in range(1, len(timestamps)):
        model += start[index] >= pump[index] - pump[index - 1]
        model += start[index] <= pump[index]
        model += start[index] <= 1 - pump[index - 1]
    for index in restricted:
        model += pump[index] == 0, f"Weekday_Restriction_{index}"
    if min_on_steps:
        if inputs.initial_pump_on:
            remaining = max(0, min_on_steps - int(inputs.elapsed_state_steps))
            for index in range(min(remaining, len(timestamps))):
                model += pump[index] == 1, f"Initial_On_Obligation_{index}"
        for index in range(len(timestamps)):
            span = min(min_on_steps, len(timestamps) - index)
            model += (
                pulp.lpSum(pump[index + offset] for offset in range(span))
                >= span * start[index]
            ), f"Minimum_Run_{index}"
    for day, indices in day_indices.items():
        model += (
            pulp.lpSum(pump[index] for index in indices) >= min_daily_steps
        ), f"Minimum_Daily_Run_{day}"
        model += (
            pulp.lpSum(pump[index] * interval_volume_ml for index in indices)
            >= inputs.daily_target_ml
        ), f"Daily_Volume_{day}"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60, gapRel=0.001)
    if not solver.available():
        raise AEMODailyOptimizerError("The open-source CBC solver is not available.")
    started = time.perf_counter()
    try:
        model.solve(solver)
    except Exception as exc:
        raise AEMODailyOptimizerError("CBC failed to run AEMO daily optimisation.") from exc
    runtime = time.perf_counter() - started
    status = pulp.LpStatus.get(model.status, "UNKNOWN").upper()
    if status != "OPTIMAL":
        raise AEMODailyOptimizerError(
            f"AEMO daily optimisation was not feasible ({status})."
        )

    pump_on = np.array([float(pulp.value(pump[index])) for index in steps])
    pump_start = np.array([float(pulp.value(start[index])) for index in steps])
    power_kw = np.where(
        pump_on > 0.5, inputs.max_power_kw, inputs.standby_power_kw
    )
    volumes = pump_on * interval_volume_ml
    costs = prices * power_kw / 1000 * step_hours

    periods: list[PumpPeriod] = []
    period_start: int | None = None
    for index, is_on in enumerate(pump_on > 0.5):
        if is_on and period_start is None:
            period_start = index
        if not is_on and period_start is not None:
            periods.append(
                PumpPeriod(
                    start=timestamps[period_start],
                    stop=timestamps[index],
                    start_price_aud_per_mwh=float(prices[period_start]),
                    stop_price_aud_per_mwh=float(prices[index - 1]),
                    volume_ml=float(volumes[period_start:index].sum()),
                )
            )
            period_start = None
    if period_start is not None:
        periods.append(
            PumpPeriod(
                start=timestamps[period_start],
                stop=timestamps[-1] + pd.Timedelta(minutes=30),
                start_price_aud_per_mwh=float(prices[period_start]),
                stop_price_aud_per_mwh=float(prices[-1]),
                volume_ml=float(volumes[period_start:].sum()),
            )
        )

    final_on = bool(pump_on[-1] > 0.5)
    terminal_steps = 0
    for state in reversed(pump_on > 0.5):
        if bool(state) != final_on:
            break
        terminal_steps += 1
    if terminal_steps == len(pump_on) and final_on == bool(inputs.initial_pump_on):
        terminal_steps += int(inputs.elapsed_state_steps)
    return AEMODailyResult(
        status=status,
        solver_name="CBC",
        objective_aud=float(costs.sum()),
        solver_runtime_s=float(runtime),
        timestamps=timestamps,
        price_aud_per_mwh=prices,
        pump_on=pump_on,
        pump_start=pump_start,
        power_kw=power_kw,
        interval_volume_ml=volumes,
        interval_cost_aud=costs,
        periods=tuple(periods),
        initial_pump_on=bool(inputs.initial_pump_on),
        initial_state_steps=int(inputs.elapsed_state_steps),
        final_pump_on=final_on,
        terminal_state_steps=terminal_steps,
        remaining_minimum_on_steps=max(0, min_on_steps-terminal_steps) if final_on else 0,
    )
