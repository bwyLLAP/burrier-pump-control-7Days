from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from burrier_llm_planner.domain.enums import StrategyMode
from burrier_llm_planner.domain.models import PlantConfig
from burrier_llm_planner.workflow.completeness import reconcile_reservoir_state_ml


class MPCSolverError(RuntimeError):
    """Raised when the deterministic PI-MPC cannot produce a solution."""


@dataclass
class MPCInputs:
    price_aud_per_mwh: np.ndarray
    demand_ml_per_step: np.ndarray
    initial_reservoir_ml: float
    initial_pump_on: bool
    elapsed_state_steps: int
    target_ml: float
    target_tolerance_ml: float
    mode: StrategyMode


@dataclass
class MPCResult:
    status: str
    objective_aud: float
    solver_runtime_s: float
    mip_gap: float
    flow_ml_per_step: np.ndarray
    power_mw: np.ndarray
    pump_on: np.ndarray
    pump_start: np.ndarray
    reservoir_ml: np.ndarray
    demand_ml_per_step: np.ndarray
    price_aud_per_mwh: np.ndarray
    target_ml: float
    target_tolerance_ml: float
    mode: str
    initial_pump_on: bool
    elapsed_on_steps: int


def run_mpc(inputs: MPCInputs, plant: PlantConfig) -> MPCResult:
    """Solve the 24-hour VFD PI-MPC with the open-source CBC MILP solver."""
    try:
        import pulp
    except ImportError as exc:
        raise MPCSolverError(
            "PuLP is required to run the PI-MPC. Use the Conda Pump environment."
        ) from exc

    prices = np.asarray(inputs.price_aud_per_mwh, dtype=float)
    demand = np.asarray(inputs.demand_ml_per_step, dtype=float)
    n = plant.interval_count
    if len(prices) != n or len(demand) != n:
        raise ValueError(f"PI-MPC requires exactly {n} price and demand intervals.")
    if not np.isfinite(prices).all() or not np.isfinite(demand).all():
        raise ValueError("PI-MPC inputs contain non-finite values.")
    if (demand < 0).any():
        raise ValueError("Demand per interval cannot be negative.")
    if not 0.0 <= inputs.target_ml <= plant.maximum_daily_pumping_ml:
        raise ValueError("Daily pumping target is outside pump capability.")
    if inputs.target_tolerance_ml < 0:
        raise ValueError("Daily pumping target tolerance cannot be negative.")
    initial_reservoir = reconcile_reservoir_state_ml(
        inputs.initial_reservoir_ml, plant
    )

    steps = range(n)
    segment_count = len(plant.flow_points_ml_per_step) - 1
    segments = range(segment_count)
    min_flow = plant.flow_points_ml_per_step[1]
    max_flow = plant.flow_points_ml_per_step[-1]
    standby_power = plant.power_points_mw[0]

    model = pulp.LpProblem("Burrier_24h_PI_MPC_CBC", pulp.LpMinimize)
    pump_on = pulp.LpVariable.dicts("pump_on", steps, cat=pulp.LpBinary)
    pump_start = pulp.LpVariable.dicts("pump_start", steps, cat=pulp.LpBinary)
    pump_at_max = pulp.LpVariable.dicts("pump_at_max", steps, cat=pulp.LpBinary)
    flow = pulp.LpVariable.dicts(
        "flow_ml_per_step", steps, lowBound=0.0, upBound=max_flow
    )
    power = pulp.LpVariable.dicts(
        "power_mw",
        steps,
        lowBound=standby_power,
        upBound=plant.power_points_mw[-1],
    )
    reservoir = pulp.LpVariable.dicts(
        "reservoir_ml",
        range(n + 1),
        lowBound=plant.min_reservoir_ml,
        upBound=plant.reservoir_capacity_ml,
    )
    segment_active = pulp.LpVariable.dicts(
        "vfd_segment_active", (steps, segments), cat=pulp.LpBinary
    )
    segment_fraction = pulp.LpVariable.dicts(
        "vfd_segment_fraction", (steps, segments), lowBound=0.0, upBound=1.0
    )

    for step in steps:
        model += (
            pulp.lpSum(segment_active[step][segment] for segment in segments)
            == pump_on[step],
            f"one_vfd_segment_when_on_{step}",
        )
        for segment in segments:
            model += (
                segment_fraction[step][segment]
                <= segment_active[step][segment],
                f"segment_fraction_requires_active_{step}_{segment}",
            )

        model += (
            flow[step]
            == pulp.lpSum(
                plant.flow_points_ml_per_step[segment]
                * segment_active[step][segment]
                + (
                    plant.flow_points_ml_per_step[segment + 1]
                    - plant.flow_points_ml_per_step[segment]
                )
                * segment_fraction[step][segment]
                for segment in segments
            ),
            f"vfd_flow_curve_{step}",
        )
        model += (
            power[step]
            == standby_power * (1 - pump_on[step])
            + pulp.lpSum(
                plant.power_points_mw[segment]
                * segment_active[step][segment]
                + (
                    plant.power_points_mw[segment + 1]
                    - plant.power_points_mw[segment]
                )
                * segment_fraction[step][segment]
                for segment in segments
            ),
            f"vfd_power_curve_{step}",
        )
        model += (
            flow[step] >= min_flow * pump_on[step],
            f"minimum_running_flow_{step}",
        )
        model += (
            flow[step] <= max_flow * pump_on[step],
            f"off_means_zero_flow_{step}",
        )

        prior_on = int(inputs.initial_pump_on) if step == 0 else pump_on[step - 1]
        model += (
            pump_start[step] >= pump_on[step] - prior_on,
            f"start_lower_{step}",
        )
        model += (
            pump_start[step] <= pump_on[step],
            f"start_upper_on_{step}",
        )
        model += (
            pump_start[step] <= 1 - prior_on,
            f"start_upper_prior_{step}",
        )

    min_on_steps = int(
        round(plant.min_on_duration_hours * 60 / plant.time_step_minutes)
    )
    if inputs.initial_pump_on:
        remaining = max(0, min_on_steps - int(inputs.elapsed_state_steps))
        for step in range(min(remaining, n)):
            model += pump_on[step] == 1, f"carryover_minimum_runtime_{step}"

    for start_step in steps:
        if start_step + min_on_steps <= n:
            model += (
                pulp.lpSum(
                    pump_on[index]
                    for index in range(start_step, start_step + min_on_steps)
                )
                >= min_on_steps * pump_start[start_step],
                f"minimum_runtime_{start_step}",
            )
        else:
            model += pump_start[start_step] == 0, f"no_late_start_{start_step}"

    model += (
        pulp.lpSum(pump_start[step] for step in steps) <= plant.max_daily_starts,
        "maximum_daily_starts",
    )
    model += reservoir[0] == initial_reservoir, "initial_reservoir"
    for step in steps:
        model += (
            reservoir[step + 1]
            == reservoir[step] + flow[step] - float(demand[step]),
            f"reservoir_balance_{step}",
        )

    total_flow = pulp.lpSum(flow[step] for step in steps)
    lower_target = max(0.0, inputs.target_ml - inputs.target_tolerance_ml)
    model += total_flow >= lower_target, "daily_target_lower"
    if inputs.mode is StrategyMode.ARBITRAGE:
        upper_target = min(
            plant.maximum_daily_pumping_ml,
            inputs.target_ml + inputs.target_tolerance_ml,
        )
        model += total_flow <= upper_target, "daily_target_upper"

    maximum_threshold = max_flow - 0.05
    epsilon = 1e-5
    for step in steps:
        model += (
            flow[step] >= maximum_threshold * pump_at_max[step],
            f"maximum_power_lower_{step}",
        )
        model += (
            flow[step]
            <= maximum_threshold - epsilon + max_flow * pump_at_max[step],
            f"maximum_power_upper_{step}",
        )
        model += (
            pump_at_max[step] <= pump_on[step],
            f"maximum_power_requires_on_{step}",
        )
    if plant.min_max_power_ratio > 0:
        model += (
            pulp.lpSum(pump_at_max[step] for step in steps)
            >= plant.min_max_power_ratio
            * pulp.lpSum(pump_on[step] for step in steps),
            "minimum_maximum_power_ratio",
        )

    model += pulp.lpSum(
        float(prices[step])
        * power[step]
        * (plant.time_step_minutes / 60.0)
        for step in steps
    )

    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=plant.solver_time_limit_s,
        gapRel=plant.solver_mip_gap,
    )
    if not solver.available():
        raise MPCSolverError(
            "CBC is not available. Use the Conda Pump environment containing PuLP/CBC."
        )
    started = time.perf_counter()
    try:
        model.solve(solver)
    except pulp.PulpSolverError as exc:
        raise MPCSolverError(f"CBC failed to run the PI-MPC: {exc}") from exc
    runtime = time.perf_counter() - started

    status_name = pulp.LpStatus.get(model.status, f"STATUS_{model.status}")
    if status_name != "Optimal":
        raise MPCSolverError(
            f"PI-MPC did not produce an optimal CBC solution: {status_name}."
        )

    def values(variables, count: int) -> np.ndarray:
        return np.array([float(pulp.value(variables[index])) for index in range(count)])

    flow_values = values(flow, n)
    power_values = values(power, n)
    reservoir_values = np.empty(n + 1, dtype=float)
    reservoir_values[0] = initial_reservoir
    reservoir_values[1:] = initial_reservoir + np.cumsum(flow_values - demand)

    return MPCResult(
        status="OPTIMAL",
        objective_aud=float(pulp.value(model.objective)),
        solver_runtime_s=float(runtime),
        mip_gap=0.0,
        flow_ml_per_step=flow_values,
        power_mw=power_values,
        pump_on=np.rint(values(pump_on, n)).astype(int),
        pump_start=np.rint(values(pump_start, n)).astype(int),
        reservoir_ml=reservoir_values,
        demand_ml_per_step=demand.copy(),
        price_aud_per_mwh=prices.copy(),
        target_ml=float(inputs.target_ml),
        target_tolerance_ml=float(inputs.target_tolerance_ml),
        mode=inputs.mode.value,
        initial_pump_on=bool(inputs.initial_pump_on),
        elapsed_on_steps=(
            int(inputs.elapsed_state_steps) if inputs.initial_pump_on else 0
        ),
    )
