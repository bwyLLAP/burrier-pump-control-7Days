from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from burrier_llm_planner.domain.models import PlantConfig
from burrier_llm_planner.optimization.pi_mpc import MPCResult


@dataclass(frozen=True)
class MPCValidationReport:
    feasible: bool
    violations: list[str]
    recalculated_cost_aud: float
    pumped_volume_ml: float
    energy_mwh: float
    pump_hours: float
    pump_starts: int
    minimum_reservoir_ml: float
    terminal_reservoir_ml: float
    active_constraints: list[str] = field(default_factory=list)


def validate_mpc_result(
    result: MPCResult,
    plant: PlantConfig,
    *,
    tolerance: float = 1e-5,
) -> MPCValidationReport:
    violations: list[str] = []
    n = plant.interval_count
    arrays = {
        "flow": np.asarray(result.flow_ml_per_step, dtype=float),
        "power": np.asarray(result.power_mw, dtype=float),
        "pump_on": np.asarray(result.pump_on, dtype=float),
        "pump_start": np.asarray(result.pump_start, dtype=float),
        "reservoir": np.asarray(result.reservoir_ml, dtype=float),
        "demand": np.asarray(result.demand_ml_per_step, dtype=float),
        "price": np.asarray(result.price_aud_per_mwh, dtype=float),
    }
    for name in ("flow", "power", "pump_on", "pump_start", "demand", "price"):
        if len(arrays[name]) != n:
            violations.append(f"{name} must contain exactly {n} intervals.")
    if len(arrays["reservoir"]) != n + 1:
        violations.append(f"reservoir must contain exactly {n + 1} states.")
    if violations:
        return _report(arrays, violations, plant, result)
    if any(not np.isfinite(values).all() for values in arrays.values()):
        violations.append("MPC result contains non-finite numeric values.")
        return _report(arrays, violations, plant, result)

    pump_on = arrays["pump_on"]
    pump_start = arrays["pump_start"]
    flow = arrays["flow"]
    power = arrays["power"]
    reservoir = arrays["reservoir"]
    demand = arrays["demand"]

    if np.max(np.abs(pump_on - np.rint(pump_on))) > tolerance:
        violations.append("Pump ON/OFF values are not binary.")
    if np.max(np.abs(pump_start - np.rint(pump_start))) > tolerance:
        violations.append("Pump-start values are not binary.")
    binary_on = np.rint(pump_on).astype(int)
    binary_start = np.rint(pump_start).astype(int)

    if (reservoir < plant.min_reservoir_ml - tolerance).any() or (
        reservoir > plant.reservoir_capacity_ml + tolerance
    ).any():
        violations.append("Predicted reservoir trajectory violates hard reservoir bounds.")

    expected_reservoir = reservoir[:-1] + flow - demand
    balance_error = np.max(np.abs(reservoir[1:] - expected_reservoir))
    if balance_error > tolerance:
        violations.append(
            f"Reservoir mass balance error {balance_error:.6g} ML exceeds tolerance."
        )

    min_running_flow = plant.flow_points_ml_per_step[1]
    max_flow = plant.flow_points_ml_per_step[-1]
    if (flow < -tolerance).any() or (flow > max_flow + tolerance).any():
        violations.append("Pump flow violates calibrated hard bounds.")
    if ((binary_on == 0) & (np.abs(flow) > tolerance)).any():
        violations.append("Pump flow is non-zero while pump status is OFF.")
    if ((binary_on == 1) & (flow < min_running_flow - tolerance)).any():
        violations.append("Pump flow is below minimum calibrated running flow.")

    expected_power = np.interp(
        flow,
        np.asarray(plant.flow_points_ml_per_step),
        np.asarray(plant.power_points_mw),
    )
    if np.max(np.abs(power - expected_power)) > 1e-4:
        violations.append("VFD power does not match the calibrated flow-power curve.")

    previous = 1 if result.initial_pump_on else 0
    derived_starts = np.zeros(n, dtype=int)
    for index in range(n):
        derived_starts[index] = max(0, binary_on[index] - previous)
        previous = binary_on[index]
    if not np.array_equal(binary_start, derived_starts):
        violations.append("Pump-start indicators do not match ON/OFF transitions.")
    if int(binary_start.sum()) > plant.max_daily_starts:
        violations.append("Daily pump-start limit is exceeded.")

    min_on_steps = int(round(plant.min_on_duration_hours * 60 / plant.time_step_minutes))
    if result.initial_pump_on:
        remaining = max(0, min_on_steps - int(result.elapsed_on_steps))
        if remaining and not binary_on[:remaining].all():
            violations.append("Initial pump minimum-runtime carry-over is violated.")
    for start_index in np.flatnonzero(binary_start):
        end = start_index + min_on_steps
        if end > n or not binary_on[start_index:end].all():
            violations.append(
                f"Minimum pump runtime is violated for start at interval {start_index}."
            )

    pumped = float(flow.sum())
    lower = max(0.0, result.target_ml - result.target_tolerance_ml)
    upper = min(
        plant.maximum_daily_pumping_ml,
        result.target_ml + result.target_tolerance_ml,
    )
    if pumped < lower - tolerance:
        violations.append("Daily pumping target lower bound is not met.")
    if result.mode == "arbitrage" and pumped > upper + tolerance:
        violations.append("Daily pumping target upper bound is exceeded in arbitrage mode.")

    recalculated_cost = float(
        np.sum(arrays["price"] * power * (plant.time_step_minutes / 60.0))
    )
    if not np.isclose(recalculated_cost, result.objective_aud, rtol=1e-7, atol=1e-4):
        violations.append(
            "Reported cost does not match the independently recalculated cost."
        )

    active: list[str] = []
    if np.isclose(reservoir.min(), plant.min_reservoir_ml, atol=1e-3):
        active.append("minimum_reservoir")
    if np.isclose(pumped, lower, atol=1e-3):
        active.append("daily_target_lower")
    if result.mode == "arbitrage" and np.isclose(pumped, upper, atol=1e-3):
        active.append("daily_target_upper")
    return _report(arrays, violations, plant, result, active)


def _report(
    arrays: dict[str, np.ndarray],
    violations: list[str],
    plant: PlantConfig,
    result: MPCResult,
    active_constraints: list[str] | None = None,
) -> MPCValidationReport:
    flow = arrays.get("flow", np.array([], dtype=float))
    power = arrays.get("power", np.array([], dtype=float))
    price = arrays.get("price", np.array([], dtype=float))
    on = arrays.get("pump_on", np.array([], dtype=float))
    starts = arrays.get("pump_start", np.array([], dtype=float))
    reservoir = arrays.get("reservoir", np.array([], dtype=float))
    count = min(len(power), len(price))
    cost = float(
        np.sum(price[:count] * power[:count] * (plant.time_step_minutes / 60.0))
    )
    return MPCValidationReport(
        feasible=not violations,
        violations=list(violations),
        recalculated_cost_aud=cost,
        pumped_volume_ml=float(flow.sum()) if len(flow) else 0.0,
        energy_mwh=float(power.sum() * plant.time_step_minutes / 60.0)
        if len(power)
        else 0.0,
        pump_hours=float(np.rint(on).sum() * plant.time_step_minutes / 60.0)
        if len(on)
        else 0.0,
        pump_starts=int(np.rint(starts).sum()) if len(starts) else 0,
        minimum_reservoir_ml=float(reservoir.min()) if len(reservoir) else float("nan"),
        terminal_reservoir_ml=float(reservoir[-1]) if len(reservoir) else float("nan"),
        active_constraints=active_constraints or [],
    )
