from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from burrier_llm_planner.domain.models import PlantConfig
from burrier_llm_planner.workflow.completeness import reconcile_reservoir_state_ml


class DailyPumpingTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    demand_ml: float
    current_storage_ml: float
    target_storage_ml: float
    storage_gap_ml: float
    storage_shift_ml: float
    transition_days: int
    estimated_minimum_transition_days: float
    requested_ml: float
    applied_ml: float
    tolerance_ml: float
    clipped: bool
    source: str


def estimate_minimum_transition_days(
    *,
    current_storage_ml: float,
    target_storage_ml: float,
    demand_ml: float,
    plant: PlantConfig,
) -> float:
    gap = target_storage_ml - current_storage_ml
    if abs(gap) < 1e-9:
        return 0.0
    if gap > 0:
        maximum_net_refill = plant.maximum_daily_pumping_ml - demand_ml
        if maximum_net_refill <= 0:
            return math.inf
        return float(math.ceil(gap / maximum_net_refill))
    if demand_ml <= 0:
        return math.inf
    return float(math.ceil(abs(gap) / demand_ml))


def generate_daily_pumping_target(
    *,
    demand_ml: float,
    current_storage_ml: float,
    target_fraction: float,
    transition_days: int,
    plant: PlantConfig,
    source: str,
) -> DailyPumpingTarget:
    if demand_ml < 0:
        raise ValueError("Demand forecast cannot be negative.")
    if not plant.min_reservoir_fraction <= target_fraction <= 1.0:
        raise ValueError("Strategic target fraction is outside hard safety bounds.")
    if not 1 <= transition_days <= 365:
        raise ValueError("Transition period must be within [1, 365] days.")

    current = reconcile_reservoir_state_ml(current_storage_ml, plant)
    target_storage = reconcile_reservoir_state_ml(
        target_fraction * plant.reservoir_capacity_ml, plant
    )
    gap = target_storage - current
    shift = gap / transition_days
    requested = float(demand_ml) + shift
    applied = min(max(requested, 0.0), plant.maximum_daily_pumping_ml)
    tolerance = (
        plant.price_index_target_tolerance_ml
        if source == "annual_price_index"
        else plant.pumping_target_tolerance_ml
    )
    return DailyPumpingTarget(
        demand_ml=demand_ml,
        current_storage_ml=current,
        target_storage_ml=target_storage,
        storage_gap_ml=gap,
        storage_shift_ml=shift,
        transition_days=transition_days,
        estimated_minimum_transition_days=estimate_minimum_transition_days(
            current_storage_ml=current,
            target_storage_ml=target_storage,
            demand_ml=demand_ml,
            plant=plant,
        ),
        requested_ml=requested,
        applied_ml=applied,
        tolerance_ml=tolerance,
        clipped=not math.isclose(requested, applied, abs_tol=1e-9),
        source=source,
    )
