from __future__ import annotations

from burrier_llm_planner.domain.models import PhysicalInputs, PlantConfig


REQUIRED_PHYSICAL_FIELDS: tuple[str, ...] = (
    "measured_reservoir_fraction",
    "measurement_time",
    "initial_pump_on",
    "elapsed_state_minutes",
    "equipment_availability",
)


def missing_physical_fields(inputs: PhysicalInputs) -> list[str]:
    missing: list[str] = []
    for field_name in REQUIRED_PHYSICAL_FIELDS:
        value = getattr(inputs, field_name)
        if value is None or (field_name == "equipment_availability" and not value):
            missing.append(field_name)
    return missing


def reconcile_reservoir_state_ml(
    value_ml: float,
    plant: PlantConfig,
    *,
    numerical_tolerance_ml: float = 1e-5,
) -> float:
    value = float(value_ml)
    lower = plant.min_reservoir_ml
    upper = plant.reservoir_capacity_ml
    if value < lower - numerical_tolerance_ml or value > upper + numerical_tolerance_ml:
        raise ValueError(
            f"Estimated state {value:.6f} ML is outside the hard reservoir range "
            f"[{lower:.6f}, {upper:.6f}] ML."
        )
    if abs(value - lower) <= numerical_tolerance_ml:
        return lower
    if abs(value - upper) <= numerical_tolerance_ml:
        return upper
    return value


def estimate_start_storage_ml(
    measured_storage_ml: float,
    scheduled_pumping_ml: float,
    forecast_demand_ml: float,
    plant: PlantConfig,
) -> float:
    estimated = (
        float(measured_storage_ml)
        + float(scheduled_pumping_ml)
        - float(forecast_demand_ml)
    )
    return reconcile_reservoir_state_ml(estimated, plant)
