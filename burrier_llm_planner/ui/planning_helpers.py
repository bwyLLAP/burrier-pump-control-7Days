from __future__ import annotations

from uuid import uuid4

import pandas as pd

from burrier_llm_planner.domain.enums import PlanKind
from burrier_llm_planner.domain.models import PlanSummary
from burrier_llm_planner.optimization.pi_mpc import MPCResult
from burrier_llm_planner.optimization.aemo_daily import AEMODailyResult
from burrier_llm_planner.optimization.result_validator import MPCValidationReport


def build_plan_summary(
    *,
    result: MPCResult,
    report: MPCValidationReport,
    kind: PlanKind,
    input_fingerprint: str,
) -> PlanSummary:
    warnings = list(report.violations)
    if result.status != "OPTIMAL":
        warnings.append(f"Solver returned {result.status}; review the MIP gap.")
    return PlanSummary(
        result_id=str(uuid4()),
        kind=kind,
        feasible=report.feasible,
        solver_status=result.status,
        input_fingerprint=input_fingerprint,
        estimated_cost_aud=report.recalculated_cost_aud,
        energy_mwh=report.energy_mwh,
        pumped_volume_ml=report.pumped_volume_ml,
        pump_hours=report.pump_hours,
        pump_starts=report.pump_starts,
        minimum_reservoir_ml=report.minimum_reservoir_ml,
        terminal_reservoir_ml=report.terminal_reservoir_ml,
        warnings=warnings,
    )


def build_schedule_frame(
    timestamps: pd.DatetimeIndex, result: MPCResult
) -> pd.DataFrame:
    if len(timestamps) != len(result.pump_on):
        raise ValueError("Schedule timestamps and MPC intervals differ.")
    return pd.DataFrame(
        {
            "DateTime": timestamps,
            "AEMOPriceAUDPerMWh": result.price_aud_per_mwh,
            "PumpStatus": ["ON" if value >= 0.5 else "OFF" for value in result.pump_on],
            "PumpStart": result.pump_start.astype(int),
            "FlowML": result.flow_ml_per_step,
            "PowerMW": result.power_mw,
            "DemandML": result.demand_ml_per_step,
            "ReservoirStartML": result.reservoir_ml[:-1],
            "ReservoirEndML": result.reservoir_ml[1:],
        }
    )


def build_schedule_csv(schedule: pd.DataFrame) -> bytes:
    """Serialize a complete operator schedule for a direct local download."""
    return schedule.to_csv(index=False).encode("utf-8-sig")


def build_aemo_daily_schedule_frame(result: AEMODailyResult) -> pd.DataFrame:
    """Build the complete interval export for the binary AEMO daily scheduler."""
    return pd.DataFrame(
        {
            "DateTime": result.timestamps,
            "AEMOPriceAUDPerMWh": result.price_aud_per_mwh,
            "PumpStatus": [
                "ON" if value >= 0.5 else "OFF" for value in result.pump_on
            ],
            "PumpStart": result.pump_start.astype(int),
            "PowerKW": result.power_kw,
            "VolumeML": result.interval_volume_ml,
            "IntervalCostAUD": result.interval_cost_aud,
        }
    )
