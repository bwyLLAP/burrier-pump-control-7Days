from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from burrier_llm_planner.domain.enums import PlanKind, StrategyMode, WorkflowStatus


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlantConfig(StrictModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    config_version: str = "burrier-v1"
    reservoir_capacity_ml: float = 3800.0
    min_reservoir_fraction: float = 0.90
    time_step_minutes: int = 30
    horizon_hours: int = 24
    min_on_duration_hours: float = 4.0
    max_daily_starts: int = 3
    min_max_power_ratio: float = 0.50
    flow_points_ml_per_step: tuple[float, ...] = (
        0.0,
        1.323,
        1.418,
        1.512,
        1.607,
        1.701,
        1.796,
        1.890,
    )
    power_points_mw: tuple[float, ...] = (
        0.005,
        0.669,
        0.823,
        0.998,
        1.197,
        1.422,
        1.671,
        1.950,
    )
    pumping_target_tolerance_ml: float = 1.0
    price_index_target_tolerance_ml: float = 0.25
    solver_time_limit_s: float = 120.0
    solver_mip_gap: float = 0.02

    @model_validator(mode="after")
    def validate_plant(self) -> "PlantConfig":
        if self.horizon_hours != 24:
            raise ValueError("The prototype horizon must be exactly 24 hours.")
        if not 0.0 < self.min_reservoir_fraction <= 1.0:
            raise ValueError("min_reservoir_fraction must lie in (0, 1].")
        if len(self.flow_points_ml_per_step) != len(self.power_points_mw):
            raise ValueError("VFD flow and power calibration lengths differ.")
        if any(
            right <= left
            for left, right in zip(
                self.flow_points_ml_per_step, self.flow_points_ml_per_step[1:]
            )
        ):
            raise ValueError("VFD flow points must be strictly increasing.")
        if any(
            right <= left
            for left, right in zip(self.power_points_mw, self.power_points_mw[1:])
        ):
            raise ValueError("VFD power points must be strictly increasing.")
        return self

    @property
    def min_reservoir_ml(self) -> float:
        return self.reservoir_capacity_ml * self.min_reservoir_fraction

    @property
    def interval_count(self) -> int:
        return self.horizon_hours * 60 // self.time_step_minutes

    @property
    def maximum_daily_pumping_ml(self) -> float:
        return self.flow_points_ml_per_step[-1] * self.interval_count


class CandidateStrategy(StrictModel):
    mode: StrategyMode
    target_fraction: float
    transition_days: int
    rationale: str = Field(min_length=1, max_length=2000)
    assumptions: list[str] = Field(default_factory=list)
    ambiguous: bool = False


class ResolvedStrategy(StrictModel):
    mode: StrategyMode
    target_fraction: float
    transition_days: int
    issued_date: date
    target_date: date
    target_fraction_source: str
    transition_days_source: str
    verification_reasons: list[str] = Field(default_factory=list)


class RequestInterpretation(StrictModel):
    scheduling_task: Literal["generate_schedule", "analyse_schedule"]
    planning_date: date
    horizon_hours: Literal[24] = 24
    requested_analysis: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    ambiguous: bool = False
    explanation: str = ""


class PhysicalInputs(StrictModel):
    measured_reservoir_fraction: float | None = None
    measurement_time: datetime | None = None
    estimated_start_reservoir_ml: float | None = None
    initial_pump_on: bool | None = None
    elapsed_state_minutes: int | None = None
    equipment_availability: dict[str, bool] | None = None
    temporary_restrictions: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("measured_reservoir_fraction")
    @classmethod
    def validate_fraction(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("Reservoir fraction must lie in [0, 1].")
        return value


class DataProvenance(StrictModel):
    source: str
    issue_time: datetime | None = None
    source_hash: str = ""
    is_fresh: bool = False
    warnings: list[str] = Field(default_factory=list)


class VerificationDecision(StrictModel):
    accepted: bool
    reasons: list[str]


class PlanSummary(StrictModel):
    result_id: str
    kind: PlanKind
    feasible: bool
    solver_status: str
    input_fingerprint: str
    estimated_cost_aud: float
    energy_mwh: float
    pumped_volume_ml: float
    pump_hours: float
    pump_starts: int
    minimum_reservoir_ml: float
    terminal_reservoir_ml: float
    warnings: list[str] = Field(default_factory=list)


class OperatorDecision(StrictModel):
    actor: str
    decision: Literal["approve", "reject", "revise"]
    timestamp: datetime
    result_id: str | None = None


class PlanningSession(StrictModel):
    session_id: str
    original_request: str
    timezone: str
    planning_date: date
    horizon_start: datetime
    horizon_end: datetime
    status: WorkflowStatus
    interpretation: RequestInterpretation | None = None
    physical_inputs: PhysicalInputs = Field(default_factory=PhysicalInputs)
    automatic_strategy: ResolvedStrategy | None = None
    candidate_strategy: CandidateStrategy | None = None
    candidate_verification: VerificationDecision | None = None
    resolved_candidate_strategy: ResolvedStrategy | None = None
    candidate_confirmed_by: str | None = None
    baseline_result: PlanSummary | None = None
    candidate_result: PlanSummary | None = None
    agent_simulation_results: list[PlanSummary] = Field(default_factory=list)
    selected_result_id: str | None = None
    final_decision: OperatorDecision | None = None
    input_fingerprint: str | None = None
    operational_export_allowed: bool = True
    created_at: datetime
    updated_at: datetime

    @classmethod
    def new(
        cls,
        request: str,
        planning_date: date,
        timezone: str = "Australia/Sydney",
    ) -> "PlanningSession":
        if not request.strip():
            raise ValueError("Operator request cannot be empty.")
        zone = ZoneInfo(timezone)
        horizon_start = datetime.combine(planning_date, datetime.min.time(), zone)
        now = datetime.now(zone)
        return cls(
            session_id=str(uuid4()),
            original_request=request.strip(),
            timezone=timezone,
            planning_date=planning_date,
            horizon_start=horizon_start,
            horizon_end=horizon_start + timedelta(hours=24),
            status=WorkflowStatus.REQUEST_CAPTURED,
            created_at=now,
            updated_at=now,
        )
