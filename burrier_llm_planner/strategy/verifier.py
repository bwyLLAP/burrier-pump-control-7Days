from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict

from burrier_llm_planner.domain.enums import StrategyMode
from burrier_llm_planner.domain.models import (
    CandidateStrategy,
    PlantConfig,
    ResolvedStrategy,
)


class CandidateVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    reasons: list[str]
    resolved: ResolvedStrategy | None = None


def verify_candidate(
    candidate: CandidateStrategy,
    *,
    plant: PlantConfig,
    current_fraction: float,
    planning_date: date,
) -> CandidateVerificationResult:
    reasons: list[str] = []
    if candidate.ambiguous:
        reasons.append("Candidate is marked ambiguous and requires clarification.")
    if not plant.min_reservoir_fraction <= candidate.target_fraction <= 1.0:
        reasons.append(
            "Candidate target fraction is outside the hard minimum-to-capacity "
            f"range [{plant.min_reservoir_fraction:.2f}, 1.00]."
        )
    if not 1 <= candidate.transition_days <= 365:
        reasons.append("Candidate transition period must be within [1, 365] days.")
    if not 0.0 <= current_fraction <= 1.0:
        reasons.append("Current reservoir fraction is outside [0, 1].")
    if candidate.mode is StrategyMode.EMERGENCY:
        if candidate.target_fraction < 0.95:
            reasons.append("Emergency mode requires a target fraction of at least 0.95.")
        if candidate.target_fraction + 1e-9 < current_fraction:
            reasons.append(
                "Emergency mode cannot request a target below the current reservoir fraction."
            )

    if reasons:
        return CandidateVerificationResult(accepted=False, reasons=reasons)

    accepted_reasons = [
        "Candidate contains only the permitted supervisory parameters.",
        "Target fraction is within the immutable plant safety range.",
        "Transition period is within the deterministic admissible range.",
        "Deterministic PI-MPC and independent result validation are still required.",
    ]
    resolved = ResolvedStrategy(
        mode=candidate.mode,
        target_fraction=candidate.target_fraction,
        transition_days=candidate.transition_days,
        issued_date=planning_date,
        target_date=planning_date + timedelta(days=candidate.transition_days - 1),
        target_fraction_source="llm_candidate_confirmed",
        transition_days_source="llm_candidate_confirmed",
        verification_reasons=accepted_reasons,
    )
    return CandidateVerificationResult(
        accepted=True,
        reasons=accepted_reasons,
        resolved=resolved,
    )
