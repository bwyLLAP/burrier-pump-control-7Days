from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from burrier_llm_planner.domain.enums import PlanKind, WorkflowStatus
from burrier_llm_planner.domain.models import (
    CandidateStrategy,
    OperatorDecision,
    PlanSummary,
    PlanningSession,
    PlantConfig,
    VerificationDecision,
)
from burrier_llm_planner.strategy.verifier import (
    CandidateVerificationResult,
    verify_candidate,
)
from burrier_llm_planner.workflow.audit import AuditLog


class WorkflowGuardError(RuntimeError):
    """Raised when a workflow transition would bypass a safety gate."""


def build_input_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PlanningWorkflow:
    def __init__(
        self,
        *,
        session: PlanningSession,
        plant: PlantConfig,
        audit_log: AuditLog | None = None,
    ) -> None:
        self.session = session
        self.plant = plant
        self.audit_log = audit_log

    def record_baseline(
        self, result: PlanSummary, *, input_fingerprint: str
    ) -> None:
        if not result.feasible:
            raise WorkflowGuardError("Only a feasible validated baseline may be recorded.")
        if result.input_fingerprint != input_fingerprint:
            raise WorkflowGuardError("Baseline result fingerprint does not match its inputs.")
        self.session.input_fingerprint = input_fingerprint
        self.session.baseline_result = result
        self.session.status = WorkflowStatus.BASELINE_READY
        self._clear_final_decision()
        self._touch()
        self._audit("baseline_recorded", "system", result.model_dump())

    def attach_candidate(self, candidate: CandidateStrategy) -> None:
        if self.session.baseline_result is None:
            raise WorkflowGuardError("An automatic baseline is required before LLM advice.")
        self.session.candidate_strategy = candidate
        self.session.candidate_verification = None
        self.session.resolved_candidate_strategy = None
        self.session.candidate_confirmed_by = None
        self.session.candidate_result = None
        self.session.status = WorkflowStatus.CANDIDATE_PROPOSED
        self._clear_final_decision()
        self._touch()
        self._audit("candidate_proposed", "llm", candidate.model_dump())

    def verify_candidate(self, *, current_fraction: float) -> CandidateVerificationResult:
        if self.session.candidate_strategy is None:
            raise WorkflowGuardError("No candidate strategy is available for verification.")
        result = verify_candidate(
            self.session.candidate_strategy,
            plant=self.plant,
            current_fraction=current_fraction,
            planning_date=self.session.planning_date,
        )
        self.session.candidate_verification = VerificationDecision(
            accepted=result.accepted,
            reasons=result.reasons,
        )
        self.session.resolved_candidate_strategy = result.resolved
        if result.accepted:
            self.session.status = WorkflowStatus.CANDIDATE_VERIFIED
        self._touch()
        self._audit(
            "candidate_verified" if result.accepted else "candidate_rejected",
            "system",
            self.session.candidate_verification.model_dump(),
        )
        return result

    def confirm_candidate_parameters(self, *, operator: str) -> None:
        verification = self.session.candidate_verification
        if verification is None or not verification.accepted:
            raise WorkflowGuardError(
                "Candidate parameters must be verified before operator confirmation."
            )
        if not operator.strip():
            raise WorkflowGuardError("Operator identity is required for confirmation.")
        self.session.candidate_confirmed_by = operator.strip()
        self.session.status = WorkflowStatus.CANDIDATE_CONFIRMED
        self._touch()
        self._audit("candidate_confirmed", operator.strip(), {})

    def record_candidate(self, result: PlanSummary) -> None:
        verification = self.session.candidate_verification
        if verification is None or not verification.accepted:
            raise WorkflowGuardError("Candidate parameters have not been verified.")
        if not self.session.candidate_confirmed_by:
            raise WorkflowGuardError("Candidate parameters have not been confirmed by an operator.")
        if not result.feasible:
            raise WorkflowGuardError("Only a feasible validated candidate may be recorded.")
        if result.input_fingerprint != self.session.input_fingerprint:
            raise WorkflowGuardError("Candidate and baseline input fingerprints differ.")
        self.session.candidate_result = result
        self.session.status = WorkflowStatus.COMPARISON_READY
        self._clear_final_decision()
        self._touch()
        self._audit("candidate_result_recorded", "system", result.model_dump())

    def record_agent_simulation(self, result: PlanSummary) -> None:
        if result.kind is not PlanKind.AGENT_SIMULATION:
            raise WorkflowGuardError("Agent simulation has the wrong plan kind.")
        if not result.feasible:
            raise WorkflowGuardError(
                "Only a feasible validated Agent simulation may be recorded."
            )
        if result.input_fingerprint != self.session.input_fingerprint:
            raise WorkflowGuardError(
                "Agent simulation fingerprint does not match the baseline inputs."
            )
        self.session.agent_simulation_results.append(result)
        self.session.status = WorkflowStatus.COMPARISON_READY
        self._clear_final_decision()
        self._touch()
        self._audit("agent_simulation_recorded", "agent", result.model_dump())

    def approve(self, *, result_id: str, operator: str) -> None:
        result = self._find_result(result_id)
        if not result.feasible:
            raise WorkflowGuardError("Only a feasible validated result can be approved.")
        if result.input_fingerprint != self.session.input_fingerprint:
            raise WorkflowGuardError("Result inputs changed and the result is stale.")
        if not operator.strip():
            raise WorkflowGuardError("Operator identity is required for approval.")
        decision = OperatorDecision(
            actor=operator.strip(),
            decision="approve",
            timestamp=datetime.now(ZoneInfo(self.session.timezone)),
            result_id=result_id,
        )
        self.session.selected_result_id = result_id
        self.session.final_decision = decision
        self.session.status = WorkflowStatus.APPROVED
        self._touch()
        self._audit("schedule_approved", operator.strip(), decision.model_dump())

    def reject(self, *, operator: str) -> None:
        self.session.final_decision = OperatorDecision(
            actor=operator.strip(),
            decision="reject",
            timestamp=datetime.now(ZoneInfo(self.session.timezone)),
        )
        self.session.selected_result_id = None
        self.session.status = WorkflowStatus.REJECTED
        self._touch()
        self._audit("schedule_rejected", operator.strip(), {})

    def update_reservoir_fraction(self, value: float) -> None:
        prior = self.session.input_fingerprint
        self.session.physical_inputs.measured_reservoir_fraction = value
        self.invalidate_results()
        self._audit(
            "physical_input_changed",
            "operator",
            {"field": "measured_reservoir_fraction", "value": value},
            prior_fingerprint=prior,
            current_fingerprint=None,
        )

    def invalidate_results(self) -> None:
        self.session.baseline_result = None
        self.session.candidate_strategy = None
        self.session.candidate_verification = None
        self.session.resolved_candidate_strategy = None
        self.session.candidate_confirmed_by = None
        self.session.candidate_result = None
        self.session.agent_simulation_results = []
        self.session.input_fingerprint = None
        self._clear_final_decision()
        self.session.status = WorkflowStatus.INPUTS_REQUIRED
        self._touch()

    def _find_result(self, result_id: str) -> PlanSummary:
        results = (
            [self.session.baseline_result, self.session.candidate_result]
            + self.session.agent_simulation_results
        )
        for result in results:
            if result is not None and result.result_id == result_id:
                return result
        raise WorkflowGuardError(f"Unknown result identifier: {result_id}")

    def _clear_final_decision(self) -> None:
        self.session.selected_result_id = None
        self.session.final_decision = None

    def _touch(self) -> None:
        self.session.updated_at = datetime.now(ZoneInfo(self.session.timezone))

    def _audit(
        self,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
        *,
        prior_fingerprint: str | None = None,
        current_fingerprint: str | None = None,
    ) -> None:
        if self.audit_log is not None:
            self.audit_log.record(
                event_type=event_type,
                actor=actor,
                payload=payload,
                prior_fingerprint=prior_fingerprint,
                current_fingerprint=current_fingerprint,
            )
