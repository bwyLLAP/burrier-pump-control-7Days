try:
    from enum import StrEnum
except ImportError:  # Python 3.10 in the existing Conda Pump environment.
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value


class WorkflowStatus(StrEnum):
    NEW = "new"
    REQUEST_CAPTURED = "request_captured"
    INPUTS_REQUIRED = "inputs_required"
    INPUTS_READY = "inputs_ready"
    BASELINE_READY = "baseline_ready"
    CANDIDATE_PROPOSED = "candidate_proposed"
    CANDIDATE_VERIFIED = "candidate_verified"
    CANDIDATE_CONFIRMED = "candidate_confirmed"
    COMPARISON_READY = "comparison_ready"
    APPROVED = "approved"
    REJECTED = "rejected"


class StrategyMode(StrEnum):
    ARBITRAGE = "arbitrage"
    EMERGENCY = "emergency"


class PlanKind(StrEnum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"
    AGENT_SIMULATION = "agent_simulation"
