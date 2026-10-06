from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from burrier_llm_planner.domain.models import (
    CandidateStrategy,
    PlanSummary,
    PlantConfig,
    ResolvedStrategy,
)
from burrier_llm_planner.services.knowledge import EvidenceReference, LocalKnowledgeIndex
from burrier_llm_planner.strategy.verifier import verify_candidate


CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
HARD_CONSTRAINT_TERMS = (
    "minimum reservoir",
    "hard constraint",
    "maximum starts",
    "最低库容",
    "硬约束",
    "启动次数上限",
)
FASTER_FILL_TERMS = (
    "fill faster",
    "faster fill",
    "fill more",
    "full reservoir",
    "pump full",
    "更快",
    "泵满",
    "储水",
    "加快",
)
LOWER_COST_TERMS = (
    "save more",
    "save electricity",
    "lower cost",
    "cheaper",
    "electricity cost",
    "省电",
    "电费",
    "节约",
    "更便宜",
)


class ControlAgentError(RuntimeError):
    """Raised when a live control-agent response cannot be used safely."""


def operator_error_message(error: Exception) -> str:
    """Return a credential-safe, actionable message for a live Agent failure."""
    detail = str(error).lower()
    if "401" in detail or "authentication" in detail or "unauthorized" in detail:
        return "The OpenAI API key was rejected. Check that the key is active and complete."
    if "402" in detail or "billing" in detail or "insufficient_quota" in detail:
        return "OpenAI API billing or quota is unavailable. Check Platform billing and retry."
    if "429" in detail or "rate limit" in detail:
        return "The OpenAI request limit was reached. Wait briefly, then retry."
    if "connection" in detail or "connect" in detail or "timeout" in detail:
        return "The application could not connect to the OpenAI API. Check network access and retry."
    if "404" in detail or "model" in detail and "not found" in detail:
        return "The configured OpenAI model is unavailable for this API project."
    return "OpenAI returned an unusable response. Retry once, then check the service status."


class AgentContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validated_inputs: bool = False
    current_reservoir_fraction: float = 0.95
    planning_date: date = Field(default_factory=date.today)
    automatic_strategy: ResolvedStrategy | None = None
    baseline_result: PlanSummary | None = None
    price_summary: str = ""
    schedule_summary: str = ""
    active_strategy: Literal["aemo_daily", "seasonal_price_index"] = (
        "seasonal_price_index"
    )
    aemo_parameters: dict[str, float] = Field(default_factory=dict)
    aemo_result_summary: dict[str, float] = Field(default_factory=dict)
    pd7day_summary: str = ""


class ScenarioOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feasible: bool
    status: str
    candidate: CandidateStrategy
    result_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    comparison: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class ScenarioSimulator(Protocol):
    def simulate(self, candidate: CandidateStrategy) -> ScenarioOutcome: ...


class AgentTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    language: Literal["en", "zh"]
    intent: Literal[
        "explain", "result", "scenario", "hard_constraint_change", "unknown"
    ]
    basis: Literal[
        "concept", "current_data", "parameter_suggestion", "validated_simulation"
    ]
    evidence: list[EvidenceReference] = Field(default_factory=list)
    proposed_parameters: list[CandidateStrategy] = Field(default_factory=list)
    scenario_result_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    refused: bool = False


class OpenAIReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    intent: Literal[
        "explain", "result", "scenario", "hard_constraint_change", "unknown"
    ]
    basis: Literal["concept", "current_data", "parameter_suggestion"]
    candidate: CandidateStrategy | None = None
    warnings: list[str] = Field(default_factory=list)
    refused: bool = False


class OpenAIControlAgent:
    provider_name = "openai"

    def __init__(
        self,
        knowledge: LocalKnowledgeIndex,
        simulator: ScenarioSimulator,
        plant: PlantConfig | None = None,
        *,
        model: str = "gpt-5.6-terra",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.simulator = simulator
        self.plant = plant or PlantConfig()
        self.model = model.strip() or "gpt-5.6-terra"
        self.provider_version = self.model
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise ControlAgentError(
                "A tester-supplied OpenAI API key is required for the live Agent."
            )
        self._client = client

    def _client_instance(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ControlAgentError(
                "The openai package is required for the live OpenAI Agent."
            ) from exc
        self._client = OpenAI(
            api_key=self.api_key,
            timeout=30.0,
            max_retries=1,
        )
        return self._client

    def answer(
        self,
        question: str,
        context: AgentContext,
        history: list[dict[str, str]] | None = None,
    ) -> AgentTurn:
        language: Literal["en", "zh"] = (
            "zh" if CHINESE_PATTERN.search(question) else "en"
        )
        normalised = question.strip().lower()
        chunks = self.knowledge.search(question, limit=4)
        evidence = [self.knowledge.evidence(chunk) for chunk in chunks]

        if any(term in normalised for term in HARD_CONSTRAINT_TERMS):
            return AgentTurn(
                answer=MockControlAgent._hard_constraint_answer(language),
                language=language,
                intent="hard_constraint_change",
                basis="concept",
                evidence=evidence,
                refused=True,
            )

        payload = {
            "question": question,
            "language": language,
            "planning_context": context.model_dump(mode="json"),
            "recent_conversation": (history or [])[-6:],
            "local_evidence": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_category": chunk.source_category,
                    "source_path": chunk.source_path,
                    "text": chunk.text,
                }
                for chunk in chunks
            ],
            "immutable_limits": {
                "minimum_reservoir_fraction": self.plant.min_reservoir_fraction,
                "maximum_daily_starts": self.plant.max_daily_starts,
                "horizon_hours": self.plant.horizon_hours,
                "live_equipment_access": False,
            },
        }
        system = (
            "You are the LLM assistant for optimisation-control software developed jointly "
            "by the University of Wollongong and Shoalhaven Water. The developer is Wenyuan Bai. "
            "When asked to introduce yourself, state those facts directly in the user's language. "
            "Answer questions about the Burrier pumping station, PI-MPC, seasonal arbitrage, AEMO "
            "prices, control strategy, software operation, supplied source-code evidence, charts, "
            "validated results, parameters, and operating safety. For unrelated general-purpose "
            "questions, briefly explain your project scope and redirect to it. Base current-plan and "
            "implementation claims on the supplied context or evidence and do not invent values. "
            "For control-change advice, a candidate may contain only mode, target_fraction, "
            "transition_days, rationale, assumptions, and ambiguous. Never change hard constraints, "
            "claim CBC ran, or issue PLC/SCADA commands. For a hard-constraint change, set intent to "
            "hard_constraint_change, candidate to null, and refused true. Otherwise use a null "
            "candidate unless the three editable supervisory parameters directly answer the request."
            " The AEMO daily strategy directly minimises the cost of a 24-hour schedule using "
            "the validated daily forecast and CBC. AEMO PD7Day NSW1 PRICESOLUTION is a separate "
            "informational outlook and is not an optimiser input; never imply that it changes a schedule."
        )
        try:
            response = self._client_instance().responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": str(payload)},
                ],
                text_format=OpenAIReply,
                reasoning={"effort": "low"},
                max_output_tokens=1200,
                store=False,
            )
            reply = response.output_parsed
            if reply is None:
                raise ControlAgentError(
                    "OpenAI returned no parsed structured response."
                )
        except ControlAgentError:
            raise
        except Exception as exc:
            raise ControlAgentError(f"OpenAI request failed: {exc}") from exc

        proposed_parameters: list[CandidateStrategy] = []
        warnings = list(reply.warnings)
        refused = reply.refused
        answer = reply.answer
        if reply.intent == "hard_constraint_change":
            refused = True
        elif reply.candidate is not None:
            verification = verify_candidate(
                reply.candidate,
                plant=self.plant,
                current_fraction=context.current_reservoir_fraction,
                planning_date=context.planning_date,
            )
            if verification.accepted:
                proposed_parameters = [reply.candidate]
            else:
                refused = True
                warnings.extend(verification.reasons)
                answer = (
                    f"{answer}\n\nThe suggested parameters were not applied because "
                    "they failed the local deterministic boundary check."
                )

        return AgentTurn(
            answer=answer,
            language=language,
            intent=reply.intent,
            basis=reply.basis,
            evidence=evidence,
            proposed_parameters=proposed_parameters,
            warnings=warnings,
            refused=refused,
        )


class MockControlAgent:
    provider_name = "mock"
    provider_version = "mock-control-agent-v1"

    def __init__(
        self,
        knowledge: LocalKnowledgeIndex,
        simulator: ScenarioSimulator,
        plant: PlantConfig | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.simulator = simulator
        self.plant = plant or PlantConfig()

    def answer(self, question: str, context: AgentContext) -> AgentTurn:
        language: Literal["en", "zh"] = (
            "zh" if CHINESE_PATTERN.search(question) else "en"
        )
        normalised = question.strip().lower()
        evidence = [
            self.knowledge.evidence(chunk)
            for chunk in self.knowledge.search(question, limit=3)
        ]

        if any(term in normalised for term in HARD_CONSTRAINT_TERMS):
            return AgentTurn(
                answer=self._hard_constraint_answer(language),
                language=language,
                intent="hard_constraint_change",
                basis="concept",
                evidence=evidence,
                refused=True,
            )

        if any(term in normalised for term in FASTER_FILL_TERMS):
            return self._run_scenario("faster_fill", language, context, evidence)

        if any(term in normalised for term in LOWER_COST_TERMS):
            return self._run_scenario("lower_cost", language, context, evidence)

        if any(term in normalised for term in ("alpha_d", "α_d", "target fraction")):
            answer = (
                "α_d 是目标库容比例。提高它通常要求增加储水和泵水量；它不能低于0.90的硬性最低库容比例。"
                if language == "zh"
                else "alpha_d is the target reservoir fraction. Raising it usually requires more storage recovery and pumping; it cannot be set below the immutable 0.90 minimum."
            )
            return AgentTurn(
                answer=answer,
                language=language,
                intent="explain",
                basis="concept",
                evidence=evidence,
            )

        if any(term in normalised for term in ("h_d", "transition", "过渡")):
            answer = (
                "H_d 是达到目标库容的过渡天数。数值越小，每天的储水调整越快；数值越大，调整越平缓。"
                if language == "zh"
                else "H_d is the transition period in days. A smaller value moves storage toward the target faster; a larger value spreads the change over more days."
            )
            return AgentTurn(
                answer=answer,
                language=language,
                intent="explain",
                basis="concept",
                evidence=evidence,
            )

        if any(term in normalised for term in ("m_d", "mode", "模式")):
            answer = (
                "m_d 是监督运行模式。arbitrage用于正常电价套利，emergency用于更快恢复储水；它不是直接的泵启停命令。"
                if language == "zh"
                else "m_d is the supervisory mode. Arbitrage is the normal price-aware mode and emergency prioritises storage recovery; it is not a direct pump command."
            )
            return AgentTurn(
                answer=answer,
                language=language,
                intent="explain",
                basis="concept",
                evidence=evidence,
            )

        if context.baseline_result is not None and any(
            term in normalised for term in ("why", "when", "凌晨", "为什么", "何时")
        ):
            answer = (
                f"当前计划依据：{context.price_summary} {context.schedule_summary}。具体可行性来自已验证的自动基线 {context.baseline_result.result_id}。"
                if language == "zh"
                else f"Current-plan evidence: {context.price_summary} {context.schedule_summary}. Feasibility comes from validated baseline {context.baseline_result.result_id}."
            )
            return AgentTurn(
                answer=answer,
                language=language,
                intent="result",
                basis="current_data",
                evidence=evidence,
            )

        answer = (
            "当前Mock Agent只能可靠回答控制参数、季节套利、当前计划和受限场景模拟问题。请更具体地询问m_d、α_d、H_d、泵运行时间或电费目标。"
            if language == "zh"
            else "The current Mock Agent reliably covers control parameters, seasonal arbitrage, current-plan questions, and bounded scenarios. Ask specifically about m_d, alpha_d, H_d, pump timing, or an electricity-cost goal."
        )
        return AgentTurn(
            answer=answer,
            language=language,
            intent="unknown",
            basis="concept",
            evidence=evidence,
            warnings=["Mock provider has bounded natural-language coverage."],
        )

    def _run_scenario(
        self,
        goal: Literal["faster_fill", "lower_cost"],
        language: Literal["en", "zh"],
        context: AgentContext,
        evidence: list[EvidenceReference],
    ) -> AgentTurn:
        if (
            not context.validated_inputs
            or context.automatic_strategy is None
            or context.baseline_result is None
        ):
            answer = (
                "这个目标涉及参数试算，但当前没有完整且已验证的24小时输入，因此没有运行MPC。请先验证电价、季节策略和泵站状态。"
                if language == "zh"
                else "This goal requires a parameter scenario, but complete validated 24-hour inputs are not available, so MPC was not run. Validate price, seasonal strategy, and plant state first."
            )
            return AgentTurn(
                answer=answer,
                language=language,
                intent="scenario",
                basis="concept",
                evidence=evidence,
                warnings=["Validated planning inputs and a baseline are required."],
            )

        automatic = context.automatic_strategy
        if goal == "faster_fill":
            candidate = CandidateStrategy(
                mode="emergency",
                target_fraction=1.0,
                # A 14-day recovery is materially faster than the annual
                # strategy while keeping the 24-hour CBC scenario responsive.
                transition_days=max(1, min(14, automatic.transition_days)),
                rationale="Move storage toward full capacity more quickly.",
            )
        else:
            candidate = CandidateStrategy(
                mode="arbitrage",
                target_fraction=max(
                    self.plant.min_reservoir_fraction,
                    round(automatic.target_fraction - 0.02, 3),
                ),
                transition_days=min(365, max(30, automatic.transition_days * 2)),
                rationale="Reduce the near-term storage shift while preserving the hard minimum.",
            )

        verification = verify_candidate(
            candidate,
            plant=self.plant,
            current_fraction=context.current_reservoir_fraction,
            planning_date=context.planning_date,
        )
        if not verification.accepted:
            return AgentTurn(
                answer=self._rejected_candidate_answer(language, verification.reasons),
                language=language,
                intent="scenario",
                basis="current_data",
                evidence=evidence,
                proposed_parameters=[candidate],
                warnings=verification.reasons,
            )

        if language == "zh":
            answer = (
                "建议值已填入右侧可编辑参数："
                f"Operating mode = {candidate.mode.value}，"
                f"Reservoir target fraction = {candidate.target_fraction:.2f}，"
                f"Transition period = {candidate.transition_days}天。"
                "CBC尚未运行；请检查或修改参数后点击 Run Agent scenario。"
            )
        else:
            answer = (
                "Suggested values were placed in the editable controls: "
                f"Operating mode = {candidate.mode.value}, "
                f"Reservoir target fraction = {candidate.target_fraction:.2f}, and "
                f"Transition period = {candidate.transition_days} days. "
                "CBC has not run; review or edit the values, then click Run Agent scenario."
            )
        return AgentTurn(
            answer=answer,
            language=language,
            intent="scenario",
            basis="parameter_suggestion",
            evidence=evidence,
            proposed_parameters=[candidate],
        )

    @staticmethod
    def _hard_constraint_answer(language: Literal["en", "zh"]) -> str:
        if language == "zh":
            return "最低库容和其他物理限制属于不可修改的硬约束。Agent不会降低这些限制，也不会运行绕过硬约束的方案。"
        return "The minimum reservoir and other physical limits are immutable hard constraints. The Agent will not lower them or run a scenario that bypasses them."

    @staticmethod
    def _rejected_candidate_answer(language: Literal["en", "zh"], reasons: list[str]) -> str:
        detail = "; ".join(reasons)
        if language == "zh":
            return f"候选参数未通过确定性边界检查，因此没有运行MPC：{detail}"
        return f"The candidate failed deterministic parameter checks, so MPC was not run: {detail}"
