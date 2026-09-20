from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from burrier_llm_planner.domain.models import (
    CandidateStrategy,
    RequestInterpretation,
)


class LLMProviderError(RuntimeError):
    """Raised when a bounded LLM operation cannot be completed."""


class LiveLLMDisabledError(LLMProviderError):
    """Raised when live LLM use has not been explicitly enabled."""


class LLMUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    timestamp_utc: datetime


T = TypeVar("T")


class LLMResult(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    value: T
    usage: LLMUsage


class StrategyExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    seasonal_reasoning: str
    key_factors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ResultExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    cost_observations: list[str] = Field(default_factory=list)
    reservoir_observations: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class LLMProvider(Protocol):
    def interpret_request(self, request: str, now: datetime) -> LLMResult[RequestInterpretation]: ...

    def explain_automatic_strategy(
        self, strategy_context: dict[str, Any]
    ) -> LLMResult[StrategyExplanation]: ...

    def explain_result(
        self, result_metrics: dict[str, Any]
    ) -> LLMResult[ResultExplanation]: ...

    def suggest_candidate(
        self,
        request: str,
        automatic_context: dict[str, Any],
        result_metrics: dict[str, Any],
    ) -> LLMResult[CandidateStrategy | None]: ...


class MockLLMProvider:
    provider_name = "mock"
    model_name = "mock-burrier-v1"

    def _result(self, value: T) -> LLMResult[T]:
        return LLMResult(
            value=value,
            usage=LLMUsage(
                provider=self.provider_name,
                model=self.model_name,
                timestamp_utc=datetime(2000, 1, 1, tzinfo=timezone.utc),
            ),
        )

    def interpret_request(
        self, request: str, now: datetime
    ) -> LLMResult[RequestInterpretation]:
        text = request.strip().lower()
        if "tomorrow" in text or "明天" in text:
            planning_date = now.date() + timedelta(days=1)
            ambiguous = False
        else:
            planning_date = now.date() + timedelta(days=1)
            ambiguous = True
        requested_analysis = []
        if any(token in text for token in ("analyse", "analyze", "explain", "分析", "解释")):
            requested_analysis.append("strategy_and_result")
        value = RequestInterpretation(
            scheduling_task="generate_schedule",
            planning_date=planning_date,
            requested_analysis=requested_analysis,
            ambiguous=ambiguous,
            explanation=(
                "The request was interpreted as a 24-hour plan beginning at "
                f"00:00 on {planning_date.isoformat()} Australia/Sydney time."
            ),
        )
        return self._result(value)

    def explain_automatic_strategy(
        self, strategy_context: dict[str, Any]
    ) -> LLMResult[StrategyExplanation]:
        target = float(strategy_context["target_fraction"])
        transition = int(strategy_context["transition_days"])
        regime = str(strategy_context.get("price_regime", "unknown"))
        return self._result(
            StrategyExplanation(
                summary=(
                    f"The automatic price-index strategy selects a {target:.1%} "
                    f"reservoir target over {transition} days."
                ),
                seasonal_reasoning=(
                    f"The recorded long-term price regime is {regime}; the target "
                    "shifts storage gradually rather than controlling individual pump intervals."
                ),
                key_factors=["long-term price index", "current storage", "forecast demand"],
            )
        )

    def explain_result(
        self, result_metrics: dict[str, Any]
    ) -> LLMResult[ResultExplanation]:
        return self._result(
            ResultExplanation(
                summary=(
                    f"The deterministic optimiser reports {result_metrics.get('solver_status', 'unknown')} "
                    f"with estimated cost AUD {float(result_metrics.get('estimated_cost_aud', 0.0)):.2f}."
                ),
                cost_observations=["Pumping is shifted toward lower forecast-price intervals."],
                reservoir_observations=[
                    "The minimum reservoir value is taken from the independently validated trajectory."
                ],
                risks=list(result_metrics.get("warnings", [])),
                assumptions=list(result_metrics.get("assumptions", [])),
            )
        )

    def suggest_candidate(
        self,
        request: str,
        automatic_context: dict[str, Any],
        result_metrics: dict[str, Any],
    ) -> LLMResult[CandidateStrategy | None]:
        text = request.lower()
        if any(token in text for token in ("bushfire", "fire risk", "山火", "火灾")):
            candidate: CandidateStrategy | None = CandidateStrategy(
                mode="emergency",
                target_fraction=1.0,
                transition_days=2,
                rationale=(
                    "High fire risk was stated; consider a faster move toward full storage."
                ),
                assumptions=["Equipment remains available during the planning horizon."],
            )
        else:
            candidate = None
        return self._result(candidate)


class OpenAILLMProvider:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        if not model.strip():
            raise LLMProviderError("A live OpenAI model name must be configured.")
        self.model = model.strip()
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise LLMProviderError(
                "A tester-supplied OpenAI API key is required for live LLM use."
            )

    def _parse(self, *, schema: type[T], system: str, payload: dict[str, Any]) -> LLMResult[T]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMProviderError("The openai package is required for live LLM use.") from exc
        started = time.perf_counter()
        try:
            client = OpenAI(api_key=self.api_key, timeout=30.0, max_retries=1)
            response = client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, default=str)},
                ],
                text_format=schema,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise LLMProviderError("The live LLM returned no structured output.")
            usage = response.usage
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Live LLM request failed: {exc}") from exc
        return LLMResult(
            value=parsed,
            usage=LLMUsage(
                provider="openai",
                model=self.model,
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                timestamp_utc=datetime.now(timezone.utc),
            ),
        )

    def interpret_request(
        self, request: str, now: datetime
    ) -> LLMResult[RequestInterpretation]:
        return self._parse(
            schema=RequestInterpretation,
            system=(
                "Interpret a Burrier pumping-station planning request. The horizon must be exactly "
                "24 hours. Do not produce pump states, VFD values, physical measurements, or constraints."
            ),
            payload={"request": request, "current_time": now.isoformat()},
        )

    def explain_automatic_strategy(
        self, strategy_context: dict[str, Any]
    ) -> LLMResult[StrategyExplanation]:
        return self._parse(
            schema=StrategyExplanation,
            system=(
                "Explain the supplied deterministic price-index strategy. Do not calculate or change controls."
            ),
            payload=strategy_context,
        )

    def explain_result(
        self, result_metrics: dict[str, Any]
    ) -> LLMResult[ResultExplanation]:
        return self._parse(
            schema=ResultExplanation,
            system=(
                "Explain only the supplied validated MPC metrics. Do not invent values or approve the plan."
            ),
            payload=result_metrics,
        )

    def suggest_candidate(
        self,
        request: str,
        automatic_context: dict[str, Any],
        result_metrics: dict[str, Any],
    ) -> LLMResult[CandidateStrategy | None]:
        class OptionalCandidate(BaseModel):
            candidate: CandidateStrategy | None

        parsed = self._parse(
            schema=OptionalCandidate,
            system=(
                "Optionally propose only mode, target_fraction, and transition_days. "
                "Never propose pump/VFD commands or physical constraint changes. The output is advice only."
            ),
            payload={
                "request": request,
                "automatic_strategy": automatic_context,
                "validated_result_metrics": result_metrics,
            },
        )
        return LLMResult(value=parsed.value.candidate, usage=parsed.usage)


def build_llm_provider(provider: str, *, model: str = "") -> LLMProvider:
    provider_name = provider.strip().lower()
    if provider_name == "mock":
        return MockLLMProvider()
    if provider_name == "openai":
        if os.getenv("BURRIER_ENABLE_LIVE_LLM", "false").lower() != "true":
            raise LiveLLMDisabledError(
                "Live LLM use requires BURRIER_ENABLE_LIVE_LLM=true."
            )
        return OpenAILLMProvider(model=model)
    raise LLMProviderError(f"Unsupported LLM provider: {provider}")
