from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from burrier_llm_planner.domain.enums import StrategyMode
from burrier_llm_planner.domain.models import DataProvenance, PlantConfig, ResolvedStrategy


class PriceIndexError(RuntimeError):
    """Base error for offline price-index strategy handling."""


class PriceIndexValidationError(PriceIndexError):
    """Raised when a strategy file is malformed or outside safe bounds."""


class StaleStrategyError(PriceIndexError):
    """Raised when live planning receives an out-of-date strategy."""


class PriceIndexMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    strategy_date: date
    parameter_source: str
    price_regime: str
    daily_mean_price: float | None = None
    backward_30d_price: float | None = None
    price_score: float | None = None
    raw_target_fraction: float | None = None
    calibration_year: int
    evaluation_type: str


class PriceIndexLoadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: ResolvedStrategy
    metadata: PriceIndexMetadata
    provenance: DataProvenance
    age_days: int
    is_fresh: bool
    operational_export_allowed: bool
    warnings: list[str] = Field(default_factory=list)


class PriceIndexService:
    def __init__(
        self,
        max_age_days: int = 7,
        plant: PlantConfig | None = None,
        timezone: str = "Australia/Sydney",
    ) -> None:
        if max_age_days < 0:
            raise ValueError("max_age_days cannot be negative.")
        self.max_age_days = max_age_days
        self.plant = plant or PlantConfig()
        self.timezone = timezone

    def load(
        self,
        path: Path,
        planning_date: date,
        live_mode: bool = True,
    ) -> PriceIndexLoadResult:
        source = Path(path)
        if not source.is_file():
            raise PriceIndexValidationError(f"Strategy file not found: {source}")

        source_bytes = source.read_bytes()
        return self.load_bytes(
            source_bytes,
            planning_date=planning_date,
            live_mode=live_mode,
            source_name=str(source.resolve()),
        )

    def load_bytes(
        self,
        source_bytes: bytes,
        *,
        planning_date: date,
        live_mode: bool = True,
        source_name: str = "uploaded strategy JSON",
    ) -> PriceIndexLoadResult:
        """Validate a strategy supplied in memory using the file safety boundary."""
        try:
            payload = json.loads(source_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PriceIndexValidationError("Strategy file is not valid UTF-8 JSON.") from exc
        if not isinstance(payload, dict):
            raise PriceIndexValidationError("Strategy file must contain one JSON object.")

        required = {
            "schema_version",
            "date",
            "mode",
            "reservoir_target_fraction",
            "default_transition_days",
            "parameter_source",
            "price_regime",
            "calibration_year",
            "evaluation_type",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise PriceIndexValidationError(
                "Strategy file is missing fields: " + ", ".join(missing)
            )

        strategy_date = self._parse_date(payload["date"])
        age_days = (planning_date - strategy_date).days
        if age_days < 0:
            raise PriceIndexValidationError(
                "Strategy date is after the requested planning date; future look-ahead is not allowed."
            )
        is_fresh = age_days <= self.max_age_days
        if live_mode and not is_fresh:
            raise StaleStrategyError(
                f"Price-index strategy is stale ({age_days} days old; maximum "
                f"{self.max_age_days} days) and cannot be used in live mode."
            )

        try:
            mode = StrategyMode(str(payload["mode"]).strip().lower())
            target_fraction = float(payload["reservoir_target_fraction"])
            transition_days = int(payload["default_transition_days"])
        except (TypeError, ValueError) as exc:
            raise PriceIndexValidationError(
                "Strategy mode, target fraction, or transition period is invalid."
            ) from exc

        if not self.plant.min_reservoir_fraction <= target_fraction <= 1.0:
            raise PriceIndexValidationError(
                "Price-index target is outside the plant admissible range "
                f"[{self.plant.min_reservoir_fraction:.2f}, 1.00]."
            )
        if not 1 <= transition_days <= 365:
            raise PriceIndexValidationError(
                "Price-index transition period must be an integer within [1, 365]."
            )
        if mode is not StrategyMode.ARBITRAGE:
            raise PriceIndexValidationError(
                "The automatic price-index strategy must use arbitrage mode."
            )

        warnings: list[str] = []
        if not is_fresh:
            warnings.append(
                "Historical strategy loaded in demonstration mode; it is not a "
                "current operational control recommendation."
            )

        strategy = ResolvedStrategy(
            mode=mode,
            target_fraction=target_fraction,
            transition_days=transition_days,
            issued_date=planning_date,
            target_date=planning_date.fromordinal(
                planning_date.toordinal() + transition_days - 1
            ),
            target_fraction_source=str(payload["parameter_source"]),
            transition_days_source="price_index_default",
            verification_reasons=[
                "Offline strategy schema validated.",
                "Target fraction lies inside the immutable plant safety range.",
            ],
        )
        metadata = PriceIndexMetadata(
            schema_version=int(payload["schema_version"]),
            strategy_date=strategy_date,
            parameter_source=str(payload["parameter_source"]),
            price_regime=str(payload["price_regime"]),
            daily_mean_price=self._optional_float(payload.get("daily_mean_price")),
            backward_30d_price=self._optional_float(payload.get("backward_30d_price")),
            price_score=self._optional_float(payload.get("price_score")),
            raw_target_fraction=self._optional_float(payload.get("raw_target_fraction")),
            calibration_year=int(payload["calibration_year"]),
            evaluation_type=str(payload["evaluation_type"]),
        )
        issue_time = datetime.combine(
            strategy_date, time.min, ZoneInfo(self.timezone)
        )
        provenance = DataProvenance(
            source=source_name,
            issue_time=issue_time,
            source_hash=hashlib.sha256(source_bytes).hexdigest(),
            is_fresh=is_fresh,
            warnings=warnings,
        )
        return PriceIndexLoadResult(
            strategy=strategy,
            metadata=metadata,
            provenance=provenance,
            age_days=age_days,
            is_fresh=is_fresh,
            operational_export_allowed=live_mode and is_fresh,
            warnings=warnings,
        )

    @staticmethod
    def _parse_date(value: Any) -> date:
        try:
            return date.fromisoformat(str(value))
        except ValueError as exc:
            raise PriceIndexValidationError(
                "Strategy date must use ISO format YYYY-MM-DD."
            ) from exc

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise PriceIndexValidationError(
                "Optional price-index metrics must be numeric or null."
            ) from exc
