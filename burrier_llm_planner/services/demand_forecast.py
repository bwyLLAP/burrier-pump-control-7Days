from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DemandForecastError(RuntimeError):
    """Raised when a deterministic demand forecast cannot be produced."""


class DemandForecastResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planning_date: date
    raw_forecast_ml: float
    corrected_forecast_ml: float
    correction_ml: float
    method: str
    source: str
    warnings: list[str] = Field(default_factory=list)


class DemandForecastService:
    def __init__(
        self,
        method: Literal["historical_actual", "weekday_average"] = "weekday_average",
        weekday_lookback_weeks: int = 4,
        correction_gain: float = 1.0,
    ) -> None:
        if weekday_lookback_weeks < 1:
            raise ValueError("weekday_lookback_weeks must be at least one.")
        if not 0.0 <= correction_gain <= 2.0:
            raise ValueError("correction_gain must lie in [0, 2].")
        self.method = method
        self.weekday_lookback_weeks = weekday_lookback_weeks
        self.correction_gain = correction_gain

    def forecast(
        self,
        history: pd.DataFrame,
        planning_date: date,
        *,
        previous_actual_ml: float | None = None,
        previous_forecast_ml: float | None = None,
        source: str = "local_demand_history",
    ) -> DemandForecastResult:
        data = self._normalise(history)
        target = pd.Timestamp(planning_date)

        if self.method == "historical_actual":
            match = data[data["Date"] == target]
            if match.empty:
                raise DemandForecastError(
                    "Historical-actual mode requires a value for the planning date."
                )
            raw = float(match.iloc[-1]["DemandML"])
        else:
            prior = data[
                (data["Date"] < target)
                & (data["Date"].dt.weekday == target.weekday())
            ].tail(self.weekday_lookback_weeks)
            if prior.empty:
                raise DemandForecastError(
                    "No prior matching weekdays are available for demand forecasting."
                )
            raw = float(prior["DemandML"].mean())

        correction = 0.0
        if previous_actual_ml is not None and previous_forecast_ml is not None:
            correction = self.correction_gain * (
                float(previous_actual_ml) - float(previous_forecast_ml)
            )
        corrected = raw + correction
        if corrected < 0:
            raise DemandForecastError("Corrected demand forecast cannot be negative.")
        return DemandForecastResult(
            planning_date=planning_date,
            raw_forecast_ml=raw,
            corrected_forecast_ml=corrected,
            correction_ml=correction,
            method=self.method,
            source=source,
        )

    @staticmethod
    def _normalise(history: pd.DataFrame) -> pd.DataFrame:
        required = {"Date", "DemandML"}
        missing = sorted(required - set(history.columns))
        if missing:
            raise DemandForecastError(
                "Demand history is missing columns: " + ", ".join(missing)
            )
        data = history.loc[:, ["Date", "DemandML"]].copy()
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
        data["DemandML"] = pd.to_numeric(data["DemandML"], errors="coerce")
        data = data.dropna().sort_values("Date").drop_duplicates("Date", keep="last")
        if data.empty:
            raise DemandForecastError("Demand history contains no valid daily values.")
        return data
