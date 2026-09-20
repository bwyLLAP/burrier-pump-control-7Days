from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Callable, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from burrier_llm_planner.domain.models import PlantConfig
from burrier_llm_planner.services.aemo_price import (
    AEMOPriceService,
    ValidatedPriceWindow,
    validate_price_window,
)
from burrier_llm_planner.services.price_index import (
    PriceIndexLoadResult,
    PriceIndexService,
)


ZONE = ZoneInfo("Australia/Sydney")
PROJECT_DIR = Path(__file__).resolve().parents[1]
GroupState = Literal["valid", "invalid"]


class PreparedPlanningInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    price_window: ValidatedPriceWindow | None = None
    strategy_load: PriceIndexLoadResult | None = None
    reservoir_fraction: float = 0.95
    demand_ml: float = 45.0
    initial_pump_on: bool = False
    elapsed_state_minutes: int = 240
    equipment_available: bool = True
    operator_id: str = "operator-1"
    measurement_time: datetime
    group_status: dict[str, GroupState]
    errors: list[str] = Field(default_factory=list)

    @property
    def all_groups_valid(self) -> bool:
        return all(value == "valid" for value in self.group_status.values())


def demonstration_prices(
    planning_date: date, plant: PlantConfig
) -> ValidatedPriceWindow:
    timestamps = pd.date_range(
        datetime.combine(planning_date, datetime.min.time(), ZONE),
        periods=plant.interval_count,
        freq=f"{plant.time_step_minutes}min",
    )
    hour = np.arange(plant.interval_count) * plant.time_step_minutes / 60.0
    prices = 65 + 34 * np.exp(-((hour - 18.0) / 2.5) ** 2) - 27 * np.exp(
        -((hour - 3.0) / 2.3) ** 2
    )
    return validate_price_window(
        pd.DataFrame({"DateTime": timestamps, "Price": prices}), planning_date
    )


def prepare_planning_inputs(
    *,
    planning_date: date,
    demo_mode: bool,
    strategy_path: Path,
    plant: PlantConfig,
    now: datetime | None = None,
    aemo_loader: Callable[[date], ValidatedPriceWindow] | None = None,
) -> PreparedPlanningInputs:
    current_time = now or datetime.now(ZONE)
    group_status: dict[str, GroupState] = {
        "AEMO forecast": "invalid",
        "Seasonal strategy": "invalid",
        "Plant state": "valid",
    }
    errors: list[str] = []
    price_window: ValidatedPriceWindow | None = None
    strategy_load: PriceIndexLoadResult | None = None

    try:
        if demo_mode:
            price_window = demonstration_prices(planning_date, plant)
        else:
            loader = aemo_loader or AEMOPriceService(
                PROJECT_DIR / "data" / "cache"
            ).fetch_or_cache
            price_window = loader(planning_date)
        group_status["AEMO forecast"] = "valid"
    except Exception as exc:
        errors.append(f"AEMO forecast: {exc}")

    try:
        strategy_load = PriceIndexService(plant=plant).load(
            Path(strategy_path), planning_date, live_mode=not demo_mode
        )
        group_status["Seasonal strategy"] = "valid"
    except Exception as exc:
        errors.append(f"Seasonal strategy: {exc}")

    return PreparedPlanningInputs(
        price_window=price_window,
        strategy_load=strategy_load,
        measurement_time=current_time.replace(microsecond=0),
        group_status=group_status,
        errors=errors,
    )
