from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class AppSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_dir: Path = Path(__file__).resolve().parents[1]
    timezone: str = "Australia/Sydney"
    strategy_max_age_days: int = 7
    enable_live_llm: bool = False
    llm_provider: str = "mock"
    llm_model: str = ""

    @classmethod
    def from_environment(cls) -> "AppSettings":
        return cls(
            enable_live_llm=os.getenv("BURRIER_ENABLE_LIVE_LLM", "false").lower()
            == "true",
            llm_provider=os.getenv("BURRIER_LLM_PROVIDER", "mock"),
            llm_model=os.getenv("BURRIER_LLM_MODEL", ""),
        )

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def audit_dir(self) -> Path:
        return self.data_dir / "audit"
