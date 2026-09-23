"""Flow Power KWatch actual and short-term forecast price retrieval."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd


ZONE = "Australia/Sydney"
BASE_URL = "https://api.kwatch.com.au/api/v1"


class KWatchPriceError(RuntimeError):
    """Raised when KWatch cannot provide a continuous price window."""


class KWatchPriceService:
    def __init__(self, api_key: str, *, session=None, timeout_seconds: float = 30.0) -> None:
        if not str(api_key).strip():
            raise ValueError("KWatch API key is required.")
        self.api_key = str(api_key).strip()
        self.timeout_seconds = float(timeout_seconds)
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    def _request(self, endpoint: str, period: int) -> pd.DataFrame:
        try:
            response = self.session.get(
                f"{BASE_URL}/{endpoint}",
                params={
                    "apikey": self.api_key,
                    "regName": "nsw",
                    "period": int(period),
                },
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise KWatchPriceError(
                f"KWatch {endpoint} connection failed."
            ) from exc
        try:
            response.raise_for_status()
        except Exception as exc:
            status = getattr(response, "status_code", None)
            suffix = f" (HTTP {status})" if status is not None else ""
            raise KWatchPriceError(
                f"KWatch {endpoint} request failed{suffix}."
            ) from exc
        payload = response.json()
        while isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, list):
            raise KWatchPriceError(f"KWatch {endpoint} returned an invalid response.")
        if not payload:
            return pd.DataFrame(columns=["DateTime", "Price"])
        frame = pd.DataFrame(payload)
        if not {"Key", "Value"}.issubset(frame.columns):
            raise KWatchPriceError(f"KWatch {endpoint} is missing Key or Value.")
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame["Key"], errors="raise"))
        timestamps = timestamps.tz_localize(ZONE) if timestamps.tz is None else timestamps.tz_convert(ZONE)
        prices = pd.to_numeric(frame["Value"], errors="raise")
        if not np.isfinite(prices.to_numpy(float)).all():
            raise KWatchPriceError(f"KWatch {endpoint} returned non-finite prices.")
        result = pd.DataFrame({"DateTime": timestamps, "Price": prices.to_numpy(float)})
        if result["DateTime"].duplicated().any():
            raise KWatchPriceError(f"KWatch {endpoint} returned duplicate timestamps.")
        return result.sort_values("DateTime").reset_index(drop=True)

    @staticmethod
    def _interval_starts(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
        result = frame.copy()
        result["DateTime"] = pd.DatetimeIndex(result["DateTime"]) - pd.Timedelta(minutes=minutes)
        return result

    @staticmethod
    def _future_start(now) -> pd.Timestamp:
        current = pd.Timestamp(now)
        current = current.tz_localize(ZONE) if current.tzinfo is None else current.tz_convert(ZONE)
        return current if current == current.floor("30min") else current.ceil("30min")

    def fetch_actual(self, *, start, now) -> pd.DataFrame:
        start = pd.Timestamp(start)
        start = start.tz_localize(ZONE) if start.tzinfo is None else start.tz_convert(ZONE)
        start = start.floor("30min")
        future_start = self._future_start(now)
        period_days = max(1, min(180, math.ceil((future_start - start) / pd.Timedelta(days=1))))

        completed = self._interval_starts(
            self._request("dispatch30mins", period_days), 30,
        )
        dispatch_five = self._interval_starts(
            self._request("dispatch5mins", 180), 5,
        )
        forecast_five = self._interval_starts(
            self._request("predispatch5mins", 60), 5,
        )
        five = pd.concat([forecast_five, dispatch_five], ignore_index=True)
        five = five.drop_duplicates("DateTime", keep="last").sort_values("DateTime")
        if five.empty:
            bridge = pd.DataFrame(columns=["DateTime", "Price"])
        else:
            five["HalfHour"] = pd.DatetimeIndex(five["DateTime"]).floor("30min")
            bridge = five.groupby("HalfHour")["Price"].agg(["mean", "count"])
            bridge = bridge[bridge["count"] == 6].reset_index()
            bridge = bridge.rename(columns={"HalfHour": "DateTime", "mean": "Price"})
            bridge = bridge.loc[:, ["DateTime", "Price"]]

        populated = [frame for frame in (bridge, completed) if not frame.empty]
        combined = (
            pd.concat(populated, ignore_index=True)
            if populated else pd.DataFrame(columns=["DateTime", "Price"])
        )
        combined = combined.drop_duplicates("DateTime", keep="last")
        combined = combined[(combined["DateTime"] >= start) & (combined["DateTime"] < future_start)]
        return combined.sort_values("DateTime").reset_index(drop=True).loc[:, ["DateTime", "Price"]]

    def fetch_predispatch(self, *, now) -> pd.DataFrame:
        future_start = self._future_start(now)
        forecast = self._interval_starts(
            self._request("predispatch30mins", 7), 30,
        )
        forecast = forecast[forecast["DateTime"] >= future_start].copy()
        if forecast.empty:
            return pd.DataFrame(columns=["DateTime", "Price"])
        return forecast.sort_values("DateTime").reset_index(drop=True).loc[:, ["DateTime", "Price"]]
