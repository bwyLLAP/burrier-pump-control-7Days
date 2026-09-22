"""Rolling NSW1 price history and forecast composition for weekly planning."""
from __future__ import annotations

import csv
import io
import math
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .aemo_price import NEMWEB_PREDISPATCH_URL, parse_predispatch_zip
from .kwatch_prices import KWatchPriceError, KWatchPriceService
from .pd7day_price import PD7DayPriceService
from .weekly_planning import cap_weekly_prices


ZONE_NAME = "Australia/Sydney"
ZONE = ZoneInfo(ZONE_NAME)
NEM_TIMEZONE = "Etc/GMT-10"  # AEMO market timestamps remain UTC+10 (AEST).
DISPATCH_CURRENT_URL = "https://www.nemweb.com.au/Reports/CURRENT/DispatchIS_Reports/"
DISPATCH_ARCHIVE_URL = "https://www.nemweb.com.au/REPORTS/ARCHIVE/DispatchIS_Reports/"
DISPATCH_DASHBOARD_URL = "https://visualisations.aemo.com.au/aemo/apps/api/report/5MIN"
DASHBOARD_CACHE_NAME = "aemo_dashboard_5min_nsw1.csv"
DISPATCH_NAME = re.compile(r"PUBLIC_DISPATCHIS_(\d{12})_\d+\.zip$", re.IGNORECASE)
DISPATCH_ARCHIVE_NAME = re.compile(r"PUBLIC_DISPATCHIS_(\d{8})\.zip$", re.IGNORECASE)


class RollingPriceError(RuntimeError):
    """Raised when a safe continuous rolling price horizon cannot be built."""


@dataclass(frozen=True)
class VolumeCorrection:
    actual_minus_planned_ml: float
    adjusted_future_target_ml: float


@dataclass(frozen=True)
class RollingPriceWindow:
    frame: pd.DataFrame
    optimisation_frame: pd.DataFrame
    allocation_frame: pd.DataFrame
    default_display_start: pd.Timestamp
    default_display_end: pd.Timestamp
    retrieved_at: pd.Timestamp
    source_names: tuple[str, ...] = ()


def calendar_week_prices(
    price_frame: pd.DataFrame,
    monday: pd.Timestamp | datetime | date,
) -> pd.DataFrame:
    """Extract a complete local Monday-to-Sunday price set for the baseline plan."""
    start = pd.Timestamp(monday)
    start = start.tz_localize(ZONE_NAME) if start.tzinfo is None else start.tz_convert(ZONE_NAME)
    start = start.normalize()
    end = start + pd.DateOffset(days=7)
    expected = pd.date_range(start, end, freq="30min", inclusive="left")
    frame = price_frame.copy()
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["DateTime"], errors="raise"))
    timestamps = timestamps.tz_localize(ZONE_NAME) if timestamps.tz is None else timestamps.tz_convert(ZONE_NAME)
    frame["DateTime"] = timestamps
    selected = frame[(frame["DateTime"] >= start) & (frame["DateTime"] < end)].copy()
    if selected["DateTime"].duplicated().any():
        raise RollingPriceError("The calendar-week baseline contains duplicate half-hours.")
    selected = selected.set_index("DateTime").reindex(expected)
    if "Price" not in selected or selected["Price"].isna().any():
        missing = int(selected["Price"].isna().sum()) if "Price" in selected else len(expected)
        raise RollingPriceError(
            f"The calendar-week baseline has {missing} missing half-hour prices."
        )
    selected.index.name = "DateTime"
    return selected.reset_index()


def calculate_volume_correction(
    base_future_target_ml: float,
    actual_to_date_ml: float,
    planned_to_date_ml: float,
) -> VolumeCorrection:
    values = np.asarray(
        [base_future_target_ml, actual_to_date_ml, planned_to_date_ml], dtype=float
    )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Pumping volumes must be finite and non-negative.")
    variance = float(actual_to_date_ml - planned_to_date_ml)
    return VolumeCorrection(
        actual_minus_planned_ml=variance,
        adjusted_future_target_ml=max(0.0, float(base_future_target_ml) - variance),
    )


def _normalise(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["DateTime", "Price", "PriceSource"])
    required = {"DateTime", "Price"}
    if not required.issubset(frame.columns):
        raise RollingPriceError(f"{source} data is missing DateTime or Price.")
    result = frame.loc[:, ["DateTime", "Price"]].copy()
    index = pd.DatetimeIndex(pd.to_datetime(result.pop("DateTime"), errors="raise"))
    index = index.tz_localize(ZONE_NAME) if index.tz is None else index.tz_convert(ZONE_NAME)
    result.insert(0, "DateTime", index)
    result["Price"] = pd.to_numeric(result["Price"], errors="raise")
    if not np.isfinite(result["Price"].to_numpy(float)).all():
        raise RollingPriceError(f"{source} prices must be finite.")
    if result["DateTime"].duplicated().any():
        raise RollingPriceError(f"{source} data contains duplicate half-hours.")
    result["PriceSource"] = source
    return result.sort_values("DateTime").reset_index(drop=True)


def compose_rolling_price_window(
    *,
    actual: pd.DataFrame,
    one_day: pd.DataFrame,
    seven_day: pd.DataFrame,
    historical_fallback: pd.DataFrame | None = None,
    actual_source: str = "Actual",
    one_day_source: str = "1-day forecast",
    now: pd.Timestamp | datetime,
    display_monday: pd.Timestamp | datetime,
) -> RollingPriceWindow:
    """Join completed actuals, short forecast, and long forecast by priority."""
    current = pd.Timestamp(now)
    current = current.tz_localize(ZONE_NAME) if current.tzinfo is None else current.tz_convert(ZONE_NAME)
    monday = pd.Timestamp(display_monday)
    monday = monday.tz_localize(ZONE_NAME) if monday.tzinfo is None else monday.tz_convert(ZONE_NAME)
    monday = monday.normalize()
    future_start = current.ceil("30min")
    if current == current.floor("30min"):
        future_start = current
    optimisation_index = pd.date_range(future_start, periods=336, freq="30min")
    optimisation_end = future_start + pd.Timedelta(days=7)
    default_end = monday + pd.Timedelta(days=7)
    display_end = max(default_end, optimisation_end)

    actual_frame = _normalise(actual, actual_source)
    one_day_frame = _normalise(one_day, one_day_source)
    seven_day_frame = _normalise(seven_day, "7-day forecast")
    fallback_frame = _normalise(
        historical_fallback, "Monday forecast fallback",
    )
    actual_frame = actual_frame[actual_frame["DateTime"] < future_start]
    # Keep the latest short forecast for the unfinished current half-hour as well.
    # Completed actual intervals still win during source-priority deduplication below.
    seven_day_frame = seven_day_frame[seven_day_frame["DateTime"] >= future_start]

    # Later concatenated sources win.  This makes the one-day forecast preferred
    # over PD7Day in its valid range, and actuals preferred for elapsed intervals.
    populated = [
        frame for frame in (
            fallback_frame, seven_day_frame, one_day_frame, actual_frame,
        ) if not frame.empty
    ]
    combined = pd.concat(populated, ignore_index=True).drop_duplicates("DateTime", keep="last")
    combined = combined[
        (combined["DateTime"] >= min(monday, future_start)) & (combined["DateTime"] < display_end)
    ].sort_values("DateTime").reset_index(drop=True)
    combined = cap_weekly_prices(combined)

    display_index = pd.date_range(
        min(monday, future_start), display_end, freq="30min", inclusive="left",
    )
    combined = combined.set_index("DateTime").reindex(display_index)
    display_missing = combined["Price"].isna()
    if display_missing.any():
        examples = ", ".join(
            stamp.strftime("%a %d %b %H:%M") for stamp in combined.index[display_missing][:5]
        )
        raise RollingPriceError(
            f"The displayed price window has {int(display_missing.sum())} missing half-hour "
            f"prices; first: {examples}."
        )
    combined.index.name = "DateTime"
    combined = combined.reset_index()

    optimisation = combined[combined["DateTime"].isin(optimisation_index)].copy()
    optimisation = optimisation.set_index("DateTime").reindex(optimisation_index)
    missing = optimisation["Price"].isna()
    if missing.any():
        examples = ", ".join(stamp.strftime("%a %d %b %H:%M") for stamp in optimisation.index[missing][:5])
        raise RollingPriceError(
            f"The future seven-day horizon has {int(missing.sum())} missing half-hour prices; first: {examples}."
        )
    optimisation.index.name = "DateTime"
    optimisation = optimisation.reset_index()

    # Preserve the underlying seven-day forecast separately.  The first-stage
    # optimiser uses it to allocate volume between days before the short-term
    # forecast is allowed to move tomorrow's operating window.
    allocation = seven_day_frame.set_index("DateTime").reindex(optimisation_index)
    missing_allocation = allocation["Price"].isna()
    if missing_allocation.any():
        combined_optimisation = optimisation.set_index("DateTime")
        allocation.loc[missing_allocation, "Price"] = combined_optimisation.loc[
            missing_allocation, "Price"
        ]
        allocation.loc[missing_allocation, "PriceSource"] = "Combined forecast fallback"
    allocation.index.name = "DateTime"
    allocation = cap_weekly_prices(allocation.reset_index())
    return RollingPriceWindow(
        frame=combined,
        optimisation_frame=optimisation,
        allocation_frame=allocation,
        default_display_start=monday,
        default_display_end=default_end,
        retrieved_at=pd.Timestamp.now(tz="UTC"),
    )


def _market_interval_starts(values: pd.Series, minutes: int) -> pd.DatetimeIndex:
    ends = pd.DatetimeIndex(pd.to_datetime(values, errors="raise", format="mixed"))
    ends = ends.tz_localize(NEM_TIMEZONE) if ends.tz is None else ends.tz_convert(NEM_TIMEZONE)
    return (ends - pd.Timedelta(minutes=minutes)).tz_convert(ZONE_NAME)


def _parse_dispatch_zip(content: bytes, source_name: str) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise RollingPriceError(f"{source_name} contains no CSV.")
            rows = list(csv.reader(io.StringIO(archive.read(csv_names[0]).decode("utf-8-sig"))))
    except (zipfile.BadZipFile, UnicodeDecodeError) as exc:
        raise RollingPriceError(f"{source_name} is not a valid Dispatch ZIP.") from exc
    header = None
    records: list[dict[str, object]] = []
    for row in rows:
        if row[:4] == ["I", "DISPATCH", "PRICE", "5"]:
            header = row[4:]
        elif header is not None and row[:4] == ["D", "DISPATCH", "PRICE", "5"]:
            record = dict(zip(header, row[4:]))
            if record.get("REGIONID") == "NSW1" and record.get("INTERVENTION") == "0":
                price = float(record["RRP"])
                if not math.isfinite(price):
                    raise RollingPriceError("Dispatch RRP is not finite.")
                records.append({"DateTime": record["SETTLEMENTDATE"], "Price": price})
    if not records:
        raise RollingPriceError(f"{source_name} contains no NSW1 Dispatch price.")
    return pd.DataFrame(records)


def _parse_dispatch_archive(content: bytes, source_name: str) -> pd.DataFrame:
    frames = []
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".zip"):
                    frames.append(_parse_dispatch_zip(archive.read(name), name))
    except zipfile.BadZipFile as exc:
        raise RollingPriceError(f"{source_name} is not a valid Dispatch archive.") from exc
    if not frames:
        raise RollingPriceError(f"{source_name} contains no Dispatch reports.")
    return pd.concat(frames, ignore_index=True)


def _dispatch_half_hours(five_minute: pd.DataFrame) -> pd.DataFrame:
    if five_minute.empty:
        return pd.DataFrame(columns=["DateTime", "Price"])
    frame = five_minute.copy()
    starts = _market_interval_starts(frame["DateTime"], 5)
    frame["DateTime"] = starts.floor("30min")
    frame["DispatchStart"] = starts
    frame = frame.drop_duplicates("DispatchStart", keep="last")
    grouped = frame.groupby("DateTime")["Price"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] == 6]
    return grouped.reset_index().rename(columns={"mean": "Price"}).loc[:, ["DateTime", "Price"]]


class RollingAEMOPriceService:
    """Retrieve the three AEMO sources needed by the rolling weekly dashboard."""

    def __init__(
        self, cache_dir: Path, *, session=None, timeout_seconds: float = 60.0,
        kwatch_api_key: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.kwatch_api_key = str(kwatch_api_key).strip() if kwatch_api_key else None
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    def _links(self, url: str) -> list[str]:
        from bs4 import BeautifulSoup
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return [
            str(anchor.get("href"))
            for anchor in BeautifulSoup(response.text, "html.parser").find_all("a", href=True)
        ]

    def _download(self, url: str) -> bytes:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        destination = self.cache_dir / url.rsplit("/", 1)[-1]
        if destination.is_file():
            return destination.read_bytes()
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        content = response.content
        if content[:2] != b"PK":
            raise RollingPriceError(f"AEMO returned an invalid ZIP for {destination.name}.")
        destination.write_bytes(content)
        return content

    def _dashboard_actual_rows(self, now: pd.Timestamp) -> pd.DataFrame:
        """Read the rolling 48-hour actual feed with retry and a recent disk fallback."""
        cache_path = self.cache_dir / DASHBOARD_CACHE_NAME
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.session.post(
                    DISPATCH_DASHBOARD_URL,
                    json={"timeScale": ["5MIN"]},
                    timeout=min(self.timeout_seconds, 15.0),
                )
                response.raise_for_status()
                records = response.json().get("5MIN", [])
                rows = pd.DataFrame([
                    {"DateTime": row.get("SETTLEMENTDATE"), "Price": row.get("RRP")}
                    for row in records
                    if row.get("REGIONID") == "NSW1" and row.get("PERIODTYPE") == "ACTUAL"
                ])
                if rows.empty or rows[["DateTime", "Price"]].isna().any().any():
                    raise RollingPriceError("AEMO 5MIN returned no complete NSW1 actual prices.")
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                rows.to_csv(cache_path, index=False)
                return rows
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))

        if cache_path.is_file():
            cached = pd.read_csv(cache_path)
            if {"DateTime", "Price"}.issubset(cached.columns) and not cached.empty:
                interval_starts = _market_interval_starts(cached["DateTime"], 5)
                current = pd.Timestamp(now)
                current = current.tz_localize(ZONE_NAME) if current.tzinfo is None else current.tz_convert(ZONE_NAME)
                if current - interval_starts.max() <= pd.Timedelta(minutes=90):
                    return cached.loc[:, ["DateTime", "Price"]]

        raise RollingPriceError(
            f"AEMO 5MIN actual prices failed after 3 attempts and no recent cache is available: {last_error}"
        ) from last_error

    def _actual_prices(self, start: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
        archive_links = [
            link for link in self._links(DISPATCH_ARCHIVE_URL)
            if DISPATCH_ARCHIVE_NAME.search(link)
        ]
        dated_archives = sorted(
            (datetime.strptime(DISPATCH_ARCHIVE_NAME.search(link).group(1), "%Y%m%d").date(), link)
            for link in archive_links
        )
        frames = []
        latest_archive_date: date | None = None
        for archive_date, link in dated_archives:
            if start.date() <= archive_date <= now.date():
                frames.append(_parse_dispatch_archive(
                    self._download(urljoin(DISPATCH_ARCHIVE_URL, link)), link.rsplit("/", 1)[-1]
                ))
                latest_archive_date = archive_date

        current_rows = self._dashboard_actual_rows(now)
        if not current_rows.empty:
            frames.append(current_rows)
        if not frames:
            return pd.DataFrame(columns=["DateTime", "Price"])
        return self._complete_actual_half_hours(
            pd.concat(frames, ignore_index=True).drop_duplicates("DateTime"),
            start=start,
            now=now,
        )

    def _one_day_prices(self) -> tuple[pd.DataFrame, str]:
        links = [link for link in self._links(NEMWEB_PREDISPATCH_URL) if link.lower().endswith(".zip")]
        if not links:
            raise RollingPriceError("AEMO PreDispatch directory contains no ZIP reports.")
        latest = max(links)
        frame = parse_predispatch_zip(self._download(urljoin(NEMWEB_PREDISPATCH_URL, latest)))
        frame["DateTime"] = _market_interval_starts(frame["DateTime"], 30)
        return frame, latest.rsplit("/", 1)[-1]

    @staticmethod
    def _report_half_hour(name: str) -> pd.Timestamp | None:
        match = DISPATCH_NAME.search(Path(name).name)
        if match is None:
            return None
        settlement = pd.Timestamp(
            datetime.strptime(match.group(1), "%Y%m%d%H%M"), tz=NEM_TIMEZONE,
        )
        return (settlement - pd.Timedelta(minutes=5)).tz_convert(ZONE_NAME).floor("30min")

    def _complete_actual_half_hours(
        self,
        five_minute: pd.DataFrame,
        *,
        start: pd.Timestamp,
        now: pd.Timestamp,
    ) -> pd.DataFrame:
        """Backfill archive/dashboard boundary gaps from exact Dispatch reports."""

        start = pd.Timestamp(start)
        start = start.tz_localize(ZONE_NAME) if start.tzinfo is None else start.tz_convert(ZONE_NAME)
        now = pd.Timestamp(now)
        now = now.tz_localize(ZONE_NAME) if now.tzinfo is None else now.tz_convert(ZONE_NAME)
        expected = pd.date_range(
            start.floor("30min"), now.floor("30min"), freq="30min", inclusive="left",
        )

        def missing_intervals(rows: pd.DataFrame) -> tuple[pd.DataFrame, set[pd.Timestamp]]:
            half_hours = _dispatch_half_hours(rows)
            present = set(pd.DatetimeIndex(half_hours["DateTime"]))
            return half_hours, set(expected).difference(present)

        combined = five_minute.copy()
        half_hours, missing = missing_intervals(combined)
        if not missing:
            return half_hours

        cached_paths = [
            path for path in self.cache_dir.glob("PUBLIC_DISPATCHIS_*.zip")
            if self._report_half_hour(path.name) in missing
        ]
        if cached_paths:
            combined = pd.concat([
                combined,
                *(_parse_dispatch_zip(path.read_bytes(), path.name) for path in cached_paths),
            ], ignore_index=True)
            half_hours, missing = missing_intervals(combined)
        if not missing:
            return half_hours

        links = self._links(DISPATCH_CURRENT_URL)
        selected_links = [
            link for link in links if self._report_half_hour(link) in missing
        ]
        if selected_links:
            combined = pd.concat([
                combined,
                *(
                    _parse_dispatch_zip(
                        self._download(urljoin(DISPATCH_CURRENT_URL, link)),
                        link.rsplit("/", 1)[-1],
                    )
                    for link in selected_links
                ),
            ], ignore_index=True)
        return _dispatch_half_hours(combined)

    def fetch(
        self, *, now: pd.Timestamp, display_monday: pd.Timestamp,
        strict_kwatch: bool = False,
    ) -> RollingPriceWindow:
        try:
            actual = None
            one_day = None
            actual_source = "AEMO actual"
            one_day_source = "AEMO 1-day forecast"
            source_names: list[str] = []
            if self.kwatch_api_key:
                kwatch = KWatchPriceService(
                    self.kwatch_api_key, session=self.session,
                    timeout_seconds=min(self.timeout_seconds, 45.0),
                )
                try:
                    actual = kwatch.fetch_actual(start=display_monday, now=now)
                    actual_source = "KWatch actual"
                    source_names.append("KWatch dispatch30mins + live 5min bridge")
                except KWatchPriceError as exc:
                    if strict_kwatch:
                        raise RollingPriceError(str(exc)) from exc
                    actual = None
                except Exception as exc:
                    if strict_kwatch:
                        raise RollingPriceError("KWatch actual retrieval failed.") from exc
                    actual = None
                try:
                    one_day = kwatch.fetch_predispatch(now=now)
                    one_day_source = "KWatch predispatch"
                    source_names.append("KWatch predispatch30mins")
                except KWatchPriceError as exc:
                    if strict_kwatch:
                        raise RollingPriceError(str(exc)) from exc
                    one_day = None
                except Exception as exc:
                    if strict_kwatch:
                        raise RollingPriceError("KWatch predispatch retrieval failed.") from exc
                    one_day = None
            if actual is None:
                actual = self._actual_prices(display_monday, now)
                source_names.append("AEMO Dispatch")
            if one_day is None:
                one_day, one_day_name = self._one_day_prices()
                source_names.append(one_day_name)
            pd7_service = PD7DayPriceService(session=self.session)
            pd7 = pd7_service.fetch_latest()
            seven_day = pd7.frame.copy()
            seven_day["DateTime"] = _market_interval_starts(seven_day["DateTime"], 30)
            monday_fallback = pd.DataFrame(columns=["DateTime", "Price"])
            monday_source: str | None = None
            try:
                monday_window = pd7_service.fetch_monday(
                    pd.Timestamp(display_monday).date(),
                )
                monday_fallback = monday_window.frame.copy()
                monday_fallback["DateTime"] = _market_interval_starts(
                    monday_fallback["DateTime"], 30,
                )
                monday_source = monday_window.source_name
            except Exception:
                # The released Monday forecast is a low-priority resilience
                # source. Actual, PreDispatch and current PD7Day remain usable
                # when that historical vintage is no longer listed by AEMO.
                pass
            window = compose_rolling_price_window(
                actual=actual,
                one_day=one_day,
                seven_day=seven_day,
                historical_fallback=monday_fallback,
                actual_source=actual_source,
                one_day_source=one_day_source,
                now=now,
                display_monday=display_monday,
            )
            source_names.append(pd7.source_name)
            if monday_source is not None:
                source_names.append(f"Monday fallback: {monday_source}")
            return RollingPriceWindow(
                frame=window.frame,
                optimisation_frame=window.optimisation_frame,
                allocation_frame=window.allocation_frame,
                default_display_start=window.default_display_start,
                default_display_end=window.default_display_end,
                retrieved_at=window.retrieved_at,
                source_names=tuple(source_names),
            )
        except RollingPriceError:
            raise
        except Exception as exc:
            raise RollingPriceError(f"Rolling AEMO price retrieval failed: {exc}") from exc
