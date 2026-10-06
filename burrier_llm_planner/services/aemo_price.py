from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd


NEMWEB_PREDISPATCH_URL = (
    "https://nemweb.com.au/Reports/Current/PredispatchIS_Reports/"
)


class AEMOError(RuntimeError):
    """Base error for AEMO pre-dispatch handling."""


class AEMODownloadError(AEMOError):
    """Raised when NEMWEB cannot be accessed or parsed."""


class AEMOValidationError(AEMOError):
    """Raised when a forecast is unsafe to use for a planning horizon."""


@dataclass(frozen=True)
class ValidatedPriceWindow:
    frame: pd.DataFrame
    coverage_complete: bool
    planning_date: date
    source_hash: str
    issue_time: datetime | None = None
    retrieved_at: datetime | None = None
    used_cache: bool = False
    requires_operator_confirmation: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _normalise_timestamps(
    values: pd.Series, timezone: str = "Australia/Sydney"
) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        bad_rows = parsed[parsed.isna()].index.tolist()[:5]
        raise AEMOValidationError(
            f"AEMO forecast contains invalid timestamps at rows {bad_rows}."
        )
    if parsed.dt.tz is None:
        try:
            return parsed.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        except (TypeError, ValueError) as exc:
            raise AEMOValidationError(
                "AEMO timestamps could not be localised to Australia/Sydney."
            ) from exc
    return parsed.dt.tz_convert(timezone)


def validate_price_window(
    frame: pd.DataFrame,
    planning_date: date,
    *,
    timezone: str = "Australia/Sydney",
    issue_time: datetime | None = None,
    retrieved_at: datetime | None = None,
    used_cache: bool = False,
) -> ValidatedPriceWindow:
    required = {"DateTime", "Price"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise AEMOValidationError(
            "AEMO forecast is missing columns: " + ", ".join(missing_columns)
        )

    data = frame.loc[:, ["DateTime", "Price"]].copy()
    data["DateTime"] = _normalise_timestamps(data["DateTime"], timezone)
    numeric_prices = pd.to_numeric(data["Price"], errors="coerce")
    if numeric_prices.isna().any():
        rows = numeric_prices[numeric_prices.isna()].index.tolist()[:5]
        raise AEMOValidationError(
            f"AEMO forecast contains non-numeric prices at rows {rows}."
        )
    data["Price"] = numeric_prices.astype(float)

    zone = ZoneInfo(timezone)
    start = pd.Timestamp(datetime.combine(planning_date, datetime.min.time(), zone))
    expected = pd.date_range(start=start, periods=48, freq="30min")
    window = data[data["DateTime"].isin(expected)].copy()

    duplicate_mask = window["DateTime"].duplicated(keep=False)
    if duplicate_mask.any():
        examples = (
            window.loc[duplicate_mask, "DateTime"]
            .dt.strftime("%Y-%m-%d %H:%M")
            .unique()
            .tolist()[:5]
        )
        raise AEMOValidationError(
            "AEMO forecast contains duplicate intervals: " + ", ".join(examples)
        )

    actual_index = pd.DatetimeIndex(window["DateTime"])
    missing_intervals = expected.difference(actual_index)
    if len(missing_intervals):
        examples = [stamp.strftime("%Y-%m-%d %H:%M") for stamp in missing_intervals[:8]]
        raise AEMOValidationError(
            "AEMO forecast does not cover the full 24-hour horizon; missing: "
            + ", ".join(examples)
        )

    window = window.sort_values("DateTime").reset_index(drop=True)
    canonical = window.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    warnings: tuple[str, ...] = ()
    if used_cache:
        warnings = (
            "Cached AEMO forecast is in use and requires explicit operator confirmation.",
        )
    return ValidatedPriceWindow(
        frame=window,
        coverage_complete=True,
        planning_date=planning_date,
        source_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        issue_time=issue_time,
        retrieved_at=retrieved_at,
        used_cache=used_cache,
        requires_operator_confirmation=used_cache,
        warnings=warnings,
    )


def parse_predispatch_filename(filename: str) -> datetime | None:
    match = re.search(r"PREDISPATCHIS_\d{12}_(\d{14})", filename.upper())
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(
        tzinfo=ZoneInfo("Australia/Sydney")
    )


def parse_predispatch_target(filename: str) -> datetime | None:
    match = re.search(r"PREDISPATCHIS_(\d{12})_", filename.upper())
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d%H%M")


def parse_predispatch_zip(content: bytes) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = [name for name in archive.namelist() if name.upper().endswith(".CSV")]
            if not names:
                raise AEMODownloadError("Pre-dispatch ZIP contains no CSV file.")
            csv_text = archive.read(names[0]).decode("utf-8-sig")
    except (zipfile.BadZipFile, UnicodeDecodeError) as exc:
        raise AEMODownloadError("Pre-dispatch download is not a valid AEMO ZIP.") from exc

    rows = list(csv.reader(io.StringIO(csv_text)))
    header: list[str] | None = None
    table_prefix: str | None = None
    header_index = -1
    for index, row in enumerate(rows):
        if row and row[0] == "I" and "REGION_PRICES" in row:
            header = [item.strip() for item in row]
            table_prefix = row[1].strip() if len(row) > 1 else None
            header_index = index
            break
    if header is None or table_prefix is None:
        raise AEMODownloadError("REGION_PRICES header not found in AEMO file.")

    try:
        region_index = header.index("REGIONID")
        time_index = header.index("DATETIME")
        price_index = header.index("RRP")
    except ValueError as exc:
        raise AEMODownloadError(
            "AEMO REGION_PRICES header lacks REGIONID, DATETIME, or RRP."
        ) from exc

    records: list[dict[str, object]] = []
    for row in rows[header_index + 1 :]:
        if len(row) <= max(region_index, time_index, price_index):
            continue
        if row[0] != "D" or row[1].strip() != table_prefix:
            continue
        if "REGION_PRICES" not in row[:4] or row[region_index].strip() != "NSW1":
            continue
        try:
            timestamp = pd.to_datetime(row[time_index].strip(), format="%Y/%m/%d %H:%M:%S")
            price = float(row[price_index].strip())
        except (TypeError, ValueError) as exc:
            raise AEMODownloadError("Malformed NSW1 price row in AEMO file.") from exc
        records.append({"DateTime": timestamp, "Price": price})
    if not records:
        raise AEMODownloadError("AEMO file contains no NSW1 REGION_PRICES rows.")
    return pd.DataFrame.from_records(records)


class AEMOPriceService:
    def __init__(
        self,
        cache_dir: Path,
        timeout_seconds: float = 30.0,
        timezone: str = "Australia/Sydney",
        session=None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.timezone = timezone
        if session is None:
            try:
                import requests
            except ImportError as exc:
                raise AEMODownloadError(
                    "Live AEMO retrieval requires the requests package."
                ) from exc
            session = requests.Session()
        self.session = session

    def _listing_links(self) -> list[str]:
        try:
            from bs4 import BeautifulSoup
            listing = self.session.get(
                NEMWEB_PREDISPATCH_URL, timeout=self.timeout_seconds
            )
            listing.raise_for_status()
            links = [
                str(link.get("href"))
                for link in BeautifulSoup(listing.text, "html.parser").find_all(
                    "a", href=True
                )
                if str(link.get("href")).upper().endswith(".ZIP")
                and "PREDISPATCHIS" in str(link.get("href")).upper()
            ]
        except Exception as exc:
            raise AEMODownloadError(
                "Unable to access the AEMO pre-dispatch directory."
            ) from exc
        if not links:
            raise AEMODownloadError("NEMWEB listing contains no pre-dispatch ZIP.")
        return links

    def fetch_manual_day(self, planning_date: date) -> ValidatedPriceWindow:
        """Fetch the newest report issued for the start of a complete calendar day."""
        day_start = datetime.combine(planning_date, datetime.min.time())
        candidates = sorted(
            (
                (target, link)
                for link in self._listing_links()
                if (target := parse_predispatch_target(link)) is not None
                and target <= day_start
            ),
            reverse=True,
        )
        if not candidates:
            raise AEMODownloadError(
                "No AEMO pre-dispatch report can cover the selected day."
            )
        last_error: Exception | None = None
        for _target, link in candidates[:8]:
            try:
                response = self.session.get(
                    urljoin(NEMWEB_PREDISPATCH_URL, link),
                    timeout=max(self.timeout_seconds, 60.0),
                )
                response.raise_for_status()
                frame = parse_predispatch_zip(response.content)
                validated = validate_price_window(
                    frame,
                    planning_date,
                    timezone=self.timezone,
                    issue_time=parse_predispatch_filename(link.rsplit("/", 1)[-1]),
                    retrieved_at=datetime.now(ZoneInfo(self.timezone)),
                )
                self._write_cache(validated)
                return validated
            except Exception as exc:
                last_error = exc
        raise AEMODownloadError(
            "AEMO reports did not contain a complete 48-interval day."
        ) from last_error

    def fetch(self, planning_date: date) -> ValidatedPriceWindow:
        try:
            zip_links = sorted(self._listing_links())
            filename = str(zip_links[-1]).split("/")[-1]
            response = self.session.get(
                urljoin(NEMWEB_PREDISPATCH_URL, str(zip_links[-1])),
                timeout=max(self.timeout_seconds, 60.0),
            )
            response.raise_for_status()
        except AEMODownloadError:
            raise
        except Exception as exc:
            raise AEMODownloadError(f"Unable to retrieve AEMO forecast: {exc}") from exc

        frame = parse_predispatch_zip(response.content)
        retrieved_at = datetime.now(ZoneInfo(self.timezone))
        issue_time = parse_predispatch_filename(filename)
        validated = validate_price_window(
            frame,
            planning_date,
            timezone=self.timezone,
            issue_time=issue_time,
            retrieved_at=retrieved_at,
        )
        self._write_cache(validated)
        return validated

    def load_cache(self, planning_date: date) -> ValidatedPriceWindow:
        csv_path = self.cache_dir / f"aemo_nsw1_{planning_date:%Y%m%d}.csv"
        metadata_path = csv_path.with_suffix(".json")
        if not csv_path.is_file() or not metadata_path.is_file():
            raise AEMODownloadError("No cached AEMO forecast is available.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(csv_path)
        issue_time = (
            datetime.fromisoformat(metadata["issue_time"])
            if metadata.get("issue_time")
            else None
        )
        retrieved_at = (
            datetime.fromisoformat(metadata["retrieved_at"])
            if metadata.get("retrieved_at")
            else None
        )
        return validate_price_window(
            frame,
            planning_date,
            timezone=self.timezone,
            issue_time=issue_time,
            retrieved_at=retrieved_at,
            used_cache=True,
        )

    def fetch_or_cache(self, planning_date: date) -> ValidatedPriceWindow:
        try:
            return self.fetch(planning_date)
        except AEMODownloadError:
            return self.load_cache(planning_date)

    def _write_cache(self, validated: ValidatedPriceWindow) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.cache_dir / f"aemo_nsw1_{validated.planning_date:%Y%m%d}.csv"
        metadata_path = csv_path.with_suffix(".json")
        metadata = {
            "issue_time": validated.issue_time.isoformat()
            if validated.issue_time
            else None,
            "retrieved_at": validated.retrieved_at.isoformat()
            if validated.retrieved_at
            else None,
            "source_hash": validated.source_hash,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".csv", delete=False, dir=self.cache_dir
        ) as csv_temp:
            validated.frame.to_csv(csv_temp, index=False)
            csv_temp_path = Path(csv_temp.name)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False, dir=self.cache_dir
        ) as meta_temp:
            json.dump(metadata, meta_temp, indent=2)
            meta_temp_path = Path(meta_temp.name)
        csv_temp_path.replace(csv_path)
        meta_temp_path.replace(metadata_path)
