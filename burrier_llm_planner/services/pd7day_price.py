from __future__ import annotations

import csv
import io
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin

import pandas as pd


PD7DAY_DIRECTORY_URL = "https://www.nemweb.com.au/REPORTS/CURRENT/PD7Day/"
ZIP_NAME_PATTERN = re.compile(r"PUBLIC_PD7DAY_\d{14}_\d+\.zip$", re.IGNORECASE)


class PD7DayError(RuntimeError):
    """Base error for the informational AEMO seven-day price outlook."""


class PD7DayValidationError(PD7DayError):
    """Raised when an AEMO PD7Day report cannot form a safe price window."""


class PD7DayDownloadError(PD7DayError):
    """Raised when the latest PD7Day archive cannot be retrieved."""


@dataclass(frozen=True)
class PD7DayPriceWindow:
    frame: pd.DataFrame
    run_datetime: pd.Timestamp
    source_name: str
    retrieved_at_utc: datetime
    interval_minutes: int = 30

    @property
    def start(self) -> pd.Timestamp:
        return self.frame["DateTime"].iloc[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.frame["DateTime"].iloc[-1]


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise PD7DayValidationError("PD7Day CSV text encoding is not supported.")


def parse_pd7day_csv(
    content: bytes,
    *,
    source_name: str = "uploaded PD7Day CSV",
    retrieved_at_utc: datetime | None = None,
) -> PD7DayPriceWindow:
    header: list[str] | None = None
    records: list[dict[str, str]] = []
    for row in csv.reader(io.StringIO(_decode(content))):
        if len(row) < 4:
            continue
        key = tuple(item.strip() for item in row[:4])
        if key == ("I", "PD7DAY", "PRICESOLUTION", "1"):
            header = [item.strip() for item in row[4:]]
            required = {
                "RUN_DATETIME",
                "INTERVENTION",
                "INTERVAL_DATETIME",
                "REGIONID",
                "RRP",
            }
            missing = required.difference(header)
            if missing:
                raise PD7DayValidationError(
                    "PD7Day PRICESOLUTION header is missing: " + ", ".join(sorted(missing))
                )
            continue
        if key != ("D", "PD7DAY", "PRICESOLUTION", "1"):
            continue
        if header is None:
            raise PD7DayValidationError(
                "PD7Day PRICESOLUTION data appeared before its header."
            )
        record = dict(zip(header, row[4:]))
        if record.get("REGIONID", "").strip() == "NSW1" and record.get(
            "INTERVENTION", ""
        ).strip() == "0":
            records.append(record)

    if header is None:
        raise PD7DayValidationError("PD7Day PRICESOLUTION table was not found.")
    if not records:
        raise PD7DayValidationError(
            "PD7Day PRICESOLUTION contains no NSW1 non-intervention prices."
        )

    parsed: list[tuple[pd.Timestamp, float, pd.Timestamp]] = []
    try:
        for record in records:
            interval = pd.Timestamp(
                datetime.strptime(record["INTERVAL_DATETIME"], "%Y/%m/%d %H:%M:%S")
            )
            run = pd.Timestamp(
                datetime.strptime(record["RUN_DATETIME"], "%Y/%m/%d %H:%M:%S")
            )
            price = float(record["RRP"])
            if not math.isfinite(price):
                raise ValueError("non-finite RRP")
            parsed.append((interval, price, run))
    except (KeyError, TypeError, ValueError) as exc:
        raise PD7DayValidationError(
            "PD7Day PRICESOLUTION contains an invalid timestamp or RRP."
        ) from exc

    run_datetimes = {item[2] for item in parsed}
    if len(run_datetimes) != 1:
        raise PD7DayValidationError(
            "PD7Day PRICESOLUTION contains more than one run timestamp."
        )

    raw = pd.DataFrame(
        {"DateTime": [item[0] for item in parsed], "Price": [item[1] for item in parsed]}
    ).sort_values("DateTime")
    conflicts = raw.groupby("DateTime")["Price"].nunique()
    if (conflicts > 1).any():
        raise PD7DayValidationError(
            "PD7Day PRICESOLUTION contains conflicting duplicate NSW1 intervals."
        )
    frame = raw.drop_duplicates("DateTime", keep="last").reset_index(drop=True)
    if len(frame) < 2:
        raise PD7DayValidationError(
            "PD7Day PRICESOLUTION needs at least two NSW1 intervals."
        )
    deltas = frame["DateTime"].diff().dropna()
    if not deltas.eq(pd.Timedelta(minutes=30)).all():
        raise PD7DayValidationError(
            "PD7Day NSW1 prices must form a continuous 30-minute series."
        )

    return PD7DayPriceWindow(
        frame=frame,
        run_datetime=next(iter(run_datetimes)),
        source_name=source_name,
        retrieved_at_utc=retrieved_at_utc or datetime.now(timezone.utc),
    )


def parse_pd7day_zip(
    content: bytes,
    *,
    source_name: str = "uploaded PD7Day ZIP",
    retrieved_at_utc: datetime | None = None,
) -> PD7DayPriceWindow:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_names = [
                name for name in archive.namelist() if name.lower().endswith(".csv")
            ]
            if not csv_names:
                raise PD7DayValidationError("PD7Day ZIP contains no CSV report.")
            preferred = sorted(
                csv_names,
                key=lambda name: ("pd7day" not in name.lower(), name.lower()),
            )[0]
            csv_content = archive.read(preferred)
    except PD7DayValidationError:
        raise
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise PD7DayValidationError("PD7Day download is not a valid ZIP archive.") from exc
    return parse_pd7day_csv(
        csv_content,
        source_name=source_name,
        retrieved_at_utc=retrieved_at_utc,
    )


def daily_price_summary(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.loc[:, ["DateTime", "Price"]].copy()
    working["Date"] = pd.to_datetime(working["DateTime"]).dt.date
    return (
        working.groupby("Date", as_index=False)["Price"]
        .agg(Minimum="min", Average="mean", Maximum="max")
        .loc[:, ["Date", "Minimum", "Average", "Maximum"]]
    )


class PD7DayPriceService:
    def fetch_monday(self, monday, *, rolling: bool = False) -> PD7DayPriceWindow:
        """Find the latest pre-week vintage that covers Monday through Sunday."""
        from .weekly_planning import select_week
        if monday.weekday() != 0:
            raise PD7DayValidationError("Select a Monday.")
        try:
            from bs4 import BeautifulSoup
            listing = self.session.get(self.directory_url, timeout=self.listing_timeout_s)
            listing.raise_for_status()
            links = [str(a.get("href", "")) for a in BeautifulSoup(listing.text, "html.parser").find_all("a", href=True)]
            # A complete Monday-to-Sunday plan must use the newest Sunday vintage
            # issued before the calendar week starts. Rolling mode remains available
            # for callers that explicitly want the earliest usable Monday vintage.
            from datetime import timedelta
            issue_date = monday if rolling else monday - timedelta(days=1)
            issue_token = issue_date.strftime("%Y%m%d")
            links = sorted(set(link for link in links if ZIP_NAME_PATTERN.search(link)
                               and ZIP_NAME_PATTERN.search(link).group(0).split("_")[2][:8] == issue_token),
                           reverse=not rolling)
            last_error = "The selected week is not in the current AEMO directory. Upload its saved snapshot."
            for link in links:
                response = self.session.get(urljoin(self.directory_url, link), timeout=self.download_timeout_s)
                response.raise_for_status()
                window = parse_pd7day_zip(response.content, source_name=link.rsplit("/", 1)[-1])
                try:
                    select_week(window, monday, rolling=rolling)
                    return window
                except ValueError as exc:
                    last_error = str(exc)
            raise PD7DayDownloadError(last_error)
        except PD7DayDownloadError:
            raise
        except Exception as exc:
            raise PD7DayDownloadError(f"Monday forecast retrieval failed: {exc}") from exc

    def __init__(
        self,
        *,
        session=None,
        directory_url: str = PD7DAY_DIRECTORY_URL,
        listing_timeout_s: float = 20.0,
        download_timeout_s: float = 60.0,
    ) -> None:
        if session is None:
            try:
                import requests
            except ImportError as exc:
                raise PD7DayDownloadError(
                    "Live PD7Day retrieval requires the requests package."
                ) from exc
            session = requests.Session()
        self.session = session
        self.directory_url = directory_url
        self.listing_timeout_s = listing_timeout_s
        self.download_timeout_s = download_timeout_s

    def fetch_latest(self) -> PD7DayPriceWindow:
        try:
            listing = self.session.get(
                self.directory_url, timeout=self.listing_timeout_s
            )
            listing.raise_for_status()
            try:
                from bs4 import BeautifulSoup
            except ImportError as exc:
                raise PD7DayDownloadError(
                    "Live PD7Day retrieval requires beautifulsoup4."
                ) from exc
            links = [
                str(anchor.get("href", ""))
                for anchor in BeautifulSoup(listing.text, "html.parser").find_all(
                    "a", href=True
                )
            ]
            archives = [link for link in links if ZIP_NAME_PATTERN.search(link)]
            if not archives:
                raise PD7DayDownloadError(
                    "AEMO PD7Day directory contains no downloadable ZIP archive."
                )
            latest = max(archives, key=lambda link: ZIP_NAME_PATTERN.search(link).group(0))
            url = urljoin(self.directory_url, latest)
            response = self.session.get(url, timeout=self.download_timeout_s)
            response.raise_for_status()
            return parse_pd7day_zip(
                response.content,
                source_name=url.rsplit("/", 1)[-1],
                retrieved_at_utc=datetime.now(timezone.utc),
            )
        except PD7DayDownloadError:
            raise
        except PD7DayValidationError as exc:
            raise PD7DayDownloadError(
                f"Latest AEMO PD7Day ZIP could not be validated: {exc}"
            ) from exc
        except Exception as exc:
            raise PD7DayDownloadError(
                "Unable to retrieve the latest AEMO PD7Day ZIP."
            ) from exc
