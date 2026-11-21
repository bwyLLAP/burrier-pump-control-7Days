"""Continuous operating windows from the active schedule; no SCADA inference."""
from __future__ import annotations

from dataclasses import dataclass, replace
import heapq

import numpy as np
import pandas as pd

from .weekly_actuals import replacement_dates_for_runs

ZONE = "Australia/Sydney"


def local_time(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(ZONE) if stamp.tzinfo is None else stamp.tz_convert(ZONE)


@dataclass(frozen=True)
class OperatingWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    on: bool | None
    source: str
    start_clipped: bool = False
    end_clipped: bool = False


def operating_windows(*, original=None, rolling=None, actual_runs=(), replaced_dates=()):
    """Original < whole-date override < entered run < latest rolling coverage.

    Rolling results start at their executable boundary. Overrides replace elapsed
    dates, while a new rolling result controls its own future coverage. Gaps stay
    unknown. Adjacent equal states merge even at midnight or a source boundary.
    """
    spans = []

    def add_plan(plan, priority, source):
        if plan is None:
            return
        if hasattr(plan, "frame"):
            times = pd.DatetimeIndex(plan.frame["DateTime"])
            values = np.asarray(plan.frame["PumpOn"], float)
        else:
            times = pd.DatetimeIndex(plan.timestamps)
            values = np.asarray(plan.pump_on, float)
        times = times.tz_localize(ZONE) if times.tz is None else times.tz_convert(ZONE)
        if len(times) != len(values) or times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError("Status requires unique ordered timestamps and matching states.")
        for stamp, value in zip(times, values):
            if np.isfinite(value):
                spans.append((stamp, stamp + pd.Timedelta(minutes=30), bool(value>.5), source, priority))

    add_plan(original, 1, "Original weekly plan")
    add_plan(rolling, 4, "Current rolling plan")
    if not spans:
        return []
    start, end = min(s[0] for s in spans), max(s[1] for s in spans)
    dates = set(replaced_dates) | replacement_dates_for_runs(actual_runs)
    for day in dates:
        a = local_time(pd.Timestamp(day))
        b = a + pd.DateOffset(days=1)
        if a < end and b > start:
            spans.append((max(a,start), min(b,end), False, "Override", 2))
    for run in actual_runs:
        a,b = local_time(run.start),local_time(run.end)
        if a < end and b > start:
            spans.append((max(a,start), min(b,end), True, "Override", 3))
    starts, points = {}, set()
    for i, (a,b,on,source,priority) in enumerate(spans):
        starts.setdefault(a,[]).append((-priority,-i,b,on,source))
        points.update((a,b))
    points = sorted(points)
    active, windows = [], []
    for a,b in zip(points[:-1], points[1:]):
        for entry in starts.get(a,[]):
            heapq.heappush(active,entry)
        while active and active[0][2] <= a:
            heapq.heappop(active)
        on, source = (active[0][3], active[0][4]) if active else (None,"No schedule coverage")
        if windows and windows[-1].on == on and windows[-1].end == a:
            sources = list(dict.fromkeys(windows[-1].source.split(" + ")+[source]))
            windows[-1] = replace(windows[-1],end=b,source=" + ".join(sources))
        else:
            windows.append(OperatingWindow(a,b,on,source))
    return [replace(w,start_clipped=(i==0 or windows[i-1].on is None),
                    end_clipped=(i==len(windows)-1 or windows[i+1].on is None))
            for i,w in enumerate(windows)]


def status_at(windows, now):
    now = local_time(now)
    for i,window in enumerate(windows):
        if window.start <= now < window.end:
            return window, windows[i+1] if i+1<len(windows) else None
        if now < window.start:
            return None, window
    return None,None


def daily_scada_text(windows, day):
    """Local-day initial state and changes; each value holds until the next.

    Only export a fully known day. Do not force OFF at midnight when a run
    crosses the day boundary, or turn missing schedule coverage into OFF.
    """
    start = local_time(day).normalize()
    end = start + pd.DateOffset(days=1)
    cursor, entries = start, []
    previous = None
    for window in windows:
        if window.end <= start or window.start >= end:
            continue
        a, b = max(start, window.start), min(end, window.end)
        if a != cursor or window.on is None:
            raise ValueError("A complete daily schedule is required for SCADA text.")
        if not entries or window.on != previous:
            entries.append(f"{a.strftime('%H:%M:%S')}  {int(window.on)}")
        previous, cursor = window.on, b
    if cursor != end:
        raise ValueError("A complete daily schedule is required for SCADA text.")
    return "\n".join(entries)
