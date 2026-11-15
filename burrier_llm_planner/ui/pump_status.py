"""Self-updating schedule status component; changes locally without fetching prices."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from burrier_llm_planner.services.pump_status import ZONE, daily_scada_text, operating_windows


def pump_status_html(windows, *, layout="strip"):
    payload = [{"start": int(w.start.timestamp()*1000), "end": int(w.end.timestamp()*1000),
                "on": w.on,"source": w.source,"startClipped": w.start_clipped,
                "endClipped": w.end_clipped} for w in windows]
    template = Path(__file__).with_name("pump_status.html").read_text(encoding="utf-8")
    return template.replace("__WINDOWS__", json.dumps(payload).replace("</","<\\/")).replace(
        'class="status"', 'class="status panel"' if layout == "panel" else 'class="status"', 1,
    )


def render_pump_status(*, original=None, rolling=None, actual_runs=(), replaced_dates=(), layout="strip"):
    windows = operating_windows(original=original, rolling=rolling,
        actual_runs=actual_runs, replaced_dates=replaced_dates)
    st.iframe(pump_status_html(windows, layout=layout), height="content")
    if layout != "panel" or not windows:
        return
    daily_text = {}
    for stamp in pd.date_range(windows[0].start.normalize(), windows[-1].end.normalize(), freq="D"):
        try:
            daily_text[stamp.date()] = daily_scada_text(windows, stamp)
        except ValueError:
            continue
    if not daily_text:
        st.caption("SCADA time entries require a complete day of schedule coverage.")
        return
    title, picker = st.columns([1, 1.2], vertical_alignment="center")
    with title:
        st.markdown("**SCADA time entries**", help="Copy the active daily schedule: 0 = OFF, 1 = ON. Each state stays in effect until the next entry. This does not send commands to SCADA.")
    dates = list(daily_text)
    today = pd.Timestamp.now(tz=ZONE).date()
    default = min(dates, key=lambda day: abs((day-today).days))
    with picker:
        day = st.selectbox("SCADA schedule date", dates, index=dates.index(default),
                          format_func=lambda d: d.strftime("%a %d %b %Y"),
                          label_visibility="collapsed", key="pump_status_scada_day")
    st.code(daily_text[day], language=None)
    st.caption("0 = OFF · 1 = ON · Sydney time. Last state holds to midnight; 00:00 sets the next day's state.")
