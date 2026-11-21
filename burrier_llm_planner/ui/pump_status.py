"""Self-updating schedule status component; changes locally without fetching prices."""
from __future__ import annotations

from html import escape
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
    scada = _scada_panel(windows) if layout == "panel" else ""
    return template.replace("__SCADA__", scada).replace("__WINDOWS__", json.dumps(payload).replace("</","<\\/")).replace(
        'class="status"', 'class="status panel"' if layout == "panel" else 'class="status"', 1,
    )


def render_pump_status(*, original=None, rolling=None, actual_runs=(), replaced_dates=(), layout="strip"):
    windows = operating_windows(original=original, rolling=rolling,
        actual_runs=actual_runs, replaced_dates=replaced_dates)
    st.iframe(pump_status_html(windows, layout=layout), height="content")


def _scada_panel(windows):
    if not windows:
        return '<aside class="scada"><b>SCADA time entries</b><small>Generate a plan to see daily entries.</small></aside>'
    daily_text = {}
    for stamp in pd.date_range(windows[0].start.normalize(), windows[-1].end.normalize(), freq="D"):
        try:
            daily_text[stamp.date()] = daily_scada_text(windows, stamp)
        except ValueError:
            continue
    if not daily_text:
        return '<aside class="scada"><b>SCADA time entries</b><small>A full day of schedule coverage is required.</small></aside>'
    today = pd.Timestamp.now(tz=ZONE).date()
    default = min(daily_text, key=lambda day: abs((day-today).days))
    options = "".join(
        f'<option value="{day}"{" selected" if day == default else ""}>{day:%a %d %b %Y}</option>'
        for day in daily_text
    )
    data = json.dumps({day.isoformat(): text for day, text in daily_text.items()})
    return f'''<aside class="scada"><label for="scada-day"><b>SCADA time entries</b></label>
        <select id="scada-day">{options}</select>
        <textarea id="scada-text" readonly aria-label="SCADA time entries for selected date">{escape(daily_text[default])}</textarea>
        <div class="scada-actions"><small>0 = OFF · 1 = ON · Sydney time</small><button id="copy" type="button">Copy</button></div>
        <script type="application/json" id="scada-data">{data}</script></aside>'''
