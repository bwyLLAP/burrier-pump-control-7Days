from __future__ import annotations

from datetime import date, datetime, time, timedelta
import hashlib
import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from burrier_llm_planner.optimization.aemo_weekly import (
    AEMOWeeklyInputs,
    run_aemo_two_stage_rolling_optimization,
    run_aemo_weekly_optimization,
)
from burrier_llm_planner.services.pd7day_price import PD7DayPriceService
from burrier_llm_planner.services.rolling_prices import (
    RollingAEMOPriceService, RollingPriceWindow, VolumeCorrection, calendar_week_prices,
)
from burrier_llm_planner.services.weekly_actuals import (
    ActualPumpRun, ActualRunSummary,
    replacement_dates_for_runs, summarise_actual_runs,
)
from burrier_llm_planner.services.weekly_comparison import (
    DailyWindowComparison, FrozenSchedule, compare_daily_windows,
    frozen_schedule_from_result, load_frozen_schedule, save_frozen_schedule,
    schedule_runs_for_day, states_for_runs,
)
from burrier_llm_planner.services.weekly_planning import (
    containing_monday, fingerprint, read_snapshot, select_week, snapshot_csv,
)


ZONE = ZoneInfo("Australia/Sydney")
LOGGER = logging.getLogger(__name__)
WEEKLY_FETCH_POLICY_VERSION = 4
WEEKLY_ACTUALS_POLICY_VERSION = 5
ENGLISH_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
ENGLISH_MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def _reference_root() -> Path:
    configured = os.getenv("BURRIER_REFERENCE_DIR")
    return (
        Path(configured) if configured
        else Path(__file__).resolve().parents[1] / "data" / "schedule_references"
    )


def format_english_date(value: date) -> str:
    return (
        f"{ENGLISH_WEEKDAYS[value.weekday()]} {value.day:02d} "
        f"{ENGLISH_MONTHS[value.month - 1]} {value.year}"
    )


def parse_english_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parts = str(value).split()
    if len(parts) != 4 or parts[2] not in ENGLISH_MONTHS:
        raise ValueError(f"Invalid weekly date: {value}")
    return date(int(parts[3]), ENGLISH_MONTHS.index(parts[2]) + 1, int(parts[1]))


def weekly_fetch_token(monday: date, now: pd.Timestamp | datetime) -> tuple:
    """Invalidate live AEMO prices when the executable half-hour advances."""

    current = _to_local_timestamp(now).floor("30min")
    return monday, current.isoformat(), WEEKLY_FETCH_POLICY_VERSION


def build_weekly_figure(
    frame, result=None, actual_runs=(), default_range=None, baseline_result=None,
    replaced_dates=(),
):
    times = pd.DatetimeIndex(frame["DateTime"])
    display = times.tz_localize(None) if times.tz else times
    working = frame.copy()
    working["DisplayTime"] = display
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=working["DisplayTime"], y=working["Price"], name="Combined price",
        mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color="#64748B", width=1.5),
    ))
    source_styles = {
        "KWatch actual": ("Actual · KWatch", "#172033"),
        "AEMO actual": ("Actual · AEMO", "#172033"),
        "Actual": ("Actual", "#172033"),
        "KWatch predispatch": ("Short-term forecast · KWatch", "#2563A6"),
        "AEMO 1-day forecast": ("Short-term forecast · AEMO", "#2563A6"),
        "1-day forecast": ("1-day forecast", "#2563A6"),
        "7-day forecast": ("7-day forecast · AEMO", "#7C3AED"),
        "Monday forecast fallback": ("Monday forecast fallback", "#A16207"),
    }
    source_values = (
        working["PriceSource"].dropna().astype(str).drop_duplicates().tolist()
        if "PriceSource" in working.columns else ["Forecast price"]
    )
    for source in source_values:
        label, color = source_styles.get(source, (source, "#64748B"))
        subset = working if "PriceSource" not in working.columns else working[working["PriceSource"] == source]
        if subset.empty:
            continue
        figure.add_trace(go.Scatter(
            x=subset["DisplayTime"], y=subset["Price"], name=label, mode="lines",
            line=dict(color=color, width=2),
            legend="legend",
            customdata=np.full(len(subset), source, dtype=object),
            hovertemplate=(
                "%{x|%a %d %b %H:%M}<br>Price: %{y:.2f} AUD/MWh"
                "<br>Source: %{customdata}<extra></extra>"
            ),
        ))
    for day in pd.date_range(display[0].normalize(), display[-1].normalize(), freq="D"):
        if day.weekday() < 5:
            figure.add_vrect(
                x0=day + pd.Timedelta(hours=16), x1=day + pd.Timedelta(hours=20),
                fillcolor="rgba(220,65,65,0.20)", line_width=0, layer="below",
            )
    override_dates = set(replaced_dates) | replacement_dates_for_runs(actual_runs)

    def add_plan_background(
        plan, fillcolor: str, linecolor: str, excluded_dates: set[date] | None = None,
        line_dash: str = "solid",
    ) -> None:
        if isinstance(plan, FrozenSchedule):
            states = plan.frame["PumpOn"].to_numpy(float) > 0.5
            result_times = pd.DatetimeIndex(plan.frame["DateTime"])
        else:
            states = np.asarray(plan.interval_volume_ml) > 0
            result_times = pd.DatetimeIndex(plan.timestamps)
        if excluded_dates:
            local_times = (
                result_times.tz_localize(ZONE) if result_times.tz is None
                else result_times.tz_convert(ZONE)
            )
            states &= np.asarray([
                timestamp.date() not in excluded_dates for timestamp in local_times
            ])
        result_display = result_times.tz_localize(None) if result_times.tz else result_times
        changes = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
        for first, stop in zip(changes[:-1], changes[1:]):
            if states[first]:
                end = result_display[stop] if stop < len(result_display) else result_display[-1] + pd.Timedelta(minutes=30)
                figure.add_vrect(
                    x0=result_display[first], x1=end,
                    fillcolor=fillcolor, line_color=linecolor, line_width=1,
                    line_dash=line_dash, layer="below",
                )

    if baseline_result is not None:
        add_plan_background(
            baseline_result, "rgba(176,218,187,0.24)", "rgba(89,151,106,0.55)",
            excluded_dates=override_dates,
        )
    if result is not None:
        add_plan_background(
            result, "rgba(137,207,155,0.42)", "rgba(63,143,84,0.75)",
            line_dash="dash",
        )
    for run in actual_runs:
        figure.add_vrect(
            x0=pd.Timestamp(run.start).tz_localize(None),
            x1=pd.Timestamp(run.end).tz_localize(None),
            fillcolor="rgba(71,126,170,0.25)", line_width=0, layer="below",
        )
    for label, fill, border in [
        ("No pumping · Mon–Fri 16:00–20:00", "rgba(220,65,65,0.20)", "rgba(220,65,65,0)"),
        ("Original weekly plan", "rgba(176,218,187,0.24)", "rgba(89,151,106,0.55)"),
        ("Adjusted rolling plan", "rgba(137,207,155,0.42)", "rgba(63,143,84,0.75)"),
        ("Override", "rgba(71,126,170,0.25)", "rgba(71,126,170,0)"),
    ]:
        figure.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=label,
            marker=dict(
                color=fill, symbol="square", size=12,
                line=dict(color=border, width=1 if border else 0),
            ),
            legend="legend2",
        ))
    figure.update_layout(
        height=470, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=15, r=15, t=125, b=20), hovermode="x unified",
        legend=dict(
            orientation="h", y=1.19, yanchor="bottom", x=0, xanchor="left",
            title=dict(text="Price", side="left"),
        ),
        legend2=dict(
            orientation="h", y=1.07, yanchor="bottom", x=0, xanchor="left",
            title=dict(text="Operating windows", side="left"),
        ),
    )
    chart_range = default_range or [display[0], display[-1] + pd.Timedelta(minutes=30)]
    chart_range = [pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo else value for value in chart_range]
    figure.update_xaxes(
        title="Australia/Sydney local time", dtick=86400000,
        tickformat="%a %d %b", range=chart_range, gridcolor="#E8EDF1",
        rangeslider=dict(visible=True, thickness=0.10), fixedrange=False,
    )
    figure.update_yaxes(title="Price (AUD/MWh)", gridcolor="#E8EDF1")
    return figure


def _result_state_and_volume(result) -> tuple[pd.Series, pd.Series]:
    times = pd.DatetimeIndex(result.timestamps)
    state_values = np.asarray(
        result.pump_on if hasattr(result, "pump_on") else result.interval_volume_ml,
        dtype=float,
    )
    states = state_values > (0.5 if hasattr(result, "pump_on") else 0.0)
    volumes = np.asarray(result.interval_volume_ml, dtype=float)
    return pd.Series(states, index=times), pd.Series(volumes, index=times)


def _format_change_windows(times: pd.DatetimeIndex, states: np.ndarray) -> str:
    states = np.asarray(states, dtype=bool)
    if not states.any():
        return "—"
    changes = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
    windows = []
    for first, stop in zip(changes[:-1], changes[1:]):
        if not states[first]:
            continue
        end = times[stop] if stop < len(times) else times[-1] + pd.Timedelta(minutes=30)
        windows.append(f"{times[first]:%H:%M}–{end:%H:%M}")
    return ", ".join(windows)


def _schedule_change_rows(before, after, start) -> list[dict[str, object]]:
    """Summarise future operating-window changes between two rolling plans."""
    if before is None or after is None:
        return []
    before_states, before_volume = _result_state_and_volume(before)
    after_states, after_volume = _result_state_and_volume(after)
    index = before_states.index.union(after_states.index).sort_values()
    boundary = _to_local_timestamp(start).ceil("30min")
    index = index[index >= boundary]
    if index.empty:
        return []
    before_states = before_states.reindex(index, fill_value=False).astype(bool)
    after_states = after_states.reindex(index, fill_value=False).astype(bool)
    before_volume = before_volume.reindex(index, fill_value=0.0)
    after_volume = after_volume.reindex(index, fill_value=0.0)
    local_index = index.tz_localize(ZONE) if index.tz is None else index.tz_convert(ZONE)
    rows: list[dict[str, object]] = []
    for day in dict.fromkeys(local_index.date):
        mask = np.asarray(local_index.date) == day
        day_times = local_index[mask]
        old = before_states.to_numpy()[mask]
        new = after_states.to_numpy()[mask]
        if np.array_equal(old, new):
            continue
        removed = old & ~new
        added = new & ~old
        rows.append({
            "Date": f"{ENGLISH_WEEKDAYS[day.weekday()]} {day:%d %b}",
            "Removed windows": _format_change_windows(day_times, removed),
            "Added windows": _format_change_windows(day_times, added),
            "Hours Δ": float((new.sum() - old.sum()) * 0.5),
            "Volume Δ (ML)": float(
                after_volume.to_numpy()[mask].sum() - before_volume.to_numpy()[mask].sum()
            ),
        })
    return rows


def _planned_run_rows(result) -> list[dict[str, object]]:
    states = np.asarray(result.interval_volume_ml) > 0
    changes = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
    rows = []
    for first, stop in zip(changes[:-1], changes[1:]):
        if not states[first]:
            continue
        end = result.timestamps[stop] if stop < len(states) else result.timestamps[-1] + pd.Timedelta(minutes=30)
        rows.append({
            "Type": "Planned", "Start": result.timestamps[first], "End": end,
            "Duration (h)": (stop - first) * 0.5,
            "Volume (ML)": float(result.interval_volume_ml[first:stop].sum()),
            "Cost (AUD)": float(result.interval_cost_aud[first:stop].sum()),
            "Cost basis": "Forecast estimate",
        })
    return rows


def build_run_schedule(result, actual_summary: ActualRunSummary) -> pd.DataFrame:
    actual = actual_summary.rows.copy()
    if not actual.empty:
        actual.insert(0, "Type", "Actual")
    planned = pd.DataFrame(_planned_run_rows(result)) if result is not None else pd.DataFrame()
    populated = [frame for frame in (actual, planned) if not frame.empty]
    combined = pd.concat(populated, ignore_index=True) if populated else pd.DataFrame()
    if combined.empty:
        return pd.DataFrame(columns=[
            "Type", "Start", "End", "Duration (h)", "Volume (ML)", "Cost (AUD)", "Cost basis",
        ])
    return combined.sort_values("Start").reset_index(drop=True)


def display_run_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Expose costs only for completed runs; forecast costs are intentionally omitted."""
    displayed = schedule.copy()
    if displayed.empty:
        return displayed.rename(columns={"Cost (AUD)": "Actual cost (AUD)"})
    planned_mask = displayed["Type"].eq("Planned")
    displayed.loc[planned_mask, "Cost (AUD)"] = np.nan
    displayed.loc[planned_mask, "Cost basis"] = ""
    for column in ("Start", "End"):
        if column in displayed.columns:
            displayed[column] = displayed[column].map(
                lambda value: (
                    value if pd.isna(value)
                    else _to_local_timestamp(value).tz_localize(None)
                )
            )
    return displayed.rename(columns={"Cost (AUD)": "Actual cost (AUD)"})


def _to_local_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(ZONE) if timestamp.tzinfo is None else timestamp.tz_convert(ZONE)


def _editor_time(value) -> time:
    if isinstance(value, time):
        return value.replace(tzinfo=None)
    if isinstance(value, datetime):
        return value.time().replace(tzinfo=None)
    try:
        return time.fromisoformat(str(value).strip()).replace(tzinfo=None)
    except ValueError as exc:
        raise ValueError(f"Invalid time value: {value}") from exc


def _runs_from_editor(editor: pd.DataFrame) -> list[ActualPumpRun]:
    runs = []
    for row_number, row in editor.iterrows():
        include = row.get("Include", True)
        if pd.notna(include) and not bool(include):
            continue
        start_date, start_time = row.get("Start date"), row.get("Start time")
        end_date, end_time = row.get("End date"), row.get("End time")
        if pd.isna(start_time) and pd.isna(end_time):
            continue
        values = (start_date, start_time, end_date, end_time)
        if all(pd.isna(value) for value in values):
            continue
        if any(pd.isna(value) for value in values):
            raise ValueError(
                f"Actual run row {row_number + 1} needs a date and time for both start and end."
            )
        start = datetime.combine(parse_english_date(start_date), _editor_time(start_time))
        end = datetime.combine(parse_english_date(end_date), _editor_time(end_time))
        override = row.get("Actual cost override (AUD)")
        runs.append(ActualPumpRun(
            start=_to_local_timestamp(start), end=_to_local_timestamp(end),
            cost_override_aud=None if pd.isna(override) else float(override),
        ))
    return runs


def _actual_editor_frame(runs, default_date: date | None = None) -> pd.DataFrame:
    rows = []
    for run in runs:
        local_start = run.start.tz_convert(ZONE)
        local_end = run.end.tz_convert(ZONE)
        rows.append({
            "Include": True,
            "Start date": format_english_date(local_start.date()),
            "Start time": local_start.time().replace(tzinfo=None),
            "End date": format_english_date(local_end.date()),
            "End time": local_end.time().replace(tzinfo=None),
            "Actual cost override (AUD)": run.cost_override_aud,
        })
    if not rows:
        default_date_label = format_english_date(default_date) if default_date else None
        rows.append({
            "Include": True,
            "Start date": default_date_label,
            "Start time": None,
            "End date": default_date_label,
            "End time": None,
            "Actual cost override (AUD)": None,
        })
    return pd.DataFrame(rows)


def _override_editor_frame(runs: list[ActualPumpRun]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Start": _to_local_timestamp(run.start).time().replace(tzinfo=None),
            "End": _to_local_timestamp(run.end).time().replace(tzinfo=None),
        }
        for run in runs
    ], columns=["Start", "End"])


def _runs_from_override_editor(
    editor: pd.DataFrame,
    selected_day: date,
    *,
    no_pumping: bool,
    daily_cost_override_aud: float | None = None,
) -> list[ActualPumpRun]:
    if no_pumping:
        return []
    raw: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for row_number, row in editor.iterrows():
        start_value, end_value = row.get("Start"), row.get("End")
        if pd.isna(start_value) and pd.isna(end_value):
            continue
        if pd.isna(start_value) or pd.isna(end_value):
            raise ValueError(f"Window {row_number + 1} needs both start and end times.")
        start_time = _editor_time(start_value)
        end_time = _editor_time(end_value)
        start = _to_local_timestamp(datetime.combine(selected_day, start_time))
        if end_time == time.min and start_time != time.min:
            end = _to_local_timestamp(datetime.combine(selected_day + timedelta(days=1), time.min))
        else:
            end = _to_local_timestamp(datetime.combine(selected_day, end_time))
        if end <= start:
            raise ValueError(f"Window {row_number + 1} end must be after its start.")
        raw.append((start, end))
    total_seconds = sum((end - start).total_seconds() for start, end in raw)
    runs = []
    for start, end in raw:
        override = None
        if daily_cost_override_aud is not None and total_seconds > 0:
            override = float(daily_cost_override_aud) * (
                (end - start).total_seconds() / total_seconds
            )
        runs.append(ActualPumpRun(start=start, end=end, cost_override_aud=override))
    return runs


def _runs_touching_day(runs: list[ActualPumpRun], selected_day: date) -> list[ActualPumpRun]:
    day_start = _to_local_timestamp(datetime.combine(selected_day, time.min))
    day_end = day_start + pd.DateOffset(days=1)
    return [run for run in runs if run.start < day_end and run.end > day_start]


def _replace_day_runs(
    existing: list[ActualPumpRun], selected_day: date, replacements: list[ActualPumpRun],
) -> list[ActualPumpRun]:
    return sorted(
        [run for run in existing if not _runs_touching_day([run], selected_day)] + replacements,
        key=lambda run: run.start,
    )


def _override_summary_rows(
    replaced_dates: set[date], runs: list[ActualPumpRun], flow_lps: float,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for day in sorted(replaced_dates):
        day_runs = _runs_touching_day(runs, day)
        if not day_runs:
            detail = "No pumping"
            meta = "0.0 h · 0.00 ML"
        else:
            windows = []
            duration_hours = 0.0
            for run in day_runs:
                local_start = _to_local_timestamp(run.start)
                local_end = _to_local_timestamp(run.end)
                windows.append(f"{local_start:%H:%M}–{local_end:%H:%M}")
                duration_hours += (run.end - run.start).total_seconds() / 3600.0
            volume_ml = duration_hours * 3600.0 * flow_lps / 1_000_000.0
            detail = ", ".join(windows)
            cost_basis = (
                "operator cost" if all(run.cost_override_aud is not None for run in day_runs)
                else "price-based cost"
            )
            meta = f"{duration_hours:.1f} h · {volume_ml:.2f} ML · {cost_basis}"
        items.append({
            "day": day,
            "date_label": format_english_date(day),
            "detail": detail,
            "meta": meta,
        })
    return items


def _empty_summary() -> ActualRunSummary:
    return ActualRunSummary(
        rows=pd.DataFrame(columns=[
            "Start", "End", "Duration (h)", "Volume (ML)", "Cost (AUD)", "Cost basis",
        ]), achieved_volume_ml=0.0, actual_cost_aud=0.0,
    )


def _current_week_balance(
    *, target_ml: float, actual_ml: float, replaced_dates: set[date],
    reference_timestamps=None, reference_volumes=None, future_start: pd.Timestamp,
) -> tuple[float, float]:
    """Return remaining weekly target and override variance at the execution boundary."""

    if reference_timestamps is None or reference_volumes is None:
        return max(0.0, float(target_ml)), 0.0
    timestamps = pd.DatetimeIndex(reference_timestamps)
    boundary = pd.Timestamp(future_start)
    if timestamps.tz is not None and boundary.tzinfo is None:
        boundary = boundary.tz_localize(timestamps.tz)
    elif timestamps.tz is None and boundary.tzinfo is not None:
        boundary = boundary.tz_localize(None)
    volumes = np.asarray(reference_volumes, dtype=float)
    elapsed = timestamps < boundary
    replaced = np.asarray([stamp.date() in replaced_dates for stamp in timestamps])
    planned_elapsed = float(volumes[elapsed].sum())
    replaced_planned_elapsed = float(volumes[elapsed & replaced].sum())
    assumed_executed = planned_elapsed - replaced_planned_elapsed + float(actual_ml)
    return (
        max(0.0, float(target_ml) - assumed_executed),
        float(actual_ml) - replaced_planned_elapsed,
    )


def _remaining_week_price_frames(
    window, frame: pd.DataFrame, monday: date, now: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build current-week 7-day allocation prices and D+1 operating prices."""

    week_end = pd.Timestamp(monday + timedelta(days=7), tz=ZONE)
    current_week = frame[frame["DateTime"] < week_end].copy().reset_index(drop=True)
    allocation_source = getattr(window, "allocation_frame", frame)
    allocation = allocation_source.set_index("DateTime").reindex(current_week["DateTime"])
    if allocation["Price"].isna().any():
        raise ValueError("The latest 7-day forecast does not fully cover the rest of this week.")
    allocation = allocation.reset_index().rename(columns={"index": "DateTime"})
    operating_prices = allocation["Price"].to_numpy(float).copy()
    tomorrow = _to_local_timestamp(now).date() + timedelta(days=1)
    tomorrow_mask = np.asarray([stamp.date() == tomorrow for stamp in current_week["DateTime"]])
    operating_prices[tomorrow_mask] = current_week.loc[tomorrow_mask, "Price"].to_numpy(float)
    return current_week, allocation, operating_prices


def _is_rolling_window(window) -> bool:
    """Support both current and Streamlit-hot-reloaded rolling window instances."""
    return all(hasattr(window, name) for name in (
        "frame", "optimisation_frame", "default_display_start", "default_display_end",
    ))


def _weekly_fingerprint(window, monday: date, parameters: dict) -> str:
    if _is_rolling_window(window):
        canonical = window.optimisation_frame.loc[:, ["DateTime", "Price", "PriceSource"]].to_csv(
            index=False, date_format="%Y-%m-%dT%H:%M:%S%z"
        )
        allocation_frame = getattr(window, "allocation_frame", None)
        if allocation_frame is not None:
            canonical += allocation_frame.loc[:, ["DateTime", "Price", "PriceSource"]].to_csv(
                index=False, date_format="%Y-%m-%dT%H:%M:%S%z"
            )
        return hashlib.sha256(
            canonical.encode("utf-8") + str((monday, parameters)).encode("utf-8")
        ).hexdigest()
    return fingerprint(window, monday, parameters)


def actual_editor_week_limits(monday: date) -> tuple[datetime, datetime]:
    """Return local, naive editor bounds covering only the selected calendar week."""
    week_start = datetime.combine(monday, time.min)
    week_end = datetime.combine(monday + timedelta(days=6), time.max).replace(microsecond=0)
    return week_start, week_end


def _create_frozen_reference(
    *, monday: date, target: float, common_params: dict[str, float], window,
) -> FrozenSchedule:
    if not _is_rolling_window(window) and hasattr(window, "run_datetime"):
        reference_window = window
    else:
        reference_window = PD7DayPriceService().fetch_monday(monday)
    reference_frame = select_week(reference_window, monday)
    result = run_aemo_weekly_optimization(AEMOWeeklyInputs(
        timestamps=pd.DatetimeIndex(reference_frame["DateTime"]),
        price_aud_per_mwh=reference_frame["Price"].to_numpy(float),
        weekly_target_ml=float(target), prorate_partial_days=False,
        **common_params,
    ))
    issued_at = _to_local_timestamp(reference_window.run_datetime)
    schedule = frozen_schedule_from_result(
        monday=monday, source_name=reference_window.source_name,
        issued_at=issued_at, weekly_target_ml=float(target),
        parameters={key: float(value) for key, value in common_params.items()
                    if isinstance(value, (int, float))},
        result=result,
    )
    save_frozen_schedule(_reference_root(), schedule)
    return schedule


def _daily_comparisons(
    *, frozen: FrozenSchedule, display_frame: pd.DataFrame,
    rolling_result=None,
    saved_runs: list[ActualPumpRun], replaced_dates: set[date], now: pd.Timestamp,
    flow: float, power: float, min_run: float, min_start_interval: float,
) -> list[DailyWindowComparison | None]:
    current = display_frame.set_index("DateTime").sort_index()
    frozen_frame = frozen.frame.set_index("DateTime").sort_index()
    rolling_frame = pd.DataFrame(
        {"PumpOn": pd.Series(dtype=float)}, index=pd.DatetimeIndex([]),
    )
    if rolling_result is not None:
        rolling_frame = pd.DataFrame(
            {"PumpOn": np.asarray(rolling_result.pump_on, dtype=float)},
            index=pd.DatetimeIndex(rolling_result.timestamps),
        ).sort_index()
    week_end = frozen.monday + timedelta(days=6)
    tomorrow = now.date() + timedelta(days=1)
    if now.date() < frozen.monday or now.date() > week_end:
        last_day = week_end
    else:
        last_day = week_end
    if (
        tomorrow > week_end
        and not rolling_frame.empty
        and tomorrow <= rolling_frame.index[-1].date()
    ):
        last_day = tomorrow
    comparisons: list[DailyWindowComparison | None] = []
    for offset in range((last_day - frozen.monday).days + 1):
        day = frozen.monday + timedelta(days=offset)
        current_day = current[np.asarray(current.index.date) == day]
        frozen_day = frozen_frame[np.asarray(frozen_frame.index.date) == day]
        rolling_day = rolling_frame[np.asarray(rolling_frame.index.date) == day]
        if current_day.empty or (frozen_day.empty and rolling_day.empty):
            comparisons.append(None)
            continue
        if not frozen_day.empty:
            target_states = frozen_day.reindex(current_day.index)["PumpOn"]
            if target_states.isna().any():
                comparisons.append(None)
                continue
            target_states = target_states.to_numpy(float)
            comparison_label = "Original weekly plan"
        else:
            target_states = rolling_day.reindex(current_day.index)["PumpOn"].fillna(0.0).to_numpy(float)
            comparison_label = "Adjusted rolling plan"
        if day in replaced_dates and not frozen_day.empty:
            day_runs = _runs_touching_day(saved_runs, day)
            comparison_states = states_for_runs(current_day.index, day_runs)
            comparison_label = "Original weekly plan · Override"
            cost_override = (
                sum(float(run.cost_override_aud) for run in day_runs)
                if day_runs and all(run.cost_override_aud is not None for run in day_runs)
                else None
            )
        else:
            comparison_states = target_states
            cost_override = None
        fixed_ideal_states = None
        if day <= now.date():
            ideal_label = "Actual-price ideal"
        elif day == tomorrow:
            ideal_label = "1-day forecast"
            if rolling_day.empty:
                comparisons.append(None)
                continue
            rolling_states = rolling_day.reindex(current_day.index)["PumpOn"]
            if rolling_states.isna().any():
                comparisons.append(None)
                continue
            fixed_ideal_states = rolling_states.to_numpy(float)
        else:
            ideal_label = "Current rolling plan"
            if rolling_day.empty:
                comparisons.append(None)
                continue
            rolling_states = rolling_day.reindex(current_day.index)["PumpOn"]
            if rolling_states.isna().any():
                comparisons.append(None)
                continue
            fixed_ideal_states = rolling_states.to_numpy(float)
        sources = (
            current_day["PriceSource"].dropna().astype(str).unique().tolist()
            if "PriceSource" in current_day else ["Forecast price"]
        )
        try:
            comparisons.append(compare_daily_windows(
                day=day,
                timestamps=pd.DatetimeIndex(current_day.index),
                prices=current_day["Price"].to_numpy(float),
                comparison_states=comparison_states,
                ideal_target_states=target_states,
                fixed_ideal_states=fixed_ideal_states,
                comparison_label=comparison_label,
                ideal_label=ideal_label,
                price_source=" + ".join(sources), flow_lps=flow, power_kw=power,
                minimum_continuous_run_hours=min_run, minimum_off_hours=2.0,
                minimum_start_interval_hours=min_start_interval,
                comparison_cost_override_aud=cost_override,
            ))
        except Exception:
            comparisons.append(None)
    return comparisons


def _window_track(states: np.ndarray, timestamps: pd.DatetimeIndex) -> str:
    timestamps = pd.DatetimeIndex(timestamps)
    if len(states) != len(timestamps):
        raise ValueError("Window states and timestamps must have equal lengths.")
    cells = "".join(
        f'<i class="window-cell {"on" if value > 0.5 else ""}" '
        f'title="{timestamp:%H:%M}–{timestamp + pd.Timedelta(minutes=30):%H:%M} · '
        f'Pump {"ON" if value > 0.5 else "OFF"}" '
        f'aria-label="{timestamp:%H:%M} Pump {"ON" if value > 0.5 else "OFF"}"></i>'
        for value, timestamp in zip(states, timestamps)
    )
    return f"<span class='window-track'>{cells}</span>"


def _render_daily_comparison(
    day: date, comparison: DailyWindowComparison | None, *, is_tomorrow: bool,
) -> None:
    if comparison is None:
        st.markdown(
            f"<div class='day-review {'tomorrow' if is_tomorrow else ''}'>"
            "<div class='match-ring empty'><span>—</span><small>MATCH</small></div>"
            f"<div><strong>{format_english_date(day)}</strong>"
            "<p>Comparison data unavailable.</p></div></div>",
            unsafe_allow_html=True,
        )
        return
    match = comparison.window_match_percent
    fill = 0.0 if match is None else min(100.0, max(0.0, match))
    match_text = "N/A" if match is None else f"{match:.0f}%"
    comparison_unit_cost = (
        "N/A" if comparison.comparison_cost_per_ml is None
        else f"${comparison.comparison_cost_per_ml:,.2f}/ML"
    )
    ideal_unit_cost = (
        "N/A" if comparison.ideal_cost_per_ml is None
        else f"${comparison.ideal_cost_per_ml:,.2f}/ML"
    )
    unit_cost_delta = (
        "N/A" if comparison.cost_per_ml_delta is None
        else f"${comparison.cost_per_ml_delta:+.2f}/ML"
    )
    tomorrow_label = " · Tomorrow" if is_tomorrow else ""
    st.markdown(
        f"<div class='day-review {'tomorrow' if is_tomorrow else ''}'>"
        f"<div class='match-ring' style='--match:{fill:.1f}%'><span>{match_text}</span><small>MATCH</small></div>"
        "<div class='day-review-body'>"
        f"<div class='day-review-head'><strong>{format_english_date(day)}{tomorrow_label}</strong></div>"
        f"<div class='window-line'><em>{comparison.comparison_label}</em>{_window_track(comparison.comparison_states, comparison.timestamps)}</div>"
        f"<div class='window-line'><em>{comparison.ideal_label}</em>{_window_track(comparison.ideal_states, comparison.timestamps)}</div>"
        f"<div class='review-stats'><span>Cost <b>${comparison.comparison_cost_aud:,.0f}</b> "
        f"vs <b>${comparison.ideal_cost_aud:,.0f}</b> · Δ <b>${comparison.cost_delta_aud:+,.0f}</b></span>"
        f"<span>Hours <b>{comparison.comparison_hours:.1f} h</b> vs "
        f"<b>{comparison.ideal_hours:.1f} h</b> · Δ <b>{comparison.hours_delta:+.1f} h</b></span>"
        f"<span>Volume <b>{comparison.comparison_volume_ml:.2f} ML</b> vs "
        f"<b>{comparison.ideal_volume_ml:.2f} ML</b> · Δ <b>{comparison.volume_delta_ml:+.2f} ML</b></span>"
        f"<span>Unit cost <b>{comparison_unit_cost}</b> vs <b>{ideal_unit_cost}</b> · "
        f"Δ <b>{unit_cost_delta}</b></span>"
        f"<span>{comparison.price_source}</span></div></div></div>",
        unsafe_allow_html=True,
    )


def render_weekly_workspace():
    now = pd.Timestamp(datetime.now(ZONE))
    st.subheader("Weekly pumping plan")
    st.caption("Actual prices, the latest one-day forecast and the seven-day outlook drive a rolling seven-day schedule.")

    controls, source_panel = st.columns([3, 5], gap="large")
    with controls:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>Planning inputs</div>", unsafe_allow_html=True)
            selected_date = st.date_input("Select any date in the week", now.date(), key="weekly_date")
            monday = containing_monday(selected_date)
            target = st.number_input(
                "Weekly pumping target (ML)", min_value=0.0, value=210.0, step=1.0,
                key="weekly_target",
            )
            st.caption(
                f"Baseline week starts Monday {monday:%d %b %Y}. The same target is then corrected "
                "by actual-versus-baseline performance within the rest of this week."
            )
            with st.expander("Pump settings"):
                flow = st.number_input("Flow rate (L/s)", min_value=1.0, value=1050.0, key="weekly_flow")
                power = st.number_input("Pump power (kW)", min_value=1.0, value=1950.0, key="weekly_power")
                standby = st.number_input("Standby power (kW)", min_value=0.0, value=12.0, key="weekly_standby")
                min_run = st.number_input(
                    "Minimum continuous run (h)", min_value=0.0, max_value=20.0,
                    value=4.0, step=0.5, key="weekly_min_run",
                )
                min_start_interval = st.number_input(
                    "Minimum interval between starts (h)", min_value=0.0, max_value=168.0,
                    value=6.0, step=0.5, key="weekly_min_start_interval",
                )
                min_daily = st.number_input(
                    "Minimum daily run (h)", min_value=0.0, max_value=20.0,
                    value=6.0, step=0.5, key="weekly_min_daily",
                )
            st.caption(
                f"Rules: at least 2 hours OFF between runs; at least {min_start_interval:g} hours "
                "between pump starts; weekdays 16:00–20:00 are blocked for future pumping."
            )

    actuals_token = (monday, WEEKLY_ACTUALS_POLICY_VERSION)
    if st.session_state.get("weekly_actuals_token") != actuals_token:
        st.session_state.weekly_actual_monday = monday
        st.session_state.weekly_actuals_token = actuals_token
        st.session_state.weekly_actual_runs = []
        st.session_state.weekly_actuals_applied = False
        st.session_state.weekly_actual_replaced_dates = []
        for key in (
            "weekly_reference_result", "weekly_baseline_result",
            "weekly_baseline_fingerprint", "weekly_pre_override_result",
            "weekly_schedule_changes", "weekly_schedule_changes_at",
            "weekly_schedule_change_reference_available",
            "weekly_reoptimisation_pending",
        ):
            st.session_state.pop(key, None)
        st.session_state.weekly_actual_editor_version = st.session_state.get("weekly_actual_editor_version", 0) + 1

    if st.session_state.get("weekly_price_week") != monday:
        for key in (
            "weekly_window", "weekly_result", "weekly_result_fingerprint",
            "weekly_load_error", "weekly_pre_override_result",
            "weekly_schedule_changes", "weekly_schedule_changes_at",
            "weekly_schedule_change_reference_available",
            "weekly_reoptimisation_pending",
        ):
            st.session_state.pop(key, None)
        st.session_state.weekly_price_week = monday
        st.session_state.pop("weekly_price_mode", None)

    frozen = load_frozen_schedule(_reference_root(), monday)
    fetch_token = weekly_fetch_token(monday, now)
    with source_panel:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>Price data</div>", unsafe_allow_html=True)
            kwatch_key = st.text_input(
                "KWatch API key", type="password", key="weekly_kwatch_key",
                help="Kept only in this Streamlit session; it is not written to a file.",
            )
            kwatch_action, aemo_action = st.columns(2, gap="small")
            load_kwatch = kwatch_action.button(
                "Load KWatch prices", type="primary", width="stretch",
                disabled=not kwatch_key.strip(), key="weekly_load_kwatch",
            )
            load_aemo = aemo_action.button(
                "Use AEMO data", type="secondary", width="stretch",
                key="weekly_load_aemo",
            )
            st.caption(
                "KWatch uses dispatch and predispatch prices, with AEMO PD7Day for the longer outlook. "
                "AEMO mode uses the public AEMO sources for the complete window."
            )
            if load_kwatch or load_aemo:
                st.session_state.pop("weekly_result", None)
                st.session_state.pop("weekly_result_fingerprint", None)
                st.session_state.pop("weekly_schedule_changes", None)
                st.session_state.pop("weekly_schedule_changes_at", None)
                selected_key = kwatch_key.strip() if load_kwatch else None
                mode = "KWatch + AEMO outlook" if load_kwatch else "AEMO public data"
                try:
                    with st.spinner(f"Loading {mode}…"):
                        st.session_state.weekly_window = RollingAEMOPriceService(
                            Path(__file__).resolve().parents[1] / "data" / "aemo_live_cache",
                            kwatch_api_key=selected_key,
                        ).fetch(
                            now=now,
                            display_monday=pd.Timestamp(monday, tz=ZONE),
                            strict_kwatch=load_kwatch,
                            original_fallback=(
                                frozen.frame.loc[:, ["DateTime", "Price"]]
                                if frozen is not None else None
                            ),
                        )
                    st.session_state.weekly_price_mode = mode
                    st.session_state.pop("weekly_load_error", None)
                except Exception as exc:
                    st.session_state.pop("weekly_window", None)
                    st.session_state.weekly_load_error = str(exc)

            window = st.session_state.get("weekly_window")
            if window is not None:
                if _is_rolling_window(window):
                    mode = st.session_state.get("weekly_price_mode", "Loaded price data")
                    st.success(f"Ready · {mode}")
                    st.caption(" · ".join(window.source_names))
                    st.download_button(
                        "Download price window", window.frame.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"burrier_price_window_{monday}.csv", mime="text/csv",
                    )
                else:
                    st.success(f"Ready · issued {window.run_datetime} AEST · {window.source_name}")
                    st.download_button(
                        "Download forecast snapshot", snapshot_csv(window),
                        file_name=f"burrier_forecast_{monday}.csv", mime="text/csv",
                    )
            elif st.session_state.get("weekly_load_error"):
                load_error = str(st.session_state.weekly_load_error)
                LOGGER.warning("Weekly price retrieval failed: %s", load_error)
                st.warning(f"Price retrieval failed: {load_error}")
            else:
                st.info("Enter the KWatch API key and load prices, or use AEMO public data.")
            with st.expander("Use a saved forecast file"):
                upload = st.file_uploader(
                    "Saved PD7Day ZIP or snapshot CSV", type=["zip", "csv"], key="weekly_upload",
                )
                st.caption("Use this fallback when live retrieval is unavailable.")

    if upload is not None:
        upload_id = hashlib.sha256(upload.getvalue()).hexdigest()
        if st.session_state.get("weekly_upload_id") != upload_id:
            st.session_state.pop("weekly_result", None)
            try:
                st.session_state.weekly_window = read_snapshot(upload.getvalue(), upload.name)
                st.session_state.weekly_upload_id = upload_id
                st.session_state.weekly_price_mode = "Uploaded forecast"
                st.session_state.pop("weekly_load_error", None)
                st.rerun()
            except Exception as exc:
                st.session_state.weekly_load_error = str(exc)

    window = st.session_state.get("weekly_window")
    frame = None
    baseline_frame = None
    display_frame = None
    default_chart_range = None
    frame_error = None
    if window is not None:
        try:
            if _is_rolling_window(window):
                frame = window.optimisation_frame.copy()
                display_frame = window.frame.copy()
                baseline_frame = calendar_week_prices(display_frame, pd.Timestamp(monday, tz=ZONE))
                default_chart_range = [window.default_display_start, window.default_display_end]
            else:
                frame = select_week(window, monday)
                baseline_frame = frame.copy()
                display_frame = frame
        except Exception as exc:
            frame_error = str(exc)
            frame = None
            baseline_frame = None

    saved_runs = st.session_state.get("weekly_actual_runs", [])
    week_date_options = [
        format_english_date(monday + timedelta(days=offset)) for offset in range(7)
    ]
    week_start = pd.Timestamp(monday, tz=ZONE)
    week_end = pd.Timestamp(monday + timedelta(days=7), tz=ZONE)
    eligible_run_dates = [
        monday + timedelta(days=offset) for offset in range(7)
        if monday + timedelta(days=offset) <= now.date()
    ]
    eligible_run_options = [format_english_date(value) for value in eligible_run_dates]
    completed_dates = [
        monday + timedelta(days=offset) for offset in range(7)
        if _to_local_timestamp(datetime.combine(
            monday + timedelta(days=offset + 1), time.min,
        )) <= now
    ]
    completed_date_options = [format_english_date(value) for value in completed_dates]
    saved_replaced_dates = set(st.session_state.get("weekly_actual_replaced_dates", []))
    actual_summary = _empty_summary()
    if frame is not None:
        horizon_start = week_start
        horizon_end = week_end
        try:
            actual_summary = summarise_actual_runs(
                saved_runs, horizon_start=horizon_start, horizon_end=horizon_end,
                now=now, flow_lps=flow, power_kw=power, price_frame=display_frame,
            )
        except Exception as exc:
            frame_error = str(exc)

    reference = st.session_state.get("weekly_baseline_result")
    reference_times = pd.DatetimeIndex(frozen.frame["DateTime"]) if frozen is not None else (
        reference.timestamps if reference is not None else None
    )
    reference_volumes = frozen.frame["IntervalVolumeML"].to_numpy(float) if frozen is not None else (
        reference.interval_volume_ml if reference is not None else None
    )
    future_start = pd.Timestamp(frame["DateTime"].iloc[0]) if frame is not None else week_start
    remaining, variance = _current_week_balance(
        target_ml=float(target), actual_ml=actual_summary.achieved_volume_ml,
        replaced_dates=saved_replaced_dates, reference_timestamps=reference_times,
        reference_volumes=reference_volumes, future_start=future_start,
    )
    correction = VolumeCorrection(variance, remaining)
    common_params = dict(
        flow_lps=flow, max_power_kw=power,
        standby_power_kw=standby, minimum_continuous_run_hours=min_run,
        minimum_off_hours=2.0, minimum_start_interval_hours=min_start_interval,
        minimum_daily_run_hours=min_daily,
        constrain_total_volume=True,
    )
    params = {
        **common_params, "minimum_daily_run_hours": 0.0,
        "weekly_target_ml": remaining, "prorate_partial_days": True,
    }
    actual_signature = tuple((str(run.start), str(run.end), run.cost_override_aud) for run in saved_runs)
    baseline_current = _weekly_fingerprint(window, monday, {
        **common_params, "weekly_target_ml": float(target), "stage": "calendar-week-baseline",
    }) if baseline_frame is not None else None
    if st.session_state.get("weekly_baseline_fingerprint") != baseline_current:
        st.session_state.pop("weekly_baseline_result", None)
        st.session_state.pop("weekly_result", None)
        reference = None
        remaining, variance = _current_week_balance(
            target_ml=float(target), actual_ml=actual_summary.achieved_volume_ml,
            replaced_dates=saved_replaced_dates,
            reference_timestamps=(pd.DatetimeIndex(frozen.frame["DateTime"])
                                  if frozen is not None else None),
            reference_volumes=(frozen.frame["IntervalVolumeML"].to_numpy(float)
                               if frozen is not None else None),
            future_start=future_start,
        )
        correction = VolumeCorrection(variance, remaining)
        params["weekly_target_ml"] = remaining
    rolling_fingerprint_params = {
        **common_params, "weekly_target_ml": float(target), "actual_runs": actual_signature,
        "replaced_dates": tuple(sorted(map(str, saved_replaced_dates))),
        "stage": "current-week-two-stage-v3",
    }
    current = _weekly_fingerprint(window, monday, rolling_fingerprint_params) if frame is not None else None
    if st.session_state.get("weekly_result_fingerprint") != current or current is None:
        st.session_state.pop("weekly_result", None)
        st.session_state.pop("weekly_allocation_result", None)

    with st.container(border=True):
        action, balance = st.columns([2, 5], gap="large", vertical_alignment="center")
        with action:
            if st.button(
                "Optimise weekly plan", type="primary",
                disabled=frame is None or baseline_frame is None,
                key="weekly_optimise",
            ):
                st.session_state.pop("weekly_result", None)
                try:
                    with st.spinner("Allocating this week's remaining volume, then refining tomorrow's window…"):
                        if frozen is None:
                            try:
                                frozen = _create_frozen_reference(
                                    monday=monday, target=float(target),
                                    common_params=common_params, window=window,
                                )
                                st.session_state.pop("weekly_reference_error", None)
                            except Exception as reference_exc:
                                st.session_state.weekly_reference_error = str(reference_exc)
                        baseline = run_aemo_weekly_optimization(AEMOWeeklyInputs(
                            timestamps=pd.DatetimeIndex(baseline_frame["DateTime"]),
                            price_aud_per_mwh=baseline_frame["Price"].to_numpy(float),
                            weekly_target_ml=float(target), prorate_partial_days=False,
                            **common_params,
                        ))
                        current_week, allocation_frame, operating_prices = _remaining_week_price_frames(
                            window, frame, monday, now,
                        )
                        balance_times = pd.DatetimeIndex(frozen.frame["DateTime"]) if frozen is not None else baseline.timestamps
                        balance_volumes = frozen.frame["IntervalVolumeML"].to_numpy(float) if frozen is not None else baseline.interval_volume_ml
                        local_remaining, local_variance = _current_week_balance(
                            target_ml=float(target), actual_ml=actual_summary.achieved_volume_ml,
                            replaced_dates=saved_replaced_dates,
                            reference_timestamps=balance_times,
                            reference_volumes=balance_volumes,
                            future_start=pd.Timestamp(current_week["DateTime"].iloc[0]),
                        )
                        local_correction = VolumeCorrection(local_variance, local_remaining)
                        future_params = {
                            **common_params,
                            "minimum_daily_run_hours": 0.0,
                            "weekly_target_ml": local_remaining,
                            "prorate_partial_days": True,
                        }
                        allocation_inputs = AEMOWeeklyInputs(
                            timestamps=pd.DatetimeIndex(current_week["DateTime"]),
                            price_aud_per_mwh=allocation_frame["Price"].to_numpy(float),
                            fixed_pump_on=np.full(len(current_week), np.nan),
                            optimisation_start_index=0, **future_params,
                        )
                        allocation_result, result = run_aemo_two_stage_rolling_optimization(
                            allocation_inputs, operating_prices,
                        )
                    st.session_state.weekly_baseline_result = baseline
                    st.session_state.weekly_baseline_fingerprint = baseline_current
                    st.session_state.weekly_result = result
                    st.session_state.weekly_allocation_result = allocation_result
                    st.session_state.weekly_result_fingerprint = current
                    if st.session_state.get("weekly_reoptimisation_pending"):
                        st.session_state.weekly_schedule_changes = _schedule_change_rows(
                            st.session_state.get("weekly_pre_override_result"), result, now,
                        )
                        st.session_state.weekly_schedule_changes_at = now
                        st.session_state.weekly_reoptimisation_pending = False
                    reference = baseline
                    correction = local_correction
                    remaining = local_remaining
                except Exception as exc:
                    st.error(f"No schedule generated: {exc} Check the remaining target and available hours.")
        with balance:
            st.markdown(
                f"<div class='balance-card'><span>Pumping variance</span><strong>{correction.actual_minus_planned_ml:+.2f} ML</strong>"
                f"<span>Remaining target this week</span><strong>{remaining:.2f} ML</strong></div>",
                unsafe_allow_html=True,
            )

    if frame_error:
        st.error(frame_error)
    if frame is None or baseline_frame is None:
        st.info("A complete saved forecast is required before actuals can be reconciled or a plan generated.")
        return

    result = st.session_state.get("weekly_result")
    future_volume = result.pumped_volume_ml if result is not None else 0.0
    future_hours = result.pump_hours if result is not None else 0.0
    metrics = [
        ("Remaining-week volume", f"{future_volume:.2f} ML"),
        ("Remaining run hours", f"{future_hours:.1f} h"),
        ("Pumping variance", f"{correction.actual_minus_planned_ml:+.2f} ML"),
        ("Data status", "Plan ready" if result is not None else "Prices ready"),
    ]
    for column, (label, value) in zip(st.columns(4), metrics):
        column.metric(label, value)

    chart_column, override_column = st.columns([5, 2], gap="large")
    with chart_column:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>Weekly price and pumping window</div>", unsafe_allow_html=True)
            st.plotly_chart(
                build_weekly_figure(
                    display_frame, result, saved_runs, default_chart_range,
                    baseline_result=frozen, replaced_dates=saved_replaced_dates,
                ),
                width="stretch",
                config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
            )
            st.caption(
                "The default view is the selected Monday-to-Sunday week. Drag the range slider to inspect "
                "the additional forecast days. Pale green is the original weekly plan; stronger green "
                "with a dashed border is the adjusted rolling plan; blue shows applied Overrides."
            )
    with override_column:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>Override actual pumping</div>", unsafe_allow_html=True)
            if frozen is None:
                st.info("Optimise once to create the original weekly plan.")
            selectable_dates = eligible_run_options or week_date_options
            default_override_day = min(max(now.date(), monday), monday + timedelta(days=6))
            selected_override_label = st.selectbox(
                "Date", options=selectable_dates,
                index=(eligible_run_dates.index(default_override_day)
                       if default_override_day in eligible_run_dates else 0),
                key=f"weekly_override_date_{monday}",
            )
            selected_override_day = parse_english_date(selected_override_label)
            applied_for_day = selected_override_day in saved_replaced_dates
            applied_runs = _runs_touching_day(saved_runs, selected_override_day)
            draft_runs = (
                applied_runs if applied_for_day else
                schedule_runs_for_day(frozen, selected_override_day) if frozen is not None else []
            )
            st.caption(
                "Applied Actual" if applied_for_day
                else "Draft from original weekly plan"
            )
            no_pumping = st.checkbox(
                "No pumping",
                value=applied_for_day and not applied_runs,
                key=(f"weekly_override_zero_{monday}_{selected_override_day}_"
                     f"{st.session_state.get('weekly_actual_editor_version', 0)}"),
            )
            override_editor = st.data_editor(
                _override_editor_frame(draft_runs), num_rows="dynamic", hide_index=True,
                width="stretch", disabled=no_pumping,
                key=(f"weekly_override_editor_{monday}_{selected_override_day}_"
                     f"{st.session_state.get('weekly_actual_editor_version', 0)}"),
                column_config={
                    "Start": st.column_config.TimeColumn("Start", format="HH:mm", step=1800),
                    "End": st.column_config.TimeColumn("End", format="HH:mm", step=1800),
                },
            )
            with st.expander("Advanced"):
                daily_cost_override = st.number_input(
                    "Actual daily cost override (AUD)", min_value=0.0, value=0.0,
                    step=10.0,
                    key=f"weekly_override_cost_{monday}_{selected_override_day}",
                )
                st.caption("Leave at $0 to calculate cost from Actual prices.")
            if st.button(
                "Apply override", type="primary", width="stretch",
                disabled=frame is None or selected_override_day > now.date(),
                key="weekly_apply_override",
            ):
                try:
                    day_replacements = _runs_from_override_editor(
                        override_editor, selected_override_day,
                        no_pumping=no_pumping,
                        daily_cost_override_aud=(
                            float(daily_cost_override) if daily_cost_override > 0 else None
                        ),
                    )
                    candidate_runs = _replace_day_runs(
                        saved_runs, selected_override_day, day_replacements,
                    )
                    summarise_actual_runs(
                        candidate_runs, horizon_start=week_start, horizon_end=week_end,
                        now=now, flow_lps=flow, power_kw=power, price_frame=display_frame,
                    )
                    replaced_dates = set(saved_replaced_dates)
                    replaced_dates.add(selected_override_day)
                    replaced_dates.update(replacement_dates_for_runs(day_replacements))
                    active_plan = st.session_state.get("weekly_result")
                    if active_plan is not None or not st.session_state.get(
                        "weekly_reoptimisation_pending", False
                    ):
                        st.session_state.weekly_pre_override_result = active_plan
                        st.session_state.weekly_schedule_change_reference_available = (
                            active_plan is not None
                        )
                    st.session_state.weekly_actual_runs = candidate_runs
                    st.session_state.weekly_actuals_applied = True
                    st.session_state.weekly_actual_replaced_dates = sorted(replaced_dates)
                    st.session_state.weekly_reoptimisation_pending = True
                    st.session_state.pop("weekly_result", None)
                    st.session_state.pop("weekly_daily_comparisons", None)
                    st.session_state.pop("weekly_schedule_changes", None)
                    st.session_state.pop("weekly_schedule_changes_at", None)
                    st.session_state.weekly_actual_editor_version = (
                        st.session_state.get("weekly_actual_editor_version", 0) + 1
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Override not applied: {exc}")

            def stage_override_removal(updated_runs, updated_dates) -> None:
                active_plan = st.session_state.get("weekly_result")
                if active_plan is not None or not st.session_state.get(
                    "weekly_reoptimisation_pending", False
                ):
                    st.session_state.weekly_pre_override_result = active_plan
                    st.session_state.weekly_schedule_change_reference_available = (
                        active_plan is not None
                    )
                st.session_state.weekly_actual_runs = updated_runs
                st.session_state.weekly_actuals_applied = bool(updated_dates)
                st.session_state.weekly_actual_replaced_dates = sorted(updated_dates)
                st.session_state.weekly_reoptimisation_pending = True
                st.session_state.pop("weekly_result", None)
                st.session_state.pop("weekly_daily_comparisons", None)
                st.session_state.pop("weekly_schedule_changes", None)
                st.session_state.pop("weekly_schedule_changes_at", None)
                st.session_state.weekly_actual_editor_version = (
                    st.session_state.get("weekly_actual_editor_version", 0) + 1
                )

            st.markdown("<div class='subpanel-title'>Applied overrides this week</div>", unsafe_allow_html=True)
            override_rows = _override_summary_rows(saved_replaced_dates, saved_runs, flow)
            if not override_rows:
                st.caption("No overrides applied this week.")
            for item in override_rows:
                with st.container(border=True):
                    summary, remove = st.columns([6, 1], vertical_alignment="center")
                    with summary:
                        st.markdown(f"**{item['date_label']}**")
                        st.caption(f"{item['detail']} · {item['meta']}")
                    if remove.button(
                        "×", key=f"weekly_remove_override_{item['day']}",
                        help=f"Cancel the override for {item['date_label']}",
                        width="stretch",
                    ):
                        restored_dates = set(saved_replaced_dates)
                        restored_dates.discard(item["day"])
                        stage_override_removal(
                            _replace_day_runs(saved_runs, item["day"], []),
                            restored_dates,
                        )
                        st.rerun()

            if st.button(
                "Restore all overrides", type="secondary", width="stretch",
                disabled=not saved_replaced_dates,
                key="weekly_restore_all_overrides",
            ):
                stage_override_removal([], set())
                st.rerun()

            if st.session_state.get("weekly_reoptimisation_pending"):
                st.info("Override saved. Reoptimise the future plan to apply the volume correction.")
            if st.button(
                "Reoptimise future plan", type="secondary", width="stretch",
                disabled=(
                    frame is None
                    or not st.session_state.get("weekly_reoptimisation_pending", False)
                ),
                key="weekly_reoptimise_after_override",
            ):
                try:
                    with st.spinner("Reallocating the rest of this week and refining tomorrow…"):
                        current_week, allocation_frame, operating_prices = _remaining_week_price_frames(
                            window, frame, monday, now,
                        )
                        allocation_inputs = AEMOWeeklyInputs(
                            timestamps=pd.DatetimeIndex(current_week["DateTime"]),
                            price_aud_per_mwh=allocation_frame["Price"].to_numpy(float),
                            fixed_pump_on=np.full(len(current_week), np.nan),
                            optimisation_start_index=0,
                            **params,
                        )
                        allocation_result, revised = run_aemo_two_stage_rolling_optimization(
                            allocation_inputs, operating_prices,
                        )
                    st.session_state.weekly_result = revised
                    st.session_state.weekly_allocation_result = allocation_result
                    st.session_state.weekly_result_fingerprint = current
                    st.session_state.weekly_schedule_changes = _schedule_change_rows(
                        st.session_state.get("weekly_pre_override_result"), revised, now,
                    )
                    st.session_state.weekly_schedule_changes_at = now
                    st.session_state.weekly_reoptimisation_pending = False
                    st.rerun()
                except Exception as exc:
                    st.error(f"Remaining-week plan not reoptimised: {exc}")

    schedule_changes = st.session_state.get("weekly_schedule_changes")
    changes_at = st.session_state.get("weekly_schedule_changes_at")
    if schedule_changes is not None:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>Schedule changes after latest override</div>", unsafe_allow_html=True)
            if changes_at is not None:
                st.caption(f"Reoptimised { _to_local_timestamp(changes_at):%a %d %b %H:%M} · compared with the plan active before the override")
            if not st.session_state.get(
                "weekly_schedule_change_reference_available", False
            ):
                st.info("The plan was reoptimised, but no earlier rolling plan was available for comparison.")
            elif schedule_changes:
                changes_frame = pd.DataFrame(schedule_changes)
                st.dataframe(
                    changes_frame, hide_index=True, width="stretch",
                    column_config={
                        "Hours Δ": st.column_config.NumberColumn(format="%+.1f h"),
                        "Volume Δ (ML)": st.column_config.NumberColumn(format="%+.2f ML"),
                    },
                )
            else:
                st.success("No future pumping windows changed after reoptimisation.")

    with st.container(border=True):
        st.markdown("<div class='panel-title'>Daily window performance</div>", unsafe_allow_html=True)
        if frozen is None:
            detail = st.session_state.get("weekly_reference_error")
            st.info("The original weekly plan is not available yet. Optimise to create it.")
            if detail:
                st.caption(detail)
        else:
            st.caption(
                f"Original weekly plan · forecast issued "
                f"{_to_local_timestamp(frozen.issued_at):%a %d %b %H:%M} · {frozen.source_name}"
            )
            comparison_columns = ["DateTime", "Price"] + (
                ["PriceSource"] if "PriceSource" in display_frame.columns else []
            )
            comparison_key = hashlib.sha256(
                display_frame[comparison_columns].to_csv(
                    index=False, date_format="%Y-%m-%dT%H:%M:%S%z"
                ).encode("utf-8")
                + str((frozen.source_name, actual_signature, tuple(sorted(saved_replaced_dates)),
                       flow, power, min_run, min_start_interval)).encode("utf-8")
                + (
                    np.asarray(result.pump_on, dtype=float).tobytes()
                    if result is not None else b"no-rolling-result"
                )
            ).hexdigest()
            if st.session_state.get("weekly_daily_comparison_key") != comparison_key:
                st.session_state.weekly_daily_comparisons = _daily_comparisons(
                    frozen=frozen, display_frame=display_frame, saved_runs=saved_runs,
                    rolling_result=result,
                    replaced_dates=saved_replaced_dates, now=now, flow=flow, power=power,
                    min_run=min_run, min_start_interval=min_start_interval,
                )
                st.session_state.weekly_daily_comparison_key = comparison_key
            daily_comparisons = st.session_state.get("weekly_daily_comparisons", [None] * 7)
            for offset, comparison in enumerate(daily_comparisons):
                comparison_day = (
                    comparison.day if comparison is not None
                    else monday + timedelta(days=offset)
                )
                _render_daily_comparison(
                    comparison_day, comparison,
                    is_tomorrow=comparison_day == now.date() + timedelta(days=1),
                )

    schedule = build_run_schedule(result, actual_summary)
    displayed_schedule = display_run_schedule(schedule)
    with st.container(border=True):
        st.markdown("<div class='panel-title'>Recent schedule</div>", unsafe_allow_html=True)
        if schedule.empty:
            st.info("Apply actual runs or optimise the remaining week to populate the schedule.")
        else:
            st.dataframe(
                displayed_schedule.round({
                    "Duration (h)": 2, "Volume (ML)": 2, "Actual cost (AUD)": 2,
                }),
                hide_index=True, width="stretch",
            )
            st.download_button(
                "Download weekly schedule CSV", displayed_schedule.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"burrier_weekly_schedule_{monday}.csv", mime="text/csv",
            )
        if result is not None:
            st.caption(
                f"Weekly target {target:.2f} ML · actual minus replaced baseline {correction.actual_minus_planned_ml:+.2f} ML · "
                f"remaining target {remaining:.2f} ML · remaining-week scheduled {future_volume:.2f} ML."
            )
