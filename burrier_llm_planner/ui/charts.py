from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from burrier_llm_planner.domain.models import PlantConfig
from burrier_llm_planner.optimization.pi_mpc import MPCResult
from burrier_llm_planner.optimization.aemo_daily import AEMODailyResult


def build_pd7day_price_figure(frame: pd.DataFrame) -> go.Figure:
    """Build the informational seven-day NSW1 regional-price outlook."""
    if not {"DateTime", "Price"}.issubset(frame.columns):
        raise ValueError("PD7Day outlook requires DateTime and Price columns.")
    if frame.empty:
        raise ValueError("PD7Day outlook cannot be empty.")

    timestamps = pd.to_datetime(frame["DateTime"])
    figure = go.Figure(
        data=[
            go.Scatter(
                x=timestamps,
                y=frame["Price"],
                name="NSW1 RRP",
                mode="lines",
                line={"color": "#2563A6", "width": 1.8},
                hovertemplate=(
                    "%{x|%a %d %b, %H:%M} NEM time"
                    "<br>AUD %{y:,.2f}/MWh<extra></extra>"
                ),
            )
        ]
    )
    figure.add_hline(y=0, line_color="#64748B", line_dash="dot", line_width=1)
    first_midnight = timestamps.iloc[0].normalize() + pd.Timedelta(days=1)
    for boundary in pd.date_range(
        first_midnight, timestamps.iloc[-1].normalize(), freq="1D"
    ):
        figure.add_vline(
            x=boundary, line_color="#CBD5E1", line_dash="dot", line_width=1
        )
    figure.update_layout(
        height=380,
        margin={"l": 18, "r": 12, "t": 20, "b": 18},
        hovermode="x unified",
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    figure.update_xaxes(
        title_text="NEM time (AEST)", showgrid=True, gridcolor="#E8EDF1"
    )
    figure.update_yaxes(
        title_text="Price (AUD/MWh)", showgrid=True, gridcolor="#E8EDF1"
    )
    return figure


def build_aemo_daily_figure(result: AEMODailyResult) -> go.Figure:
    """Build the legacy AEMO price and binary pump-schedule operator view."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=result.timestamps,
            y=result.price_aud_per_mwh,
            name="AEMO price",
            mode="lines",
            line={"color": "#2563A6", "width": 2.3},
            hovertemplate="%{x|%H:%M}<br>AUD %{y:,.2f}/MWh<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=result.timestamps,
            y=result.pump_on,
            name="Pump status",
            mode="lines",
            line={"color": "#D97706", "width": 1.5, "shape": "hv"},
            fill="tozeroy",
            fillcolor="rgba(217,119,6,0.45)",
            customdata=np.asarray(result.power_kw).reshape(-1, 1),
            hovertemplate=(
                "%{x|%H:%M}<br>Pump: %{y:.0f}<br>Power: "
                "%{customdata[0]:.0f} kW<extra></extra>"
            ),
        ),
        secondary_y=True,
    )
    figure.update_layout(
        height=440,
        margin={"l": 20, "r": 20, "t": 28, "b": 20},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    figure.update_xaxes(
        title_text="NEM time (AEST)", showgrid=True, gridcolor="#E8EDF1"
    )
    figure.update_yaxes(
        title_text="Price (AUD/MWh)",
        showgrid=True,
        gridcolor="#E8EDF1",
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="Pump status",
        tickvals=[0, 1],
        ticktext=["OFF", "ON"],
        range=[-0.08, 1.08],
        showgrid=False,
        secondary_y=True,
    )
    return figure


def build_price_forecast_figure(frame: pd.DataFrame) -> go.Figure:
    """Build a compact operator preview of the validated 24-hour price window."""
    if not {"DateTime", "Price"}.issubset(frame.columns):
        raise ValueError("Price forecast requires DateTime and Price columns.")

    figure = go.Figure(
        data=[
            go.Scatter(
                x=frame["DateTime"],
                y=frame["Price"],
                name="Forecast price",
                mode="lines",
                line={"color": "#2563A6", "width": 2.4},
                fill="tozeroy",
                fillcolor="rgba(37,99,166,0.08)",
                hovertemplate="%{x|%H:%M}<br>$%{y:.2f}/MWh<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        height=285,
        margin={"l": 18, "r": 12, "t": 18, "b": 18},
        hovermode="x unified",
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    figure.update_xaxes(
        title_text="Sydney time", showgrid=True, gridcolor="#E8EDF1"
    )
    figure.update_yaxes(
        title_text="Price (AUD/MWh)", showgrid=True, gridcolor="#E8EDF1"
    )
    return figure


def build_plan_figure(
    *,
    timestamps: pd.DatetimeIndex,
    result: MPCResult,
    plant: PlantConfig,
    label: str,
) -> go.Figure:
    """Build the operator view of forecast price and scheduled pump state."""
    if len(timestamps) != plant.interval_count:
        raise ValueError(f"Expected {plant.interval_count} timestamps.")

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=result.price_aud_per_mwh,
            name="AEMO price",
            mode="lines",
            line={"color": "#2563A6", "width": 2.4},
            hovertemplate="%{x|%H:%M}<br>$%{y:.2f}/MWh<extra></extra>",
        ),
        secondary_y=False,
    )
    power_hover = np.asarray(result.power_mw, dtype=float).reshape(-1, 1)
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=result.pump_on,
            name="Pump status",
            mode="lines",
            line={"color": "#D97706", "width": 1.5, "shape": "hv"},
            fill="tozeroy",
            fillcolor="rgba(217,119,6,0.48)",
            customdata=power_hover,
            hovertemplate=(
                "%{x|%H:%M}<br>Pump: %{y:.0f}<br>VFD: "
                "%{customdata[0]:.3f} MW<extra></extra>"
            ),
        ),
        secondary_y=True,
    )
    figure.update_layout(
        title={"text": label, "x": 0.01, "xanchor": "left"},
        height=440,
        margin={"l": 20, "r": 20, "t": 52, "b": 20},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#E8EDF1", title_text="Sydney time")
    figure.update_yaxes(
        title_text="Price (AUD/MWh)",
        showgrid=True,
        gridcolor="#E8EDF1",
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="Pump status",
        tickvals=[0, 1],
        ticktext=["OFF", "ON"],
        range=[-0.08, 1.08],
        showgrid=False,
        secondary_y=True,
    )
    return figure


def build_power_figure(
    *, timestamps: pd.DatetimeIndex, result: MPCResult, label: str
) -> go.Figure:
    """Build the engineering-detail view of scheduled VFD power."""
    if len(timestamps) != len(result.power_mw):
        raise ValueError("Timestamp count must match the VFD power schedule.")

    figure = go.Figure(
        data=[
            go.Scatter(
                x=timestamps,
                y=result.power_mw,
                name="VFD power",
                mode="lines",
                line={"color": "#187682", "width": 2.2, "shape": "hv"},
                fill="tozeroy",
                fillcolor="rgba(24,118,130,0.16)",
                hovertemplate="%{x|%H:%M}<br>%{y:.3f} MW<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title={"text": label, "x": 0.01, "xanchor": "left"},
        height=300,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        hovermode="x unified",
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    figure.update_xaxes(
        title_text="Sydney time", showgrid=True, gridcolor="#E8EDF1"
    )
    figure.update_yaxes(
        title_text="Power (MW)", showgrid=True, gridcolor="#E8EDF1", rangemode="tozero"
    )
    return figure
