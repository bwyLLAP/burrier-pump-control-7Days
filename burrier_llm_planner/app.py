from __future__ import annotations

import io
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from burrier_llm_planner.domain.enums import PlanKind, WorkflowStatus
from burrier_llm_planner.domain.models import (
    CandidateStrategy,
    PhysicalInputs,
    PlanningSession,
    PlantConfig,
)
from burrier_llm_planner.optimization.pi_mpc import MPCInputs, MPCResult, run_mpc
from burrier_llm_planner.optimization.aemo_daily import (
    AEMODailyInputs,
    AEMODailyResult,
    run_aemo_daily_optimization,
)
from burrier_llm_planner.optimization.result_validator import validate_mpc_result
from burrier_llm_planner.services.aemo_price import (
    AEMOPriceService,
    ValidatedPriceWindow,
    validate_price_window,
)
from burrier_llm_planner.services.pd7day_price import (
    PD7DayPriceService,
    PD7DayPriceWindow,
    daily_price_summary,
    parse_pd7day_csv,
    parse_pd7day_zip,
)
from burrier_llm_planner.services.control_agent import (
    AgentContext,
    ControlAgentError,
    OpenAIControlAgent,
    ScenarioOutcome,
    operator_error_message,
)
from burrier_llm_planner.services.knowledge import KnowledgeSource, LocalKnowledgeIndex
from burrier_llm_planner.services.llm_provider import build_llm_provider
from burrier_llm_planner.services.planning_inputs import prepare_planning_inputs
from burrier_llm_planner.services.price_index import PriceIndexLoadResult, PriceIndexService
from burrier_llm_planner.strategy.target_generator import generate_daily_pumping_target
from burrier_llm_planner.strategy.verifier import verify_candidate
from burrier_llm_planner.ui.charts import (
    build_aemo_daily_figure,
    build_pd7day_price_figure,
    build_plan_figure,
    build_power_figure,
    build_price_forecast_figure,
)
from burrier_llm_planner.ui.planning_helpers import (
    build_aemo_daily_schedule_frame,
    build_plan_summary,
    build_schedule_csv,
    build_schedule_frame,
)
from burrier_llm_planner.workflow.audit import AuditLog
from burrier_llm_planner.ui.weekly import render_weekly_workspace
from burrier_llm_planner.workflow.planning import PlanningWorkflow, build_input_fingerprint


ZONE = ZoneInfo("Australia/Sydney")
PLANT = PlantConfig()
DEFAULT_STRATEGY = WORKSPACE_DIR / "Price_Index_Outputs" / "latest_price_strategy.json"
STRATEGY_AEMO = "Daily Planning"
STRATEGY_WEEKLY = "Weekly Planning"
STRATEGY_SEASONAL = "Seasonal Price Index"


def configured_llm_provider():
    return build_llm_provider(
        os.getenv("BURRIER_LLM_PROVIDER", "mock"),
        model=os.getenv("BURRIER_LLM_MODEL", ""),
    )


def initialise_state() -> None:
    defaults: dict[str, object] = {
        "session": None,
        "price_window": None,
        "strategy_load": None,
        "prepared_inputs": None,
        "prepared_demo_mode": None,
        "inputs_confirmed": False,
        "baseline_raw": None,
        "agent_raw_results": {},
        "agent_mode": "arbitrage",
        "agent_target_fraction": 1.0,
        "agent_transition_days": 90,
        "pending_agent_parameters": None,
        "latest_agent_result_id": None,
        "agent_run_notice": None,
        "chat_messages": [
            {
                "role": "assistant",
                "content": (
                    "I am the LLM assistant for the Burrier optimisation-control "
                    "software. Ask me about the model, strategy, code, current plan, "
                    "results, parameters, or operating safety."
                ),
                "evidence": [],
            }
        ],
        "openai_api_key": "",
        "export_result": None,
        "active_strategy": STRATEGY_WEEKLY,
        "aemo_auto_fetch_attempted": False,
        "aemo_daily_price_window": None,
        "aemo_daily_fetch_error": None,
        "aemo_daily_fetch_notice": None,
        "aemo_daily_planning_date": datetime.now(ZONE).date(),
        "aemo_daily_result": None,
        "aemo_daily_flow_lps": 1050.0,
        "aemo_daily_max_power_kw": 1950.0,
        "aemo_daily_standby_power_kw": 0.0,
        "aemo_daily_min_continuous_hours": 4.0,
        "aemo_daily_min_hours": 6.0,
        "aemo_daily_target_ml": 40.0,
        "pd7day_auto_fetch_attempted": False,
        "pd7day_window": None,
        "pd7day_fetch_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def workflow() -> PlanningWorkflow:
    session: PlanningSession = st.session_state.session
    return PlanningWorkflow(
        session=session,
        plant=PLANT,
        audit_log=AuditLog(
            PROJECT_DIR / "data" / "audit" / f"{session.session_id}.jsonl",
            session.session_id,
        ),
    )


@st.cache_resource(show_spinner=False)
def knowledge_index() -> LocalKnowledgeIndex:
    sources = [
        KnowledgeSource(
            root=PROJECT_DIR / "knowledge", category="strategy", authority_rank=3
        ),
        KnowledgeSource(root=PROJECT_DIR, category="current_code", authority_rank=4),
    ]
    legacy = WORKSPACE_DIR / "Pump control software"
    if legacy.exists():
        sources.append(
            KnowledgeSource(root=legacy, category="legacy_code", authority_rank=5)
        )
    for filename in ("LLM_ready_MPC_VFD.py", "LLM_ready_MPC_VFD_price_index.py"):
        source = WORKSPACE_DIR / filename
        if source.exists():
            sources.append(
                KnowledgeSource(root=source, category="research_code", authority_rank=4)
            )
    return LocalKnowledgeIndex.build(sources)


def run_strategy_mpc(strategy, kind: PlanKind) -> tuple[MPCResult, object]:
    session: PlanningSession = st.session_state.session
    price_window: ValidatedPriceWindow = st.session_state.price_window
    daily_demand = float(st.session_state.confirmed_demand_ml)
    initial_storage = float(st.session_state.confirmed_storage_ml)
    source = {
        PlanKind.BASELINE: "annual_price_index",
        PlanKind.CANDIDATE: "operator_candidate",
        PlanKind.AGENT_SIMULATION: "agent_simulation",
    }[kind]
    target = generate_daily_pumping_target(
        demand_ml=daily_demand,
        current_storage_ml=initial_storage,
        target_fraction=strategy.target_fraction,
        transition_days=strategy.transition_days,
        plant=PLANT,
        source=source,
    )
    result = run_mpc(
        MPCInputs(
            price_aud_per_mwh=price_window.frame["Price"].to_numpy(dtype=float),
            demand_ml_per_step=np.full(
                PLANT.interval_count, daily_demand / PLANT.interval_count
            ),
            initial_reservoir_ml=initial_storage,
            initial_pump_on=bool(session.physical_inputs.initial_pump_on),
            elapsed_state_steps=int(session.physical_inputs.elapsed_state_minutes or 0)
            // PLANT.time_step_minutes,
            target_ml=target.applied_ml,
            target_tolerance_ml=target.tolerance_ml,
            mode=strategy.mode,
        ),
        PLANT,
    )
    report = validate_mpc_result(result, PLANT)
    return result, report


def render_metrics(summary) -> None:
    unit_cost = (
        summary.estimated_cost_aud / summary.pumped_volume_ml
        if summary.pumped_volume_ml > 0
        else 0.0
    )
    values = (
        ("Run time", f"{summary.pump_hours:.1f} h"),
        ("Pumped volume", f"{summary.pumped_volume_ml:.1f} ML"),
        ("Total cost", f"AUD {summary.estimated_cost_aud:,.0f}"),
        ("Average unit cost", f"AUD {unit_cost:.2f}/ML"),
    )
    for offset in range(0, len(values), 2):
        columns = st.columns(2)
        for column, (label, value) in zip(columns, values[offset : offset + 2]):
            column.metric(label, value)


def show_result(summary, raw: MPCResult, label: str) -> None:
    render_metrics(summary)
    timestamps = pd.DatetimeIndex(st.session_state.price_window.frame["DateTime"])
    schedule = build_schedule_frame(timestamps, raw)
    st.plotly_chart(
        build_plan_figure(timestamps=timestamps, result=raw, plant=PLANT, label=label),
        width="stretch",
    )
    with st.expander("Engineering details"):
        st.caption(
            "Reservoir values remain part of deterministic safety validation and the "
            "interval table; they are omitted from the primary chart for readability."
        )
        st.write(
            {
                "Energy (MWh)": round(summary.energy_mwh, 2),
                "Pump starts": summary.pump_starts,
                "Minimum reservoir (ML)": round(summary.minimum_reservoir_ml, 1),
                "Independent validation": "Passed" if summary.feasible else "Failed",
            }
        )
        st.plotly_chart(
            build_power_figure(timestamps=timestamps, result=raw, label="VFD power"),
            width="stretch",
        )
        st.dataframe(schedule, width="stretch", hide_index=True)
    source_name = (
        "baseline" if summary.kind is PlanKind.BASELINE else "agent_scenario"
    )
    button_label = (
        "Download baseline CSV"
        if summary.kind is PlanKind.BASELINE
        else "Download Agent scenario CSV"
    )
    st.download_button(
        button_label,
        data=build_schedule_csv(schedule),
        file_name=(
            f"burrier_{source_name}_{st.session_state.session.planning_date.isoformat()}.csv"
        ),
        mime="text/csv",
        width="stretch",
    )


def reset_run_results() -> None:
    session: PlanningSession | None = st.session_state.session
    if session is not None and (
        session.baseline_result is not None or session.agent_simulation_results
    ):
        workflow().invalidate_results()
    st.session_state.baseline_raw = None
    st.session_state.agent_raw_results = {}
    st.session_state.latest_agent_result_id = None
    st.session_state.agent_run_notice = None
    st.session_state.export_result = None


def prepare_request(request: str, planning_day: date, demo_mode: bool) -> None:
    session = PlanningSession.new(request=request, planning_date=planning_day)
    session.operational_export_allowed = not demo_mode
    session.interpretation = configured_llm_provider().interpret_request(
        request, datetime.now(ZONE)
    ).value
    prepared = prepare_planning_inputs(
        planning_date=planning_day,
        demo_mode=demo_mode,
        strategy_path=DEFAULT_STRATEGY,
        plant=PLANT,
    )
    st.session_state.session = session
    st.session_state.prepared_inputs = prepared
    st.session_state.price_window = prepared.price_window
    st.session_state.strategy_load = prepared.strategy_load
    st.session_state.inputs_confirmed = False
    st.session_state.baseline_raw = None
    st.session_state.agent_raw_results = {}
    st.session_state.reservoir_fraction = prepared.reservoir_fraction
    st.session_state.demand_ml = prepared.demand_ml
    st.session_state.initial_on = prepared.initial_pump_on
    st.session_state.elapsed_minutes = prepared.elapsed_state_minutes
    st.session_state.equipment_available = prepared.equipment_available
    st.session_state.operator_id = prepared.operator_id
    st.session_state.measurement_time = prepared.measurement_time.isoformat()
    if prepared.strategy_load is not None:
        session.automatic_strategy = prepared.strategy_load.strategy
        st.session_state.agent_mode = prepared.strategy_load.strategy.mode.value
        st.session_state.agent_target_fraction = (
            prepared.strategy_load.strategy.target_fraction
        )
        st.session_state.agent_transition_days = (
            prepared.strategy_load.strategy.transition_days
        )
        session.operational_export_allowed = (
            prepared.strategy_load.operational_export_allowed and not demo_mode
        )


def ensure_planning_session(demo_mode: bool) -> PlanningSession:
    planning_day = datetime.now(ZONE).date() + timedelta(days=1)
    session: PlanningSession | None = st.session_state.session
    if (
        session is None
        or st.session_state.prepared_demo_mode != demo_mode
        or session.planning_date != planning_day
    ):
        prepare_request(
            "Prepare a cost-aware 24-hour pumping schedule.",
            planning_day,
            demo_mode,
        )
        st.session_state.prepared_demo_mode = demo_mode
    return st.session_state.session


def validate_current_inputs(uploaded_prices, uploaded_strategy, demo_mode: bool) -> None:
    session: PlanningSession = st.session_state.session
    try:
        if uploaded_prices is not None:
            frame = pd.read_csv(io.BytesIO(uploaded_prices.getvalue()))
            price_window = validate_price_window(frame, session.planning_date)
        else:
            price_window = st.session_state.price_window
        if price_window is None:
            raise ValueError("A complete 48-interval AEMO price window is required.")

        strategy_service = PriceIndexService(plant=PLANT)
        if uploaded_strategy is not None:
            strategy_load = strategy_service.load_bytes(
                uploaded_strategy.getvalue(),
                planning_date=session.planning_date,
                live_mode=not demo_mode,
                source_name=uploaded_strategy.name,
            )
        else:
            strategy_load = strategy_service.load(
                DEFAULT_STRATEGY,
                session.planning_date,
                live_mode=not demo_mode,
            )
        measurement = datetime.fromisoformat(st.session_state.measurement_time)
        if measurement.tzinfo is None:
            measurement = measurement.replace(tzinfo=ZONE)
        if not st.session_state.equipment_available:
            raise ValueError("Pump and VFD must be available before optimisation.")

        session.physical_inputs = PhysicalInputs(
            measured_reservoir_fraction=float(st.session_state.reservoir_fraction),
            measurement_time=measurement,
            estimated_start_reservoir_ml=float(st.session_state.reservoir_fraction)
            * PLANT.reservoir_capacity_ml,
            initial_pump_on=bool(st.session_state.initial_on),
            elapsed_state_minutes=int(st.session_state.elapsed_minutes),
            equipment_availability={"pump": True, "vfd": True},
        )
        fingerprint = build_input_fingerprint(
            {
                "planning_date": session.planning_date,
                "prices": price_window.source_hash,
                "strategy": strategy_load.provenance.source_hash,
                "demand_ml": float(st.session_state.demand_ml),
                "physical_inputs": session.physical_inputs.model_dump(),
                "plant_config": PLANT.model_dump(),
            }
        )
        if session.input_fingerprint and session.input_fingerprint != fingerprint:
            reset_run_results()
        st.session_state.price_window = price_window
        st.session_state.strategy_load = strategy_load
        st.session_state.confirmed_demand_ml = float(st.session_state.demand_ml)
        st.session_state.confirmed_storage_ml = (
            float(st.session_state.reservoir_fraction) * PLANT.reservoir_capacity_ml
        )
        st.session_state.pending_fingerprint = fingerprint
        st.session_state.inputs_confirmed = True
        session.automatic_strategy = strategy_load.strategy
        session.operational_export_allowed = (
            strategy_load.operational_export_allowed and not demo_mode
        )
        session.status = WorkflowStatus.INPUTS_READY
        st.success(
            "Inputs validated. The AEMO window, seasonal strategy, and plant state are ready."
        )
    except Exception as exc:
        st.session_state.inputs_confirmed = False
        st.error(f"Inputs not accepted: {exc}")


def render_input_editor(session: PlanningSession, demo_mode: bool) -> None:
    prepared = st.session_state.prepared_inputs
    for error in prepared.errors if prepared is not None else []:
        st.warning(error)

    st.subheader("AEMO forecast")
    price_window: ValidatedPriceWindow | None = st.session_state.price_window
    source = "Demonstration profile" if demo_mode else "AEMO pre-dispatch"
    if price_window is not None:
        st.caption(f"{source} · {len(price_window.frame)} validated intervals")
    price_data_column, price_chart_column = st.columns([2, 5], gap="large")
    with price_data_column:
        if price_window is not None:
            price_preview = price_window.frame.copy()
            price_preview["Time"] = pd.to_datetime(price_preview["DateTime"]).dt.strftime(
                "%H:%M"
            )
            price_preview["Price"] = price_preview["Price"].round(2)
            st.dataframe(
                price_preview[["Time", "Price"]],
                width="stretch",
                height=285,
                hide_index=True,
                column_config={
                    "Time": st.column_config.TextColumn("Time", width="small"),
                    "Price": st.column_config.NumberColumn(
                        "AUD/MWh", format="%.2f", width="small"
                    ),
                },
            )
        uploaded_prices = st.file_uploader(
            "Replace forecast CSV",
            type="csv",
            key="prices_upload",
            help="Optional 48-interval file with DateTime and Price columns.",
        )
    with price_chart_column:
        if price_window is not None:
            st.plotly_chart(
                build_price_forecast_figure(price_window.frame),
                width="stretch",
                config={"displayModeBar": False},
            )

    st.divider()
    st.subheader("Seasonal strategy")
    st.caption(
        "The validated default price-index strategy is used unless a replacement JSON is uploaded."
    )
    loaded: PriceIndexLoadResult | None = st.session_state.strategy_load
    if loaded is not None:
        st.markdown(
            f"**Operating mode** ($m_d$): `{loaded.strategy.mode.value}`  \n"
            f"**Reservoir target fraction** ($\\alpha_d$): "
            f"`{loaded.strategy.target_fraction:.2f}`  \n"
            f"**Transition period** ($H_d$): "
            f"`{loaded.strategy.transition_days} days`  \n"
            f"**Price regime:** `{loaded.metadata.price_regime}`"
        )
        for warning in loaded.warnings:
            st.warning(warning)
    uploaded_strategy = st.file_uploader(
        "Replace strategy JSON",
        type="json",
        key="strategy_upload",
        help="Optional replacement. The existing safety schema and bounds are applied.",
    )

    st.divider()
    st.subheader("Plant state & demand")
    st.caption("Current storage, demand, and pump state used by the 24-hour optimiser.")
    st.number_input(
        "Measured reservoir fraction",
        min_value=PLANT.min_reservoir_fraction,
        max_value=1.0,
        step=0.001,
        format="%.3f",
        key="reservoir_fraction",
    )
    st.number_input(
        "24-hour demand forecast (ML)",
        min_value=0.0,
        step=0.5,
        key="demand_ml",
    )
    st.text_input("Operator / engineer ID", key="operator_id")
    st.toggle("Pump currently ON", key="initial_on")
    st.number_input(
        "Minutes in current pump state",
        min_value=0,
        step=30,
        key="elapsed_minutes",
    )
    st.toggle("Pump and VFD available", key="equipment_available")
    st.text_input("Measurement timestamp (Sydney)", key="measurement_time")

    if st.button("Validate inputs", type="primary"):
        validate_current_inputs(uploaded_prices, uploaded_strategy, demo_mode)


class StreamlitScenarioSimulator:
    def simulate(self, candidate: CandidateStrategy) -> ScenarioOutcome:
        session: PlanningSession = st.session_state.session
        decision = verify_candidate(
            candidate,
            plant=PLANT,
            current_fraction=float(st.session_state.reservoir_fraction),
            planning_date=session.planning_date,
        )
        if not decision.accepted or decision.resolved is None:
            return ScenarioOutcome(
                feasible=False,
                status="parameter_rejected",
                candidate=candidate,
                reasons=decision.reasons,
            )
        raw, report = run_strategy_mpc(
            decision.resolved, PlanKind.AGENT_SIMULATION
        )
        if not report.feasible:
            return ScenarioOutcome(
                feasible=False,
                status="validation_failed",
                candidate=candidate,
                reasons=list(report.violations),
            )
        summary = build_plan_summary(
            result=raw,
            report=report,
            kind=PlanKind.AGENT_SIMULATION,
            input_fingerprint=session.input_fingerprint or "",
        )
        workflow().record_agent_simulation(summary)
        raw_results = dict(st.session_state.agent_raw_results)
        raw_results[summary.result_id] = raw
        st.session_state.agent_raw_results = raw_results
        baseline = session.baseline_result
        comparison = {
            "cost_delta_aud": summary.estimated_cost_aud
            - (baseline.estimated_cost_aud if baseline else 0.0),
            "pump_hours_delta": summary.pump_hours
            - (baseline.pump_hours if baseline else 0.0),
        }
        return ScenarioOutcome(
            feasible=True,
            status="validated",
            candidate=candidate,
            result_id=summary.result_id,
            metrics={
                "estimated_cost_aud": summary.estimated_cost_aud,
                "pump_hours": summary.pump_hours,
                "minimum_reservoir_ml": summary.minimum_reservoir_ml,
            },
            comparison=comparison,
        )


def agent_context(session: PlanningSession | None) -> AgentContext:
    if session is None:
        return AgentContext(validated_inputs=False)
    if st.session_state.active_strategy == STRATEGY_AEMO:
        window: ValidatedPriceWindow | None = st.session_state.aemo_daily_price_window
        result: AEMODailyResult | None = st.session_state.aemo_daily_result
        pd7: PD7DayPriceWindow | None = st.session_state.pd7day_window
        price_summary = ""
        if window is not None:
            cheapest = window.frame.nsmallest(3, "Price")
            price_summary = "Lowest 24-hour forecast intervals: " + ", ".join(
                stamp.strftime("%H:%M") for stamp in cheapest["DateTime"]
            )
        pd7_summary = ""
        if pd7 is not None:
            pd7_summary = (
                f"NSW1 seven-day outlook spans {pd7.start:%Y-%m-%d %H:%M} to "
                f"{pd7.end:%Y-%m-%d %H:%M}; informational only."
            )
        return AgentContext(
            active_strategy="aemo_daily",
            validated_inputs=window is not None,
            planning_date=session.planning_date,
            price_summary=price_summary,
            schedule_summary=(
                f"Pump ON for {result.pump_hours:.1f} hours."
                if result is not None
                else "No AEMO daily schedule has been run."
            ),
            aemo_parameters={
                "daily_target_ml": float(st.session_state.aemo_daily_target_ml),
                "flow_lps": float(st.session_state.aemo_daily_flow_lps),
                "max_power_kw": float(st.session_state.aemo_daily_max_power_kw),
                "minimum_continuous_run_hours": float(
                    st.session_state.aemo_daily_min_continuous_hours
                ),
                "minimum_daily_run_hours": float(st.session_state.aemo_daily_min_hours),
            },
            aemo_result_summary=(
                {
                    "pump_hours": result.pump_hours,
                    "pumped_volume_ml": result.pumped_volume_ml,
                    "cost_aud": result.objective_aud,
                    "start_count": float(result.start_count),
                }
                if result is not None
                else {}
            ),
            pd7day_summary=pd7_summary,
        )
    price_summary = ""
    price_window: ValidatedPriceWindow | None = st.session_state.price_window
    if price_window is not None:
        frame = price_window.frame.nsmallest(3, "Price")
        times = ", ".join(stamp.strftime("%H:%M") for stamp in frame["DateTime"])
        price_summary = f"Lowest forecast intervals: {times}."
    schedule_summary = ""
    raw: MPCResult | None = st.session_state.baseline_raw
    if raw is not None:
        on_indices = np.flatnonzero(raw.pump_on >= 0.5)
        schedule_summary = (
            f"Pump ON in {len(on_indices)} of {PLANT.interval_count} half-hour intervals."
        )
    return AgentContext(
        active_strategy="seasonal_price_index",
        validated_inputs=bool(st.session_state.inputs_confirmed),
        current_reservoir_fraction=float(
            st.session_state.get("reservoir_fraction", 0.95)
        ),
        planning_date=session.planning_date,
        automatic_strategy=session.automatic_strategy,
        baseline_result=session.baseline_result,
        price_summary=price_summary,
        schedule_summary=schedule_summary,
    )


def audit_agent_turn(
    question: str, turn, *, provider_name: str, provider_version: str
) -> None:
    session: PlanningSession | None = st.session_state.session
    if session is None:
        return
    AuditLog(
        PROJECT_DIR / "data" / "audit" / f"{session.session_id}.jsonl",
        session.session_id,
    ).record(
        event_type="agent_turn",
        actor="operator",
        payload={
            "question": question,
            "provider": provider_name,
            "provider_version": provider_version,
            "intent": turn.intent,
            "basis": turn.basis,
            "evidence": [item.model_dump() for item in turn.evidence],
            "proposed_parameters": [
                item.model_dump() for item in turn.proposed_parameters
            ],
            "scenario_result_ids": turn.scenario_result_ids,
            "warnings": turn.warnings,
        },
        current_fingerprint=session.input_fingerprint,
    )


def render_agent_panel(session: PlanningSession | None) -> None:
    st.subheader("Burrier Control Agent")
    st.caption(f"Context: {st.session_state.active_strategy}")
    session_key = str(st.session_state.openai_api_key).strip()
    api_key = session_key
    if api_key:
        st.caption("GPT-5.6 Terra · project knowledge connected")
    else:
        st.caption("GPT-5.6 Terra · API key required")
    with st.expander("Connect OpenAI"):
        st.text_input(
            "OpenAI API key",
            type="password",
            key="openai_api_key",
            help=(
                "Use your own restricted OpenAI project key. It is held only in "
                "this Streamlit session and is not written to files, chat history, "
                "audit records, or downloads."
            ),
        )
        st.caption(
            "Your key is sent over HTTPS to this Streamlit server and is used only "
            "for its OpenAI requests. Model: gpt-5.6-terra. Requests include your "
            "question, recent chat, validated plan summary, and selected local "
            "evidence. Use a restricted project key with a small budget."
        )
        session_key = str(st.session_state.openai_api_key).strip()
        api_key = session_key
    if not api_key:
        st.error("OpenAI API key is required before the Agent can answer questions.")
    messages = list(st.session_state.chat_messages)
    chat_history = st.container(height=540, border=False)
    with chat_history:
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                evidence = message.get("evidence", [])
                if evidence:
                    with st.expander("Evidence"):
                        for item in evidence:
                            st.caption(
                                f"{item['source_category']} · "
                                f"{Path(item['source_path']).name}"
                            )
                            st.write(item["excerpt"])

    question = st.chat_input(
        "Ask the Agent", key="agent_question", disabled=not bool(api_key)
    )
    if question:
        with chat_history:
            with st.chat_message("user"):
                st.markdown(question)
        simulator = StreamlitScenarioSimulator()
        agent = OpenAIControlAgent(
            knowledge=knowledge_index(),
            simulator=simulator,
            plant=PLANT,
            model="gpt-5.6-terra",
            api_key=api_key,
        )
        with st.spinner("Checking strategy evidence and current planning context…"):
            try:
                history = [
                    {"role": item["role"], "content": item["content"]}
                    for item in messages[-6:]
                ]
                turn = agent.answer(
                    question, agent_context(session), history=history
                )
            except ControlAgentError as exc:
                st.error(operator_error_message(exc))
                turn = None
        if turn is not None:
            evidence_payload = [item.model_dump() for item in turn.evidence]
            with chat_history:
                with st.chat_message("assistant"):
                    st.markdown(turn.answer)
                    if turn.refused:
                        st.warning(
                            "Request refused at the immutable hard-constraint boundary"
                        )
                    if turn.evidence:
                        with st.expander("Evidence"):
                            for item in turn.evidence:
                                st.caption(
                                    f"{item.source_category} · "
                                    f"{Path(item.source_path).name}"
                                )
                                st.write(item.excerpt)
            st.session_state.chat_messages = messages + [
                {"role": "user", "content": question, "evidence": []},
                {
                    "role": "assistant",
                    "content": turn.answer,
                    "evidence": evidence_payload,
                },
            ]
            audit_agent_turn(
                question,
                turn,
                provider_name=agent.provider_name,
                provider_version=agent.provider_version,
            )
            if len(turn.proposed_parameters) == 1:
                st.session_state.pending_agent_parameters = (
                    turn.proposed_parameters[0].model_dump(mode="json")
                )
                st.rerun()


def render_agent_scenario_workspace(session: PlanningSession | None) -> None:
    st.subheader("Agent scenario")
    st.caption(
        "GPT suggestions populate these editable supervisory parameters. "
        "Only Run Agent scenario starts the local CBC solve."
    )
    pending = st.session_state.pop("pending_agent_parameters", None)
    if pending is not None:
        st.session_state.agent_mode = str(pending["mode"])
        st.session_state.agent_target_fraction = float(pending["target_fraction"])
        st.session_state.agent_transition_days = int(pending["transition_days"])

    mode_column, target_column, transition_column = st.columns(3)
    with mode_column:
        st.selectbox(
            r"Operating mode ($m_d$)",
            ["arbitrage", "emergency"],
            key="agent_mode",
        )
    with target_column:
        st.number_input(
            r"Reservoir target fraction ($\alpha_d$)",
            min_value=PLANT.min_reservoir_fraction,
            max_value=1.0,
            step=0.01,
            format="%.2f",
            key="agent_target_fraction",
        )
    with transition_column:
        st.number_input(
            r"Transition period ($H_d$)",
            min_value=1,
            max_value=365,
            step=1,
            key="agent_transition_days",
        )
    can_run = bool(
        session is not None
        and st.session_state.inputs_confirmed
        and session.baseline_result is not None
    )
    if st.button("Run Agent scenario", disabled=not can_run, type="primary"):
        candidate = CandidateStrategy(
            mode=st.session_state.agent_mode,
            target_fraction=float(st.session_state.agent_target_fraction),
            transition_days=int(st.session_state.agent_transition_days),
            rationale="Customer-edited Agent scenario parameters.",
        )
        try:
            outcome = StreamlitScenarioSimulator().simulate(candidate)
            if not outcome.feasible or not outcome.result_id:
                detail = "; ".join(outcome.reasons) or outcome.status
                st.error(f"Agent scenario not accepted: {detail}")
            else:
                st.session_state.latest_agent_result_id = outcome.result_id
                st.session_state.agent_run_notice = (
                    "Validated Agent scenario passed CBC and independent checks."
                )
                st.rerun()
        except Exception as exc:
            st.error(f"Agent scenario not accepted: {exc}")
    if st.session_state.agent_run_notice:
        st.success(st.session_state.agent_run_notice)

    latest_id = st.session_state.latest_agent_result_id
    if session is not None and latest_id:
        latest_summary = next(
            (
                result
                for result in session.agent_simulation_results
                if result.result_id == latest_id
            ),
            None,
        )
        latest_raw = st.session_state.agent_raw_results.get(latest_id)
        if latest_summary is not None and latest_raw is not None:
            show_result(latest_summary, latest_raw, "Agent scenario")


def render_price_index_workspace(demo_mode: bool) -> None:
    session = ensure_planning_session(demo_mode)
    render_input_editor(session, demo_mode)

    st.subheader("Automatic baseline")
    st.caption(
        "The price index supplies the daily target; CBC PI-MPC selects 48 pump/VFD intervals."
    )
    if st.button(
        "Run automatic baseline", disabled=not st.session_state.inputs_confirmed
    ):
        try:
            with st.spinner("Solving and independently validating the 24-hour PI-MPC…"):
                raw, report = run_strategy_mpc(
                    session.automatic_strategy, PlanKind.BASELINE
                )
            summary = build_plan_summary(
                result=raw,
                report=report,
                kind=PlanKind.BASELINE,
                input_fingerprint=st.session_state.pending_fingerprint,
            )
            workflow().record_baseline(
                summary, input_fingerprint=st.session_state.pending_fingerprint
            )
            st.session_state.baseline_raw = raw
            st.success(
                "Feasible automatic baseline passed independent constraint validation."
            )
        except Exception as exc:
            st.error(f"Baseline not accepted: {exc}")

    if session.baseline_result is not None and st.session_state.baseline_raw is not None:
        show_result(session.baseline_result, st.session_state.baseline_raw, "Automatic baseline")

    render_agent_scenario_workspace(session)


def _live_fetch_disabled() -> bool:
    return os.getenv("BURRIER_DISABLE_AUTO_FETCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def ensure_automatic_pd7day_data() -> None:
    """Fetch only the informational PD7Day product automatically."""
    if not st.session_state.pd7day_auto_fetch_attempted:
        st.session_state.pd7day_auto_fetch_attempted = True
        if not _live_fetch_disabled():
            try:
                st.session_state.pd7day_window = PD7DayPriceService().fetch_latest()
                st.session_state.pd7day_fetch_error = None
            except Exception as exc:
                st.session_state.pd7day_fetch_error = str(exc)


def _replace_daily_prices(uploaded, planning_day: date) -> None:
    if uploaded is None:
        return
    frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    st.session_state.aemo_daily_price_window = validate_price_window(
        frame, planning_day
    )
    st.session_state.aemo_daily_fetch_error = None
    st.session_state.aemo_daily_result = None


def _replace_pd7day(uploaded) -> None:
    if uploaded is None:
        return
    content = uploaded.getvalue()
    if uploaded.name.lower().endswith(".zip") or content[:2] == b"PK":
        window = parse_pd7day_zip(content, source_name=uploaded.name)
    else:
        window = parse_pd7day_csv(content, source_name=uploaded.name)
    st.session_state.pd7day_window = window
    st.session_state.pd7day_fetch_error = None


def render_aemo_daily_result(result: AEMODailyResult) -> None:
    metric_values = (
        ("Run time", f"{result.pump_hours:.1f} h"),
        ("Pumped volume", f"{result.pumped_volume_ml:.1f} ML"),
        ("Total cost", f"AUD {result.objective_aud:,.0f}"),
        (
            "Average unit cost",
            f"AUD {result.objective_aud / result.pumped_volume_ml:.2f}/ML"
            if result.pumped_volume_ml
            else "—",
        ),
    )
    for offset in range(0, 4, 2):
        columns = st.columns(2)
        for column, (label, value) in zip(columns, metric_values[offset : offset + 2]):
            column.metric(label, value)
    st.plotly_chart(build_aemo_daily_figure(result), width="stretch")
    periods = pd.DataFrame(
        [
            {
                "Start": period.start,
                "Stop": period.stop,
                "Start price (AUD/MWh)": period.start_price_aud_per_mwh,
                "Stop price (AUD/MWh)": period.stop_price_aud_per_mwh,
                "Volume (ML)": period.volume_ml,
            }
            for period in result.periods
        ]
    )
    if not periods.empty:
        st.dataframe(periods, width="stretch", hide_index=True)
    schedule = build_aemo_daily_schedule_frame(result)
    st.download_button(
        "Download AEMO schedule CSV",
        data=build_schedule_csv(schedule),
        file_name=f"burrier_aemo_daily_{result.timestamps[0].date()}.csv",
        mime="text/csv",
        width="stretch",
    )


def render_pd7day_outlook() -> None:
    st.subheader("7-Day NSW Price Outlook")
    st.caption(
        "AEMO PD7Day · NSW1 PRICESOLUTION · informational only—not used by the optimiser."
    )
    uploaded = st.file_uploader(
        "Replace PD7Day ZIP or CSV",
        type=["zip", "csv"],
        key="pd7day_upload",
    )
    if st.button("Refresh 7-day outlook"):
        st.session_state.pd7day_auto_fetch_attempted = False
        st.session_state.pd7day_fetch_error = None
        st.rerun()
    if uploaded is not None:
        try:
            _replace_pd7day(uploaded)
        except Exception as exc:
            st.error(f"PD7Day file not accepted: {exc}")
    error = st.session_state.pd7day_fetch_error
    window: PD7DayPriceWindow | None = st.session_state.pd7day_window
    if error:
        st.warning(f"Automatic PD7Day retrieval was unavailable: {error}")
    if window is None:
        st.info("No validated PD7Day outlook is available. Upload the latest AEMO ZIP or CSV.")
        return
    st.caption(
        f"Forecast run {window.run_datetime:%Y-%m-%d %H:%M} · "
        f"{window.start:%Y-%m-%d %H:%M} to {window.end:%Y-%m-%d %H:%M}"
    )
    st.plotly_chart(build_pd7day_price_figure(window.frame), width="stretch")
    summary = daily_price_summary(window.frame).copy()
    for column in ("Minimum", "Average", "Maximum"):
        summary[column] = summary[column].round(2)
    st.dataframe(summary, width="stretch", hide_index=True)


def render_aemo_daily_workspace(demo_mode: bool) -> None:
    session = ensure_planning_session(demo_mode)
    ensure_automatic_pd7day_data()
    controls, output = st.columns([2, 5], gap="large")
    with controls:
        st.subheader("System Parameters")
        st.number_input("Flow rate (L/s)", min_value=1.0, key="aemo_daily_flow_lps")
        st.number_input(
            "Maximum pump power (kW)", min_value=1.0, key="aemo_daily_max_power_kw"
        )
        st.number_input(
            "Standby power (kW)", min_value=0.0, key="aemo_daily_standby_power_kw"
        )
        st.number_input(
            "Minimum continuous run (h)",
            min_value=0.0,
            max_value=24.0,
            step=0.5,
            key="aemo_daily_min_continuous_hours",
        )
        st.number_input(
            "Minimum daily run (h)",
            min_value=0.0,
            max_value=24.0,
            step=0.5,
            key="aemo_daily_min_hours",
        )
        st.number_input(
            "Daily pumping target (ML)",
            min_value=0.0,
            step=1.0,
            key="aemo_daily_target_ml",
        )
        window: ValidatedPriceWindow | None = st.session_state.aemo_daily_price_window
        if window is not None:
            st.success(
                f"{len(window.frame)} validated AEMO intervals ready for "
                f"{window.planning_date:%Y-%m-%d}."
            )
        if st.session_state.aemo_daily_fetch_notice:
            st.info(st.session_state.aemo_daily_fetch_notice)
        if st.session_state.aemo_daily_fetch_error:
            st.warning(
                "AEMO retrieval is unavailable. Use Fetch AEMO Data again, "
                "or upload a validated 48-interval forecast CSV."
            )
        fetch_type = "primary" if window is None else "secondary"
        if st.button("Fetch AEMO Data", type=fetch_type, width="stretch"):
            planning_day = datetime.now(ZONE).date()
            try:
                with st.spinner("Fetching the complete AEMO daily forecast…"):
                    st.session_state.aemo_daily_price_window = AEMOPriceService(
                        PROJECT_DIR / "data" / "cache"
                    ).fetch_manual_day(planning_day)
                st.session_state.aemo_daily_planning_date = planning_day
                st.session_state.aemo_daily_fetch_error = None
                st.session_state.aemo_daily_fetch_notice = (
                    f"AEMO data fetched manually for {planning_day:%Y-%m-%d}."
                )
                st.session_state.aemo_daily_result = None
                st.rerun()
            except Exception as exc:
                st.session_state.aemo_daily_fetch_error = str(exc)
                st.session_state.aemo_daily_fetch_notice = None
        uploaded = st.file_uploader(
            "Replace 24-hour AEMO forecast CSV",
            type="csv",
            key="aemo_daily_upload",
        )
        if uploaded is not None:
            try:
                _replace_daily_prices(
                    uploaded, st.session_state.aemo_daily_planning_date
                )
                window = st.session_state.aemo_daily_price_window
            except Exception as exc:
                st.error(f"AEMO forecast not accepted: {exc}")
        if st.button("Run AEMO optimisation", type="primary", disabled=window is None):
            try:
                with st.spinner("Solving the 24-hour AEMO schedule with CBC…"):
                    st.session_state.aemo_daily_result = run_aemo_daily_optimization(
                        AEMODailyInputs(
                            timestamps=pd.DatetimeIndex(window.frame["DateTime"]),
                            price_aud_per_mwh=window.frame["Price"].to_numpy(float),
                            daily_target_ml=float(st.session_state.aemo_daily_target_ml),
                            flow_lps=float(st.session_state.aemo_daily_flow_lps),
                            max_power_kw=float(st.session_state.aemo_daily_max_power_kw),
                            standby_power_kw=float(st.session_state.aemo_daily_standby_power_kw),
                            minimum_continuous_run_hours=float(
                                st.session_state.aemo_daily_min_continuous_hours
                            ),
                            minimum_daily_run_hours=float(st.session_state.aemo_daily_min_hours),
                        )
                    )
            except Exception as exc:
                st.error(f"AEMO optimisation not accepted: {exc}")
    with output:
        st.subheader("Optimisation Result & Schedule")
        result: AEMODailyResult | None = st.session_state.aemo_daily_result
        if result is None:
            window = st.session_state.aemo_daily_price_window
            if window is not None:
                st.info(
                    "Validated prices and default parameters are ready. Run the optimiser to create the schedule."
                )
                st.plotly_chart(
                    build_price_forecast_figure(window.frame),
                    width="stretch",
                    config={"displayModeBar": False},
                )
            else:
                st.info(
                    "A complete 48-interval price window is required before optimisation."
                )
        else:
            render_aemo_daily_result(result)
    st.divider()
    render_pd7day_outlook()


def render_brand_header(active_page: str) -> None:
    title, logos = st.columns([5, 3], vertical_alignment="center")
    with title:
        st.title("Burrier Pump Control")
        st.caption(f"{active_page} · AEMO price-aware operations · Australia/Sydney")
    with logos:
        columns = st.columns(3, vertical_alignment="center")
        for column, filename in zip(
            columns, ("uow.png", "shoalhaven_water.png", "arc_future_grids.png")
        ):
            column.image(PROJECT_DIR / "assets" / filename, width="stretch")


def render_strategy_workspace(demo_mode: bool) -> None:
    with st.sidebar:
        st.markdown("<div class='sidebar-brand'>Burrier Control</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-caption'>Planning workspace</div>", unsafe_allow_html=True)
        st.radio(
            "Workspace",
            [STRATEGY_WEEKLY, STRATEGY_AEMO],
            key="active_strategy",
            label_visibility="collapsed",
        )
    render_brand_header(st.session_state.active_strategy)
    if st.session_state.active_strategy == STRATEGY_WEEKLY:
        render_weekly_workspace()
    else:
        render_aemo_daily_workspace(demo_mode)



st.set_page_config(page_title="Burrier Pump Control", page_icon="💧", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink:#17313a; --muted:#60747b; --teal:#187682; --teal-soft:#e8f3f3; --surface:#f5f8f9; --line:#d8e2e5; --amber:#d97706; }
    ::selection { background:#cce5e6; color:var(--ink); }
    .stApp { background:#fbfcfc; color:var(--ink); }
    .block-container { max-width:1480px; padding-top:3.75rem; padding-bottom:5rem; }
    .block-container h1 { color:var(--ink); font-size:1.65rem; line-height:1.18; font-weight:680; letter-spacing:-0.025em; margin-bottom:.15rem; text-wrap:balance; }
    .block-container h2 { color:var(--ink); font-size:1.3rem; line-height:1.25; font-weight:650; letter-spacing:-0.02em; text-wrap:balance; }
    .block-container h3 { color:var(--ink); font-size:1.08rem; line-height:1.3; font-weight:650; letter-spacing:-0.015em; text-wrap:balance; }
    [data-testid="stCaptionContainer"] { color:var(--muted); font-size:.82rem; line-height:1.45; }
    [data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-color:var(--line); border-radius:12px; box-shadow:0 3px 14px rgba(20,55,64,.035); }
    [data-testid="stVerticalBlockBorderWrapper"] > div { padding:.1rem; }
    .panel-title { color:var(--ink); font-size:.94rem; line-height:1.35; font-weight:680; margin:0 0 .25rem; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:0.85rem 1rem; box-shadow:0 3px 14px rgba(20,55,64,.04); }
    [data-testid="stMetricLabel"] { font-size:.78rem; }
    [data-testid="stMetricValue"] { font-size:1.35rem; }
    [data-testid="stSidebar"] { background:#eef3f4; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.35rem; }
    .sidebar-brand { color:var(--ink); font-size:1.08rem; line-height:1.25; font-weight:720; letter-spacing:-.015em; margin:0 0 .25rem; }
    .sidebar-caption { color:var(--muted); font-size:.78rem; line-height:1.4; margin:0 0 1rem; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap:.35rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { width:100%; min-height:2.45rem; border-radius:8px; padding:.48rem .65rem; margin:0; font-size:.88rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:#fff; color:var(--teal); box-shadow:0 2px 9px rgba(20,55,64,.06); }
    [data-testid="stChatMessage"] { border-bottom:1px solid var(--line); padding-bottom:0.8rem; }
    .safety-note { border:1px solid var(--line); background:var(--surface); padding:0.9rem 1rem; border-radius:8px; margin-bottom:1rem; }
    .stButton > button[kind="primary"] { background:var(--teal); border-color:var(--teal); }
    button:focus-visible, input:focus-visible, textarea:focus-visible, [tabindex]:focus-visible { outline:3px solid rgba(24,118,130,.28); outline-offset:2px; }
    button[role="switch"][aria-checked="true"] { background:var(--teal); }
    .balance-card { min-height:3rem; display:flex; align-items:center; justify-content:flex-end; gap:.7rem; color:var(--muted); }
    .balance-card strong { color:var(--ink); font-size:1.12rem; margin-right:1.2rem; }
    .day-review { display:grid; grid-template-columns:5.2rem minmax(0,1fr); gap:1rem; align-items:center; padding:.8rem 0; border-top:1px solid var(--line); }
    .day-review:first-of-type { border-top:0; }
    .day-review.tomorrow { background:#f1f8f8; margin-inline:-.65rem; padding-inline:.65rem; border-radius:8px; }
    .match-ring { --match:0%; width:4.6rem; height:4.6rem; border-radius:50%; display:grid; place-content:center; text-align:center; background:radial-gradient(circle at center,#fff 59%,transparent 61%),conic-gradient(var(--teal) var(--match),#dfe8ea 0); color:var(--ink); font-variant-numeric:tabular-nums; }
    .match-ring.empty { background:radial-gradient(circle at center,#fff 59%,transparent 61%),conic-gradient(#dfe8ea 100%,#dfe8ea 0); color:var(--muted); }
    .match-ring span { font-size:1.05rem; line-height:1; font-weight:720; }
    .match-ring small { margin-top:.22rem; font-size:.58rem; line-height:1; color:var(--muted); letter-spacing:.04em; }
    .day-review-body { min-width:0; }
    .day-review-head { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:.38rem; }
    .day-review-head strong { font-size:.89rem; }
    .day-review-head span { color:var(--teal); font-size:.74rem; font-weight:650; }
    .day-review p { margin:.2rem 0 0; color:var(--muted); font-size:.8rem; }
    .window-line { display:grid; grid-template-columns:10.5rem minmax(12rem,1fr); align-items:center; gap:.55rem; margin:.2rem 0; }
    .window-line em { overflow:hidden; color:var(--muted); font-size:.72rem; font-style:normal; text-overflow:ellipsis; white-space:nowrap; }
    .window-track { display:grid; grid-template-columns:repeat(48,minmax(1px,1fr)); gap:1px; height:.48rem; background:#edf2f3; }
    .window-cell { display:block; min-width:1px; background:#e3eaec; }
    .window-cell.on { background:var(--teal); }
    .review-stats { display:flex; flex-wrap:wrap; gap:.25rem 1rem; margin-top:.38rem; color:var(--muted); font-size:.75rem; font-variant-numeric:tabular-nums; }
    .review-stats b { color:var(--ink); font-weight:650; }
    .ai-launcher { position:fixed; right:1.35rem; bottom:1.2rem; z-index:9999; display:flex; align-items:center; gap:.55rem; background:#fff; color:var(--ink); border:1px solid var(--line); border-radius:999px; padding:.72rem 1rem; box-shadow:0 8px 28px rgba(20,55,64,.16); font-size:.88rem; font-weight:650; }
    .ai-launcher .dot { width:.62rem; height:.62rem; border-radius:50%; background:var(--teal); box-shadow:0 0 0 4px var(--teal-soft); }
    .ai-launcher small { color:var(--muted); font-weight:500; }
    [data-testid="stImage"] { display:flex; align-items:center; justify-content:center; }
    [data-testid="stImage"] img { width:auto !important; max-width:100%; max-height:72px; object-fit:contain; }
    @media (max-width:760px) {
      .block-container { padding-bottom:5.5rem; }
      .ai-launcher { right:.75rem; bottom:.75rem; width:2.75rem; height:2.75rem; padding:0; justify-content:center; }
      .ai-launcher span:not(.dot), .ai-launcher small { display:none; }
      .day-review { grid-template-columns:4.3rem minmax(0,1fr); gap:.65rem; }
      .match-ring { width:4rem; height:4rem; }
      .window-line { grid-template-columns:1fr; gap:.18rem; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { scroll-behavior:auto !important; transition:none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
initialise_state()
demo_mode = True

render_strategy_workspace(demo_mode)
st.markdown(
    "<div class='ai-launcher' role='status' aria-label='AI Assistant coming soon'>"
    "<span class='dot'></span><span>AI Assistant</span><small>Coming soon</small></div>",
    unsafe_allow_html=True,
)
