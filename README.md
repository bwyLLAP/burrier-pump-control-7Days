# Burrier Pump Control Dashboard — Public Test

The light Streamlit dashboard opens directly in the Weekly Planning workspace. The three partner logos are retained, and the AI Assistant is currently a minimised placeholder for a later release.

Weekly Planning identifies the Monday for any selected date. Price retrieval is operator-triggered: enter a KWatch API key for KWatch actual and predispatch prices plus the AEMO PD7Day outlook, or explicitly choose the public AEMO-only route. The key remains in the current Streamlit session and is not written to the repository. The planner first optimises the complete Monday-to-Sunday calendar week, including Monday morning, and freezes that released schedule for execution and comparison. It then optimises a second rolling seven-day plan from the current executable interval. Its default view is Monday-to-Sunday, with a range slider for the additional forecast days. Prices are capped to −100 through 200 AUD/MWh. Red chart backgrounds mark weekday 16:00–20:00 no-pumping periods, pale green shows Monday's frozen schedule, stronger green shows the adjusted rolling plan, and blue marks operator Overrides.

Because the application has no live SCADA connection, operator-entered actual pumping records each completed pump run using its start and end time. Water volume is calculated from the configured flow. An included actual run replaces the original baseline display for that local date, while the unchanged baseline is retained internally for reconciliation. The dashboard compares actual volume with that baseline plan. Any shortfall is added to the next rolling seven-day target and any surplus is subtracted. Forecast cost is not shown as a planning KPI.

The optimiser uses half-hour intervals, a default four-hour minimum continuous run, a configurable minimum interval between pump starts, a six-hour daily minimum, and at least two continuous hours OFF between pumping runs. Infeasible targets are rejected. All operating times use Australia/Sydney.

If automatic retrieval is unavailable, a saved AEMO PD7Day ZIP/CSV can still be uploaded as a legacy calendar-week fallback. Missing future prices are never filled.

Actual records are stored only in the active Streamlit session in this version. This is planning software and is not connected to PLC or SCADA control.

## Deploy on Streamlit Community Cloud

1. Create an empty **private GitHub** repository.
2. Commit and push the contents of this folder as the repository root.
3. In Streamlit Community Cloud, select **Create app** and choose that repository and branch.
4. Set the main file path to `burrier_llm_planner/app.py`.
5. In Advanced settings, select **Python 3.10**. No API key needs to be committed or added to deployment settings; the operator enters it in the password field for the active Streamlit session.
6. Deploy the app, then open its sharing settings and make the app **public**.
7. Test Weekly Planning in a private browser window.

Upload the contents of this folder to the GitHub repository root. The included dependency and Streamlit configuration files are intended for that layout.

Anyone with the public URL can access the app, so do not place confidential operating data in this test deployment. Public apps may also be discoverable by search engines.

On Linux, `packages.txt` asks the host to install the open-source `coinor-cbc` solver. No Gurobi dependency or licence is used.
