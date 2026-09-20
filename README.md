# Burrier Pump Control Dashboard — Public Test

The light Streamlit dashboard combines the Daily Planning and Weekly Planning workspaces. The three partner logos are retained, and the AI Assistant is currently a minimised placeholder for a later release.

Weekly Planning automatically identifies the Monday for any selected date. It first optimises the complete Monday-to-Sunday calendar week, including Monday morning, and then optimises a second rolling seven-day plan from the current executable interval. The chart combines completed NSW1 actual prices, the latest PreDispatch one-day forecast, and PD7Day for the rest of the horizon. Its default view is Monday-to-Sunday, with a range slider for the additional forecast days. Prices are capped to −100 through 200 AUD/MWh. Red chart backgrounds mark weekday 16:00–20:00 no-pumping periods, pale green shows the calendar-week baseline, stronger green shows the adjusted rolling plan, and blue marks operator-entered actual pumping.

Because the application has no live SCADA connection, operators can add each completed pump run using its start and end time. Water volume is calculated from the configured flow. An included actual run replaces the original baseline display for that local date, while the unchanged baseline is retained internally for reconciliation. The dashboard compares actual volume with that baseline plan. Any shortfall is added to the next rolling seven-day target and any surplus is subtracted. Forecast cost is not shown as a planning KPI.

The optimiser uses half-hour intervals, a default four-hour minimum continuous run, a configurable minimum interval between pump starts, a six-hour daily minimum, and at least two continuous hours OFF between pumping runs. Infeasible targets are rejected. All operating times use Australia/Sydney.

If automatic retrieval is unavailable, a saved AEMO PD7Day ZIP/CSV can still be uploaded as a legacy calendar-week fallback. Missing future prices are never filled.

Actual records are stored only in the active Streamlit session in this version. This is planning software and is not connected to PLC or SCADA control.

## Deploy on Streamlit Community Cloud

1. Create an empty **private GitHub** repository.
2. Commit and push the contents of this folder as the repository root.
3. In Streamlit Community Cloud, select **Create app** and choose that repository and branch.
4. Set the main file path to `burrier_llm_planner/app.py`.
5. In Advanced settings, select **Python 3.10**. Leave Secrets empty.
6. Deploy the app, then open its sharing settings and make the app **public**.
7. Test both Daily Planning and Weekly Planning in a private browser window.

The recommended layout puts the contents of this folder at the GitHub repository root. If you keep the outer `burrier_streamlit_public` folder inside a larger repository, use `burrier_streamlit_public/burrier_llm_planner/app.py` as the entrypoint. A duplicate requirements file is included beside `app.py` for that layout, but `packages.txt` and `.streamlit/config.toml` still work most reliably at the repository root.

Anyone with the public URL can access the app, so do not place confidential operating data in this test deployment. Public apps may also be discoverable by search engines.

## Run the bundle locally

With Python 3.10 and the dependencies installed:

```bash
python runapp.py
```

On Linux, `packages.txt` asks the host to install the open-source `coinor-cbc` solver. No Gurobi dependency or licence is used.
