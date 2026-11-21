# PESCO | Burrier Pump Control

PESCO stands for Pumping Energy Savings through Control Optimisation. The Streamlit dashboard opens directly in the Weekly Planning workspace. The three partner logos are retained.

For the SHW trial, see [PESCO Operator and Technical Guide](PESCO_Operator_and_Technical_Guide.docx) and [SHW Trial Checklist](SHW_TRIAL_CHECKLIST.md).

Weekly Planning identifies the Monday-to-Sunday week containing the selected date. Price retrieval is operator-triggered. KWatch mode combines completed KWatch actual prices, the available KWatch short-term forecast and the AEMO seven-day outlook. The short-term forecast is required and is not silently replaced by seven-day data. The separate AEMO route and saved snapshot upload remain available as fallbacks. The KWatch key remains in the active Streamlit session and is not written to the repository.

The first optimisation creates and freezes the **Original weekly plan** for the complete calendar week. A separate **Current rolling plan** begins at the current executable half-hour. It keeps the remaining current-week plan, then assigns the weekly target minus that remaining volume to the rolling-horizon days after Sunday. This is one rolling target, not a second full weekly target. The chart defaults to Monday-to-Sunday and its range slider exposes the additional forecast days.

Because the application has no live SCADA connection, the operator records completed pumping as an **Override**. The entered start and end times are converted to volume from the configured flow, and the selected date replaces that day's Original weekly plan. The operator may mark a date as No pumping, remove one override with its x button, or restore all overrides. Overrides are held only in the active Streamlit session.

After an override, **Days available for catch-up** controls which future dates may change. Checkbox changes are draft selections and do not change the displayed plan until **Reoptimise selected dates** is pressed. The resulting change report compares the updated schedule with the plan that was visible immediately before the update. A shortfall adds future pumping and excess completed pumping removes future pumping where the selected dates and constraints allow it.

The optimiser uses half-hour intervals. Pump settings include flow, pumping and standby power, minimum continuous run time and minimum continuous stop time. **Balanced** mode applies a minimum daily run; **Flexible** mode removes that daily minimum. Weekday 16:00–20:00 intervals are blocked for future pumping. Prices are capped to −100 through 200 AUD/MWh. All operating times use Australia/Sydney.

Daily window comparison shows the app schedule alongside fixed pumping from 07:00 to 15:30 Sydney time, delivering 32.13 ML per day at the default flow. Week-to-date savings use actual electricity prices for completed half-hours in the selected week. The schedule CSV lists completed overrides and future planned runs in time order.

This is planning software. It does not connect to or command SCADA or the PLC. The operator must check every schedule against current plant conditions, reservoir requirements and site procedures.

## Deploy on Streamlit Community Cloud

1. Create an empty **private GitHub** repository.
2. Commit and push the contents of this folder as the repository root.
3. In Streamlit Community Cloud, select **Create app** and choose that repository and branch.
4. Set the main file path to `burrier_llm_planner/app.py`.
5. In Advanced settings, select **Python 3.10**. Do not commit the KWatch API key; the operator enters it in the password field for the active Streamlit session.
6. Deploy the app, then open its sharing settings and make the app **public**.
7. Test Weekly Planning in a private browser window.

Upload the contents of this folder to the GitHub repository root. The included dependency and Streamlit configuration files are intended for that layout.

Anyone with the public URL can access the app, so do not place confidential operating data in this test deployment. Public apps may also be discoverable by search engines.

On Linux, `packages.txt` asks the host to install the open-source `coinor-cbc` solver. No Gurobi dependency or licence is used.
