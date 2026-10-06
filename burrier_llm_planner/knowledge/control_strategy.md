# Burrier Control Strategy

## Authority and purpose

This document is the authoritative explanation layer for the Burrier planning assistant. The deterministic optimiser and validator remain the authority for feasibility and safety. The Agent may explain and simulate, but it has no PLC or SCADA connection.

## Supervisory parameters

### m_d: operating mode

`m_d` selects the supervisory operating mode for planning day `d`. `arbitrage` is the normal price-aware mode. `emergency` prioritises storage recovery and requires a target reservoir fraction of at least 0.95. The mode does not directly switch the pump.

### alpha_d: target reservoir fraction

`alpha_d` is the desired reservoir fraction used by the daily target generator. Increasing `alpha_d` asks the planner to move storage upward and normally increases required pumping. Decreasing it can reduce near-term pumping, but it can never be set below the immutable minimum reservoir fraction of 0.90.

### H_d: transition days

`H_d` is the number of days over which the reservoir is moved toward `alpha_d`. A smaller value produces a larger daily storage adjustment and can increase near-term pumping and cost. A larger value spreads the adjustment over more days and reduces the response to short-term conditions. The admissible range is 1 to 365 days.

## Daily target interface

The daily pumping target replaces forecast demand and adds a bounded storage-shift term:

`daily target = daily demand + (alpha_d * reservoir capacity - current storage) / H_d`

The target generator clips the requested volume to physical daily pumping capability. The 24-hour PI-MPC then chooses 48 half-hour pump/VFD intervals that meet the applied target within tolerance.

## Seasonal price-index strategy

The annual price index provides the default daily supervisory strategy. During sustained lower-price seasons, the strategy can raise `alpha_d` and replenish storage. During sustained higher-price seasons, it can reduce the target toward the hard minimum so stored water substitutes for expensive pumping. `H_d` prevents excessive reactions to short-term price changes by distributing storage adjustment over time.

The daily PI-MPC is distinct from the annual strategy. The annual strategy supplies the target and transition. The daily optimiser uses the next 24 hours of AEMO prices to select economical pump intervals while satisfying the verified target and all hard constraints.

## Immutable hard constraints

The Agent cannot change:

- reservoir capacity or the 0.90 minimum reservoir fraction;
- reservoir mass balance;
- pump and VFD availability;
- calibrated flow and power operating points;
- minimum pump run time;
- maximum daily starts;
- target tolerance rules;
- the 24-hour, 48-interval planning horizon;
- independent feasibility and result validation.

Requests to change these values must be refused. A parameter suggestion is not safe merely because it is within its allowed numerical range. It is safe for the current planning data only after CBC PI-MPC returns a feasible result and the independent validator passes.

## Agent simulation boundary

The Agent may translate customer goals into `m_d`, `alpha_d`, and `H_d`, run local scenario simulations, compare them with the automatic baseline, and explain the results. Simulations are labelled `Agent simulation` and never overwrite the automatic baseline. Export is a user action. No function in the Agent service sends a live equipment command.
