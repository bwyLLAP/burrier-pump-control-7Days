# Paper Method Digest

**Source:** `C:\Users\BWY\OneDrive - University of Wollongong\Projects\2026 TIA special issue\Digest-SI-TPSO.docx`

**Title:** Verified LLM-Assisted PI-MPC for Operator-Centred Pumping-Station Control

**Extracted:** 2026-07-28

## Framework

The paper proposes a verified LLM-assisted physics-informed MPC framework for water-pumping stations. Natural-language requests are translated into bounded supervisory intent. A deterministic verifier rejects invalid, conflicting, or unauthorised requests before optimisation. The validated strategy contains only operating mode `m_d`, target reservoir fraction `alpha_d`, and adjustment period `H_d`.

The LLM is restricted from modifying physical limits, protection settings, or pump commands. The verified strategy is converted into a daily pumping target that combines forecast water demand with a gradual movement from current storage toward `alpha_d` times reservoir capacity over `H_d` days.

The lower PI-MPC uses reservoir mass balance, calibrated pump/VFD power, forecast electricity prices, fixed reservoir-security limits, operating ranges, minimum run time, and switching constraints to produce a feasible short-term schedule.

## Seasonal arbitrage result

The 2023 annual replay used a target reservoir fraction between 0.90 and 1.00. When the sustained price index increased, the target was reduced toward 0.90 so stored water could substitute for expensive pumping. When the price index declined, the target moved back toward 1.00 to replenish storage. The transition period produced a smoother actual reservoir response than the strategic target.

The replay kept the reservoir above the 0.90 hard minimum. It reported 15,061.18 ML pumped for AUD 316,878.02, or 21.04 AUD/ML, compared with a historical statistical unit cost of 37.80 AUD/ML. These figures are paper evaluation results, not a guarantee for a future 24-hour plan.

## Application interpretation

For this local application, the automatic price-index path remains the default. The Agent can explain the strategy and run bounded local simulations. Feasibility and safety statements for the current day come from the current CBC solve and independent validator, not from the annual paper result or from language-model judgement.
