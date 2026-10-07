# ev-charging-simulator

Simulates unmanaged home EV charging for Axle's six driver archetypes, for one
driver or a whole population, and reports what flexibility the plugged-in fleet
offers. Behaviour is drawn from the Intelligent Octopus CNZ report (May 2022)
and the archetype spreadsheet.

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync                                   # install (also refreshes uv.lock)
uv run marimo run app.py                  # interactive dashboard in the browser
uv run python scripts/run_population.py   # HTML figures -> figures/, tables -> data/outputs/
uv run python scripts/validate.py         # compare with held-out report figures
uv run pytest                             # tests
```

`run_population.py` takes `--n-drivers`, `--n-runs`, `--seed`, `--day-type`
and `--p-overnight`. The default (1,000 drivers, 200 runs) takes about 10 s.

## What it does

**One driver.** Each driver gets an archetype plus their own annual mileage
and target SoC. A simulated day decides how many times they plug in, when, at
what SoC, and charges them at their charger's rate up to their target.

**A population.** N drivers are drawn once (split by archetype share, the
spreadsheet's by default; the app's "Population mix" panel changes it, e.g. to
match the chargers in one area) and the
same population is simulated for R independent days. Outputs, per half hour
over a noon-to-noon window:

- share of drivers plugged in;
- combined SoC of plugged-in cars (kWh stored over kWh of battery capacity);
  per-driver SoC mean and percentiles are in the summary table;
- charging demand, unmanaged and with the same sessions charged as late as
  possible (each car starts at the last moment that still delivers its
  energy by plug-out);
- charging that could be deferred to a later time: what the unmanaged fleet
  has charged by each time that could have waited. kW and kWh are
  totals for the population, since grid flexibility is managed in aggregate;
- in the summary table: charger kW connected, kWh still needed to reach
  target, hours until plug-out, and spare hours (time until plug-out minus
  time needed to reach target);
- every metric also has a 5th to 95th percentile band across runs, i.e. the
  day-to-day range a planner should expect for this fleet.

## Model in brief

Full spec and reasoning: [PLAN.md](PLAN.md).

| Piece | Source | How |
|---|---|---|
| Sessions per day | Table 2 | Overnight session with probability p = 0.74. Given that, an extra same-day session with probability q = 1/p - 1, so the mean is one session a day. Infrequent charging: p = 0.2, never twice. Each driver's p is then adjusted for their maximum nights between charges (next row). |
| Maximum nights between charges | Assumed | A driver goes at most K nights between charges, K = how many average days fit between their target SoC and a 20% reserve (5% for Infrequent charging). The gap is min(geometric(p), K), and the driver's own p is set so sessions still match gaps: p_i = p / (1 - (1-p)^K). A heavy driver with K = 1 plugs in every night, once. |
| Session times | Fig 4 (digitised) | Heatmap cells split by whether the session spans 03:00: overnight vs same-day shapes. |
| Target SoC | Fig 2 (digitised) | Drawn once per driver from the target-SoC marginal. |
| Energy use | Spreadsheet | Days since the last overnight charge = capped geometric (above). Daily SoC use ~ gamma (shape 2) with the driver's mean. Mileage lognormal (mean = sheet, CV 0.4). |
| Charging | Assumed | Charger power to 80% SoC, linear taper to 2 kW at 100%, 90% efficiency. Closed-form SoC, so no time-stepping. |

All defaults live in `src/ev_charging_simulator/config.py`. Each assumption
below has an ID, and the code is tagged `ASSUMPTION A<n>` where it is applied
(`grep -rn "ASSUMPTION A" src`).

## Key assumptions

| ID | Assumption | Where in code |
|---|---|---|
| A1 | Days are independent, so the 24 h window is circular: a session running past the window start wraps to the left edge. Population plots run noon to noon (overnight block whole); single-driver plots run 06:00 to 06:00 (daytime sessions shown before the overnight one, matching how SoC is computed). | `models/driver.py` (DayResult), `simulation.py` |
| A2 | Spreadsheet values are archetype means. Each driver's annual mileage is lognormal with mean = spreadsheet and CV 0.4, the same relative spread for every archetype. | `config.mileage_cv`, `models/driver.py` (sample_drivers) |
| A3 | Daily SoC use is gamma (shape 2) with mean = miles ÷ efficiency ÷ battery. We compute it rather than use the sheet's requirement column, which differs slightly for Intelligent Octopus average (0.28 vs 0.30). Gamma because daily mileage is right-skewed: most days are short and a few are long. | `config.gamma_shape`, `models/archetypes.py` |
| A4 | Sessions per day: overnight with probability p = 0.74 (Table 2), then a same-day session with q = 1/p - 1, so one a day on average. Drawn independently each day (no habitual twice-daily drivers). Infrequent charging: p = sheet frequency, never twice. | `config.p_overnight`, `models/driver.py` (plug_in_model) |
| A5 | Maximum nights between charges: drivers don't go more nights without charging than their average driving allows before SoC would drop below 20%. Infrequent chargers run low by design, so their reserve is the 5% floor. Each driver's p is raised to keep sessions consistent with gaps. | `config.gap_reserve_soc`, `models/driver.py` (plug_in_model) |
| A6 | The last overnight session ended at target SoC. Plug-in SoC is floored at 5%. | `config.soc_floor`, `models/driver.py` (simulate_day) |
| A7 | On a two-session day, the day's SoC use is split by time away from the charger before and after the same-day session. | `models/driver.py` (simulate_day) |
| A8 | Fig 4 gives the shape of session times only, split by whether a session spans 03:00. Its hourly resolution, the exclusion of sessions under 10 minutes and the lack of sessions over 24 h carry over. | `models/distributions.py` |
| A9 | **Scheduled charging** is a late plug-in, late plug-out driver (Fig 4 overnight times shifted +4 h in, +2 h out to about 22:00 to 09:00), with times fixed per driver plus ±15 min daily noise. If it really means a timer on a car plugged in earlier, its flexibility is understated. | `models/driver.py` (sample_drivers), `config.scheduled_noise_minutes` |
| A10 | **Always plugged-in** is connected all day at target SoC with zero demand. The spreadsheet row (connected 24 h yet 9,435 mi/yr) can't be met in full; the plug-in pattern defines the archetype. It is 1% of the fleet. | `models/driver.py`, `models/archetypes.py` |
| A11 | Fig 2's Intelligent Octopus charge limits (target SoC) apply to every driver. | `models/driver.py` (sample_drivers) |
| A12 | Charging: charger power to 80% SoC, linear taper to 2 kW at 100%, 90% efficiency. | `config.ChargingConfig`, `models/charging.py` |
| A13 | Charging is unmanaged. Smart and bump charging are not modelled: this is the baseline flexibility is measured against. | `models/charging.py` |
| A14 | Across-run bands cover day-to-day randomness only, not uncertainty in the parameters or the digitised figures. | `simulation.py` (run_population) |

## Validation

The report's other figures were not used for fitting, so they are independent
checks. They describe Intelligent Octopus customers, so the comparison uses
that archetype. Bars were read exactly from the PDF's vector graphics
(`scripts/digitise_validation.py`).

| Figure | Histogram overlap (1 = identical) | Before limiting nights between charges |
|---|---|---|
| Fig 8, share of overnight time spent charging | 0.95 | 0.93 |
| Fig 9, overnight top-up | 0.85 | 0.82 |
| Fig 6, plug duration | 0.78 | 0.87 |
| Fig 7, plug-in SoC | 0.75 | 0.71 |

Fig 5 (median duration by plug-in time) matches in shape (mean absolute
error 1.7 h). Fig 6 got worse because heavy drivers now plug in every night
but rarely twice, so there are fewer short daytime sessions.

**Known limitations.**

- About 7% of Intelligent Octopus sessions still start at the 5% floor
  (9% without the cap). These are heavy drivers on heavy days, which the
  gamma's tail allows; in reality they would top up at a public charger.
  The floor trims delivered energy: -6% for Intelligent Octopus average and
  -3% for Infrequent charging, against -12% and -27% without the cap. The
  other archetypes are within 2%. A test checks every archetype.
- Simulated plug-in SoC still runs high in the body (median 63% against the
  report's 52%).
- Infrequent charging plugs in 0.27 times a day rather than the sheet's 0.2.
  That is what the cap needs to deliver the sheet's energy without running
  flat.

## What was prioritised

- A behaviour model that is calibrated to published data and checked against
  figures it never saw, over adding more features.
- Population outputs that matter for flexibility (demand charged as late as
  possible, shiftable energy, connected kW, energy needed, time left) with
  both driver spread and day-to-day bounds.
- Clear, testable code: config in Pydantic, numpy in the hot path, tests per
  module, and every assumption visible where it is used.

## Layout

```
app.py                       marimo dashboard
src/ev_charging_simulator/
  config.py                  all settings and defaults
  models/                    archetypes, distributions, charging, driver/population
  simulation.py              profiles, aggregation, population runs
  plotting.py                plotly figures
  validation.py              comparison with held-out report figures
  data/                      digitised Fig 2, Fig 4 and validation CSVs
scripts/                     digitising, population run, validation
tests/
```
