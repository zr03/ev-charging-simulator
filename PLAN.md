# PLAN: EV driver behaviour simulator

Design spec for the Axle Energy take-home. It records what we agreed, the defaults, the assumptions, and the build order. Update it when a decision changes.

## 1. Goal

An agent-based simulator of EV driver behaviour that returns, per driver:
- when the car is plugged in;
- the battery state of charge (SoC) at plug-in (and the SoC profile while plugged in).

Run over a population of drivers built from the six archetypes, it should reproduce population-level observations from the CNZ / Intelligent Octopus report and show variation across drivers and days. The end use is estimating **flexibility**: energy that can be shifted while cars are connected.

## 2. Sources

| Source | Used for |
|---|---|
| `data/inputs/Axle Take Home Test_ Archetypes.xlsx` | Archetype parameters and population shares |
| CNZ report Fig 4 (weekday + weekend plug-in x plug-out heatmaps) | Session times (**input**) |
| CNZ report Fig 2 (target SoC x deadline heatmap) | Target SoC, deadline axis collapsed (**input**) |
| CNZ report Table 2 | Calibrating `p` (26.3% of gaps < 12 h gives p ~ 0.74) |
| CNZ report Figs 5, 6, 7, 9, 10 | **Validation only**, never fitted |
| DfT EV driver survey 2022 | Sense check: daytime charging is mostly ad hoc |

Digitised figure data lives as CSVs with the script that produced them (section 9), never as hand-typed constants.

## 3. Archetypes

One `Archetype` Pydantic model per spreadsheet row (frozen, validated). The spreadsheet's `Average SoC drop/day` column is the mean daily drop: miles/yr / 365 / efficiency / battery.

| # | Archetype | Share | Special handling |
|---|---|---|---|
| 1 | Average (UK) | 40% | none |
| 2 | Intelligent Octopus average | 30% | none |
| 3 | Infrequent charging | 10% | p = 0.2, q = 0 |
| 4 | Infrequent driving | 10% | none |
| 5 | Scheduled charging | 9% | overnight times shifted and drawn **once per driver** (section 5.3) |
| 6 | Always plugged-in | 1% | connected 24 h, SoC held at target, **zero energy demand** (kind flag) |

All behaviour lives in one `Driver` class. Archetypes 5 and 6 differ only by an `ArchetypeKind` flag read from the archetype name, so there are no subclasses. Simulation is vectorised over a `Population` (driver parameters stacked into arrays), and a single driver is a population of one, so there is one code path.

## 4. Driver-level attributes (drawn once, when the driver is created)

- **Annual mileage:** lognormal with **mean** = spreadsheet miles/yr and CV = 0.4 (configurable). This sets the driver's mean daily SoC drop. The mean, not the median, matches the spreadsheet, so annual energy is preserved.
- **Target SoC:** sampled from the Fig 2 marginal over target SoC.
- **Archetype 5 only:** the overnight plug-in/plug-out time, from the shifted overnight distribution.
- Battery size, efficiency and charger kW come from the archetype (not varied).

## 5. Day-level simulation (one simulated day per call, independent across days)

### 5.1 Number of sessions: symmetric plug-in model

Per driver-day: plug in overnight with probability `p`. If plugged in, add a same-day session with probability `q = 1/p - 1`.

| Sessions | Probability | At p = 0.74 |
|---|---|---|
| 0 | 1 - p | 26% |
| 1 | 2p - 1 | 48% |
| 2 | 1 - p | 26% |

The mean is exactly 1 per day and the median is 1 for p > 0.5. Default `p = 0.74`, configurable. For a target frequency `f` < 1 (archetype 3): `p = f`, `q = 0`. Archetype 6: `p = 1`, `q = 0`.

**Per-driver adjustment (maximum nights between charges, see 5.4).** The `p` above is the base rate. Each driver's actual overnight probability is `p_i = p / (1 - (1 - p)^K)`, where `K` is their longest possible gap, and `q_i = 1/p_i - 1` (still 0 for archetype 3). Drivers with K = 1 get `p_i = 1`, `q_i = 0`: a heavy driver plugs in every night and only once. Low-mileage drivers have large K, so `p_i` is about `p`.

### 5.2 Session times from Fig 4

Split each Fig 4 heatmap by whether the session **spans 03:00**:
- **Overnight part:** sessions that span 03:00 (main 17:00 to 07:00 cluster).
- **Same-day part:** all other cells (the diagonal, including short evening sessions).

Each part is renormalised and used for **shape only**. How often each kind happens is set by `p` and `q`. Use the weekday or weekend heatmap according to the day flag. Sample an hourly cell, then add uniform jitter within the hour.

A same-day session is redrawn if it overlaps the driver's overnight session in the window.

### 5.3 Scheduled charging (archetype 5)

Shift the overnight distribution so its peak moves from about 18:00/07:00 to the spreadsheet's 22:00/09:00 (plug-in +4 h, plug-out +2 h). Each driver draws their times **once**. Each day adds small noise (default +/- 15 min, configurable, 0 allowed).

### 5.4 SoC drop and plug-in SoC

- **Longest gap K (per driver):** `K = max(1, floor((target - reserve) / mean daily drop))`, with reserve 20% (`gap_reserve_soc`). Archetype 3 uses the 5% floor as its reserve, since those drivers deliberately run low. ASSUMPTION: drivers don't go more nights without charging than their average driving allows before SoC would fall below the reserve.
- **Gap since the last overnight session:** `k = min(Geometric(p), K)`. Its mean is `(1 - (1 - p)^K) / p` (the geometric's tail beyond K is folded onto K), so `p_i` in 5.1 is 1 over this mean, which keeps sessions consistent with gaps and energy balanced.
- **Drops** use a gamma with shape `alpha = 2` and mean = the driver's daily drop:
  - earlier days: `Gamma((k - 1) * alpha, theta)` (zero when k = 1);
  - today: `Gamma(alpha, theta)`.
- **Single-session day:** plug-in SoC = target - (earlier + today's drop).
- **Two-session day:** see 5.5.
- **Floor:** clip plug-in SoC at 5%. With the limit on nights between charges this binds rarely: about 7% of Intelligent Octopus sessions (heavy drivers on heavy days, read as public or en-route charging) and 13% for archetype 3.
- **Starting point:** the last overnight session ended at target. At 7 kW over 10 to 15 h this is almost always true.

### 5.5 Plot window and session order (noon to noon)

The plotted day is the window **12:00 to 12:00**, the time when the fewest cars are plugged in. Because days are i.i.d., the window is treated as **circular**: a session that runs past noon wraps to the start of the window, which is what a steady-state day looks like.

Session order is worked out in a day anchored at 03:00, when every overnight session is connected:
- The overnight session stands in for last night's too (days are i.i.d.). A same-day session must start after its plug-out and end before tonight's plug-in; it is redrawn until it fits (dropped after 50 tries, which is rare).
- Today's drop is split by time away from the charger before and after the same-day session (proxy for driving). The same-day session takes the earlier days' drop plus the "before" share; the overnight session takes the "after" share.
- Same-day sessions are then placed by clock time, so a morning one appears as the next morning in the window.

### 5.6 Charging (unmanaged)

Charging starts at plug-in and runs continuously until **target SoC** or plug-out, whichever comes first.
- 7 kW constant up to 80% SoC, then a linear taper to about 2 kW at 100%.
- Charging efficiency 90%: grid kWh = battery kWh / 0.9.
- Target SoC clips the profile, since it represents what the driver actually wants.

Integrate in minutes. Report half-hourly.

## 6. Population runs and outputs

- **Fixed population:** build N drivers (default 1,000) by archetype share, with a seed.
- **Runs:** simulate the chosen day type (weekday or weekend) R times (default 200). Run seeds come from one master seed (`numpy.random.SeedSequence.spawn`), so runs differ but the whole set is reproducible.

**Per driver:** plugged-in intervals, SoC at plug-in and plug-out, and the SoC profile (the brief's first sketch).

**Population, per half-hour, over 12:00 to 12:00:**
- % of drivers plugged in;
- SoC of **plugged-in cars only**: mean and percentile band **across drivers** (the brief's second sketch);
- **across-run** band (e.g. 5th to 95th percentile over runs) for % plugged in and flexibility, i.e. lower and upper bounds for this population;
- **flexibility outputs:** kW connected, kWh still needed to reach target, and hours left until plug-out.

When showing percentiles, blank out half-hours with too few plugged-in cars, or show how many cars each point is based on.

## 7. Defaults (all configurable in `config.py`)

| Parameter | Default |
|---|---|
| p (overnight plug-in probability) | 0.74 |
| q | 1/p - 1 |
| Gap reserve SoC (sets K) | 20% (archetype 3: the 5% floor) |
| Gamma shape alpha | 2 |
| Mileage CV | 0.4 |
| SoC floor | 5% |
| Charger power | 7 kW |
| Taper | linear from 80% to about 2 kW at 100% |
| Charging efficiency | 90% |
| Scheduled-charging daily noise | +/- 15 min |
| Drivers N | 1,000 |
| Runs R | 200 |
| Time step | 30 min output; charging is closed form, so state is exact at bin midpoints and power is the exact bin average |

## 8. Assumptions to state in the README

1. Days are independent. The only cross-day link is the capped geometric gap used for energy.
2. Spreadsheet values are treated as means. Variation is added around them.
3. Fig 4 is used for the shape of session times only. Its hourly resolution, the exclusion of sessions under 10 minutes, and the lack of sessions over 24 h all carry over.
4. Same-day sessions are occasional within a driver (no habitual twice-daily drivers).
5. Time away from the charger is a proxy for driving when splitting a day's drop.
6. **Scheduled charging** is read literally as a late plug-in, late plug-out driver connected from about 22:00 to 09:00. If it actually represents a charge timer on a car plugged in earlier, connected time and flexibility are understated.
7. **Always plugged-in:** the spreadsheet row is internally inconsistent (connected 24 h yet 9,435 mi/yr). We keep the plug-in pattern, which defines the archetype, and set energy demand to zero. Impact is small at 1% of the population.
8. Intelligent Octopus target SoC preferences (Fig 2) are applied to all drivers as their charge limit.
9. Bump charging is not modelled. It is specific to Intelligent Octopus.
10. Charging is unmanaged throughout: this is the baseline from which flexibility is measured.
11. The archetype 2 spreadsheet values (0.28 requirement, 2.5 h) differ slightly from its own arithmetic (0.30, 3.1 h). We use the computed daily drop column.
12. Across-run bounds cover day-to-day randomness only, not uncertainty in parameters or digitised figures.

## 9. Repo layout

```
src/ev_charging_simulator/
  __init__.py
  config.py            # Pydantic SimulationConfig with the defaults above
  models/
    archetypes.py      # Archetype model + xlsx loader
    driver.py          # Driver, Population, build_population, simulate_day
    distributions.py   # Fig 4 / Fig 2 samplers, gamma, geometric, mileage
    charging.py        # charging curve and SoC integration
  simulation.py        # population building, runs, half-hourly aggregation
  plotting.py          # shared by notebook and scripts
  validation.py        # comparison with held-out Figs 5 to 9
  utils/               # time helpers, seeding
  data/                # digitised CSVs, shipped with the package
scripts/
  digitise_figures.py  # renders report pages and extracts the Fig 2 / Fig 4 grids
  run_population.py    # CLI: writes plots to figures/ and tables to data/outputs/
  digitise_validation.py  # reads Figs 5 to 9 bar heights from the PDF vectors
  validate.py          # CLI: simulated vs held-out report figures
tests/
app.py                 # marimo notebook, served locally with `marimo run`
```

The digitised CSVs sit inside the package so the notebook, scripts and tests can load them via `importlib.resources` from any working directory. Raw inputs stay in `data/inputs/`.

## 10. Build order

1. **Repo setup:** add a `[build-system]` to `pyproject.toml` so `uv sync` installs the package. Add dependencies: `pydantic` and an Excel reader (`fastexcel` for `polars.read_excel`); dev: `pytest`, plus `pymupdf` for digitising.
2. **Digitise** Fig 4 (both panels) and Fig 2 into CSVs. Plot them back next to the originals as a visual check.
3. **Models:** `config.py`, `archetypes.py` with the xlsx loader, and validation tests.
4. **Distributions:** samplers, with tests (plug-in model mean = 1; gamma mean = daily drop).
5. **Charging** curve and SoC integration, with tests (energy balance).
6. **simulate_day()** over a `Population` (archetype kinds as flags, no subclasses).
7. **Population and aggregation.** Annual energy per archetype must match the spreadsheet (test).
8. **Plots and CLI script.**
9. **Validation:** compare simulated Figs 5, 6, 7, 9 and 10 with the report and record the differences.
10. **marimo app,** run locally with `uv run marimo run app.py` (no WASM export).
11. **README:** how to run, the assumptions (section 8), what was prioritised.

## 11. Risks and checks

- **Performance:** vectorise across drivers with numpy, with no per-session Python objects in the hot loop.
