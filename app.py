import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="EV charging simulator")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from numpy.random import default_rng

    from ev_charging_simulator.config import SimulationConfig
    from ev_charging_simulator.models.archetypes import (
        load_archetypes,
        with_population_shares,
    )
    from ev_charging_simulator.models.distributions import load_distributions
    from ev_charging_simulator.models.driver import sample_drivers
    from ev_charging_simulator.plotting import (
        archetype_demand_figure,
        driver_figure,
        driver_many_days_figure,
        population_figure,
    )
    from ev_charging_simulator.simulation import (
        ALL_DRIVERS,
        run_population,
        simulate_driver_days,
    )
    from ev_charging_simulator.validation import (
        compare,
        simulate_sessions,
        summary_table,
        validation_figure,
    )

    return (
        ALL_DRIVERS,
        SimulationConfig,
        archetype_demand_figure,
        compare,
        default_rng,
        driver_figure,
        driver_many_days_figure,
        load_archetypes,
        load_distributions,
        mo,
        pl,
        population_figure,
        run_population,
        sample_drivers,
        simulate_driver_days,
        simulate_sessions,
        summary_table,
        validation_figure,
        with_population_shares,
    )


@app.cell
def _(SimulationConfig, load_archetypes, load_distributions):
    # Inputs are read once; every widget change reuses them.
    defaults = SimulationConfig()
    archetypes = load_archetypes(defaults.archetypes_path)
    dists = load_distributions()
    archetype_names = [a.name for a in archetypes]
    return archetype_names, archetypes, defaults, dists


@app.cell
def _(mo):
    mo.md("""
    # EV charging simulator

    Simulates unmanaged home charging for Axle's six driver archetypes, one
    driver at a time or as a population. Session times come from the
    Intelligent Octopus CNZ report (Fig 4), target SoC from Fig 2, and energy
    use from the archetype spreadsheet. Population plots run noon to noon;
    single-driver plots run 06:00 to 06:00 so each day reads in order.
    Assumptions are listed in the last tab and in the README.
    """)
    return


@app.cell
def _(defaults, mo):
    # Shared settings: visible ones first, model parameters tucked away.
    day_type = mo.ui.dropdown(
        ["weekday", "weekend"], value=defaults.day_type, label="Day type"
    )
    seed = mo.ui.number(value=defaults.seed, start=0, stop=10**9, label="Seed")
    p_overnight = mo.ui.slider(
        0.55,
        1.0,
        step=0.01,
        value=defaults.p_overnight,
        label="Overnight plug-in probability p",
        show_value=True,
    )
    gamma_shape = mo.ui.slider(
        0.5,
        10.0,
        step=0.5,
        value=defaults.gamma_shape,
        label="Daily SoC use, gamma shape",
        show_value=True,
    )
    mileage_cv = mo.ui.slider(
        0.0,
        1.0,
        step=0.05,
        value=defaults.mileage_cv,
        label="Mileage spread (CV)",
        show_value=True,
    )
    gap_reserve = mo.ui.slider(
        0.05,
        0.5,
        step=0.05,
        value=defaults.gap_reserve_soc,
        label="Gap reserve SoC",
        show_value=True,
    )
    advanced = mo.accordion(
        {
            "Advanced model settings": mo.vstack(
                [
                    p_overnight,
                    gamma_shape,
                    mileage_cv,
                    gap_reserve,
                    mo.md(
                        """
                        - **p:** chance of an overnight session (Table 2: 0.74).
                          Same-day sessions follow so the mean is one a day.
                        - **Gamma shape:** lower means more variable daily
                          driving.
                        - **Mileage spread:** standard deviation / mean of
                          annual mileage within an archetype (lognormal).
                        - **Gap reserve:** drivers charge before their average
                          driving would take them below this SoC.
                        """
                    ).style({"font-size": "0.85em"}),
                ]
            )
        }
    )
    mo.vstack([mo.hstack([day_type, seed], justify="start"), advanced])
    return day_type, gamma_shape, gap_reserve, mileage_cv, p_overnight, seed


@app.cell
def _(
    SimulationConfig, day_type, gamma_shape, gap_reserve, mileage_cv, p_overnight, seed
):
    base_config = SimulationConfig(
        day_type=day_type.value,
        seed=int(seed.value),
        p_overnight=p_overnight.value,
        gamma_shape=gamma_shape.value,
        mileage_cv=mileage_cv.value,
        gap_reserve_soc=gap_reserve.value,
    )
    return (base_config,)


@app.cell
def _(archetype_names, mo):
    driver_archetype = mo.ui.dropdown(
        archetype_names, value=archetype_names[0], label="Archetype"
    )
    driver_index = mo.ui.number(value=0, start=0, stop=10_000, label="Driver number")
    plot_mode = mo.ui.radio(
        ["Few days", "Many days"], value="Few days", label="Plot", inline=True
    )
    few_days = mo.ui.slider(1, 10, value=5, label="Days", show_value=True)
    many_days = mo.ui.slider(
        50, 1000, step=50, value=200, label="Days", show_value=True
    )
    return driver_archetype, driver_index, few_days, many_days, plot_mode


@app.cell
def _(
    archetypes,
    base_config,
    default_rng,
    dists,
    driver_archetype,
    driver_figure,
    driver_index,
    driver_many_days_figure,
    few_days,
    many_days,
    mo,
    plot_mode,
    sample_drivers,
    simulate_driver_days,
):
    _archetype = next(a for a in archetypes if a.name == driver_archetype.value)
    # Each driver number gives a different person from the same archetype.
    _rng = default_rng(base_config.seed + int(driver_index.value))
    _driver = sample_drivers(_archetype, 1, base_config, dists, _rng)[0]
    _many = plot_mode.value == "Many days"
    _n_days = many_days if _many else few_days
    _days = simulate_driver_days(_driver, _n_days.value, base_config, dists)
    _figure = (driver_many_days_figure if _many else driver_figure)(
        _days, _archetype.name
    )
    _facts = mo.md(
        f"""
        **This driver:** {_driver.annual_miles:,.0f} miles/yr,
        target SoC {_driver.target_soc:.0%},
        mean daily SoC use {_driver.mean_daily_soc_drop:.1%},
        {_archetype.charger_kw:g} kW charger, {_archetype.battery_kwh:g} kWh battery.
        """
    )
    driver_tab = mo.vstack(
        [
            mo.hstack(
                [driver_archetype, driver_index, plot_mode, _n_days], justify="start"
            ),
            _facts,
            _figure,
        ]
    )
    return (driver_tab,)


@app.cell
def _(ALL_DRIVERS, archetype_names, archetypes, defaults, mo):
    # Population mix, defaulting to the spreadsheet. Lets a user model the
    # chargers in a particular area, whose mix may differ from the UK's.
    mix = mo.ui.dictionary(
        {
            a.name: mo.ui.number(
                value=round(100 * a.population_share, 1),
                start=0,
                stop=100,
                step=0.5,
                label=a.name,
            )
            for a in archetypes
        }
    )
    n_drivers = mo.ui.number(
        value=defaults.n_drivers, start=10, stop=50_000, step=10, label="Drivers (N)"
    )
    n_runs = mo.ui.number(value=100, start=1, stop=1_000, label="Runs (R)")
    group = mo.ui.dropdown(
        [ALL_DRIVERS, *archetype_names], value=ALL_DRIVERS, label="Show"
    )
    return group, mix, n_drivers, n_runs


@app.cell
def _(
    archetypes,
    base_config,
    dists,
    mix,
    mo,
    n_drivers,
    n_runs,
    run_population,
    with_population_shares,
):
    # Shares are relative weights, normalised to 100%; all zero falls back
    # to the spreadsheet mix.
    _total = sum(mix.value.values())
    if _total > 0:
        mixed_archetypes = with_population_shares(archetypes, mix.value)
        mix_message = (
            ""
            if abs(_total - 100) < 0.05
            else f"Shares add up to {_total:g}%, so they are scaled to 100%."
        )
    else:
        mixed_archetypes = archetypes
        mix_message = "All shares are zero, so the spreadsheet mix is used."

    # Reruns whenever a setting changes; 1,000 drivers x 100 runs takes ~5 s.
    config = base_config.model_copy(
        update={"n_drivers": int(n_drivers.value), "n_runs": int(n_runs.value)}
    )
    with mo.status.spinner(
        f"Simulating {config.n_runs} days for {config.n_drivers} drivers"
    ):
        result = run_population(config, mixed_archetypes, dists)
    return mix_message, result


@app.cell
def _(
    ALL_DRIVERS,
    archetype_demand_figure,
    group,
    mix,
    mix_message,
    mo,
    n_drivers,
    n_runs,
    pl,
    population_figure,
    result,
):
    _counts = dict(
        result.summary.group_by("group").agg(pl.col("n_drivers").first()).iter_rows()
    )
    _mix_panel = mo.accordion(
        {
            "Population mix (default: spreadsheet shares)": mo.vstack(
                [
                    mo.md(
                        "Share of drivers in each archetype, in %. Change it to "
                        "match the chargers in a particular area. Shares are "
                        "normalised to 100%."
                    ).style({"font-size": "0.85em"}),
                    mo.hstack(
                        [
                            mo.vstack(list(mix.elements.values())),
                            mo.ui.table(
                                [
                                    {"Archetype": name, "Drivers": _counts.get(name, 0)}
                                    for name in mix.elements
                                ],
                                selection=None,
                            ),
                        ],
                        justify="start",
                        gap=2,
                    ),
                    mo.md(mix_message),
                ]
            )
        }
    )
    _controls = mo.hstack([n_drivers, n_runs, group], justify="start")
    # An archetype set to 0% has no drivers to show; fall back to everyone.
    _group = group.value if _counts.get(group.value, 0) else ALL_DRIVERS
    _warning = (
        mo.callout(
            mo.md(f"No drivers in **{group.value}** with this mix; showing all."),
            kind="warn",
        )
        if _group != group.value
        else mo.md("")
    )
    _peak = (
        result.summary.filter(pl.col("group") == _group).sort("power_kw_per_ev").tail(1)
    )
    _shift = (
        result.summary.filter(pl.col("group") == _group)
        .sort("shiftable_kwh_per_ev")
        .tail(1)
    )
    _scale = _counts[_group]
    _per = f"for {_scale:,} EVs"
    _stats = mo.md(
        f"N drivers are split by the population mix above. "
        f"Peak unmanaged demand **{_scale * _peak['power_kw_per_ev'][0]:,.0f} kW "
        f"{_per}** at {_peak['clock'][0]} (runs 5th to 95th pct: "
        f"{_scale * _peak['power_kw_per_ev_run_p5'][0]:,.0f} to "
        f"{_scale * _peak['power_kw_per_ev_run_p95'][0]:,.0f}). "
        f"Most charging that could be deferred **"
        f"{_scale * _shift['shiftable_kwh_per_ev'][0]:,.0f} kWh {_per}** "
        f"at {_shift['clock'][0]}."
    )
    _flex_note = mo.md(
        "**Flexibility.** The dashed demand line charges the same sessions as "
        "late as possible: each car starts at the last moment that still "
        "delivers its energy by plug-out. The bottom panel is the energy the "
        "unmanaged fleet has charged by each time that could have waited, "
        "i.e. the demand that can be shifted later."
    ).style({"font-size": "0.85em"})
    population_tab = mo.vstack(
        [
            _controls,
            _mix_panel,
            _warning,
            _stats,
            mo.ui.tabs(
                {
                    "Overview": mo.vstack(
                        [
                            _flex_note,
                            population_figure(
                                result,
                                _group,
                            ),
                        ]
                    ),
                    "By archetype": mo.vstack(
                        [
                            archetype_demand_figure(result),
                            mo.ui.table(
                                result.summary.filter(pl.col("group") == _group),
                                page_size=10,
                            ),
                        ]
                    ),
                }
            ),
        ]
    )
    return (population_tab,)


@app.cell
def _(archetype_names, mo):
    validation_archetype = mo.ui.dropdown(
        archetype_names, value="Intelligent Octopus average", label="Archetype"
    )
    return (validation_archetype,)


@app.cell
def _(
    archetypes,
    base_config,
    compare,
    dists,
    mo,
    simulate_sessions,
    summary_table,
    validation_archetype,
    validation_figure,
):
    _archetype = next(a for a in archetypes if a.name == validation_archetype.value)
    _comparison = compare(simulate_sessions(_archetype, base_config, dists))
    validation_tab = mo.vstack(
        [
            mo.md(
                "Report figures held out from fitting. The report covers "
                "Intelligent Octopus customers, so that archetype is the fair "
                "comparison; Fig 6 is the 'next day is a weekday' series."
            ),
            validation_archetype,
            mo.ui.table(summary_table(_comparison)),
            validation_figure(_comparison, _archetype.name),
        ]
    )
    return (validation_tab,)


@app.cell
def _(mo):
    assumptions_tab = mo.md(
        """
        - **Sessions per day:** overnight plug-in with probability p (default
          0.74, from Table 2); given that, an extra same-day session with
          probability q = 1/p - 1, so the mean is one session per day.
          Infrequent charging plugs in with probability 0.2 and never twice.
        - **Maximum nights between charges:** a driver goes at most K nights between charges, where K
          is how many average days fit between their target SoC and the
          reserve (default 20%; 5% for Infrequent charging). Each driver's
          overnight probability is raised to p / (1 - (1 - p)^K) so sessions
          match gaps; a driver with K = 1 plugs in every night, once.
        - **Session times:** Fig 4, split by whether the session spans 03:00.
          Scheduled charging shifts the overnight times +4 h in and +2 h out
          (read as late plug-in, late plug-out), drawn once per driver.
        - **Always plugged-in:** connected all day at target SoC with no demand;
          the spreadsheet row can't be both always connected and driving.
        - **Plug-in SoC:** days since the last overnight charge are
          min(geometric(p), K); daily SoC use is gamma with the spreadsheet
          mean; floor at 5%.
        - **Drivers differ:** annual mileage is lognormal with mean = spreadsheet
          and CV 0.4 (the same relative spread for every archetype); target SoC
          drawn from Fig 2.
        - **Population mix:** N drivers are split by the spreadsheet's
          population shares by default (exact counts, largest-remainder
          rounding); the Population tab lets you change the mix.
        - **Charging:** charger power to 80% SoC, then a linear taper to 2 kW at
          100%; 90% efficiency.
        - **Days are independent**, so the window is circular: a session running
          past the window start wraps to its left edge. Population plots run
          noon to noon (overnight block whole); single-driver plots run 06:00
          to 06:00 (daytime sessions before the overnight one).
        """
    )
    return (assumptions_tab,)


@app.cell
def _(assumptions_tab, driver_tab, mo, population_tab, validation_tab):
    mo.ui.tabs(
        {
            "Population": population_tab,
            "Individual driver": driver_tab,
            "Validation": validation_tab,
            "Assumptions": assumptions_tab,
        }
    )
    return


if __name__ == "__main__":
    app.run()
