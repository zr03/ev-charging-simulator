import warnings

import numpy as np
import polars as pl

from ev_charging_simulator.config import SimulationConfig
from ev_charging_simulator.models.charging import hours_to_reach
from ev_charging_simulator.models.driver import Population, simulate_day
from ev_charging_simulator.simulation import (
    ALL_DRIVERS,
    TimeGrid,
    build_profiles,
    nan_percentiles,
    run_population,
)


def test_profile_energy_matches_sessions(population: Population, config, dists) -> None:
    """Energy summed over the circular window equals energy charged in sessions."""
    day = simulate_day(population, "weekday", np.random.default_rng(5), config, dists)
    grid = TimeGrid.from_config(config, step_minutes=15)
    profiles = build_profiles(day, population, grid, config)
    profile_kwh = profiles.power_kw.sum(axis=1) * grid.step_hours
    gained = np.nan_to_num(
        day.overnight_soc_out - day.overnight_soc_in
    ) + np.nan_to_num(day.same_day_soc_out - day.same_day_soc_in)
    session_kwh = gained * population.battery_kwh / config.charging.efficiency
    assert np.allclose(profile_kwh, session_kwh, atol=1e-9)


def test_power_never_exceeds_charger(population: Population, config, dists) -> None:
    day = simulate_day(population, "weekday", np.random.default_rng(5), config, dists)
    profiles = build_profiles(day, population, TimeGrid.from_config(config), config)
    assert np.all(profiles.power_kw <= population.charger_kw[:, None] + 1e-9)
    assert np.all(profiles.power_kw >= -1e-12)


def test_energy_envelope(population: Population, config, dists) -> None:
    """Charging as late as possible delivers the same energy, just later."""
    day = simulate_day(population, "weekday", np.random.default_rng(5), config, dists)
    profiles = build_profiles(day, population, TimeGrid.from_config(config), config)
    asap = np.nan_to_num(profiles.energy_left_asap_kwh)
    alap = np.nan_to_num(profiles.energy_left_alap_kwh)
    # Same energy per driver, just later.
    assert np.allclose(
        profiles.power_alap_kw.sum(axis=1), profiles.power_kw.sum(axis=1), atol=1e-9
    )
    assert np.all(profiles.power_alap_kw <= population.charger_kw[:, None] + 1e-9)
    # Late charging never has less left to do than early charging.
    assert np.all(alap >= asap - 1e-9)
    # Before any charging, the late profile still owes the whole session.
    gained = np.nan_to_num(
        day.overnight_soc_out - day.overnight_soc_in
    ) + np.nan_to_num(day.same_day_soc_out - day.same_day_soc_in)
    session_kwh = gained * population.battery_kwh / config.charging.efficiency
    # Only sessions with at least a bin of slack: a car that can't finish
    # charges from plug-in either way, so part is delivered by the first bin.
    grid = TimeGrid.from_config(config)
    need_h = hours_to_reach(
        day.overnight_soc_in,
        population.target_soc,
        population.battery_kwh,
        population.charger_kw,
        config.charging,
    )
    slack = (
        np.isnan(day.same_day_in)
        & ~np.isnan(day.overnight_in)
        & (need_h < day.overnight_hours - grid.step_hours)
    )
    assert slack.sum() > 100
    assert np.allclose(alap.max(axis=1)[slack], session_kwh[slack], atol=1e-9)


def test_nan_percentiles_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((300, 30))
    x[rng.random(x.shape) < 0.6] = np.nan
    x[:, 3] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.nanpercentile(x, (5, 50, 95), axis=0)
    assert np.allclose(nan_percentiles(x, (5, 50, 95)), expected, equal_nan=True)


def test_run_population_reproducible() -> None:
    config = SimulationConfig(n_drivers=200, n_runs=3, seed=11)
    a, b = run_population(config), run_population(config)
    assert a.runs.equals(b.runs)
    summary = a.summary.filter(pl.col("group") == ALL_DRIVERS)
    assert summary.height == 48
    assert (summary["share_plugged_run_p5"] <= summary["share_plugged_run_p95"]).all()
    # Combined SoC is a capacity-weighted mean of plugged-in SoCs.
    runs = a.runs.filter(pl.col("group") == ALL_DRIVERS).drop_nans("soc_fleet")
    assert runs["soc_fleet"].is_between(0.05, 1.0).all()


def test_daily_energy_matches_spreadsheet(
    population: Population, archetypes, config, dists
) -> None:
    """Delivered SoC per day averages the sheet's daily SoC drop.

    The gap cap (PLAN §5.4) keeps most drivers off the 5% floor. Heavy
    Intelligent Octopus drivers and infrequent chargers still hit it on long
    gaps or heavy days (read as public or en-route charging), so they get a
    looser tolerance.
    """
    rng = np.random.default_rng(9)
    days = [simulate_day(population, "weekday", rng, config, dists) for _ in range(10)]
    gained = np.mean(
        [
            np.nan_to_num(d.overnight_soc_out - d.overnight_soc_in)
            + np.nan_to_num(d.same_day_soc_out - d.same_day_soc_in)
            for d in days
        ],
        axis=0,
    )
    loose = {"Intelligent Octopus average": 0.08, "Infrequent charging": 0.08}
    for a in archetypes:
        if a.name == "Always plugged-in":  # zero demand by assumption
            continue
        mask = population.archetype_id == a.id
        simulated = gained[mask].mean()
        assert abs(simulated / a.mean_daily_soc_drop - 1) < loose.get(a.name, 0.03), (
            a.name
        )
