import numpy as np

from ev_charging_simulator.config import SimulationConfig
from ev_charging_simulator.models.archetypes import Archetype
from ev_charging_simulator.models.distributions import Distributions
from ev_charging_simulator.models.driver import Population, plug_in_model, simulate_day
from ev_charging_simulator.utils.clock import HOURS_PER_DAY


def _day(
    pop: Population, config: SimulationConfig, dists: Distributions, seed: int = 3
):
    return simulate_day(pop, "weekday", np.random.default_rng(seed), config, dists)


def test_population_mix_is_exact(
    population: Population, archetypes: list[Archetype]
) -> None:
    for a in archetypes:
        assert (population.archetype_id == a.id).sum() == round(
            a.population_share * 20_000
        )


def test_sessions_per_day(
    population: Population, archetypes: list[Archetype], config, dists
) -> None:
    day = _day(population, config, dists)
    sessions = day.sessions_per_driver
    by_name = {a.name: population.archetype_id == a.id for a in archetypes}
    # Symmetric model: one session per day on average.
    assert abs(sessions[by_name["Average (UK)"]].mean() - 1.0) < 0.03
    # Infrequent chargers plug in at their capped rate p_i, never twice.
    infrequent = by_name["Infrequent charging"]
    p_i = plug_in_model(population, config).p_driver[infrequent].mean()
    assert abs(sessions[infrequent].mean() - p_i) < 0.03
    assert sessions[infrequent].max() == 1
    assert np.all(sessions[by_name["Always plugged-in"]] == 1)


def test_sessions_do_not_overlap(population: Population, config, dists) -> None:
    day = _day(population, config, dists)
    both = ~np.isnan(day.same_day_in)
    # Same-day start must lie outside the overnight session on the circle.
    rel = (day.same_day_in[both] - day.overnight_in[both]) % HOURS_PER_DAY
    end = rel + day.same_day_hours[both]
    assert np.all(rel >= day.overnight_hours[both])
    assert np.all(end <= HOURS_PER_DAY + 1e-9)


def test_soc_bounds(population: Population, config, dists) -> None:
    day = _day(population, config, dists)
    for soc in (day.overnight_soc_in, day.overnight_soc_out, day.same_day_soc_in):
        valid = soc[~np.isnan(soc)]
        assert valid.min() >= config.soc_floor - 1e-12
        assert valid.max() <= 1.0
    has = ~np.isnan(day.overnight_in)
    assert np.all(day.overnight_soc_out[has] <= population.target_soc[has] + 1e-12)


def test_same_seed_same_day(population: Population, config, dists) -> None:
    a, b = _day(population, config, dists, 7), _day(population, config, dists, 7)
    assert np.array_equal(a.overnight_in, b.overnight_in, equal_nan=True)
    assert np.array_equal(a.same_day_soc_in, b.same_day_soc_in, equal_nan=True)


def test_gap_cap_mean_matches_formula() -> None:
    """Mean of min(Geometric(p), K) is (1 - (1 - p)^K) / p, so p_i is its inverse."""
    rng = np.random.default_rng(1)
    p = 0.74
    for k in (1, 2, 5):
        simulated = np.minimum(rng.geometric(p, 200_000), k).mean()
        assert abs(simulated - (1 - (1 - p) ** k) / p) < 0.01


def test_plug_in_model_caps_heavy_drivers(population: Population, config) -> None:
    model = plug_in_model(population, config)
    one_day = model.max_gap_days == 1
    # A driver who can't last two nights plugs in every night, once.
    assert np.allclose(model.p_driver[one_day], 1.0)
    assert np.allclose(model.q_driver[one_day], 0.0)
    regular = population.plug_ins_per_day >= 1.0
    assert np.all(model.p_driver[regular] >= config.p_overnight - 1e-12)
    # Mean sessions stay at one a day for regular chargers.
    assert np.allclose(model.p_driver[regular] * (1 + model.q_driver[regular]), 1.0)
