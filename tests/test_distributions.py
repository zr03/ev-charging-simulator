import numpy as np

from ev_charging_simulator.models.distributions import (
    OVERNIGHT_ANCHOR_HOUR,
    Distributions,
    sample_annual_mileage,
    sample_soc_drop,
)
from ev_charging_simulator.utils.clock import hours_until

RNG_SEED = 1


def test_overnight_sessions_span_anchor(dists: Distributions) -> None:
    rng = np.random.default_rng(RNG_SEED)
    for sampler in dists.session_times.values():
        s = sampler.sample_overnight(rng, 10_000)
        to_anchor = hours_until(s.plug_in_hour, OVERNIGHT_ANCHOR_HOUR)
        # Jitter within the anchor hour can land either side; allow 1 h slack.
        assert np.all(to_anchor <= s.duration_hours + 1)
        same = sampler.sample_same_day(rng, 10_000)
        assert np.all(same.duration_hours >= 0)
        assert np.all(same.duration_hours < 24)


def test_target_soc_in_figure_range(dists: Distributions) -> None:
    targets = dists.target_soc.sample(np.random.default_rng(RNG_SEED), 10_000)
    assert targets.min() >= 0.6
    assert targets.max() <= 1.0


def test_soc_drop_mean_and_zero_days() -> None:
    rng = np.random.default_rng(RNG_SEED)
    n = 200_000
    days = np.full(n, 3.0)
    days[:10] = 0.0
    drop = sample_soc_drop(rng, days, np.full(n, 0.1), shape=2.0)
    assert np.all(drop[:10] == 0)
    assert np.isclose(drop[10:].mean(), 0.3, rtol=0.01)


def test_mileage_mean_preserved() -> None:
    miles = sample_annual_mileage(
        np.random.default_rng(RNG_SEED), np.full(200_000, 8000.0), 0.4
    )
    assert np.isclose(miles.mean(), 8000, rtol=0.01)
    assert np.isclose(miles.std() / miles.mean(), 0.4, rtol=0.03)
