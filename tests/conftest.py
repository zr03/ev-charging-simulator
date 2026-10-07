"""Shared fixtures: inputs are loaded once per test session."""

import numpy as np
import pytest

from ev_charging_simulator.config import SimulationConfig
from ev_charging_simulator.models.archetypes import Archetype, load_archetypes
from ev_charging_simulator.models.distributions import Distributions, load_distributions
from ev_charging_simulator.models.driver import Population, build_population


@pytest.fixture(scope="session")
def config() -> SimulationConfig:
    return SimulationConfig(n_drivers=20_000, n_runs=5)


@pytest.fixture(scope="session")
def archetypes(config: SimulationConfig) -> list[Archetype]:
    return load_archetypes(config.archetypes_path)


@pytest.fixture(scope="session")
def dists() -> Distributions:
    return load_distributions()


@pytest.fixture(scope="session")
def population(
    archetypes: list[Archetype], config: SimulationConfig, dists: Distributions
) -> Population:
    return build_population(archetypes, config, dists, np.random.default_rng(0))
