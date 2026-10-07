import polars as pl
import pytest

from ev_charging_simulator.config import SimulationConfig
from ev_charging_simulator.models.archetypes import (
    Archetype,
    ArchetypeKind,
    with_population_shares,
)


def test_loads_six_archetypes_with_special_kinds(archetypes: list[Archetype]) -> None:
    assert len(archetypes) == 6
    kinds = {a.name: a.kind for a in archetypes}
    assert kinds["Scheduled charging"] is ArchetypeKind.SCHEDULED
    assert kinds["Always plugged-in"] is ArchetypeKind.ALWAYS_PLUGGED_IN
    assert sum(a.population_share for a in archetypes) == 1.0


def test_daily_soc_drop_matches_spreadsheet(
    archetypes: list[Archetype], config: SimulationConfig
) -> None:
    sheet = pl.read_excel(config.archetypes_path)["Average SoC drop/day"].to_list()
    for archetype, expected in zip(archetypes, sheet, strict=True):
        assert abs(archetype.mean_daily_soc_drop - expected) < 1e-9


def test_with_population_shares_normalises(archetypes: list[Archetype]) -> None:
    names = [a.name for a in archetypes]
    mixed = with_population_shares(archetypes, {names[0]: 30, names[1]: 10})
    shares = {a.name: a.population_share for a in mixed}
    assert shares[names[0]] == pytest.approx(0.75)
    assert shares[names[1]] == pytest.approx(0.25)
    assert sum(shares.values()) == pytest.approx(1.0)
    # The originals are untouched.
    assert archetypes[0].population_share != pytest.approx(0.75)
    with pytest.raises(ValueError):
        with_population_shares(archetypes, {})
