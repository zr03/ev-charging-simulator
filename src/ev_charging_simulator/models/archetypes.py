"""Driver archetypes loaded from Axle's spreadsheet."""

import logging
from datetime import datetime, time
from enum import StrEnum
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365


class ArchetypeKind(StrEnum):
    """How an archetype's sessions are generated."""

    STANDARD = "standard"
    # Overnight times drawn once per driver, then reused with small jitter.
    SCHEDULED = "scheduled"
    # Connected all day, no energy demand (ASSUMPTION A10).
    ALWAYS_PLUGGED_IN = "always_plugged_in"


# The spreadsheet has no "kind" column, so map the special rows by name.
SPECIAL_KINDS: dict[str, ArchetypeKind] = {
    "Scheduled charging": ArchetypeKind.SCHEDULED,
    "Always plugged-in": ArchetypeKind.ALWAYS_PLUGGED_IN,
}


class Archetype(BaseModel):
    """One row of the archetypes spreadsheet.

    Spreadsheet values are treated as population means for the archetype;
    variation is added around them (ASSUMPTION A2).
    """

    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    population_share: float = Field(ge=0.0, le=1.0, description="Fraction, 0 to 1.")
    miles_per_year: float = Field(gt=0.0)
    battery_kwh: float = Field(gt=0.0)
    efficiency_mi_per_kwh: float = Field(gt=0.0)
    plug_ins_per_day: float = Field(gt=0.0, le=1.0)
    charger_kw: float = Field(gt=0.0)
    plug_in_time: time
    plug_out_time: time
    kind: ArchetypeKind = ArchetypeKind.STANDARD

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_daily_soc_drop(self) -> float:
        """Mean SoC used per day of driving (fraction of battery).

        ASSUMPTION A3: computed from miles, efficiency and battery, not the
        sheet's own requirement column (which differs slightly for row 2).
        """
        kwh_per_day = self.miles_per_year / DAYS_PER_YEAR / self.efficiency_mi_per_kwh
        return kwh_per_day / self.battery_kwh

    @model_validator(mode="after")
    def _check_special_kinds(self) -> "Archetype":
        if self.kind is ArchetypeKind.SCHEDULED and self.plug_ins_per_day < 1.0:
            raise ValueError("Scheduled archetype must plug in daily.")
        return self


def _as_time(value: datetime | time) -> time:
    """Excel times arrive as datetimes on 1899-12-31; keep the clock part."""
    return value.time() if isinstance(value, datetime) else value


def load_archetypes(path: Path) -> list[Archetype]:
    """Read the archetypes spreadsheet.

    Args:
        path: Path to the xlsx file.

    Returns:
        Archetypes in spreadsheet order.

    Raises:
        ValueError: If population shares don't sum to 100%.
    """
    frame = pl.read_excel(path)
    archetypes = [
        Archetype(
            id=row["#"],
            name=row["Name"],
            population_share=row["% of population"] / 100,
            miles_per_year=row["Miles/yr"],
            battery_kwh=row["Battery (kWh)"],
            efficiency_mi_per_kwh=row["Efficiency (mi/kWh)"],
            plug_ins_per_day=row["Plug-in frequency (per day)"],
            charger_kw=row["Charger kW"],
            plug_in_time=_as_time(row["Plug-in time"]),
            plug_out_time=_as_time(row["Plug-out time"]),
            kind=SPECIAL_KINDS.get(row["Name"], ArchetypeKind.STANDARD),
        )
        for row in frame.iter_rows(named=True)
    ]
    total_share = sum(a.population_share for a in archetypes)
    if abs(total_share - 1.0) > 1e-6:
        raise ValueError(f"Population shares sum to {total_share:.3f}, not 1.")
    logger.info("Loaded %d archetypes from %s", len(archetypes), path.name)
    return archetypes


def with_population_shares(
    archetypes: list[Archetype], weights: dict[str, float]
) -> list[Archetype]:
    """Return copies of ``archetypes`` with a different population mix.

    Lets a user model a fleet whose mix differs from the spreadsheet, e.g.
    the chargers in one area. Weights are relative (percentages need not sum
    to 100) and are normalised; archetypes missing from ``weights`` get 0.

    Args:
        archetypes: Archetypes as loaded.
        weights: Non-negative weight per archetype name.

    Raises:
        ValueError: If a weight is negative or all weights are zero.
    """
    values = [weights.get(a.name, 0.0) for a in archetypes]
    if any(v < 0 for v in values):
        raise ValueError("Population weights must be non-negative.")
    total = sum(values)
    if total <= 0:
        raise ValueError("At least one archetype needs a positive weight.")
    return [
        a.model_copy(update={"population_share": v / total})
        for a, v in zip(archetypes, values, strict=True)
    ]
