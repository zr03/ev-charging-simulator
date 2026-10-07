"""Simulation settings and their defaults.

Every tunable number in the model lives here, so the notebook, scripts and
tests share one source of truth. See PLAN.md section 7 for the defaults table.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

DayType = Literal["weekday", "weekend"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHETYPES_PATH = REPO_ROOT / "data/inputs/Axle Take Home Test_ Archetypes.xlsx"


class ChargingConfig(BaseModel):
    """Unmanaged charging curve shared by all drivers."""

    model_config = ConfigDict(frozen=True)

    taper_start_soc: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="ASSUMPTION A12: full charger power up to this SoC.",
    )
    taper_end_kw: float = Field(
        2.0,
        gt=0.0,
        description="ASSUMPTION A12: power at 100% SoC; linear taper above.",
    )
    efficiency: float = Field(
        0.9,
        gt=0.0,
        le=1.0,
        description="ASSUMPTION A12: battery kWh delivered per grid kWh.",
    )


class SimulationConfig(BaseModel):
    """All model parameters. Defaults follow PLAN.md section 7."""

    model_config = ConfigDict(frozen=True)

    # Plug-in behaviour
    p_overnight: float = Field(
        0.74,
        gt=0.5,
        le=1.0,
        description=(
            "ASSUMPTION A4: daily probability of an overnight session for "
            "1/day archetypes. Calibrated so 1 - p matches Table 2's 26.3% of "
            "gaps under 12 h."
        ),
    )

    # Energy use
    gamma_shape: float = Field(
        2.0,
        gt=0.0,
        description="ASSUMPTION A3: shape of the daily SoC-drop gamma (right skew).",
    )
    mileage_cv: float = Field(
        0.4,
        ge=0.0,
        description="ASSUMPTION A2: coefficient of variation of annual mileage.",
    )
    gap_reserve_soc: float = Field(
        0.20,
        ge=0.0,
        lt=1.0,
        description=(
            "ASSUMPTION A5: drivers never go more nights without "
            "charging than their average driving allows before SoC would fall "
            "below this reserve. Infrequent chargers use soc_floor instead."
        ),
    )
    soc_floor: float = Field(
        0.05,
        ge=0.0,
        lt=1.0,
        description="ASSUMPTION A6: minimum plug-in SoC.",
    )

    # Scheduled-charging archetype
    scheduled_noise_minutes: float = Field(
        15.0,
        ge=0.0,
        description="ASSUMPTION A9: uniform +/- daily jitter on a scheduled driver's fixed times.",
    )

    charging: ChargingConfig = ChargingConfig()

    # Population runs
    n_drivers: int = Field(1000, gt=0)
    n_runs: int = Field(200, gt=0)
    seed: int = 305
    day_type: DayType = "weekday"

    # Time grid. The window runs noon to noon: the fewest cars are plugged in
    # at midday, so almost every session fits without wrapping.
    window_start_hour: int = Field(12, ge=0, lt=24)
    # Single-driver plots run 06:00 to 06:00 instead. Almost no daytime
    # session starts before 06:00, so each day reads in the order its SoC is
    # computed: morning plug-out, daytime session, evening overnight session.
    # The cost is that most overnight sessions wrap their tail to the left edge.
    driver_window_start_hour: int = Field(6, ge=0, lt=24)
    output_step_minutes: int = Field(30, gt=0)

    archetypes_path: Path = DEFAULT_ARCHETYPES_PATH

    @computed_field  # type: ignore[prop-decorator]
    @property
    def q_same_day(self) -> float:
        """Probability of an extra same-day session, given an overnight one.

        ASSUMPTION A4: chosen so expected sessions per day = p(1 + q) = 1.
        """
        return 1.0 / self.p_overnight - 1.0
