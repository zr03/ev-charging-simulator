"""Drivers (agents) and their day-level behaviour.

Each ``Driver`` carries its own parameters (archetype, mileage-derived energy
use, target SoC, habits). Simulation is vectorised over a ``Population`` of
drivers for speed; simulating one driver is a population of one, so there is
a single code path.

A simulated day follows PLAN.md section 5:

1. Overnight session with the driver's probability ``p_i`` (``p`` adjusted
   so high-mileage drivers can't skip too many nights).
2. If so, an extra same-day session with probability ``q_i = 1/p_i - 1``.
3. Session times from Fig 4 (overnight / same-day parts).
4. SoC drop since the last overnight session: a geometric gap of ``k`` days,
   gamma-distributed drops, split between sessions on two-session days.
5. Unmanaged charging to the driver's target SoC.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ev_charging_simulator.config import DayType, SimulationConfig
from ev_charging_simulator.models.archetypes import Archetype, ArchetypeKind
from ev_charging_simulator.models.charging import soc_after_charging
from ev_charging_simulator.models.distributions import (
    OVERNIGHT_ANCHOR_HOUR,
    REFERENCE_PLUG_IN_HOUR,
    REFERENCE_PLUG_OUT_HOUR,
    Distributions,
    sample_annual_mileage,
    sample_soc_drop,
)
from ev_charging_simulator.utils.clock import (
    HOURS_PER_DAY,
    hours_until,
    signed_hour_offset,
    time_to_hours,
)

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365
MAX_SAME_DAY_REDRAWS = 50

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Driver:
    """One EV driver (agent) with fixed personal parameters.

    Attributes:
        archetype: The archetype the driver belongs to.
        annual_miles: The driver's own annual mileage.
        target_soc: SoC the driver charges to (fraction).
        fixed_plug_in_hour: Scheduled drivers only: habitual plug-in (clock h).
        fixed_duration_hours: Scheduled drivers only: habitual session length.
    """

    archetype: Archetype
    annual_miles: float
    target_soc: float
    fixed_plug_in_hour: float = float("nan")
    fixed_duration_hours: float = float("nan")

    @property
    def mean_daily_soc_drop(self) -> float:
        """Mean SoC used per day (fraction of battery)."""
        if self.archetype.kind is ArchetypeKind.ALWAYS_PLUGGED_IN:
            # ASSUMPTION A10: always-plugged-in drivers have zero demand.
            return 0.0
        a = self.archetype
        return (
            self.annual_miles / DAYS_PER_YEAR / a.efficiency_mi_per_kwh / a.battery_kwh
        )

    def simulate_day(
        self,
        day_type: DayType,
        rng: np.random.Generator,
        config: SimulationConfig,
        dists: Distributions,
    ) -> "DayResult":
        """Simulate one day for this driver alone."""
        return simulate_day(
            Population.from_drivers([self]), day_type, rng, config, dists
        )


@dataclass(frozen=True)
class Population:
    """Driver parameters stored as arrays (one entry per driver)."""

    drivers: list[Driver]
    archetype_id: NDArray[np.int64] = field(init=False)
    kind: NDArray[np.str_] = field(init=False)
    battery_kwh: FloatArray = field(init=False)
    charger_kw: FloatArray = field(init=False)
    plug_ins_per_day: FloatArray = field(init=False)
    mean_daily_drop: FloatArray = field(init=False)
    target_soc: FloatArray = field(init=False)
    fixed_plug_in_hour: FloatArray = field(init=False)
    fixed_duration_hours: FloatArray = field(init=False)
    shift_in_hours: FloatArray = field(init=False)
    shift_out_hours: FloatArray = field(init=False)

    @classmethod
    def from_drivers(cls, drivers: list[Driver]) -> "Population":
        """Stack individual drivers into arrays."""
        return cls(drivers)

    def __post_init__(self) -> None:
        d = self.drivers

        def put(name: str, values: list) -> None:
            object.__setattr__(self, name, np.asarray(values))

        put("archetype_id", [x.archetype.id for x in d])
        put("kind", [x.archetype.kind.value for x in d])
        put("battery_kwh", [x.archetype.battery_kwh for x in d])
        put("charger_kw", [x.archetype.charger_kw for x in d])
        put("plug_ins_per_day", [x.archetype.plug_ins_per_day for x in d])
        put("mean_daily_drop", [x.mean_daily_soc_drop for x in d])
        put("target_soc", [x.target_soc for x in d])
        put("fixed_plug_in_hour", [x.fixed_plug_in_hour for x in d])
        put("fixed_duration_hours", [x.fixed_duration_hours for x in d])
        put("shift_in_hours", [_overnight_shift(x.archetype)[0] for x in d])
        put("shift_out_hours", [_overnight_shift(x.archetype)[1] for x in d])

    def __len__(self) -> int:
        return len(self.drivers)

    @property
    def is_always_plugged_in(self) -> NDArray[np.bool_]:
        return self.kind == ArchetypeKind.ALWAYS_PLUGGED_IN.value

    @property
    def is_scheduled(self) -> NDArray[np.bool_]:
        return self.kind == ArchetypeKind.SCHEDULED.value


def _overnight_shift(archetype: Archetype) -> tuple[float, float]:
    """Hours to shift Fig 4 overnight samples to match the archetype's times.

    Zero for archetypes at the Fig 4 mode (18:00 / 07:00); +4 h / +2 h for
    scheduled charging (22:00 / 09:00).
    """
    if archetype.kind is ArchetypeKind.ALWAYS_PLUGGED_IN:
        return 0.0, 0.0
    shift_in = signed_hour_offset(
        time_to_hours(archetype.plug_in_time), REFERENCE_PLUG_IN_HOUR
    )
    shift_out = signed_hour_offset(
        time_to_hours(archetype.plug_out_time), REFERENCE_PLUG_OUT_HOUR
    )
    return shift_in, shift_out


def sample_drivers(
    archetype: Archetype,
    n: int,
    config: SimulationConfig,
    dists: Distributions,
    rng: np.random.Generator,
) -> list[Driver]:
    """Draw ``n`` drivers of one archetype with their personal parameters."""
    if n == 0:
        return []
    miles = sample_annual_mileage(
        rng, np.full(n, archetype.miles_per_year), config.mileage_cv
    )
    # ASSUMPTION A2: spreadsheet mileage is the archetype mean.
    # ASSUMPTION A11: Fig 2 (Intelligent Octopus) charge limits apply to all.
    targets = dists.target_soc.sample(rng, n)
    fixed_in = np.full(n, np.nan)
    fixed_dur = np.full(n, np.nan)
    if archetype.kind is ArchetypeKind.SCHEDULED:
        # ASSUMPTION A9: a schedule is a habit (late plug-in, late plug-out),
        # so each driver's times are drawn once.
        habit = dists.session_times["weekday"].sample_overnight(rng, n)
        shift_in, shift_out = _overnight_shift(archetype)
        fixed_in = (habit.plug_in_hour + shift_in) % HOURS_PER_DAY
        fixed_dur = habit.duration_hours + shift_out - shift_in
    return [
        Driver(archetype, float(miles[i]), float(targets[i]), fixed_in[i], fixed_dur[i])
        for i in range(n)
    ]


def build_population(
    archetypes: list[Archetype],
    config: SimulationConfig,
    dists: Distributions,
    rng: np.random.Generator,
) -> Population:
    """Create ``config.n_drivers`` drivers split by archetype population share.

    Counts use largest-remainder rounding, so the mix is exact for a fixed N.
    """
    shares = np.array([a.population_share for a in archetypes])
    raw = shares * config.n_drivers
    counts = np.floor(raw).astype(int)
    shortfall = config.n_drivers - counts.sum()
    counts[np.argsort(raw - counts)[::-1][:shortfall]] += 1

    drivers: list[Driver] = []
    for archetype, n in zip(archetypes, counts, strict=True):
        drivers += sample_drivers(archetype, int(n), config, dists, rng)
    logger.info(
        "Built population of %d drivers: %s",
        len(drivers),
        {a.name: int(c) for a, c in zip(archetypes, counts, strict=True)},
    )
    return Population.from_drivers(drivers)


@dataclass(frozen=True)
class DayResult:
    """Sessions for one simulated day, one entry per driver.

    Plug-in times are hours since the window start (noon by default), in
    [0, 24). ASSUMPTION A1: the window is treated as circular: a session
    running past the window end wraps to its start, valid because days are
    i.i.d.
    NaN means no such session. Each driver has at most one overnight and one
    same-day session, and the two never overlap.
    """

    overnight_in: FloatArray
    overnight_hours: FloatArray
    overnight_soc_in: FloatArray
    overnight_soc_out: FloatArray
    same_day_in: FloatArray
    same_day_hours: FloatArray
    same_day_soc_in: FloatArray
    same_day_soc_out: FloatArray

    @property
    def sessions_per_driver(self) -> NDArray[np.int64]:
        """Number of sessions each driver had (0, 1 or 2)."""
        return (~np.isnan(self.overnight_in)).astype(int) + (
            ~np.isnan(self.same_day_in)
        ).astype(int)


@dataclass(frozen=True)
class PlugInModel:
    """Each driver's nightly plug-in behaviour (PLAN §5.1 and §5.4).

    A driver plugs in on any night with probability ``p_base`` but never
    lets more than ``max_gap_days`` pass. The gap since the last overnight
    session is then Geometric(p_base) capped at K, with mean
    (1 - (1 - p_base)^K) / p_base, and the long-run share of nights plugged
    in is ``p_driver`` = 1 / mean gap. Using ``p_driver`` for tonight keeps
    delivered energy equal to driving: p_driver x E[gap] x daily drop.
    """

    p_base: FloatArray
    max_gap_days: FloatArray  # K; inf for drivers with no demand
    p_driver: FloatArray
    q_driver: FloatArray  # extra same-day session, given an overnight one


def plug_in_model(pop: Population, config: SimulationConfig) -> PlugInModel:
    """Per-driver plug-in probabilities, limiting nights between charges."""
    infrequent = pop.plug_ins_per_day < 1.0
    # ASSUMPTION A4: infrequent archetypes plug in with probability f and
    # never twice; others use the symmetric model around p_overnight, drawn
    # independently each day (no habitual twice-daily chargers).
    p_base = np.where(infrequent, pop.plug_ins_per_day, config.p_overnight)
    # ASSUMPTION A5: the longest gap is the number of days of
    # average driving between target SoC and a reserve. Infrequent chargers
    # run their battery lower by design, so their reserve is the floor.
    reserve = np.where(infrequent, config.soc_floor, config.gap_reserve_soc)
    usable = np.maximum(pop.target_soc - reserve, 0.0)
    with np.errstate(divide="ignore"):
        days = np.where(pop.mean_daily_drop > 0, usable / pop.mean_daily_drop, np.inf)
    max_gap = np.maximum(np.floor(days), 1.0)

    mean_gap = (1.0 - (1.0 - p_base) ** max_gap) / p_base
    p_driver = 1.0 / mean_gap
    q_driver = np.where(infrequent, 0.0, 1.0 / p_driver - 1.0)

    always = pop.is_always_plugged_in
    return PlugInModel(
        p_base=p_base,
        max_gap_days=max_gap,
        p_driver=np.where(always, 1.0, p_driver),
        q_driver=np.where(always, 0.0, q_driver),
    )


def simulate_day(
    pop: Population,
    day_type: DayType,
    rng: np.random.Generator,
    config: SimulationConfig,
    dists: Distributions,
) -> DayResult:
    """Simulate one day for every driver in the population (PLAN §5).

    Args:
        pop: Drivers to simulate.
        day_type: Selects the weekday or weekend Fig 4 heatmap.
        rng: Random generator for this day.
        config: Model settings.
        dists: Empirical samplers.

    Returns:
        Each driver's sessions in window time, with plug-in/plug-out SoC.
    """
    n = len(pop)
    sampler = dists.session_times[day_type]

    # 1. How many sessions today, from each driver's own plug-in model.
    plug_in = plug_in_model(pop, config)
    p, q = plug_in.p_driver, plug_in.q_driver
    has_overnight = rng.random(n) < p
    has_same_day = has_overnight & (rng.random(n) < q)

    # 2. Overnight session times (clock hours), shifted for scheduled drivers.
    overnight = sampler.sample_overnight(rng, n)
    on_in = (overnight.plug_in_hour + pop.shift_in_hours) % HOURS_PER_DAY
    on_dur = overnight.duration_hours + pop.shift_out_hours - pop.shift_in_hours
    sched = pop.is_scheduled
    if sched.any():
        jitter = config.scheduled_noise_minutes / 60
        j_in = rng.uniform(-jitter, jitter, n)
        j_out = rng.uniform(-jitter, jitter, n)
        on_in = np.where(sched, (pop.fixed_plug_in_hour + j_in) % HOURS_PER_DAY, on_in)
        on_dur = np.where(sched, pop.fixed_duration_hours + j_out - j_in, on_dur)

    # 3. Same-day session, redrawn until it fits between this morning's
    #    plug-out and tonight's plug-in. Times measured from 03:00, when every
    #    overnight session is connected. The overnight session stands in for
    #    last night's (days are i.i.d.).
    on_in_a = hours_until(OVERNIGHT_ANCHOR_HOUR, on_in)
    on_out_a = hours_until(OVERNIGHT_ANCHOR_HOUR, on_in + on_dur)
    sd_in_a = np.full(n, np.nan)
    sd_dur = np.full(n, np.nan)
    pending = has_same_day.copy()
    for _ in range(MAX_SAME_DAY_REDRAWS):
        if not pending.any():
            break
        draw = sampler.sample_same_day(rng, int(pending.sum()))
        start = hours_until(OVERNIGHT_ANCHOR_HOUR, draw.plug_in_hour)
        fits = (start >= on_out_a[pending]) & (
            start + draw.duration_hours <= on_in_a[pending]
        )
        idx = np.flatnonzero(pending)[fits]
        sd_in_a[idx] = start[fits]
        sd_dur[idx] = draw.duration_hours[fits]
        pending[idx] = False
    if pending.any():
        logger.debug("Dropped %d same-day sessions that never fitted", pending.sum())
        has_same_day &= ~pending

    # 4. Energy. k days since the last overnight session: Geometric(p_base),
    #    capped at the driver's max gap. Earlier days and today are drawn
    #    separately so today's drop can be split around a same-day session.
    k = np.minimum(rng.geometric(plug_in.p_base), plug_in.max_gap_days)
    drop_earlier = sample_soc_drop(
        rng, k - 1.0, pop.mean_daily_drop, config.gamma_shape
    )
    drop_today = sample_soc_drop(
        rng, np.ones(n), pop.mean_daily_drop, config.gamma_shape
    )
    # ASSUMPTION A7: time away from the charger is a proxy for
    # driving, so today's drop is split by away time before/after the session.
    away_before = sd_in_a - on_out_a
    away_after = on_in_a - (sd_in_a + sd_dur)
    total_away = away_before + away_after
    frac_before = np.where(total_away > 0, away_before / total_away, 0.5)

    # ASSUMPTION A6: the last overnight session ended at target.
    target = pop.target_soc
    floor = config.soc_floor
    sd_soc_in = np.maximum(target - drop_earlier - frac_before * drop_today, floor)
    sd_soc_out = soc_after_charging(
        sd_dur, sd_soc_in, target, pop.battery_kwh, pop.charger_kw, config.charging
    )
    on_soc_in = np.where(
        has_same_day,
        sd_soc_out - (1 - frac_before) * drop_today,
        target - drop_earlier - drop_today,
    )
    on_soc_in = np.maximum(on_soc_in, floor)
    on_soc_out = soc_after_charging(
        on_dur, on_soc_in, target, pop.battery_kwh, pop.charger_kw, config.charging
    )

    # 5. Place sessions in the (circular) window. Same-day sessions keep
    #    their clock time, so morning ones represent the next morning.
    start_h = config.window_start_hour
    on_in_w = hours_until(start_h, on_in)
    sd_in_w = hours_until(start_h, sd_in_a + OVERNIGHT_ANCHOR_HOUR)

    nan = np.nan
    result = DayResult(
        overnight_in=np.where(has_overnight, on_in_w, nan),
        overnight_hours=np.where(has_overnight, on_dur, nan),
        overnight_soc_in=np.where(has_overnight, on_soc_in, nan),
        overnight_soc_out=np.where(has_overnight, on_soc_out, nan),
        same_day_in=np.where(has_same_day, sd_in_w, nan),
        same_day_hours=np.where(has_same_day, sd_dur, nan),
        same_day_soc_in=np.where(has_same_day, sd_soc_in, nan),
        same_day_soc_out=np.where(has_same_day, sd_soc_out, nan),
    )
    return _apply_always_plugged_in(result, pop)


def _apply_always_plugged_in(result: DayResult, pop: Population) -> DayResult:
    """Always-plugged-in drivers: connected all window, held at target SoC."""
    always = pop.is_always_plugged_in
    if not always.any():
        return result
    t = pop.target_soc
    nan = np.nan
    return DayResult(
        overnight_in=np.where(always, 0.0, result.overnight_in),
        overnight_hours=np.where(always, HOURS_PER_DAY, result.overnight_hours),
        overnight_soc_in=np.where(always, t, result.overnight_soc_in),
        overnight_soc_out=np.where(always, t, result.overnight_soc_out),
        same_day_in=np.where(always, nan, result.same_day_in),
        same_day_hours=np.where(always, nan, result.same_day_hours),
        same_day_soc_in=np.where(always, nan, result.same_day_soc_in),
        same_day_soc_out=np.where(always, nan, result.same_day_soc_out),
    )
