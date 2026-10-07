"""Turn simulated sessions into time profiles, and run whole populations.

Profiles are evaluated on a circular 24 h window (noon to noon by default).
State (plugged in, SoC, flexibility) is read at each bin's midpoint; power is
the exact average over the bin, from the SoC change across it. The closed-form
charging curve makes this exact at any bin width, so no minute grid is needed.
"""

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ev_charging_simulator.config import SimulationConfig
from ev_charging_simulator.models.archetypes import Archetype, load_archetypes
from ev_charging_simulator.models.charging import hours_to_reach, soc_after_charging
from ev_charging_simulator.models.distributions import Distributions, load_distributions
from ev_charging_simulator.models.driver import (
    DayResult,
    Driver,
    Population,
    build_population,
    simulate_day,
)
from ev_charging_simulator.utils.clock import HOURS_PER_DAY
from ev_charging_simulator.utils.seeding import spawn_generators

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
ALL_DRIVERS = "All drivers"
SOC_PERCENTILES = (5, 50, 95)
RUN_BAND_PERCENTILES = (5, 95)
DRIVER_STEP_MINUTES = 5


@dataclass(frozen=True)
class TimeGrid:
    """Bins covering the 24 h window.

    Attributes:
        start_hour: Clock hour at which the window starts.
        step_hours: Bin width (h).
    """

    start_hour: int
    step_hours: float

    @property
    def n_bins(self) -> int:
        return round(HOURS_PER_DAY / self.step_hours)

    @property
    def edges(self) -> FloatArray:
        """Bin edges in window hours, 0 to 24."""
        return np.arange(self.n_bins + 1, dtype=float) * self.step_hours

    @property
    def mids(self) -> FloatArray:
        """Bin midpoints in window hours."""
        return self.edges[:-1] + self.step_hours / 2

    def clock_labels(self) -> list[str]:
        """``HH:MM`` clock time at each bin start."""
        minutes = np.round((self.edges[:-1] + self.start_hour) * 60).astype(int)
        return [f"{(m // 60) % 24:02d}:{m % 60:02d}" for m in minutes]

    @classmethod
    def from_config(
        cls, config: SimulationConfig, step_minutes: int | None = None
    ) -> "TimeGrid":
        step = config.output_step_minutes if step_minutes is None else step_minutes
        return cls(config.window_start_hour, step / 60)


@dataclass(frozen=True)
class Profiles:
    """Per-driver time profiles for one day, shape (drivers, bins).

    All values are grid-side (kW, kWh) except SoC. NaN where not plugged in.
    """

    plugged: NDArray[np.bool_]
    soc: FloatArray
    power_kw: FloatArray  # average over the bin; 0 when not charging
    # The same sessions charged as late as possible (see _session_profile).
    power_alap_kw: FloatArray
    energy_to_target_kwh: FloatArray
    # Energy this session will still deliver (kWh at the wall), if charging
    # runs as soon as possible (unmanaged) or as late as possible. The gap
    # between them is energy that can be moved later than this time.
    energy_left_asap_kwh: FloatArray
    energy_left_alap_kwh: FloatArray
    hours_to_plug_out: FloatArray
    # Plug-out time minus time needed to reach target: how far charging
    # could be delayed without missing the target.
    flexible_hours: FloatArray
    connected_kw: FloatArray  # charger rating if plugged in, else 0


@dataclass(frozen=True)
class _Session:
    plugged: NDArray[np.bool_]
    soc: FloatArray
    delta_soc: FloatArray
    delta_soc_alap: FloatArray
    hours_left: FloatArray
    soc_alap: FloatArray
    soc_end: FloatArray


def _session_profile(
    start: FloatArray,
    hours: FloatArray,
    soc_in: FloatArray,
    pop: Population,
    grid: TimeGrid,
    config: SimulationConfig,
) -> _Session:
    """Evaluate one session per driver (NaN start = no session) on the grid."""
    s = np.nan_to_num(start)[:, None]
    d = np.nan_to_num(hours)[:, None]
    s0 = np.nan_to_num(soc_in)[:, None]
    args = (
        pop.target_soc[:, None],
        pop.battery_kwh[:, None],
        pop.charger_kw[:, None],
        config.charging,
    )

    # As late as possible: the same charge, started at the last moment that
    # still delivers it by plug-out. A car that can't reach its target in
    # time has no slack, so its two profiles coincide.
    delay = d - np.minimum(hours_to_reach(s0, *args), d)

    def soc_at(rel: FloatArray) -> FloatArray:
        return soc_after_charging(np.minimum(rel, d), s0, *args)

    def soc_alap_at(rel: FloatArray) -> FloatArray:
        return soc_after_charging(np.minimum(rel, d) - delay, s0, *args)

    rel_mid = (grid.mids - s) % HOURS_PER_DAY
    plugged = rel_mid < d

    # ASSUMPTION A1: energy across each bin. A session that starts inside a
    # bin wraps round the circular window, so add its head separately.
    rel_a = (grid.edges[:-1] - s) % HOURS_PER_DAY
    rel_b = rel_a + grid.step_hours
    wraps = rel_b > HOURS_PER_DAY

    def gained(curve: Callable[[FloatArray], FloatArray]) -> FloatArray:
        head = np.where(wraps, curve(rel_b - HOURS_PER_DAY) - s0, 0.0)
        return curve(rel_b) - curve(rel_a) + head

    return _Session(
        plugged=plugged,
        soc=np.where(plugged, soc_at(rel_mid), np.nan),
        delta_soc=gained(soc_at),
        delta_soc_alap=gained(soc_alap_at),
        hours_left=np.where(plugged, d - rel_mid, np.nan),
        soc_alap=np.where(plugged, soc_alap_at(rel_mid), np.nan),
        soc_end=np.broadcast_to(soc_at(d), plugged.shape),
    )


def build_profiles(
    day: DayResult, pop: Population, grid: TimeGrid, config: SimulationConfig
) -> Profiles:
    """Per-driver plug-in, SoC, power and flexibility profiles for one day."""
    sessions = [
        _session_profile(
            day.overnight_in,
            day.overnight_hours,
            day.overnight_soc_in,
            pop,
            grid,
            config,
        ),
        _session_profile(
            day.same_day_in, day.same_day_hours, day.same_day_soc_in, pop, grid, config
        ),
    ]
    # The two sessions never overlap, so take whichever is active.
    plugged = sessions[0].plugged | sessions[1].plugged
    soc = np.where(sessions[0].plugged, sessions[0].soc, sessions[1].soc)
    hours_left = np.where(
        sessions[0].plugged, sessions[0].hours_left, sessions[1].hours_left
    )
    soc_alap = np.where(sessions[0].plugged, sessions[0].soc_alap, sessions[1].soc_alap)
    soc_end = np.where(sessions[0].plugged, sessions[0].soc_end, sessions[1].soc_end)
    delta_soc = sessions[0].delta_soc + sessions[1].delta_soc
    delta_soc_alap = sessions[0].delta_soc_alap + sessions[1].delta_soc_alap

    battery = pop.battery_kwh[:, None]
    eff = config.charging.efficiency
    target = pop.target_soc[:, None]
    need_h = hours_to_reach(
        soc, target, battery, pop.charger_kw[:, None], config.charging
    )
    return Profiles(
        plugged=plugged,
        soc=soc,
        power_kw=delta_soc * battery / eff / grid.step_hours,
        power_alap_kw=delta_soc_alap * battery / eff / grid.step_hours,
        energy_to_target_kwh=np.maximum(target - soc, 0.0) * battery / eff,
        energy_left_asap_kwh=np.maximum(soc_end - soc, 0.0) * battery / eff,
        energy_left_alap_kwh=np.maximum(soc_end - soc_alap, 0.0) * battery / eff,
        hours_to_plug_out=hours_left,
        flexible_hours=hours_left - need_h,
        connected_kw=np.where(plugged, pop.charger_kw[:, None], 0.0),
    )


def aggregate_profiles(
    profiles: Profiles,
    pop: Population,
    groups: dict[str, NDArray[np.bool_]],
    grid: TimeGrid,
) -> pl.DataFrame:
    """Summarise per-driver profiles by driver group, one row per group and bin.

    Per-EV quantities divide by every driver in the group (plugged in or
    not), so multiplying by fleet size gives the fleet total. SoC and
    time-left statistics are over plugged-in drivers only. ``soc_fleet`` is
    the plugged-in cars' combined state of charge: kWh stored over kWh of
    battery capacity, so bigger batteries count for more.
    """
    frames = []
    labels = grid.clock_labels()
    for name, mask in groups.items():
        if not mask.any():
            continue
        plugged = profiles.plugged[mask]
        soc = profiles.soc[mask]
        battery = pop.battery_kwh[mask][:, None]
        capacity_kwh = (plugged * battery).sum(axis=0)
        with warnings.catch_warnings():
            # Bins where nobody in the group is plugged in give NaN, as intended.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            soc_pct = nan_percentiles(soc, SOC_PERCENTILES)
            columns = {
                # NaN where nobody is plugged in; summarise_runs skips those.
                "soc_fleet": np.nansum(soc * battery, axis=0) / capacity_kwh,
                "soc_mean": np.nanmean(soc, axis=0),
                **{
                    f"soc_p{q}": v
                    for q, v in zip(SOC_PERCENTILES, soc_pct, strict=True)
                },
                "hours_to_plug_out_mean": np.nanmean(
                    profiles.hours_to_plug_out[mask], axis=0
                ),
                "flexible_hours_mean": np.nanmean(
                    profiles.flexible_hours[mask], axis=0
                ),
            }
        frames.append(
            pl.DataFrame(
                {
                    "group": name,
                    "bin": np.arange(grid.n_bins),
                    "window_hour": grid.edges[:-1],
                    "clock": labels,
                    "n_drivers": int(mask.sum()),
                    "n_plugged": plugged.sum(axis=0),
                    "share_plugged": plugged.mean(axis=0),
                    "power_kw_per_ev": profiles.power_kw[mask].mean(axis=0),
                    "power_alap_kw_per_ev": profiles.power_alap_kw[mask].mean(axis=0),
                    "connected_kw_per_ev": profiles.connected_kw[mask].mean(axis=0),
                    "energy_to_target_kwh_per_ev": np.nansum(
                        profiles.energy_to_target_kwh[mask], axis=0
                    )
                    / mask.sum(),
                    "energy_left_asap_kwh_per_ev": np.nansum(
                        profiles.energy_left_asap_kwh[mask], axis=0
                    )
                    / mask.sum(),
                    "energy_left_alap_kwh_per_ev": np.nansum(
                        profiles.energy_left_alap_kwh[mask], axis=0
                    )
                    / mask.sum(),
                    "shiftable_kwh_per_ev": np.nansum(
                        profiles.energy_left_alap_kwh[mask]
                        - profiles.energy_left_asap_kwh[mask],
                        axis=0,
                    )
                    / mask.sum(),
                    **columns,
                }
            )
        )
    return pl.concat(frames)


def nan_percentiles(values: FloatArray, percentiles: tuple[int, ...]) -> FloatArray:
    """Column-wise percentiles ignoring NaN (linear interpolation).

    Same result as ``np.nanpercentile(values, percentiles, axis=0)`` but much
    faster for many columns, which matters inside the run loop.
    """
    ordered = np.sort(values, axis=0)  # NaN sorts last
    count = (~np.isnan(values)).sum(axis=0)
    cols = np.arange(values.shape[1])
    out = np.full((len(percentiles), values.shape[1]), np.nan)
    has = count > 0
    for i, q in enumerate(percentiles):
        pos = q / 100 * np.maximum(count - 1, 0)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, np.maximum(count - 1, 0))
        frac = pos - lo
        value = ordered[lo, cols] * (1 - frac) + ordered[hi, cols] * frac
        out[i] = np.where(has, value, np.nan)
    return out


def driver_groups(
    pop: Population, archetypes: list[Archetype]
) -> dict[str, NDArray[np.bool_]]:
    """Masks for the whole population and for each archetype."""
    groups = {ALL_DRIVERS: np.ones(len(pop), dtype=bool)}
    groups |= {a.name: pop.archetype_id == a.id for a in archetypes}
    return groups


@dataclass(frozen=True)
class PopulationResult:
    """Output of :func:`run_population`.

    Attributes:
        runs: Per-run aggregates (one row per run, group and bin).
        summary: Across-run mean and percentile band of every metric.
        population: The fixed set of drivers that was simulated.
        archetypes: Archetypes the population was drawn from.
        grid: Time bins used.
        config: Settings used.
    """

    runs: pl.DataFrame
    summary: pl.DataFrame
    population: Population
    archetypes: list[Archetype]
    grid: TimeGrid
    config: SimulationConfig


def run_population(
    config: SimulationConfig,
    archetypes: list[Archetype] | None = None,
    dists: Distributions | None = None,
) -> PopulationResult:
    """Simulate a fixed population of drivers for ``config.n_runs`` days.

    The population is drawn once; each run is an independent day of the same
    type. Spread across runs is the day-to-day (planning) uncertainty; spread
    across drivers within a run is the SoC percentile band. ASSUMPTION A14:
    run bands cover day-to-day randomness only, not parameter uncertainty.

    Args:
        config: Model settings, including ``n_drivers``, ``n_runs`` and ``seed``.
        archetypes: Archetypes to use; read from ``config.archetypes_path`` if None.
        dists: Empirical samplers; loaded from the package if None.

    Returns:
        Per-run and summarised profiles.
    """
    archetypes = (
        load_archetypes(config.archetypes_path) if archetypes is None else archetypes
    )
    dists = load_distributions() if dists is None else dists
    grid = TimeGrid.from_config(config)

    # One stream builds the population, one per run simulates a day.
    pop_rng, *run_rngs = spawn_generators(config.seed, config.n_runs + 1)
    pop = build_population(archetypes, config, dists, pop_rng)
    groups = driver_groups(pop, archetypes)

    logger.info(
        "Running %d %s days for %d drivers (seed %d)",
        config.n_runs,
        config.day_type,
        len(pop),
        config.seed,
    )
    frames = []
    for run, rng in enumerate(run_rngs):
        day = simulate_day(pop, config.day_type, rng, config, dists)
        profiles = build_profiles(day, pop, grid, config)
        frames.append(
            aggregate_profiles(profiles, pop, groups, grid).with_columns(run=run)
        )
    runs = pl.concat(frames)
    return PopulationResult(runs, summarise_runs(runs), pop, archetypes, grid, config)


def summarise_runs(runs: pl.DataFrame) -> pl.DataFrame:
    """Across-run mean and percentile band for every metric, by group and bin.

    SoC and time-left metrics are NaN in a run where nobody in the group is
    plugged in at that time. Those runs are skipped rather than allowed to
    turn the whole bin NaN, so the summary is "across runs where anyone was
    plugged in".
    """
    keys = ["group", "bin", "window_hour", "clock", "n_drivers"]
    metrics = [c for c in runs.columns if c not in {*keys, "run"}]
    lo, hi = RUN_BAND_PERCENTILES
    return (
        runs.with_columns(pl.col(metrics).fill_nan(None))
        .group_by(keys, maintain_order=True)
        .agg(
            *[pl.col(m).mean().alias(f"{m}") for m in metrics],
            *[pl.col(m).quantile(lo / 100).alias(f"{m}_run_p{lo}") for m in metrics],
            *[pl.col(m).quantile(hi / 100).alias(f"{m}_run_p{hi}") for m in metrics],
        )
        .sort(keys)
    )


def simulate_driver_days(
    driver: Driver,
    n_days: int,
    config: SimulationConfig,
    dists: Distributions | None = None,
    step_minutes: int = DRIVER_STEP_MINUTES,
) -> pl.DataFrame:
    """Simulate ``n_days`` independent days for one driver, on a fine grid.

    Args:
        driver: The driver to simulate.
        n_days: Number of independent days (each is a separate draw).
        config: Model settings; ``seed``, ``day_type`` and
            ``driver_window_start_hour`` are used.
        dists: Empirical samplers; loaded from the package if None.
        step_minutes: Profile resolution.

    Returns:
        One row per day and time step: plugged in, SoC and grid power.
    """
    dists = load_distributions() if dists is None else dists
    # One driver's days use their own window (see config) so sessions appear
    # in the order they happen.
    config = config.model_copy(
        update={"window_start_hour": config.driver_window_start_hour}
    )
    grid = TimeGrid.from_config(config, step_minutes)
    # The same driver repeated: days are i.i.d., so one vectorised call.
    pop = Population.from_drivers([driver] * n_days)
    day = simulate_day(
        pop, config.day_type, np.random.default_rng(config.seed), config, dists
    )
    profiles = build_profiles(day, pop, grid, config)
    n_bins = grid.n_bins
    return pl.DataFrame(
        {
            "day": np.repeat(np.arange(n_days), n_bins),
            "window_hour": np.tile(grid.edges[:-1], n_days),
            "clock": grid.clock_labels() * n_days,
            "plugged": profiles.plugged.ravel(),
            "soc": profiles.soc.ravel(),
            "power_kw": profiles.power_kw.ravel(),
        }
    )
