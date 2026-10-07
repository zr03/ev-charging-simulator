"""Probability distributions behind driver behaviour.

Session times come from the digitised CNZ Figure 4 heatmaps, target SoC from
Figure 2, and energy use from gamma / lognormal / geometric draws calibrated
to the archetype spreadsheet. See PLAN.md sections 4 and 5.
"""

import logging
from dataclasses import dataclass
from importlib import resources

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ev_charging_simulator.config import DayType
from ev_charging_simulator.utils.clock import hours_until

logger = logging.getLogger(__name__)

DATA_PACKAGE = "ev_charging_simulator.data"

# ASSUMPTION A8: Fig 4 gives session-time shapes only. Sessions that run
# through 03:00 are "overnight"; all others are "same-day".
OVERNIGHT_ANCHOR_HOUR = 3.0

# Modal overnight cell of the weekday Fig 4 heatmap. Archetypes whose
# spreadsheet times differ (scheduled charging) shift the samples by the gap.
REFERENCE_PLUG_IN_HOUR = 18.0
REFERENCE_PLUG_OUT_HOUR = 7.0


@dataclass(frozen=True)
class SessionSample:
    """Sampled sessions: clock plug-in time and duration, both in hours."""

    plug_in_hour: NDArray[np.float64]
    duration_hours: NDArray[np.float64]


@dataclass(frozen=True)
class _CellTable:
    """Heatmap cells of one session type, with renormalised probabilities."""

    plug_in_hour: NDArray[np.int64]
    duration_hours: NDArray[np.int64]  # whole hours from plug-in to plug-out cell
    prob: NDArray[np.float64]

    def sample(self, rng: np.random.Generator, n: int) -> SessionSample:
        """Draw cells, then jitter plug-in and plug-out uniformly within the hour."""
        idx = rng.choice(self.prob.size, size=n, p=self.prob)
        u_in, u_out = rng.random(n), rng.random(n)
        whole = self.duration_hours[idx]
        # In a same-hour cell the plug-out must come after the plug-in.
        same_hour = whole == 0
        u_in[same_hour], u_out[same_hour] = (
            np.minimum(u_in, u_out)[same_hour],
            np.maximum(u_in, u_out)[same_hour],
        )
        return SessionSample(
            plug_in_hour=self.plug_in_hour[idx] + u_in,
            duration_hours=whole + u_out - u_in,
        )


class SessionTimeSampler:
    """Samples overnight and same-day session times for one day type (Fig 4).

    Fig 4 is used for the *shape* of session times only; how often each
    session type happens is set by the plug-in model (p, q).
    """

    def __init__(self, day_type: DayType) -> None:
        cells = _read_csv(f"fig4_{day_type}.csv")
        plug_in = cells["plug_in_hour"].to_numpy()
        duration = hours_until(plug_in, cells["plug_out_hour"].to_numpy()).astype(int)
        weight = cells["pct"].to_numpy()
        # A session spans 03:00 if 03:00 falls within its whole-hour cells.
        to_anchor = hours_until(plug_in, OVERNIGHT_ANCHOR_HOUR).astype(int)
        overnight = (to_anchor >= 1) & (to_anchor <= duration)

        self.day_type = day_type
        self.overnight = _cell_table(plug_in, duration, weight, overnight)
        self.same_day = _cell_table(plug_in, duration, weight, ~overnight)
        share = weight[overnight].sum() / weight.sum()
        logger.debug("Fig 4 %s: %.0f%% of events are overnight", day_type, 100 * share)

    def sample_overnight(self, rng: np.random.Generator, n: int) -> SessionSample:
        """Draw ``n`` overnight sessions."""
        return self.overnight.sample(rng, n)

    def sample_same_day(self, rng: np.random.Generator, n: int) -> SessionSample:
        """Draw ``n`` same-day sessions."""
        return self.same_day.sample(rng, n)


class TargetSocSampler:
    """Samples a driver's target SoC from Fig 2, deadline axis collapsed."""

    def __init__(self) -> None:
        marginal = (
            _read_csv("fig2_target_soc.csv")
            .group_by("target_soc")
            .agg(pl.col("pct").sum())
            .sort("target_soc")
        )
        self.targets = marginal["target_soc"].to_numpy()
        weights = marginal["pct"].to_numpy()
        self.prob = weights / weights.sum()

    def sample(self, rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        """Draw ``n`` target SoCs (fractions)."""
        return rng.choice(self.targets, size=n, p=self.prob)


@dataclass(frozen=True)
class Distributions:
    """All empirical samplers, loaded once and shared."""

    session_times: dict[DayType, SessionTimeSampler]
    target_soc: TargetSocSampler


def load_distributions() -> Distributions:
    """Load the digitised figure data bundled with the package."""
    return Distributions(
        session_times={d: SessionTimeSampler(d) for d in ("weekday", "weekend")},
        target_soc=TargetSocSampler(),
    )


def sample_annual_mileage(
    rng: np.random.Generator, mean: NDArray[np.float64], cv: float
) -> NDArray[np.float64]:
    """Lognormal annual mileage with the given mean and coefficient of variation.

    The *mean* (not median) matches the spreadsheet, so annual energy for the
    archetype is preserved; the median sits a little lower (right skew).
    """
    if cv == 0:
        return mean.astype(float).copy()
    sigma2 = np.log1p(cv**2)
    mu = np.log(mean) - sigma2 / 2
    return rng.lognormal(mu, np.sqrt(sigma2))


def sample_soc_drop(
    rng: np.random.Generator,
    days: NDArray[np.float64],
    mean_daily_drop: NDArray[np.float64],
    shape: float,
) -> NDArray[np.float64]:
    """Total SoC drop over ``days`` days of driving.

    Daily drops are Gamma(shape, mean/shape); a sum of ``days`` of them is
    Gamma(days * shape, mean/shape), so one draw covers the whole gap.
    Zero days or zero mean gives zero drop.
    """
    total_shape = days * shape
    scale = mean_daily_drop / shape
    drop = np.zeros_like(scale, dtype=float)
    active = (total_shape > 0) & (scale > 0)
    drop[active] = rng.gamma(total_shape[active], scale[active])
    return drop


def _read_csv(name: str) -> pl.DataFrame:
    """Read a CSV shipped in the package's data folder."""
    with resources.files(DATA_PACKAGE).joinpath(name).open("rb") as handle:
        return pl.read_csv(handle)


def _cell_table(
    plug_in: NDArray[np.int64],
    duration: NDArray[np.int64],
    weight: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> _CellTable:
    w = weight[mask]
    return _CellTable(plug_in[mask], duration[mask], w / w.sum())
