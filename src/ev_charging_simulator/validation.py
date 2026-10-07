"""Compare simulated sessions with CNZ report figures held out from fitting.

Inputs to the model are Fig 4 (session times), Fig 2 (target SoC), Table 2
(via p) and the archetype sheet. Figs 5 to 9 are not used to fit anything, so
they are independent checks. The report's data are Intelligent Octopus
customers, so the natural comparison is that archetype.

Caveats (PLAN §8): IO charging is smart, not unmanaged, but plug-in SoC,
top-up and plug durations don't depend on when charging happens. Fig 8
(share of plugged-in time spent charging) also holds if charging power is
the same whether smart or not.
"""

import logging
from dataclasses import dataclass
from importlib import resources

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ev_charging_simulator.config import DayType, SimulationConfig
from ev_charging_simulator.models.archetypes import Archetype
from ev_charging_simulator.models.charging import hours_to_reach
from ev_charging_simulator.models.distributions import Distributions
from ev_charging_simulator.models.driver import build_population, simulate_day
from ev_charging_simulator.utils.clock import HOURS_PER_DAY
from ev_charging_simulator.utils.seeding import spawn_generators

logger = logging.getLogger(__name__)

VALIDATION_PACKAGE = "ev_charging_simulator.data.validation"
DEFAULT_ARCHETYPE = "Intelligent Octopus average"
SOC_BIN = 0.05
MIN_SESSIONS_FOR_MEDIAN = 30


@dataclass(frozen=True)
class Sessions:
    """Simulated sessions pooled over many days (one entry per session)."""

    plug_in_clock: np.ndarray  # hours since midnight
    duration_hours: np.ndarray
    soc_in: np.ndarray
    soc_out: np.ndarray
    hours_charging: np.ndarray
    overnight: np.ndarray  # bool


def simulate_sessions(
    archetype: Archetype,
    config: SimulationConfig,
    dists: Distributions,
    day_type: DayType = "weekday",
    n_drivers: int = 2000,
    n_days: int = 20,
) -> Sessions:
    """Pool sessions from ``n_days`` simulated days of one archetype."""
    one = archetype.model_copy(update={"population_share": 1.0})
    sized = config.model_copy(update={"n_drivers": n_drivers})
    pop_rng, *day_rngs = spawn_generators(config.seed, n_days + 1)
    pop = build_population([one], sized, dists, pop_rng)

    parts: dict[str, list[np.ndarray]] = {k: [] for k in Sessions.__dataclass_fields__}
    for rng in day_rngs:
        day = simulate_day(pop, day_type, rng, config, dists)
        for start, hours, s_in, s_out, overnight in (
            (
                day.overnight_in,
                day.overnight_hours,
                day.overnight_soc_in,
                day.overnight_soc_out,
                True,
            ),
            (
                day.same_day_in,
                day.same_day_hours,
                day.same_day_soc_in,
                day.same_day_soc_out,
                False,
            ),
        ):
            has = ~np.isnan(start)
            parts["plug_in_clock"].append(
                (start[has] + config.window_start_hour) % HOURS_PER_DAY
            )
            parts["duration_hours"].append(hours[has])
            parts["soc_in"].append(s_in[has])
            parts["soc_out"].append(s_out[has])
            parts["hours_charging"].append(
                hours_to_reach(
                    s_in[has],
                    s_out[has],
                    pop.battery_kwh[has],
                    pop.charger_kw[has],
                    config.charging,
                )
            )
            parts["overnight"].append(np.full(has.sum(), overnight))
    sessions = Sessions(**{k: np.concatenate(v) for k, v in parts.items()})
    logger.info(
        "Pooled %d %s sessions for %s", sessions.soc_in.size, day_type, archetype.name
    )
    return sessions


def load_reference(name: str) -> pl.DataFrame:
    """A digitised validation figure (see scripts/digitise_validation.py)."""
    with (
        resources.files(VALIDATION_PACKAGE).joinpath(f"{name}.csv").open("rb") as handle
    ):
        return pl.read_csv(handle)


def _histogram(values: np.ndarray, n_bins: int, width: float) -> np.ndarray:
    counts, _ = np.histogram(values, bins=np.arange(n_bins + 1) * width)
    return counts / max(values.size, 1)


def compare(sessions: Sessions) -> dict[str, pl.DataFrame]:
    """Reference vs simulated values for each validation figure.

    Returns:
        Figure name -> frame with ``x``, ``reference`` and ``simulated``.
    """
    out: dict[str, pl.DataFrame] = {}

    fig5 = load_reference("fig5_median_duration")
    half_hour = np.floor(sessions.plug_in_clock * 2).astype(int)
    medians = []
    for b in fig5["bar"].to_list():
        durations = sessions.duration_hours[half_hour == b]
        enough = durations.size >= MIN_SESSIONS_FOR_MEDIAN
        medians.append(float(np.median(durations)) if enough else None)
    out["fig5"] = pl.DataFrame(
        {"x": fig5["bar"] / 2, "reference": fig5["weekday"], "simulated": medians}
    )

    fig6 = load_reference("fig6_duration")
    out["fig6"] = pl.DataFrame(
        {
            "x": fig6["bar"].cast(pl.Float64),
            "reference": fig6["next_day_weekday"],
            "simulated": _histogram(sessions.duration_hours, fig6.height, 1.0),
        }
    )

    soc_bins = np.arange(20) * SOC_BIN
    on = sessions.overnight
    for name, ref, values in (
        ("fig7", "fig7_plug_in_soc", sessions.soc_in),
        # Fig 8 is overnight sessions only.
        (
            "fig8",
            "fig8_time_charging",
            sessions.hours_charging[on] / sessions.duration_hours[on],
        ),
        ("fig9", "fig9_overnight_topup", (sessions.soc_out - sessions.soc_in)[on]),
    ):
        reference = load_reference(ref)["probability"]
        out[name] = pl.DataFrame(
            {
                "x": soc_bins,
                "reference": reference,
                # Include values of exactly 1 in the last bin.
                "simulated": _histogram(np.minimum(values, 1 - 1e-9), 20, SOC_BIN),
            }
        )
    return out


TITLES = {
    "fig5": (
        "Fig 5: median plug duration by plug-in time (weekday)",
        "Plug-in time (h)",
        "Hours",
    ),
    "fig6": ("Fig 6: plug duration", "Duration (h)", "Share of sessions"),
    "fig7": ("Fig 7: SoC at plug-in", "SoC", "Share of sessions"),
    "fig8": ("Fig 8: share of overnight plug-in time spent charging", "Share", "Share"),
    "fig9": ("Fig 9: overnight top-up", "SoC added", "Share of sessions"),
}


def validation_figure(comparison: dict[str, pl.DataFrame], label: str) -> go.Figure:
    """Grid of reference (bars) vs simulated (line) for each figure."""
    names = list(comparison)
    fig = make_subplots(
        rows=len(names),
        cols=1,
        subplot_titles=[TITLES[n][0] for n in names],
        vertical_spacing=0.06,
    )
    for row, name in enumerate(names, start=1):
        frame = comparison[name]
        width = frame["x"][1] - frame["x"][0]
        centres = (frame["x"] + width / 2).to_list()
        fig.add_trace(
            go.Bar(
                x=centres,
                y=frame["reference"].to_list(),
                width=width * 0.9,
                name="CNZ report",
                marker_color="#9db4c0",
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=centres,
                y=frame["simulated"].to_list(),
                mode="lines+markers",
                name=f"Simulated ({label})",
                line={"color": "#c0392b"},
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        fig.update_xaxes(title_text=TITLES[name][1], row=row, col=1)
        fig.update_yaxes(title_text=TITLES[name][2], row=row, col=1)
    fig.update_layout(
        height=320 * len(names),
        template="plotly_white",
        title="Validation against held-out CNZ report figures",
    )
    return fig


def summary_table(comparison: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """One row per figure: overlap of the two distributions, or median error."""
    rows = []
    for name, frame in comparison.items():
        if name == "fig5":
            both = frame.drop_nulls()
            error = (both["simulated"] - both["reference"]).abs().mean()
            rows.append(
                {"figure": name, "metric": "mean abs error (h)", "value": error}
            )
        else:
            # Overlap of two histograms: 1 = identical, 0 = disjoint.
            ref = frame["reference"] / frame["reference"].sum()
            overlap = np.minimum(ref.to_numpy(), frame["simulated"].to_numpy()).sum()
            rows.append(
                {"figure": name, "metric": "histogram overlap", "value": overlap}
            )
    return pl.DataFrame(rows)
