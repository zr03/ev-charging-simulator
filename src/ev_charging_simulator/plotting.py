"""Plotly figures shared by the marimo app and the CLI script."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ev_charging_simulator.simulation import (
    ALL_DRIVERS,
    PopulationResult,
    nan_percentiles,
)

# Percentiles over fewer plugged-in cars than this are too noisy to show.
MIN_PLUGGED_FOR_PERCENTILES = 20
TICK_EVERY_HOURS = 3

COLOURS = {
    "main": "#1f6f8b",
    "band": "rgba(31, 111, 139, 0.18)",
    "run_band": "rgba(230, 126, 34, 0.25)",
    "run_line": "#e67e22",
    "power": "#c0392b",
}
ARCHETYPE_COLOURS = ["#1f6f8b", "#e67e22", "#27ae60", "#8e44ad", "#c0392b", "#7f8c8d"]


def _time_axis(fig: go.Figure, frame: pl.DataFrame) -> None:
    """Label every window-hour x axis with clock times every few hours."""
    ticks = frame.filter(pl.col("window_hour") % TICK_EVERY_HOURS == 0).unique(
        "window_hour", maintain_order=True
    )
    fig.update_xaxes(
        tickvals=ticks["window_hour"].to_list(),
        ticktext=ticks["clock"].to_list(),
        range=[0, 24],
    )


def _band(
    fig: go.Figure,
    x: list[float],
    lower: list[float | None],
    upper: list[float | None],
    colour: str,
    name: str,
    row: int,
    showlegend: bool = True,
) -> None:
    """Shaded band between two lines."""
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            legendgroup=name,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=colour,
            name=name,
            legendgroup=name,
            showlegend=showlegend,
        ),
        row=row,
        col=1,
    )


def _bin_centres(frame: pl.DataFrame, step_hours: float) -> list[float]:
    return (frame["window_hour"] + step_hours / 2).to_list()


def driver_figure(days: pl.DataFrame, title: str) -> go.Figure:
    """Charging power for one driver over several independent days.

    One row per day: the shaded area is when the car is plugged in, the line
    is the power it draws. Days are independent draws, so they are shown
    separately rather than joined end to end.

    Args:
        days: Output of :func:`simulate_driver_days`.
        title: Figure title.
    """
    n_days = days["day"].n_unique()
    fig = make_subplots(
        rows=n_days,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.25 / max(n_days, 1),
        subplot_titles=[f"Day {d + 1}" for d in range(n_days)],
    )
    # Shading reaches just above the highest power so the line sits inside it.
    top = max(float(np.nanmax(days["power_kw"].to_numpy())), 1.0) * 1.1
    for row, (_, frame) in enumerate(days.group_by("day", maintain_order=True), 1):
        x = frame["window_hour"].to_list()
        fig.add_trace(
            go.Scatter(
                x=x,
                # Plugged in is read at bin midpoints; also shade a bin that
                # draws power, so a session starting mid-bin is covered.
                y=(
                    (frame["plugged"] | (frame["power_kw"] > 0)).cast(pl.Float64) * top
                ).to_list(),
                mode="lines",
                line={"width": 0, "shape": "hv"},
                fill="tozeroy",
                fillcolor=COLOURS["band"],
                name="Plugged in",
                legendgroup="plugged",
                showlegend=row == 1,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=frame["power_kw"].to_list(),
                mode="lines",
                line={"color": COLOURS["power"], "shape": "hv"},
                name="Charging power (kW)",
                legendgroup="power",
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(range=[0, top], title_text="kW", row=row, col=1)
    _time_axis(fig, days)
    fig.update_layout(
        title=f"{title}: {n_days} independent days",
        height=140 * n_days + 160,
        hovermode="x unified",
        template="plotly_white",
        legend={
            "orientation": "h",
            "y": 1.0,
            "yanchor": "bottom",
            "x": 1,
            "xanchor": "right",
        },
    )
    return fig


def driver_many_days_figure(days: pl.DataFrame, title: str) -> go.Figure:
    """One driver's typical day, summarised over many independent days.

    Faint lines are individual days; the band and median are across the days
    on which the car is plugged in at that time, so they show "if plugged
    in, what SoC", matching the population view.
    """
    n_days = days["day"].n_unique()
    step = days["window_hour"][1] - days["window_hour"][0]
    first = days.filter(pl.col("day") == 0)
    x = _bin_centres(first, step)
    n_bins = len(x)
    soc = days["soc"].to_numpy().reshape(n_days, n_bins) * 100
    plugged = days["plugged"].to_numpy().reshape(n_days, n_bins).astype(float)
    power = days["power_kw"].to_numpy().reshape(n_days, n_bins)

    # Percentiles only where enough days are plugged in to be meaningful:
    # at least 20 days and 10% of them, so rare daytime sessions don't
    # produce a jagged band.
    p5, p50, p95 = nan_percentiles(soc, (5, 50, 95))
    min_days = max(MIN_PLUGGED_FOR_PERCENTILES, 0.1 * n_days)
    enough = np.sum(~np.isnan(soc), axis=0) >= min_days

    def masked(values: np.ndarray) -> list[float | None]:
        return [float(v) if ok else None for v, ok in zip(values, enough, strict=True)]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Chance of being plugged in (%)",
            "State of charge while plugged in (%)",
            "Expected charging power (kW, mean over days)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=(plugged.mean(axis=0) * 100).tolist(),
            name="Plugged in",
            line={"color": COLOURS["main"]},
        ),
        row=1,
        col=1,
    )
    # All days in one trace, separated by gaps, keeps the figure light.
    xs: list[float | None] = []
    ys: list[float | None] = []
    for row in soc:
        xs += [*x, None]
        ys += [None if np.isnan(v) else float(v) for v in row] + [None]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": "rgba(31, 111, 139, 0.12)", "width": 1},
            name="Individual days",
            hoverinfo="skip",
            connectgaps=False,
        ),
        row=2,
        col=1,
    )
    _band(fig, x, masked(p5), masked(p95), COLOURS["band"], "5th to 95th pct", row=2)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=masked(p50),
            name="Median",
            line={"color": COLOURS["main"], "width": 2.5},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=first["window_hour"].to_list(),
            y=power.mean(axis=0).tolist(),
            name="Mean power",
            line={"color": COLOURS["power"], "shape": "hv"},
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="%", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="SoC (%)", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="kW", rangemode="tozero", row=3, col=1)
    _time_axis(fig, days)
    fig.update_layout(
        title=f"{title} (typical day from {n_days} independent days)",
        height=780,
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def population_figure(result: PopulationResult, group: str = ALL_DRIVERS) -> go.Figure:
    """Share plugged in, SoC of plugged-in cars, demand and shiftable energy.

    Demand is shown twice: unmanaged (each car charges as soon as it plugs
    in) and as late as possible (each car starts at the last moment that
    still delivers the same energy by plug-out). The bottom panel is the
    energy that can be moved later than each time, the gap between the two.
    Demand and deferrable energy are totals for the group's EVs, since grid
    flexibility is managed in aggregate.

    Orange bands are the 5th to 95th percentile across runs (the day-to-day
    range a planner should expect).

    Args:
        result: Population run.
        group: ``ALL_DRIVERS`` or an archetype name.
    """
    frame = result.summary.filter(pl.col("group") == group)
    x = _bin_centres(frame, result.grid.step_hours)
    n_evs = int(frame["n_drivers"][0])
    unit = f"total for {n_evs:,} EVs"

    def scaled(col: str) -> list[float | None]:
        # Per-EV means times the group's size give the group total.
        return (frame[col] * n_evs).to_list()

    def pct(col: str) -> list[float | None]:
        return (frame[col] * 100).to_list()

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Drivers plugged in (%)",
            "Combined SoC of plugged-in cars: kWh stored / kWh capacity (%)",
            f"Charging demand (kW, {unit})",
            f"Charging that could be deferred to a later time (kWh, {unit})",
        ),
    )
    # 1. Share plugged in.
    _band(
        fig,
        x,
        (frame["share_plugged_run_p5"] * 100).to_list(),
        (frame["share_plugged_run_p95"] * 100).to_list(),
        COLOURS["run_band"],
        "Across runs, 5th to 95th pct",
        row=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=(frame["share_plugged"] * 100).to_list(),
            name="Drivers plugged in",
            line={"color": COLOURS["main"]},
        ),
        row=1,
        col=1,
    )

    # 2. Combined SoC of the plugged-in cars (kWh stored / kWh capacity),
    # with its spread across runs.
    _band(
        fig,
        x,
        pct("soc_fleet_run_p5"),
        pct("soc_fleet_run_p95"),
        COLOURS["run_band"],
        "Across runs",
        row=2,
        showlegend=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pct("soc_fleet"),
            name="Combined SoC of plugged-in cars",
            line={"color": COLOURS["main"]},
        ),
        row=2,
        col=1,
    )

    # 3. Demand: unmanaged against the same sessions charged as late as
    # possible. The run band is drawn for unmanaged only, to keep it legible.
    _band(
        fig,
        x,
        scaled("power_kw_per_ev_run_p5"),
        scaled("power_kw_per_ev_run_p95"),
        COLOURS["run_band"],
        "Across runs",
        row=3,
        showlegend=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=scaled("power_kw_per_ev"),
            name="Demand, unmanaged",
            line={"color": COLOURS["power"]},
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=scaled("power_alap_kw_per_ev"),
            name="Demand, charged as late as possible",
            line={"color": COLOURS["main"], "dash": "dash"},
        ),
        row=3,
        col=1,
    )

    # 4. Shiftable energy: what the unmanaged fleet has already charged by
    # this time that the latest-possible schedule has not yet needed to.
    _band(
        fig,
        x,
        scaled("shiftable_kwh_per_ev_run_p5"),
        scaled("shiftable_kwh_per_ev_run_p95"),
        COLOURS["run_band"],
        "Across runs",
        row=4,
        showlegend=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=scaled("shiftable_kwh_per_ev"),
            name="Charging that could be deferred",
            line={"color": COLOURS["main"]},
        ),
        row=4,
        col=1,
    )

    fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_yaxes(rangemode="tozero", row=3, col=1)
    fig.update_yaxes(rangemode="tozero", row=4, col=1)
    _time_axis(fig, frame)
    n_drivers = frame["n_drivers"][0]
    fig.update_layout(
        title=(
            f"{group}: {n_drivers} drivers, {result.config.n_runs} "
            f"{result.config.day_type} runs"
        ),
        height=1050,
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def archetype_demand_figure(result: PopulationResult) -> go.Figure:
    """Fleet demand, stacked by archetype."""
    frame = result.summary.filter(pl.col("group") != ALL_DRIVERS)
    total = result.summary.filter(pl.col("group") == ALL_DRIVERS)["n_drivers"][0]
    fig = go.Figure()
    for i, name in enumerate(a.name for a in result.archetypes):
        part = frame.filter(pl.col("group") == name)
        if part.is_empty():
            continue
        # Group total = per-EV mean x number of drivers in the group.
        fig.add_trace(
            go.Scatter(
                x=_bin_centres(part, result.grid.step_hours),
                y=(part["power_kw_per_ev"] * part["n_drivers"][0]).to_list(),
                name=name,
                stackgroup="demand",
                line={"color": ARCHETYPE_COLOURS[i % len(ARCHETYPE_COLOURS)]},
            )
        )
    _time_axis(fig, frame)
    fig.update_layout(
        title=f"Unmanaged demand by archetype, stacked (total for {total:,} EVs)",
        yaxis_title="kW",
        height=450,
        hovermode="x unified",
        template="plotly_white",
    )
    return fig
