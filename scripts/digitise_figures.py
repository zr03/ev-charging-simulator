"""Digitise the CNZ report heatmaps used as simulator inputs.

Figures 2 and 4 are embedded in the PDF as raster images on a regular grid,
with a vector-drawn colour bar beside each. For every grid cell we take the
centre pixel and find the colour-bar position with the closest luminance,
which maps linearly to a percentage.

Outputs (written to ``src/ev_charging_simulator/data/``):
    fig4_weekday.csv, fig4_weekend.csv: plug-in hour x plug-out hour, % of events.
    fig2_target_soc.csv: target SoC x deadline, % of customers.

A side-by-side check plot is written to ``figures/digitised_check.png``.

Usage:
    uv run python scripts/digitise_figures.py
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymupdf
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
PDF = REPO / "data/inputs/Intelligent-Octopus-CNZ-Report-May-2022.pdf"
OUT_DIR = REPO / "src/ev_charging_simulator/data"
CHECK_PLOT = REPO / "figures/digitised_check.png"

RENDER_DPI = 600


@dataclass(frozen=True)
class HeatmapSpec:
    """Where a heatmap and its colour bar sit in the PDF.

    Colour-bar geometry is in PDF points. ``bar_x`` is a vertical line through
    the bar; ``top_pt``/``bottom_pt`` are the y-centres of the top and bottom
    tick labels, with values ``top_value``/``bottom_value`` (percent).
    """

    name: str
    page: int  # zero-based
    xref: int  # image object id in the PDF
    n_rows: int
    n_cols: int
    bar_x: float
    bar_y_range: tuple[float, float]
    top_pt: float
    top_value: float
    bottom_pt: float
    bottom_value: float


# Geometry read from the PDF's text layer (tick-label bounding boxes).
FIG4_WEEKDAY = HeatmapSpec(
    name="fig4_weekday", page=10, xref=109, n_rows=24, n_cols=24,
    bar_x=521.0, bar_y_range=(405.0, 570.0),
    top_pt=421.05, top_value=2.5, bottom_pt=556.9, bottom_value=0.0,
)  # fmt: skip
FIG4_WEEKEND = HeatmapSpec(
    name="fig4_weekend", page=10, xref=113, n_rows=24, n_cols=24,
    bar_x=521.0, bar_y_range=(585.0, 750.0),
    top_pt=601.3, top_value=2.5, bottom_pt=737.2, bottom_value=0.0,
)  # fmt: skip
FIG2 = HeatmapSpec(
    name="fig2", page=9, xref=102, n_rows=9, n_cols=16,
    bar_x=515.0, bar_y_range=(455.0, 700.0),
    top_pt=472.0, top_value=8.0, bottom_pt=683.4, bottom_value=0.0,
)  # fmt: skip

# Fig 2 axes as printed. Note the deadline axis skips 02:30.
FIG2_TARGETS = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
FIG2_DEADLINES = ["02:00"] + [f"{h:02d}:{m:02d}" for h in range(3, 10) for m in (0, 30)]
FIG2_DEADLINES += ["10:00"]


def _render_rgb(page: pymupdf.Page, clip: pymupdf.Rect) -> NDArray[np.int32]:
    """Render a clipped page region to an RGB array."""
    pix = page.get_pixmap(clip=clip, dpi=RENDER_DPI)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    return arr.reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(np.int32)


def read_colour_bar(
    doc: pymupdf.Document, spec: HeatmapSpec
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Sample the colour bar as (colours, values in percent), top to bottom."""
    page = doc[spec.page]
    y0, y1 = spec.bar_y_range
    clip = pymupdf.Rect(spec.bar_x - 1, y0, spec.bar_x + 1, y1)
    column = _render_rgb(page, clip)[:, 1]
    coloured = np.where(column.sum(axis=1) < 750)[0]
    colours = column[coloured.min() : coloured.max() + 1]
    scale = RENDER_DPI / 72
    y_pt = y0 + (np.arange(coloured.min(), coloured.max() + 1) + 0.5) / scale
    # Linear interpolation between the two tick labels; the bar may run
    # slightly past the top label, which simply extrapolates.
    slope = (spec.top_value - spec.bottom_value) / (spec.top_pt - spec.bottom_pt)
    values = spec.bottom_value + (y_pt - spec.bottom_pt) * slope
    return colours, values


def read_heatmap(doc: pymupdf.Document, spec: HeatmapSpec) -> NDArray[np.float64]:
    """Return the heatmap as an (n_rows, n_cols) array in percent, top row first."""
    pix = pymupdf.Pixmap(doc, spec.xref)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(np.int32)
    ys = ((np.arange(spec.n_rows) + 0.5) * pix.height / spec.n_rows).astype(int)
    xs = ((np.arange(spec.n_cols) + 0.5) * pix.width / spec.n_cols).astype(int)
    cells = img[ys][:, xs]

    bar_colours, bar_values = read_colour_bar(doc, spec)
    # Match on luminance, not full RGB: the embedded image is more saturated
    # than the rendered bar (up to ~40 RGB units off), but lightness agrees.
    cell_lum = _relative_luminance(cells)
    bar_lum = _relative_luminance(bar_colours)
    gap = np.abs(cell_lum[:, :, None] - bar_lum[None, None, :])
    nearest = gap.argmin(axis=-1)
    logger.info("%s: worst luminance mismatch %.3f", spec.name, gap.min(-1).max())
    return np.clip(bar_values[nearest], 0.0, None)


def _relative_luminance(rgb: NDArray[np.int32]) -> NDArray[np.float64]:
    """sRGB relative luminance (0 to 1) of an array of RGB triples."""
    c = rgb / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return linear @ np.array([0.2126, 0.7152, 0.0722])


def fig4_to_frame(grid: NDArray[np.float64]) -> pl.DataFrame:
    """Convert a Fig 4 grid (top row = plug-out 23) to tidy in/out-hour rows."""
    # Image rows run plug-out hour 23..0 top to bottom; columns plug-in 0..23.
    out_hours = np.arange(23, -1, -1)
    rows = [
        {"plug_in_hour": i, "plug_out_hour": int(out_hours[r]), "pct": grid[r, i]}
        for r in range(24)
        for i in range(24)
    ]
    return pl.DataFrame(rows).sort(["plug_in_hour", "plug_out_hour"])


def fig2_to_frame(grid: NDArray[np.float64]) -> pl.DataFrame:
    """Convert the Fig 2 grid to tidy target-SoC/deadline rows."""
    rows = [
        {"target_soc": t, "deadline": dl, "pct": grid[r, c]}
        for r, t in enumerate(FIG2_TARGETS)
        for c, dl in enumerate(FIG2_DEADLINES)
    ]
    return pl.DataFrame(rows)


def save_check_plot(
    grids: dict[str, NDArray[np.float64]], doc: pymupdf.Document
) -> None:
    """Plot each digitised grid next to the original image."""
    specs = {s.name: s for s in (FIG4_WEEKDAY, FIG4_WEEKEND, FIG2)}
    fig, axes = plt.subplots(len(grids), 2, figsize=(10, 4 * len(grids)))
    for row, (name, grid) in enumerate(grids.items()):
        pix = pymupdf.Pixmap(doc, specs[name].xref)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        axes[row, 0].imshow(img[:, :, :3], aspect="auto")
        axes[row, 0].set_title(f"{name}: original")
        im = axes[row, 1].imshow(
            grid, cmap="Blues", aspect="auto", vmin=0, vmax=specs[name].top_value
        )
        axes[row, 1].set_title(f"{name}: digitised (%)")
        fig.colorbar(im, ax=axes[row, 1])
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    CHECK_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHECK_PLOT, dpi=100)
    logger.info("Check plot written to %s", CHECK_PLOT)


def main() -> None:
    """Digitise Figures 2 and 4 and write CSVs plus a check plot."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    doc = pymupdf.open(PDF)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grids: dict[str, NDArray[np.float64]] = {}
    for spec in (FIG4_WEEKDAY, FIG4_WEEKEND):
        grid = read_heatmap(doc, spec)
        grids[spec.name] = grid
        frame = fig4_to_frame(grid)
        frame.with_columns(pl.col("pct").round(4)).write_csv(
            OUT_DIR / f"{spec.name}.csv"
        )
        logger.info("%s: cells sum to %.1f%%", spec.name, frame["pct"].sum())

    grid2 = read_heatmap(doc, FIG2)
    grids[FIG2.name] = grid2
    frame2 = fig2_to_frame(grid2)
    frame2.with_columns(pl.col("pct").round(4)).write_csv(
        OUT_DIR / "fig2_target_soc.csv"
    )
    logger.info("fig2: cells sum to %.1f%%", frame2["pct"].sum())
    # Report text: 80% / 90% / 100% targets are 25% / 24% / 23% of preferences.
    by_target = frame2.group_by("target_soc").agg(pl.col("pct").sum())
    logger.info(
        "fig2 target marginals (%%): %s", dict(by_target.sort("target_soc").rows())
    )

    save_check_plot(grids, doc)


if __name__ == "__main__":
    main()
