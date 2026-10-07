"""Extract the CNZ report's bar charts used to validate the simulator.

Figures 5 to 9 are vector graphics, so bar heights are read exactly from the
PDF drawing commands (no image processing). The y axis is calibrated from
two tick labels per chart. Figures 5 and 6 overlay two series: the darker
colour is where they overlap, so each series' value is the top of its own
colour if present, otherwise the top of the overlap bar.

Writes CSVs to ``src/ev_charging_simulator/data/validation/``. Run with::

    uv run python scripts/digitise_validation.py
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import polars as pl
import pymupdf

from ev_charging_simulator.config import REPO_ROOT

logger = logging.getLogger("digitise_validation")

PDF_PATH = REPO_ROOT / "data/inputs/Intelligent-Octopus-CNZ-Report-May-2022.pdf"
OUT_DIR = REPO_ROOT / "src/ev_charging_simulator/data/validation"

Colour = tuple[float, float, float]
OVERLAP: Colour = (0.275, 0.318, 0.667)
BLUE: Colour = (0.373, 0.557, 0.851)
PURPLE: Colour = (0.678, 0.529, 0.792)
NAVY: Colour = (0.176, 0.208, 0.29)
BASELINE_TOL = 1.5  # pt


@dataclass(frozen=True)
class BarChart:
    """Where a chart sits on its page and how to read its y axis.

    Attributes:
        name: Output file stem.
        page: Zero-based page index.
        clip: (x0, y0, x1, y1) region holding the bars, in PDF points.
        zero_y: y of the 0 tick label centre.
        top: (value, y) of the highest tick label.
        series: Output column name -> colour of that series' own bars.
    """

    name: str
    page: int
    clip: tuple[float, float, float, float]
    zero_y: float
    top: tuple[float, float]
    series: dict[str, Colour]


CHARTS = [
    # Fig 5: median plug duration (h) by half-hour of plug-in, 00:00 to 23:30.
    BarChart(
        "fig5_median_duration",
        11,
        (280, 130, 560, 368),
        365.8,
        (16, 144.5),
        {"weekday": BLUE, "weekend": PURPLE},
    ),
    # Fig 6: plug duration (whole hours, 0 to 48), split by "next day is weekend".
    BarChart(
        "fig6_duration",
        11,
        (78, 440, 560, 680),
        677.5,
        (0.20, 456.2),
        {"next_day_weekday": BLUE, "next_day_weekend": PURPLE},
    ),
    # Fig 7: SoC at plug-in, 5% bins.
    BarChart(
        "fig7_plug_in_soc",
        12,
        (105, 260, 560, 480),
        477.8,
        (0.08, 280.9),
        {"probability": NAVY},
    ),
    # Fig 8: share of overnight plug-in time spent charging, 5% bins.
    BarChart(
        "fig8_time_charging",
        13,
        (85, 500, 560, 720),
        716.7,
        (0.10, 532.9),
        {"probability": BLUE},
    ),
    # Fig 9: overnight top-up (% SoC added), 5% bins.
    BarChart(
        "fig9_overnight_topup",
        14,
        (105, 280, 560, 500),
        498.5,
        (0.10, 301.6),
        {"probability": (0.855, 1.0, 0.659)},
    ),
]


def _close(a: Colour, b: Colour) -> bool:
    return max(abs(x - y) for x, y in zip(a, b, strict=True)) < 0.01


def extract(doc: pymupdf.Document, chart: BarChart) -> pl.DataFrame:
    """Bar values of one chart, ordered left to right."""
    clip = pymupdf.Rect(chart.clip)
    rects = [
        (d["rect"], tuple(d["fill"]))
        for d in doc[chart.page].get_drawings()
        if d.get("fill") is not None
        and d["rect"].intersects(clip)
        and d["rect"].width < 60
    ]
    # Group stacked rectangles by bar position.
    bars: dict[int, list] = defaultdict(list)
    for rect, fill in rects:
        bars[round((rect.x0 + rect.x1) / 2)].append((rect, fill))
    # Keep groups standing on the baseline (drops legend swatches).
    baseline = max(r.y1 for r, _ in rects)
    positions = sorted(
        x
        for x, group in bars.items()
        if any(abs(r.y1 - baseline) < BASELINE_TOL for r, _ in group)
    )
    scale = chart.top[0] / (chart.zero_y - chart.top[1])

    def value(group: list, colour: Colour) -> float:
        own = [r.y0 for r, f in group if _close(f, colour)]
        overlap = [r.y0 for r, f in group if _close(f, OVERLAP)]
        top = min(own) if own else min(overlap)
        return (baseline - top) * scale

    columns = {
        name: [value(bars[x], c) for x in positions] for name, c in chart.series.items()
    }
    return pl.DataFrame({"bar": np.arange(len(positions)), **columns})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF_PATH)
    for chart in CHARTS:
        frame = extract(doc, chart)
        frame.write_csv(OUT_DIR / f"{chart.name}.csv", float_precision=5)
        sums = {c: round(frame[c].sum(), 3) for c in chart.series}
        logger.info("%s: %d bars, column sums %s", chart.name, frame.height, sums)


if __name__ == "__main__":
    main()
