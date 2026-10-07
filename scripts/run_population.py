"""Run the population simulation from the command line.

Writes interactive HTML figures to ``figures/`` and tables to
``data/outputs/``. Example::

    uv run python scripts/run_population.py --n-drivers 1000 --n-runs 200
"""

import argparse
import logging

import numpy as np

from ev_charging_simulator.config import REPO_ROOT, SimulationConfig
from ev_charging_simulator.models.archetypes import load_archetypes
from ev_charging_simulator.models.distributions import load_distributions
from ev_charging_simulator.models.driver import sample_drivers
from ev_charging_simulator.plotting import (
    archetype_demand_figure,
    driver_figure,
    population_figure,
)
from ev_charging_simulator.simulation import run_population, simulate_driver_days

logger = logging.getLogger("run_population")

FIGURES_DIR = REPO_ROOT / "figures"
OUTPUTS_DIR = REPO_ROOT / "data/outputs"
EXAMPLE_DRIVER_DAYS = 5


def parse_args() -> argparse.Namespace:
    defaults = SimulationConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-drivers", type=int, default=defaults.n_drivers)
    parser.add_argument("--n-runs", type=int, default=defaults.n_runs)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--day-type", choices=["weekday", "weekend"], default=defaults.day_type
    )
    parser.add_argument("--p-overnight", type=float, default=defaults.p_overnight)
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = SimulationConfig(
        n_drivers=args.n_drivers,
        n_runs=args.n_runs,
        seed=args.seed,
        day_type=args.day_type,
        p_overnight=args.p_overnight,
    )
    archetypes = load_archetypes(config.archetypes_path)
    dists = load_distributions()
    result = run_population(config, archetypes, dists)

    FIGURES_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{config.day_type}_n{config.n_drivers}_r{config.n_runs}_s{config.seed}"
    result.summary.write_csv(OUTPUTS_DIR / f"population_summary_{tag}.csv")
    result.runs.write_parquet(OUTPUTS_DIR / f"population_runs_{tag}.parquet")

    figures = {
        "population": population_figure(result),
        "demand_by_archetype": archetype_demand_figure(result),
    }
    # One example driver per archetype, drawn like the population's drivers.
    example_rng = np.random.default_rng(config.seed)
    for archetype in archetypes:
        driver = sample_drivers(archetype, 1, config, dists, example_rng)[0]
        days = simulate_driver_days(driver, EXAMPLE_DRIVER_DAYS, config, dists)
        slug = (
            archetype.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        )
        figures[f"driver_{archetype.id}_{slug}"] = driver_figure(days, archetype.name)

    for name, fig in figures.items():
        fig.write_html(FIGURES_DIR / f"{name}_{tag}.html", include_plotlyjs="cdn")
    logger.info(
        "Wrote %d figures to %s and tables to %s",
        len(figures),
        FIGURES_DIR,
        OUTPUTS_DIR,
    )


if __name__ == "__main__":
    main()
