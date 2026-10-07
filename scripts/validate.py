"""Compare simulated sessions with held-out CNZ report figures (Figs 5 to 9).

Writes ``figures/validation_<archetype>.html`` and a summary CSV to
``data/outputs/``. Example::

    uv run python scripts/validate.py --archetype "Intelligent Octopus average"
"""

import argparse
import logging

from ev_charging_simulator.config import REPO_ROOT, SimulationConfig
from ev_charging_simulator.models.archetypes import load_archetypes
from ev_charging_simulator.models.distributions import load_distributions
from ev_charging_simulator.validation import (
    DEFAULT_ARCHETYPE,
    compare,
    simulate_sessions,
    summary_table,
    validation_figure,
)

logger = logging.getLogger("validate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archetype", default=DEFAULT_ARCHETYPE)
    parser.add_argument("--seed", type=int, default=SimulationConfig().seed)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    config = SimulationConfig(seed=args.seed)
    archetypes = {a.name: a for a in load_archetypes(config.archetypes_path)}
    sessions = simulate_sessions(
        archetypes[args.archetype], config, load_distributions()
    )
    comparison = compare(sessions)
    table = summary_table(comparison)
    logger.info("Validation summary:\n%s", table)

    slug = args.archetype.lower().replace(" ", "_")
    (REPO_ROOT / "figures").mkdir(exist_ok=True)
    (REPO_ROOT / "data/outputs").mkdir(parents=True, exist_ok=True)
    validation_figure(comparison, args.archetype).write_html(
        REPO_ROOT / f"figures/validation_{slug}.html", include_plotlyjs="cdn"
    )
    table.write_csv(REPO_ROOT / f"data/outputs/validation_{slug}.csv")


if __name__ == "__main__":
    main()
