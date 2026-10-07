"""Reproducible random-number streams."""

import numpy as np


def spawn_generators(seed: int, n: int) -> list[np.random.Generator]:
    """Return ``n`` independent generators derived from one master seed.

    Each run gets its own stream, so runs differ but the whole set is
    reproducible from ``seed`` alone.
    """
    children = np.random.SeedSequence(seed).spawn(n)
    return [np.random.default_rng(child) for child in children]
