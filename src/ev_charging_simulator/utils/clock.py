"""Clock-time arithmetic on a 24-hour circle. All times are in hours."""

from datetime import time

import numpy as np
from numpy.typing import ArrayLike, NDArray

HOURS_PER_DAY = 24.0


def time_to_hours(value: time) -> float:
    """Convert a clock time to decimal hours since midnight."""
    return value.hour + value.minute / 60 + value.second / 3600


def hours_until(start: ArrayLike, end: ArrayLike) -> NDArray[np.float64]:
    """Hours from clock time ``start`` forward to the next ``end`` (0 to 24)."""
    return np.mod(np.asarray(end, dtype=float) - np.asarray(start, dtype=float), 24)


def signed_hour_offset(value: float, reference: float) -> float:
    """Shortest signed offset from ``reference`` to ``value``, in [-12, 12)."""
    return float((value - reference + 12) % HOURS_PER_DAY - 12)
