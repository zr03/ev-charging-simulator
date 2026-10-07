"""Unmanaged charging: SoC as a function of time since plug-in.

Charging starts at plug-in and runs until the driver's target SoC (or
plug-out). Power is the charger rating up to ``taper_start_soc``, then falls
linearly to ``taper_end_kw`` at 100%. With that linear taper, SoC follows a
closed-form exponential, so no time-stepping is needed.

ASSUMPTION A12: the curve shape and efficiency (see ChargingConfig).
ASSUMPTION A13: charging is unmanaged; smart and bump charging are not
modelled, since this is the baseline flexibility is measured against.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ev_charging_simulator.config import ChargingConfig

FloatArray = NDArray[np.float64]


def _curve(
    battery_kwh: ArrayLike, charger_kw: ArrayLike, cfg: ChargingConfig
) -> tuple[FloatArray, FloatArray]:
    """Charging-curve rates in SoC per hour.

    Below the knee d(SoC)/dt = rate; above it the linear power taper gives
    d(SoC)/dt = rate - decay * (SoC - knee).
    """
    battery = np.asarray(battery_kwh, dtype=float)
    power = np.asarray(charger_kw, dtype=float)
    rate = cfg.efficiency * power / battery
    end_kw = np.minimum(cfg.taper_end_kw, power)
    decay = cfg.efficiency * (power - end_kw) / ((1.0 - cfg.taper_start_soc) * battery)
    return rate, np.maximum(decay, 1e-12)  # avoid 0/0 when there is no taper


def _taper_time(
    soc: FloatArray, rate: FloatArray, decay: FloatArray, knee: float
) -> FloatArray:
    """Hours of taper charging needed to get from the knee to ``soc``."""
    above = np.clip(soc - knee, 0.0, None)
    return -np.log1p(-np.minimum(above * decay / rate, 1 - 1e-12)) / decay


def soc_after_charging(
    hours: ArrayLike,
    soc_start: ArrayLike,
    soc_target: ArrayLike,
    battery_kwh: ArrayLike,
    charger_kw: ArrayLike,
    cfg: ChargingConfig,
) -> FloatArray:
    """SoC after charging for ``hours`` from ``soc_start``, capped at target.

    All array arguments broadcast together.

    Args:
        hours: Time spent charging (h); negative values are treated as 0.
        soc_start: SoC at plug-in (fraction).
        soc_target: Driver's target SoC (fraction). Charging stops here.
        battery_kwh: Battery capacity (kWh).
        charger_kw: Charger rating at the wall (kW).
        cfg: Charging-curve settings.

    Returns:
        SoC (fraction), never below ``soc_start``.
    """
    t = np.maximum(np.asarray(hours, dtype=float), 0.0)
    s0 = np.asarray(soc_start, dtype=float)
    target = np.asarray(soc_target, dtype=float)
    knee = cfg.taper_start_soc
    rate, decay = _curve(battery_kwh, charger_kw, cfg)

    # Time spent below the knee; a start above the knee is expressed as an
    # equivalent time already spent in the taper.
    t_to_knee = np.maximum(knee - s0, 0.0) / rate
    tau = _taper_time(s0, rate, decay, knee) + np.maximum(t - t_to_knee, 0.0)
    soc_taper = knee + rate / decay * (1.0 - np.exp(-decay * tau))

    soc = np.where(t <= t_to_knee, s0 + rate * t, soc_taper)
    # A car already at or above its target draws nothing.
    return np.where(s0 >= target, s0, np.minimum(soc, target))


def hours_to_reach(
    soc_start: ArrayLike,
    soc_target: ArrayLike,
    battery_kwh: ArrayLike,
    charger_kw: ArrayLike,
    cfg: ChargingConfig,
) -> FloatArray:
    """Hours of unmanaged charging needed to go from ``soc_start`` to target.

    The inverse of :func:`soc_after_charging`; 0 if already at target.
    """
    s0 = np.asarray(soc_start, dtype=float)
    target = np.maximum(np.asarray(soc_target, dtype=float), s0)
    knee = cfg.taper_start_soc
    rate, decay = _curve(battery_kwh, charger_kw, cfg)

    linear = (np.minimum(target, knee) - np.minimum(s0, knee)) / rate
    taper = _taper_time(target, rate, decay, knee) - _taper_time(s0, rate, decay, knee)
    return linear + taper
