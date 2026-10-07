import numpy as np

from ev_charging_simulator.config import ChargingConfig
from ev_charging_simulator.models.charging import hours_to_reach, soc_after_charging

CFG = ChargingConfig()


def test_full_power_below_knee() -> None:
    # 7 kW for 1 h at 90% efficiency into 60 kWh adds 6.3 kWh = 0.105 SoC.
    soc = soc_after_charging(1.0, 0.2, 1.0, 60.0, 7.0, CFG)
    assert np.isclose(soc, 0.2 + 0.9 * 7 / 60)


def test_taper_power_at_full_is_end_kw() -> None:
    dt = 1e-4
    soc = soc_after_charging(np.array([0.0, dt]), 0.9999, 1.0, 60.0, 7.0, CFG)
    power = (soc[1] - soc[0]) * 60 / CFG.efficiency / dt
    assert np.isclose(power, CFG.taper_end_kw, rtol=1e-2)


def test_capped_at_target_and_monotonic() -> None:
    t = np.linspace(0, 20, 200)
    soc = soc_after_charging(t, 0.1, 0.85, 60.0, 7.0, CFG)
    assert np.all(np.diff(soc) >= 0)
    assert soc.max() == 0.85


def test_no_charging_above_target() -> None:
    assert soc_after_charging(5.0, 0.9, 0.8, 60.0, 7.0, CFG) == 0.9


def test_hours_to_reach_inverts_curve() -> None:
    s0 = np.array([0.05, 0.5, 0.85, 0.95])
    target = np.array([1.0, 0.9, 0.95, 0.9])
    hours = hours_to_reach(s0, target, 60.0, 7.0, CFG)
    reached = soc_after_charging(hours, s0, target, 60.0, 7.0, CFG)
    assert np.allclose(reached, np.maximum(target, s0))
    assert np.all(
        soc_after_charging(hours - 0.01, s0, target, 60.0, 7.0, CFG)[:3] < target[:3]
    )
