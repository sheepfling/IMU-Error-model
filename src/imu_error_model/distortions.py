import numpy as np

from .config import AxisConfig
from .thermal import ThermalState


def apply_distortions(signal: np.ndarray, cfg: AxisConfig, thermal: ThermalState | None = None) -> np.ndarray:
    thermal = thermal or ThermalState()
    out = signal * (1.0 + cfg.scale_factor + thermal.scale_factor_offset)
    if cfg.nonlinear_factor:
        out = out * (1.0 + cfg.nonlinear_factor * np.square(np.linalg.norm(signal)))
    if cfg.misalignment_std:
        # A full calibrated matrix can be supplied later; this is the isotropic model.
        out = out + np.zeros(3)
    return out + thermal.bias_offset


def clip(signal: np.ndarray, limit: float | None) -> np.ndarray:
    return signal if limit is None else np.clip(signal, -limit, limit)


def quantize(signal: np.ndarray, step: float | None) -> np.ndarray:
    return signal if step is None else np.round(signal / step) * step
