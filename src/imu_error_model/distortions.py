from numpy import asarray, ndarray, clip as numpy_clip, round as numpy_round
from numpy.linalg import norm

from .config import AxisConfig, AxisValue
from .thermal import ThermalState


def apply_distortions(
        signal: ndarray,
        cfg: AxisConfig,
        thermal: ThermalState,
        scale_factor: ndarray | None = None,
) -> ndarray:
    """Apply a scale-factor, nonlinearity, and thermal distortions to a rate."""
    thermal = thermal or ThermalState()
    scale_factor = asarray(cfg.scale_factor) if scale_factor is None else scale_factor
    out = signal * (
            1.0 + scale_factor + asarray(thermal.scale_factor_offset)
    )
    if cfg.nonlinear_factor:
        out = out * (1.0 + cfg.nonlinear_factor * norm(signal) ** 2)
    ####
    return out + thermal.bias_offset
####

def clip(signal: ndarray, limit: float | None) -> ndarray:
    """Symmetrically clip a rate to `[-limit, limit]` when configured."""
    return signal if limit is None else numpy_clip(signal, -limit, limit)
####

def quantize(signal: ndarray, step: AxisValue | ndarray | None) -> ndarray:
    """Round a rate to the configured quantization step when provided."""
    if step is None:
        return signal
    ####
    step_values = asarray(step, dtype=float)
    return numpy_round(signal / step_values) * step_values
####
