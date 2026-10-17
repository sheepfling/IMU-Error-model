from numpy import asarray, ndarray, clip as numpy_clip, round as numpy_round
from numpy.linalg import norm

from .config import AxisConfig, AxisValue
from .thermal import ThermalState


def apply_distortions(
        signal: ndarray,
        cfg: AxisConfig,
        thermal: ThermalState,
        scale_factor: ndarray | None = None,
        fixed_bias: ndarray | None = None,
        sensitivity_matrix: ndarray | None = None,
) -> ndarray:
    """Apply fixed intrinsic, nonlinear, and thermal distortions to a rate."""
    thermal = thermal or ThermalState()
    scale_factor = asarray(cfg.scale_factor) if scale_factor is None else scale_factor
    fixed_bias = asarray(cfg.fixed_bias) if fixed_bias is None else fixed_bias
    if sensitivity_matrix is None and cfg.sensitivity_matrix is not None:
        sensitivity_matrix = asarray(cfg.sensitivity_matrix, dtype=float)
    ####
    out = signal * (
            1.0 + scale_factor + asarray(thermal.scale_factor_offset)
    )
    if cfg.nonlinear_factor:
        out = out * (1.0 + cfg.nonlinear_factor * norm(signal) ** 2)
    ####
    if sensitivity_matrix is not None:
        out = sensitivity_matrix @ out
    ####
    if thermal.sensitivity_matrix_offset is not None:
        out = out + asarray(thermal.sensitivity_matrix_offset, dtype=float) @ signal
    ####
    return out + fixed_bias + thermal.bias_offset
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
