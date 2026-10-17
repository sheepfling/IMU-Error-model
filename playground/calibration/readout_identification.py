"""Known-input readout identification; fitting never receives injected kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _samples(value: FloatArray, name: str) -> FloatArray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 6 or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape (n, 6)")
    ####
    return result
####


@dataclass(frozen=True, slots=True)
class ReadoutKernelEstimate:
    """Effective per-channel kernels, not a separate physical-clock/delay estimate.

    A pure delay and leading zero taps have the same observable response. This
    estimator deliberately does not claim to identify them separately. Known
    reference signals must already use a common physical time base.

    ``coefficient_covariance_by_channel`` stores the OLS covariance of each
    channel's ``[kernel taps..., intercept]`` coefficient vector. Different
    channels are fit independently, so cross-channel readout-fit covariance is
    intentionally zero unless a future joint estimator supplies it.
    """

    kernels: FloatArray
    intercept: FloatArray
    rank_by_channel: tuple[int, ...]
    condition_by_channel: tuple[float, ...]
    residual_rms_by_channel: tuple[float, ...]
    coefficient_covariance_by_channel: FloatArray
    covariance_assumption: str = "independent_channel_ols_iid_rows"

    def __post_init__(self) -> None:
        kernels = np.array(self.kernels, dtype=float, copy=True)
        intercept = np.array(self.intercept, dtype=float, copy=True)
        covariance = np.array(self.coefficient_covariance_by_channel, dtype=float, copy=True)
        if kernels.ndim != 2 or kernels.shape[0] != 6 or kernels.shape[1] < 1 or not np.isfinite(kernels).all():
            raise ValueError("kernels must have finite shape (6, L), L >= 1")
        ####
        length = kernels.shape[1]
        if intercept.shape != (6,) or not np.isfinite(intercept).all():
            raise ValueError("intercept must have finite shape (6,)")
        ####
        if covariance.shape != (6, length + 1, length + 1) or not np.isfinite(covariance).all():
            raise ValueError("coefficient covariance must have finite shape (6, L+1, L+1)")
        ####
        if not np.allclose(covariance, np.swapaxes(covariance, 1, 2), rtol=0.0, atol=1e-12):
            raise ValueError("readout coefficient covariance must be symmetric by channel")
        ####
        if any(np.linalg.eigvalsh(block).min() < -1e-12 for block in covariance):
            raise ValueError("readout coefficient covariance must be positive semidefinite")
        ####
        for values, name in (
            (self.rank_by_channel, "rank_by_channel"),
            (self.condition_by_channel, "condition_by_channel"),
            (self.residual_rms_by_channel, "residual_rms_by_channel"),
        ):
            if len(values) != 6:
                raise ValueError(f"{name} must contain six entries")
            ####
        ####
        if any(type(value) is not int or value < 0 for value in self.rank_by_channel):
            raise ValueError("readout ranks must be nonnegative integers")
        ####
        if any(not np.isfinite(value) or value < 0 for value in self.condition_by_channel):
            raise ValueError("readout conditions must be finite and nonnegative")
        ####
        if any(not np.isfinite(value) or value < 0 for value in self.residual_rms_by_channel):
            raise ValueError("readout residual RMS values must be finite and nonnegative")
        ####
        if not self.covariance_assumption:
            raise ValueError("covariance_assumption must not be empty")
        ####
        kernels.setflags(write=False)
        intercept.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "coefficient_covariance_by_channel", covariance)
    ####

    @property
    def tap_covariance(self) -> FloatArray:
        """Return block-diagonal covariance of the six flattened kernel vectors."""
        length = self.kernels.shape[1]
        result = np.zeros((6 * length, 6 * length))
        for axis in range(6):
            block = slice(axis * length, (axis + 1) * length)
            result[block, block] = self.coefficient_covariance_by_channel[axis, :length, :length]
        ####
        result.setflags(write=False)
        return result
    ####
####


def fit_readout_kernel(
        reference_rates: FloatArray, measured_rates: FloatArray,
        *, kernel_length: int, maximum_condition: float = 1e8,
) -> ReadoutKernelEstimate:
    """Fit six independent FIR+offset models using full-rank known excitation.

    Rates are aligned arrays. The first L-1 samples are excluded because their
    complete reference history is not available. Constant excitation is rejected,
    not silently regularized into a supposedly identified filter.
    """
    reference = _samples(reference_rates, "reference_rates")
    measured = _samples(measured_rates, "measured_rates")
    if reference.shape != measured.shape:
        raise ValueError("reference and measured shapes must match")
    ####
    if isinstance(kernel_length, bool) or not isinstance(kernel_length, int) or kernel_length < 1:
        raise ValueError("kernel_length must be a positive integer")
    ####
    if not np.isfinite(maximum_condition) or maximum_condition <= 1.0:
        raise ValueError("maximum_condition must exceed one")
    ####
    n = reference.shape[0]
    if n < 2 * kernel_length + 2:
        raise ValueError("record too short for FIR plus offset and residual degrees of freedom")
    ####
    kernels = np.zeros((6, kernel_length))
    bias = np.zeros(6)
    covariance = np.zeros((6, kernel_length + 1, kernel_length + 1))
    ranks: list[int] = []
    conditions: list[float] = []
    residuals: list[float] = []
    for axis in range(6):
        design = np.column_stack([
            reference[kernel_length - 1 - lag:n - lag, axis] for lag in range(kernel_length)
        ] + [np.ones(n - kernel_length + 1)])
        target = measured[kernel_length - 1:, axis]
        fit, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else float("inf")
        if rank < kernel_length + 1 or condition > maximum_condition:
            raise ValueError(f"readout excitation is unobservable/ill-conditioned on channel {axis}")
        ####
        prediction = design @ fit
        residual = target - prediction
        dof = len(target) - (kernel_length + 1)
        if dof <= 0:
            raise ValueError("readout fit needs residual degrees of freedom")
        ####
        influence = np.linalg.pinv(design)
        variance = float(residual @ residual / dof)
        covariance[axis] = variance * influence @ influence.T
        kernels[axis] = fit[:-1]
        bias[axis] = fit[-1]
        ranks.append(int(rank))
        conditions.append(condition)
        residuals.append(float(np.sqrt(np.mean(residual ** 2))))
    ####
    return ReadoutKernelEstimate(
        kernels,
        bias,
        tuple(ranks),
        tuple(conditions),
        tuple(residuals),
        covariance,
    )
####


def predict_readout(estimate: ReadoutKernelEstimate, reference_rates: FloatArray) -> tuple[FloatArray, int]:
    """Predict held-out rates; return the first index with complete FIR history."""
    reference = _samples(reference_rates, "reference_rates")
    length = estimate.kernels.shape[1]
    if reference.shape[0] < length:
        raise ValueError("prediction needs at least one complete reference history")
    ####
    predicted = np.column_stack([
        np.convolve(reference[:, axis], estimate.kernels[axis], mode="full")[:len(reference)]
        + estimate.intercept[axis] for axis in range(6)
    ])
    return predicted, length - 1
####
