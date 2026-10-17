"""Time-series covariance helpers for static calibration and north finding."""

from __future__ import annotations

from dataclasses import dataclass

from numpy import asarray, diag, isfinite, ndarray, zeros
from numpy.linalg import eigh


@dataclass(frozen=True, slots=True)
class MeanCovarianceEstimate:
    """Estimated covariance of a multichannel sample mean."""

    covariance: ndarray
    instantaneous_covariance: ndarray
    effective_sample_size: ndarray
    sample_count: int
    max_lag: int
    method: str

    def __post_init__(self) -> None:
        covariance = asarray(self.covariance, dtype=float).copy()
        instantaneous = asarray(self.instantaneous_covariance, dtype=float).copy()
        effective = asarray(self.effective_sample_size, dtype=float).copy()
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be square")
        ####
        if instantaneous.shape != covariance.shape:
            raise ValueError("instantaneous_covariance must match covariance")
        ####
        if effective.shape != (covariance.shape[0],):
            raise ValueError("effective_sample_size must match the channel count")
        ####
        if not bool(isfinite(covariance).all() and isfinite(instantaneous).all() and isfinite(effective).all()):
            raise ValueError("mean covariance results must be finite")
        ####
        for array in (covariance, instantaneous, effective):
            array.setflags(write=False)
        ####
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "instantaneous_covariance", instantaneous)
        object.__setattr__(self, "effective_sample_size", effective)
    ####
####


def _nearest_positive_semidefinite(matrix: ndarray) -> ndarray:
    """Return the symmetric positive-semidefinite projection of a matrix."""
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = eigh(symmetric)
    clipped = eigenvalues.clip(min=0.0)
    return (eigenvectors * clipped) @ eigenvectors.T
####


def estimate_mean_covariance(samples: ndarray, *, max_lag: int = 0) -> MeanCovarianceEstimate:
    """Estimate covariance of a sample mean with a Bartlett HAC estimator.

    ``max_lag=0`` gives the ordinary IID sample-covariance-over-count result.
    Positive ``max_lag`` retains lagged covariance and is useful for slowly
    correlated bias. The estimator is projected to the nearest positive
    semidefinite matrix because finite records can yield small negative modes.
    """
    values = asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("samples must have shape (n, channels) with n >= 2")
    ####
    if not bool(isfinite(values).all()):
        raise ValueError("samples must be finite")
    ####
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 0:
        raise ValueError("max_lag must be a non-negative integer")
    ####
    sample_count, channel_count = values.shape
    if max_lag >= sample_count:
        raise ValueError("max_lag must be smaller than the sample count")
    ####
    centered = values - values.mean(axis=0)
    instantaneous = centered.T @ centered / (sample_count - 1)
    long_run = centered.T @ centered / sample_count
    for lag in range(1, max_lag + 1):
        lag_covariance = centered[lag:].T @ centered[:-lag] / sample_count
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run += weight * (lag_covariance + lag_covariance.T)
    ####
    long_run = _nearest_positive_semidefinite(long_run)
    covariance = long_run / sample_count
    instantaneous_diagonal = diag(instantaneous)
    long_run_diagonal = diag(long_run)
    effective = zeros(channel_count)
    positive = long_run_diagonal > 0.0
    effective[positive] = sample_count * instantaneous_diagonal[positive] / long_run_diagonal[positive]
    effective[~positive] = float(sample_count)
    effective = effective.clip(min=1.0, max=float(sample_count))
    return MeanCovarianceEstimate(
        covariance=covariance,
        instantaneous_covariance=instantaneous,
        effective_sample_size=effective,
        sample_count=sample_count,
        max_lag=max_lag,
        method="iid" if max_lag == 0 else "bartlett_hac",
    )
####
