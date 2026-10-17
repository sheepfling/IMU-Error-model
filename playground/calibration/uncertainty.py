"""Linear calibration uncertainty and observability diagnostics.

These helpers deliberately sit in the playground. They quantify the affine
known-reference fit without changing the public ``imu_error_model`` contract.
The covariance calculation is the classical multivariate least-squares result
under independent, identically distributed residual rows. Colored residuals
require a bootstrap, generalized least squares, or repeated-run campaign.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy import asarray, diag, isfinite, kron, ndarray, sqrt
from numpy.linalg import matrix_rank, pinv, svd


@dataclass(frozen=True, slots=True)
class LinearFitDiagnostics:
    """Diagnostics for one multivariate linear least-squares design.

    ``coefficient_covariance`` uses column-major vectorization of the fitted
    coefficient matrix: ``coefficients.reshape(-1, order="F")``. For
    ``Y = X B + E``, the covariance is
    ``kron(residual_covariance, pinv(X.T @ X))``.
    """

    rank: int
    parameter_count: int
    output_count: int
    sample_count: int
    singular_values: ndarray
    condition_number: float
    residual_rms: ndarray
    residual_variance: ndarray
    residual_covariance: ndarray
    coefficient_standard_error: ndarray
    coefficient_covariance: ndarray

    @property
    def full_rank(self) -> bool:
        """Return whether all fitted columns are observable from this design."""
        return self.rank == self.parameter_count
    ####
####


def linear_fit_diagnostics(
        design: ndarray,
        observations: ndarray,
        coefficients: ndarray,
) -> LinearFitDiagnostics:
    """Return classical multivariate least-squares uncertainty diagnostics.

    ``observations`` may have one or more output columns. Residual covariance
    between output axes is retained, while all rows are assumed independent
    and identically distributed for this analytical covariance estimate.
    """
    matrix = asarray(design, dtype=float)
    values = asarray(observations, dtype=float)
    beta = asarray(coefficients, dtype=float)
    if matrix.ndim != 2 or values.ndim != 2 or beta.ndim != 2:
        raise ValueError("design, observations, and coefficients must be two-dimensional")
    ####
    if matrix.shape[0] != values.shape[0] or matrix.shape[1] != beta.shape[0] or values.shape[1] != beta.shape[1]:
        raise ValueError("least-squares array dimensions are inconsistent")
    ####
    if not bool(isfinite(matrix).all() and isfinite(values).all() and isfinite(beta).all()):
        raise ValueError("least-squares arrays must be finite")
    ####
    sample_count, parameter_count = matrix.shape
    output_count = values.shape[1]
    rank = int(matrix_rank(matrix))
    singular_values = svd(matrix, full_matrices=False, compute_uv=False)
    smallest = float(singular_values[-1]) if singular_values.size else 0.0
    condition_number = float("inf") if smallest <= 0.0 else float(singular_values[0] / smallest)
    residual = values - matrix @ beta
    residual_rms = sqrt((residual ** 2).mean(axis=0))
    degrees_of_freedom = sample_count - rank
    divisor = degrees_of_freedom if degrees_of_freedom > 0 else max(sample_count, 1)
    residual_covariance = residual.T @ residual / divisor
    residual_covariance = 0.5 * (residual_covariance + residual_covariance.T)
    residual_variance = diag(residual_covariance)
    geometry_covariance = pinv(matrix.T @ matrix)
    coefficient_covariance = kron(residual_covariance, geometry_covariance)
    coefficient_covariance = 0.5 * (coefficient_covariance + coefficient_covariance.T)
    coefficient_standard_error = sqrt(
        coefficient_covariance.diagonal().clip(min=0.0)
    ).reshape((parameter_count, output_count), order="F")

    singular_values = singular_values.copy()
    residual_rms = residual_rms.copy()
    residual_variance = residual_variance.copy()
    residual_covariance = residual_covariance.copy()
    coefficient_standard_error = coefficient_standard_error.copy()
    coefficient_covariance = coefficient_covariance.copy()
    for array in (
            singular_values,
            residual_rms,
            residual_variance,
            residual_covariance,
            coefficient_standard_error,
            coefficient_covariance,
    ):
        array.setflags(write=False)
    ####
    return LinearFitDiagnostics(
        rank=rank,
        parameter_count=parameter_count,
        output_count=output_count,
        sample_count=sample_count,
        singular_values=singular_values,
        condition_number=condition_number,
        residual_rms=residual_rms,
        residual_variance=residual_variance,
        residual_covariance=residual_covariance,
        coefficient_standard_error=coefficient_standard_error,
        coefficient_covariance=coefficient_covariance,
    )
####
