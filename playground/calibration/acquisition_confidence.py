"""Model-conditional white-noise planning with explicit systematic-error bounds."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sqrt
from statistics import NormalDist
from typing import Literal

import numpy as np

from imu_error_model import AcquisitionConfig, filtered_white_mean_covariance
from .earth_reference import WGS84_EARTH_ROTATION_RADPS


@dataclass(frozen=True, slots=True)
class AcquisitionConfidencePoint:
    """A planning bound, not evidence that the hardware satisfies its assumptions."""

    retained_samples: int
    sampled_acquisition_s: float
    random_heading_std_rad: float
    systematic_heading_bound_rad: float | None
    half_width_rad: float | None
    target_met: bool
    status: Literal["target_met", "more_data_required", "systematic_floor", "missing_systematic_bound"]
    planned_looks: int
####


def plan_filtered_heading(
        *, config: AcquisitionConfig, retained_samples: int,
        gyro_noise_density_radps_sqrt_s: float, latitude_rad: float,
        target_half_width_rad: float, systematic_heading_bound_rad: float | None,
        calibration_heading_std_rad: float = 0.0, planned_looks: int = 1,
        confidence: float = 0.95, earth_rotation_radps: float = WGS84_EARTH_ROTATION_RADPS,
) -> AcquisitionConfidencePoint:
    """Plan near-north, level heading precision after filter warm-up.

    Uses the east gyro component and first-order heading sensitivity. Prefilter
    noise is independent Gaussian rate noise with covariance density**2/dt.
    Calibration error is modeled as independent Gaussian heading error. The
    deterministic bound is ADDED, not converted into an IID random variance.
    Bonferroni adjustment supports a predeclared finite number of looks, assuming
    each model-based Gaussian interval has its stated marginal coverage. It is
    not an anytime-valid confidence sequence or a guarantee under fitted noise.
    """
    values = (gyro_noise_density_radps_sqrt_s, latitude_rad, target_half_width_rad,
              calibration_heading_std_rad, confidence, earth_rotation_radps)
    if not all(isfinite(x) for x in values):
        raise ValueError("planning inputs must be finite")
    ####
    if gyro_noise_density_radps_sqrt_s < 0.0 or calibration_heading_std_rad < 0.0:
        raise ValueError("noise and calibration standard deviation must be nonnegative")
    ####
    if target_half_width_rad <= 0.0 or earth_rotation_radps <= 0.0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid target, Earth rate, or confidence")
    ####
    if abs(latitude_rad) >= np.pi/2 or abs(cos(latitude_rad)) < 1e-6:
        raise ValueError("latitude is outside the supported north-finding geometry")
    ####
    if isinstance(planned_looks, bool) or not isinstance(planned_looks, int) or planned_looks < 1:
        raise ValueError("planned_looks must be a positive integer")
    ####
    if systematic_heading_bound_rad is not None and (
            not isfinite(systematic_heading_bound_rad) or systematic_heading_bound_rad < 0.0
    ):
        raise ValueError("systematic bound must be nonnegative or explicitly unknown")
    ####
    rate_covariance = np.zeros((6, 6))
    rate_covariance[3:, 3:] = np.eye(3)*gyro_noise_density_radps_sqrt_s**2/config.sample_period_s
    covariance = filtered_white_mean_covariance(rate_covariance, config, retained_samples)
    sigma = sqrt(covariance[4, 4]/(earth_rotation_radps*cos(latitude_rad))**2
                 + calibration_heading_std_rad**2)
    probability = 1.0 - (1.0-confidence)/(2.0*planned_looks)
    if probability == 1.0:
        raise ValueError("requested tail probability is below floating-point resolution")
    ####
    z = NormalDist().inv_cdf(probability)
    warmup = max(
        len(config.accelerometer_filter.taps) + max(config.accelerometer_delay_samples),
        len(config.gyroscope_filter.taps) + max(config.gyroscope_delay_samples),
    ) - 1
    half_width = None if systematic_heading_bound_rad is None else z*sigma + systematic_heading_bound_rad
    status: Literal["target_met", "more_data_required", "systematic_floor", "missing_systematic_bound"]
    if half_width is None:
        status = "missing_systematic_bound"
    elif half_width <= target_half_width_rad:
        status = "target_met"
    elif systematic_heading_bound_rad is not None and (
            systematic_heading_bound_rad + z*calibration_heading_std_rad >= target_half_width_rad
    ):
        status = "systematic_floor"
    else:
        status = "more_data_required"
    ####
    return AcquisitionConfidencePoint(
        retained_samples, (retained_samples+warmup)*config.sample_period_s,
        sigma, systematic_heading_bound_rad, half_width, status == "target_met", status, planned_looks,
    )
####
