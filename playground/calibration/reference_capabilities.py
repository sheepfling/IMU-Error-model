"""Explicit field-to-evidence audit for reference-equipment functionality."""

from __future__ import annotations

from .reference_equipment import AngleLaw, TableGeometry
from .reference_observations import EncoderCorrection, EncoderErrorConfig


# Test files are checked to exist; statuses distinguish identification from injection.
EVIDENCE: dict[str, tuple[str, str]] = {
    "AngleLaw.offset_rad": ("fitted_effective", "tests/test_reference_observations.py"),
    "AngleLaw.rate_radps": ("fitted_effective", "tests/test_reference_observations.py"),
    "AngleLaw.sine_rad": ("fitted_effective", "tests/test_reference_observations.py"),
    "AngleLaw.cosine_rad": ("fitted_effective", "tests/test_reference_observations.py"),
    "AngleLaw.frequency_hz": ("declared_protocol", "tests/test_reference_observations.py"),
    "TableGeometry.schema_version": ("metadata", "tests/test_reference_capabilities.py"),
    "TableGeometry.base_rotvec_ned_rad": ("external_survey", "tests/test_reference_equipment.py"),
    "TableGeometry.tilt_rotvec_ned_rad": ("injected_and_surveyed", "tests/test_reference_equipment.py"),
    "TableGeometry.platform_from_sensor_rotvec_rad": ("external_anchor_required", "tests/test_reference_identification.py"),
    "TableGeometry.wobble_x_rad": ("injected_and_surveyed", "tests/test_reference_equipment.py"),
    "TableGeometry.wobble_y_rad": ("injected_and_surveyed", "tests/test_reference_equipment.py"),
    "TableGeometry.wobble_order": ("injected_and_surveyed", "tests/test_reference_equipment.py"),
    "TableGeometry.wobble_phase_rad": ("injected_and_surveyed", "tests/test_reference_equipment.py"),
    "TableGeometry.lever_arm_sensor_m": ("conditional_fit", "tests/test_reference_identification.py"),
    "EncoderErrorConfig.schema_version": ("metadata", "tests/test_reference_capabilities.py"),
    "EncoderErrorConfig.angle_offset_rad": ("external_anchor_required", "tests/test_reference_observations.py"),
    "EncoderErrorConfig.scale_error_ppm": ("external_anchor_required", "tests/test_reference_identification.py"),
    "EncoderErrorConfig.cyclic_error_rad": ("injected_model_mismatch", "tests/test_reference_observations.py"),
    "EncoderErrorConfig.cyclic_order": ("injected_model_mismatch", "tests/test_reference_observations.py"),
    "EncoderErrorConfig.cyclic_phase_rad": ("injected_model_mismatch", "tests/test_reference_observations.py"),
    "EncoderErrorConfig.angle_noise_std_rad": ("injected_not_identified", "tests/test_reference_observations.py"),
    "EncoderErrorConfig.quantization_rad": ("injected_not_invertible", "tests/test_reference_observations.py"),
    "EncoderErrorConfig.clock_offset_s": ("relative_delay_conditional_fit", "tests/test_reference_identification.py"),
    "EncoderErrorConfig.clock_frequency_error_ppm": ("relative_multichannel_clock_fit_requires_independent_reference", "tests/test_matched_acquisition_calibration.py"),
    "EncoderErrorConfig.timestamp_jitter_bound_s": ("tag_jitter_only", "tests/test_reference_observations.py"),
    "EncoderCorrection.angle_offset_rad": ("external_estimate", "tests/test_reference_observations.py"),
    "EncoderCorrection.scale_error_ppm": ("external_estimate", "tests/test_reference_observations.py"),
    "EncoderCorrection.clock_offset_s": ("external_estimate", "tests/test_reference_observations.py"),
    "EncoderCorrection.clock_frequency_error_ppm": ("external_estimate", "tests/test_reference_observations.py"),
}


def unclassified_reference_fields() -> tuple[str, ...]:
    actual={f"{model.__name__}.{name}" for model in (AngleLaw,TableGeometry,EncoderErrorConfig,EncoderCorrection) for name in model.model_fields}
    return tuple(sorted(actual-set(EVIDENCE)))
####
