from pathlib import Path

import numpy as np

from imu_error_model import load_yaml_profile


PROFILE_DIR = Path(__file__).parents[1] / "examples" / "imu_profiles" / "hardware-estimates"


def load_example(name: str):
    return load_yaml_profile(PROFILE_DIR / name)


def test_all_hardware_examples_load_with_provenance() -> None:
    paths = sorted(PROFILE_DIR.glob("*.yaml"))
    assert len(paths) >= 10
    for path in paths:
        profile = load_yaml_profile(path)
        assert profile.model_name
        assert profile.sample_period_s > 0
        assert profile.reference_links or profile.metadata.get("source")
        assert profile.config.accelerometer.measurement_range is not None
        assert profile.config.gyroscope.measurement_range is not None


def test_all_hardware_examples_are_explicitly_notional() -> None:
    for path in sorted(PROFILE_DIR.glob("*.yaml")):
        text = path.read_text(encoding="utf-8").lower()
        assert "notional" in text
        assert "approximate modeling estimates" in text
        assert "not official" in text


def test_hg1700_ag58_uses_product_sheet_range_rate_and_noise() -> None:
    profile = load_example("hg1700ag58.yaml")
    accel = profile.config.accelerometer
    gyro = profile.config.gyroscope

    assert np.isclose(profile.sample_period_s, 0.01)
    assert np.isclose(accel.measurement_range, 37.0 * 9.80665)
    assert np.isclose(accel.white_noise_density, 0.02 / 60.0)
    assert np.isclose(gyro.measurement_range, np.deg2rad(1074.0))
    assert np.isclose(gyro.white_noise_density, np.deg2rad(0.125) / 60.0)
    assert any("HG1700-SPAN58" in link for link in profile.reference_links)


    profile = load_example("hg1700ag71.yaml")
    accel = profile.config.accelerometer
    gyro = profile.config.gyroscope

    assert np.isclose(profile.sample_period_s, 0.01)
    assert np.isclose(accel.measurement_range, 70.0 * 9.80665)
    assert np.isclose(accel.scale_factor, 300e-6)
    assert np.isclose(accel.nonlinear_factor, 500e-6)
    assert np.isclose(accel.white_noise_density, 0.065 * 0.3048 / 60.0)
    assert np.isclose(gyro.measurement_range, np.deg2rad(1017.0))
    assert np.isclose(gyro.scale_factor, 150e-6)
    assert np.isclose(gyro.nonlinear_factor, 150e-6)
    assert np.isclose(gyro.white_noise_density, np.deg2rad(0.125) / 60.0)


def test_hg5700_profiles_include_published_scale_factors() -> None:
    for name in ("hg5700ca01.yaml", "hg5700ba01.yaml", "hg5700aa01.yaml"):
        profile = load_example(name)
        assert np.isclose(profile.config.accelerometer.scale_factor, 120e-6)
        assert np.isclose(profile.config.gyroscope.scale_factor, 40e-6)
        assert np.isclose(profile.config.accelerometer.white_noise_density, 0.065 * 0.3048 / 60.0)


def test_hg5700_ca01_prefers_detailed_brochure_arw() -> None:
    profile = load_example("hg5700ca01.yaml")
    assert np.isclose(profile.config.gyroscope.white_noise_density, np.deg2rad(0.0062) / 60.0)


def test_adis_profile_exposes_explicit_allan_curve_choices() -> None:
    profile = load_example("ADIS16470.yaml")
    accel = profile.config.accelerometer
    gyro = profile.config.gyroscope

    assert np.isclose(accel.white_noise_density, 0.037 / 60.0)
    assert np.isclose(gyro.white_noise_density, np.deg2rad(0.34) / 60.0)
    assert accel.flicker_components == gyro.flicker_components == 16
    assert accel.flicker_min_correlation_time == gyro.flicker_min_correlation_time == 1.0
    assert accel.flicker_max_correlation_time == gyro.flicker_max_correlation_time == 1000.0
    assert np.isclose(accel.flicker_bias_std, 8e-6 * 9.80665)
    assert np.isclose(gyro.flicker_bias_std, np.deg2rad(8.0) / 3600.0)


def test_hg9900_uses_rev_c_product_function_specification() -> None:
    profile = load_example("hg9900.yaml")
    accel = profile.config.accelerometer
    gyro = profile.config.gyroscope

    assert np.isclose(profile.sample_period_s, 1.0 / 300.0)
    assert np.isclose(accel.scale_factor, 117e-6)
    assert np.isclose(accel.white_noise_density, 0.0025 * 0.3048 / 60.0)
    assert np.isclose(accel.turn_on_bias_std, 25e-6 * 9.80665)
    assert np.isclose(accel.bias_std, 100e-6 * 9.80665)
    assert np.isclose(gyro.scale_factor, 5e-6)
    assert np.isclose(gyro.white_noise_density, np.deg2rad(0.0021) / 60.0)
    assert np.isclose(gyro.turn_on_bias_std, np.deg2rad(0.0041) / 3600.0)
