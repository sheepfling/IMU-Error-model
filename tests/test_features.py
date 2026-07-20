import numpy as np
from pathlib import Path

from imu_error_model import AxisConfig, ImuConfig, ImuModel, load_profile, load_yaml_profile, save_profile


def _baseline(model: ImuModel) -> None:
    model.measure(0.0, np.zeros(3), np.eye(3))


def test_profile_round_trip(tmp_path) -> None:
    config = ImuConfig(accelerometer=AxisConfig(quantization_step=.01))
    path = tmp_path / "profile.json"
    save_profile(config, path)
    loaded = load_profile(path)
    assert loaded.accelerometer.quantization_step == .01


def test_temperature_changes_bias_only_when_configured() -> None:
    config = ImuConfig(accelerometer=AxisConfig(thermal_bias_coefficient=.1))
    model = ImuModel(config, np.random.default_rng(1))
    _baseline(model)
    output = model.measure(.1, np.zeros(3), np.eye(3), temperature=35.0)
    np.testing.assert_allclose(output.acceleration, [1.0, 1.0, 1.0])


def test_correlated_covariance_is_reproducible() -> None:
    covariance = ((1.0, .5, 0.0), (.5, 1.0, 0.0), (0.0, 0.0, 1.0))
    config = ImuConfig(accelerometer=AxisConfig(noise_covariance=covariance))
    a, b = ImuModel(config, np.random.default_rng(5)), ImuModel(config, np.random.default_rng(5))
    _baseline(a); _baseline(b)
    first = a.measure(.01, np.zeros(3), np.eye(3))
    second = b.measure(.01, np.zeros(3), np.eye(3))
    np.testing.assert_array_equal(first.delta_v, second.delta_v)


def test_correlated_covariance_controls_cross_axis_noise() -> None:
    covariance = ((1.0, .6, 0.0), (.6, 1.0, 0.0), (0.0, 0.0, 1.0))
    config = ImuConfig(
        accelerometer=AxisConfig(noise_covariance=covariance, apply_clipping=False),
    )
    model = ImuModel(config, np.random.default_rng(17))
    _baseline(model)
    samples = np.array([
        model.measure((index + 1) * .01, np.zeros(3), np.eye(3)).acceleration
        for index in range(5000)
    ])
    measured_correlation = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
    assert np.isclose(measured_correlation, .6, atol=.05)


def test_thermal_noise_coefficient_changes_noise_level() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(
            white_noise_density=.2,
            thermal_noise_coefficient=.1,
        ),
    )
    model = ImuModel(config, np.random.default_rng(19))
    _baseline(model)
    samples = np.array([
        model.measure((index + 1) * .01, np.zeros(3), np.eye(3), temperature=35.0).acceleration
        for index in range(4000)
    ])
    expected_variance = (.2 * 2.0) ** 2 / .01
    assert np.isclose(np.var(samples[:, 0]), expected_variance, rtol=.12)


def test_quantization_is_applied_to_reported_rate() -> None:
    config = ImuConfig(accelerometer=AxisConfig(quantization_step=.5))
    model = ImuModel(config)
    _baseline(model)
    output = model.measure(1.0, np.array([.6, 0, 0]), np.eye(3))
    np.testing.assert_allclose(output.acceleration, [.5, 0, 0])


def test_white_noise_variance_matches_density() -> None:
    density = .2
    dt = .01
    config = ImuConfig(accelerometer=AxisConfig(white_noise_density=density))
    model = ImuModel(config, np.random.default_rng(22))
    _baseline(model)
    samples = np.array([
        model.measure((index + 1) * dt, np.zeros(3), np.eye(3)).acceleration
        for index in range(4000)
    ])
    variance = np.var(samples[:, 0])
    assert np.isclose(variance, density**2 / dt, rtol=.12)


def test_yaml_profile_preserves_provenance() -> None:
    path = "examples/imu_profiles/hardware-estimates/hg9900.yaml"
    profile = load_yaml_profile(path)
    assert profile.model_name == "HG9900"
    assert profile.sample_period_s == .0033333333
    assert profile.metadata["vendor"] == "Honeywell"
    assert profile.reference_links
    assert profile.config.accelerometer.output_scale == 1048576.0
    assert profile.config.gyroscope.bias_correlation_time == 3600.0


def test_temperature_vectors_are_retained() -> None:
    profile = load_yaml_profile("examples/imu_profiles/hardware-estimates/sbg_pulse_40.yaml")
    assert profile.config.accelerometer.thermal_bias_coefficient == (.0005, .0005, .0005)
    assert profile.config.gyroscope.thermal_scale_factor_coefficient == (.00003, .00003, .00003)


def test_all_reference_profiles_load() -> None:
    paths = sorted(Path("examples/imu_profiles/hardware-estimates").glob("*.yaml"))
    assert len(paths) == 14
    profiles = [load_yaml_profile(path) for path in paths]
    assert {profile.model_name for profile in profiles} >= {"HG9900", "HG5700CA01", "HG1700AG61"}
