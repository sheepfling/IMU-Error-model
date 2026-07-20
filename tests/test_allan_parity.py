from dataclasses import replace

import numpy as np

from examples.allan_variance import allan_deviation
from imu_error_model import ImuModel, load_profile


def collect_acceleration(config, duration: float = 100.0, dt: float = .01) -> np.ndarray:
    model = ImuModel(config, rng=np.random.default_rng(42))
    samples = []
    for index in range(int(duration / dt) + 1):
        output = model.measure(index * dt, np.zeros(3), np.eye(3))
        if output.dt:
            samples.append(output.acceleration[0])
    return np.asarray(samples)


def test_white_noise_allan_level_matches_configured_density() -> None:
    config = load_profile("profiles/test/short-correlation.json")
    config = replace(config, accelerometer=replace(config.accelerometer, bias_std=0.0))
    samples = collect_acceleration(config)
    tau, deviation = allan_deviation(samples, .01, np.array([1]))
    assert tau[0] == .01
    assert np.isclose(deviation[0], config.accelerometer.white_noise_density / np.sqrt(.01), rtol=.08)


def test_gauss_markov_allan_curve_has_short_knee() -> None:
    config = load_profile("profiles/test/short-correlation.json")
    config = replace(config, accelerometer=replace(config.accelerometer, white_noise_density=0.0))
    samples = collect_acceleration(config)
    cluster_sizes = np.array([1, 5, 10, 25, 50, 100, 200, 400])
    tau, deviation = allan_deviation(samples, .01, cluster_sizes)
    early_slope = np.diff(np.log(deviation[1:4])) / np.diff(np.log(tau[1:4]))
    late_slope = np.diff(np.log(deviation[-3:])) / np.diff(np.log(tau[-3:]))
    assert np.mean(early_slope) > 0.1
    assert np.mean(late_slope) < -0.1


def test_flicker_band_matches_target_near_geometric_center() -> None:
    config = load_profile("profiles/test/flicker-band.json")
    samples = collect_acceleration(config, duration=100.0)
    tau, deviation = allan_deviation(samples, .01, np.array([50, 100]))
    center_index = int(np.argmin(np.abs(tau - 1.0)))
    assert np.isclose(deviation[center_index], config.accelerometer.flicker_bias_std, rtol=.25)
