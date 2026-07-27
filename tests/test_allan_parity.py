from numpy import absolute, argmin, array, asarray, diff, eye, isclose, log, mean, ndarray, sqrt, zeros
from numpy.random import default_rng

from examples.allan_variance import allan_deviation
from imu_error_model import ImuConfig, ImuModel


def collect_acceleration(config: ImuConfig, duration: float = 100.0, dt: float = .01) -> ndarray:
    model = ImuModel(config, rng=default_rng(42))
    samples: list[float] = []
    for index in range(int(duration / dt) + 1):
        output = model.measure(index * dt, zeros(3), eye(3))
        if output.dt:
            samples.append(float(output.acceleration[0]))
        ####
    return asarray(samples)
####

def test_white_noise_allan_level_matches_configured_density(short_correlation_profile: ImuConfig) -> None:
    config = short_correlation_profile
    config = config.model_copy(update={"accelerometer": config.accelerometer.model_copy(update={"bias_std": 0.0})})
    samples = collect_acceleration(config)
    tau, deviation = allan_deviation(samples, .01, array([1]))
    assert isclose(tau[0], .01)
    assert isclose(deviation[0], config.accelerometer.white_noise_density / sqrt(.01), rtol=.08)
####

def test_gauss_markov_allan_curve_has_short_knee(short_correlation_profile: ImuConfig) -> None:
    config = short_correlation_profile
    config = config.model_copy(
        update={"accelerometer": config.accelerometer.model_copy(update={"white_noise_density": 0.0})})
    samples = collect_acceleration(config)
    cluster_sizes = array([1, 5, 10, 25, 50, 100, 200, 400])
    tau, deviation = allan_deviation(samples, .01, cluster_sizes)
    early_slope = diff(log(deviation[1:4])) / diff(log(tau[1:4]))
    late_slope = diff(log(deviation[-3:])) / diff(log(tau[-3:]))
    assert mean(early_slope) > 0.1
    assert mean(late_slope) < -0.1
####

def test_flicker_band_matches_target_near_geometric_center(flicker_band_profile: ImuConfig) -> None:
    config = flicker_band_profile
    samples = collect_acceleration(config, duration=100.0)
    tau, deviation = allan_deviation(samples, .01, array([50, 100]))
    center_index = int(argmin(absolute(tau - 1.0)))
    assert isclose(deviation[center_index], config.accelerometer.flicker_bias_std, rtol=.25)
####
