from numpy import random, zeros

from playground.calibration.time_series import estimate_mean_covariance


def test_iid_mean_covariance_tracks_variance_over_sample_count() -> None:
    generator = random.default_rng(4)
    samples = generator.normal(0.0, 2.0, size=(5000, 2))

    estimate = estimate_mean_covariance(samples, max_lag=0)

    expected = 4.0 / samples.shape[0]
    assert 0.8 < estimate.covariance[0, 0] / expected < 1.2
    assert 0.8 < estimate.covariance[1, 1] / expected < 1.2
    assert estimate.method == "iid"
    assert estimate.effective_sample_size.min() > 4900
####


def test_hac_covariance_detects_positive_serial_correlation() -> None:
    generator = random.default_rng(5)
    innovations = generator.normal(0.0, 1.0, size=4000)
    samples = zeros((innovations.size, 1))
    for index in range(1, innovations.size):
        samples[index, 0] = 0.92 * samples[index - 1, 0] + innovations[index]
    ####

    iid = estimate_mean_covariance(samples, max_lag=0)
    hac = estimate_mean_covariance(samples, max_lag=80)

    assert hac.covariance[0, 0] > 8.0 * iid.covariance[0, 0]
    assert hac.effective_sample_size[0] < 0.2 * samples.shape[0]
    assert hac.method == "bartlett_hac"
####
