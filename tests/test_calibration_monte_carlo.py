from imu_error_model import AxisConfig, ImuConfig
from playground.calibration.monte_carlo import run_monte_carlo_recovery


def test_monte_carlo_recovery_is_unbiased_and_uncertainty_is_reasonable() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(
            white_noise_density=0.002,
            fixed_bias=(0.08, -0.05, 0.03),
            sensitivity_matrix=(
                (1.012, 0.004, -0.002),
                (-0.003, 0.993, 0.005),
                (0.001, -0.004, 1.007),
            ),
        ),
        gyroscope=AxisConfig(
            white_noise_density=2e-5,
            fixed_bias=(0.001, -0.002, 0.0015),
            sensitivity_matrix=(
                (1.008, 0.003, -0.002),
                (-0.004, 0.996, 0.002),
                (0.001, -0.003, 1.005),
            ),
        ),
        gyroscope_specific_force_sensitivity=(
            (1e-5, -2e-5, 0.5e-5),
            (-1e-5, 1.5e-5, 2e-5),
            (0.25e-5, -0.5e-5, 1e-5),
        ),
    )

    summary = run_monte_carlo_recovery(
        config,
        trial_count=60,
        samples_per_segment=30,
        sample_period_s=0.02,
        seed=583,
    )

    assert summary.parameter_count == 33
    assert abs(summary.mean_error).max() < 8e-4
    assert summary.mean_coverage_95 > 0.85
    ratio = summary.predicted_standard_error / summary.empirical_standard_deviation
    assert ratio.min() > 0.55
    assert ratio.max() < 1.8
####
