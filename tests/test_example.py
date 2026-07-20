from pathlib import Path

from examples.dead_reckoning import run_case
from examples.allan_variance import allan_deviation, cluster_sizes_for_scale

import numpy as np


def test_dead_reckoning_example_runs_one_profile() -> None:
    result = run_case(Path("examples/imu_profiles/hardware-estimates/hg9900.yaml"), duration=.02, seed=0)
    assert result["model_name"] == "HG9900"
    assert result["final_position_error_m"] >= 0


def test_allan_deviation_white_noise_returns_expected_shape() -> None:
    tau, deviation = allan_deviation(np.ones(32), .01)
    assert tau.shape == deviation.shape
    np.testing.assert_allclose(deviation, 0)


def test_allan_cluster_scale_is_configurable() -> None:
    clusters = cluster_sizes_for_scale(.01, 100, .1, 10, 12)
    assert clusters[0] >= 10
    assert clusters[-1] <= 100 // 4 / .01
    assert len(clusters) <= 12
