import numpy as np
import pytest

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from imu_error_model.kinematics import rotation_vector_from_matrix, validate_orientation
from imu_error_model.processes import gauss_markov, white_noise


@pytest.mark.parametrize(
    "kwargs",
    [
        {"white_noise_density": -1},
        {"turn_on_bias_std": -1},
        {"bias_std": -1},
        {"misalignment_std": -1},
        {"bias_correlation_time": 0},
        {"measurement_range": 0},
        {"quantization_step": 0},
    ],
)
def test_axis_configuration_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        AxisConfig(**kwargs)


def test_configuration_rejects_bad_covariance_and_scale() -> None:
    with pytest.raises(ValueError):
        AxisConfig(noise_covariance=((1, 2, 0), (0, 1, 0), (0, 0, 1)))
    with pytest.raises(ValueError):
        ImuConfig(output_scale_gyroscope=0)


def test_processes_handle_zero_dt_and_random_walk() -> None:
    rng = np.random.default_rng(8)
    np.testing.assert_array_equal(white_noise(1, 0, rng), np.zeros(3))
    initial = np.ones(3)
    np.testing.assert_array_equal(gauss_markov(initial, 1, 10, 0, rng), initial)
    result = gauss_markov(initial, 1, None, 1, rng)
    assert result.shape == (3,)


def test_kinematics_validates_and_handles_identity() -> None:
    np.testing.assert_array_equal(rotation_vector_from_matrix(np.eye(3)), np.zeros(3))
    with pytest.raises(ValueError):
        validate_orientation(np.eye(2))
    with pytest.raises(ValueError):
        validate_orientation(np.diag([1.0, 1.0, -1.0]))


def test_model_rejects_bad_truth_inputs_and_can_reset() -> None:
    model = ImuModel()
    with pytest.raises(ValueError):
        model.measure(0, np.array([np.nan, 0, 0]), np.eye(3))
    with pytest.raises(ValueError):
        model.measure(0, np.zeros(3), np.eye(3), temperature=np.inf)
    model.measure(0, np.zeros(3), np.eye(3))
    model.reset()
    assert model.measure(0, np.zeros(3), np.eye(3)).dt == 0
