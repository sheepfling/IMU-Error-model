import numpy as np

from imu_error_model import BaseImuModel, ImuModelProtocol, ImuOutput


class CustomModel(BaseImuModel):
    def reset(self) -> None:
        pass

    def measure(self, timestamp, velocity_without_gravity, orientation_world_from_body, temperature=25.0):
        return ImuOutput(np.zeros(3), np.zeros(3), timestamp, temperature)


def test_custom_model_can_implement_public_contract() -> None:
    model: ImuModelProtocol = CustomModel()
    output = model.measure(.01, np.ones(3), np.eye(3))
    np.testing.assert_array_equal(output.delta_v, np.zeros(3))
