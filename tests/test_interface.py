from numpy import eye, ndarray, ones, testing, zeros

from imu_error_model import ImuModelProtocol, ImuOutput


class CustomModel:
    def reset(self) -> None:
        pass
    ####

    def measure(
            self,
            timestamp: float,
            velocity_without_gravity: ndarray,
            orientation_world_from_body: ndarray,
            temperature_celsius: float | None = None,
    ) -> ImuOutput:
        return ImuOutput(zeros(3), zeros(3), 0.0, timestamp, temperature_celsius)
    ####
####

def test_custom_model_can_implement_public_contract() -> None:
    model = CustomModel()
    assert isinstance(model, ImuModelProtocol)
    typed_model: ImuModelProtocol = model
    output = typed_model.measure(.01, ones(3), eye(3))
    testing.assert_array_equal(output.delta_v, zeros(3))
####
