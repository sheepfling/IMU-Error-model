from numpy import eye, ndarray, ones, testing, zeros

from imu_error_model import IdealInputImuModelProtocol, ImuModel, ImuModelProtocol, ImuOutput


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


def test_builtin_model_supports_the_ideal_input_protocol() -> None:
    model = ImuModel()
    assert isinstance(model, IdealInputImuModelProtocol)
    typed_model: IdealInputImuModelProtocol = model
    output = typed_model.measure_ideal_rates(
        start_time=0.0,
        end_time=0.1,
        specific_force_body=zeros(3),
        angular_rate_body=zeros(3),
    )
    testing.assert_array_equal(output.delta_v, zeros(3))
    testing.assert_array_equal(output.delta_theta, zeros(3))
####
