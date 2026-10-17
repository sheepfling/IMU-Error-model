from numpy import abs as np_abs

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.excitation import multi_amplitude_reference_program
from playground.calibration.known_reference import simulate_reference_program


def test_multi_amplitude_program_exposes_nonlinearity_and_clipping() -> None:
    program = multi_amplitude_reference_program(
        acceleration_magnitudes_mps2=(1.0, 5.0),
        angular_rate_magnitudes_radps=(0.1,),
        sample_count=2,
    )
    nonlinear = simulate_reference_program(
        ImuModel(ImuConfig(accelerometer=AxisConfig(nonlinear_factor=0.01))),
        program,
    )
    clipped = simulate_reference_program(
        ImuModel(ImuConfig(accelerometer=AxisConfig(measurement_range=2.0))),
        program,
    )

    accel_rows = [index for index, label in enumerate(nonlinear.labels) if label.startswith("acc_")]
    assert float(np_abs(nonlinear.measured_acceleration[accel_rows] - nonlinear.specific_force_body[accel_rows]).max()) > 0.1
    assert float(np_abs(clipped.measured_acceleration[accel_rows]).max()) <= 2.0
####
