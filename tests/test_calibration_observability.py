from imu_error_model import ImuModel
from playground.calibration.known_reference import (
    ReferenceSegment,
    ferraris_style_reference_program,
    simulate_reference_program,
)
from playground.calibration.observability import analyze_known_reference_observability


def test_standard_reference_program_is_full_rank() -> None:
    dataset = simulate_reference_program(ImuModel(), ferraris_style_reference_program())
    accelerometer, gyroscope = analyze_known_reference_observability(dataset)

    assert accelerometer.full_rank
    assert gyroscope.full_rank
    assert accelerometer.rank == 4
    assert gyroscope.rank == 7
####


def test_static_only_program_exposes_missing_gyro_excitation() -> None:
    static_program = tuple(
        segment
        for segment in ferraris_style_reference_program()
        if segment.label == "zero_reference" or segment.label.startswith("static_")
    )
    dataset = simulate_reference_program(ImuModel(), static_program)
    accelerometer, gyroscope = analyze_known_reference_observability(dataset)

    assert accelerometer.full_rank
    assert not gyroscope.full_rank
    assert gyroscope.rank == 4
####


def test_tiny_rotation_excitation_is_full_rank_but_badly_conditioned() -> None:
    weak_program = tuple(
        ReferenceSegment(
            label=segment.label,
            duration_s=segment.duration_s,
            specific_force_body=segment.specific_force_body,
            angular_rate_body=segment.angular_rate_body * 1e-6,
            temperature_celsius=segment.temperature_celsius,
            sample_count=segment.sample_count,
        )
        for segment in ferraris_style_reference_program()
    )
    dataset = simulate_reference_program(ImuModel(), weak_program)
    _, gyroscope = analyze_known_reference_observability(dataset)

    assert gyroscope.full_rank
    assert gyroscope.condition_number > 1e6
####
