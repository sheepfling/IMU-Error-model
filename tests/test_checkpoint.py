from pathlib import Path

from numpy import array, eye, testing, zeros
from numpy.random import MT19937, PCG64, PCG64DXSM, Philox, SFC64, Generator, default_rng

from imu_error_model import (
    AxisConfig,
    CheckpointCodecProtocol,
    CheckpointableImuModelProtocol,
    ImuConfig,
    ImuModel,
    ImuModelCheckpoint,
    ImuModelProtocol,
    PydanticJsonCheckpointCodec,
    SerializableCheckpointableImuModelProtocol,
)


def _config() -> ImuConfig:
    axis = AxisConfig(
        white_noise_density=0.02,
        turn_on_bias_std=0.1,
        bias_std=0.03,
        bias_correlation_time=2.0,
        flicker_bias_std=0.01,
        flicker_min_correlation_time=0.1,
        flicker_max_correlation_time=10.0,
        flicker_components=3,
        misalignment_std=1e-3,
        thermal_bias_coefficient=0.001,
    )
    return ImuConfig(accelerometer=axis, gyroscope=axis)
####


def _measure(model: ImuModelProtocol, timestamp: float):
    velocity = array([0.2 * timestamp, -0.1 * timestamp, 0.05 * timestamp])
    return model.measure(timestamp, velocity, eye(3), temperature_celsius=30.0)
####


def test_checkpoint_resume_reproduces_the_next_samples() -> None:
    model = ImuModel(_config(), default_rng(42))
    _measure(model, 0.0)
    _measure(model, 0.1)
    checkpoint = model.snapshot()

    expected = [_measure(model, timestamp) for timestamp in (0.2, 0.35, 0.5)]
    resumed = ImuModel.from_checkpoint(checkpoint)
    actual = [_measure(resumed, timestamp) for timestamp in (0.2, 0.35, 0.5)]

    for expected_output, actual_output in zip(expected, actual):
        testing.assert_array_equal(expected_output.delta_v, actual_output.delta_v)
        testing.assert_array_equal(expected_output.delta_theta, actual_output.delta_theta)
        assert expected_output.start_time == actual_output.start_time
        assert expected_output.end_time == actual_output.end_time
        assert expected_output.temperature_celsius == actual_output.temperature_celsius
    ####
####


def test_checkpoint_round_trip_is_json_serializable(tmp_path: Path) -> None:
    model = ImuModel(_config(), default_rng(7))
    _measure(model, 0.0)
    checkpoint_path = tmp_path / "imu-checkpoint.json"

    model.save_checkpoint(checkpoint_path)
    parsed = ImuModelCheckpoint.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
    restored = ImuModel.load_checkpoint(checkpoint_path)
    typed_model: CheckpointableImuModelProtocol[ImuModelCheckpoint] = restored
    typed_serializable_model: SerializableCheckpointableImuModelProtocol[ImuModelCheckpoint] = restored

    assert parsed == model.snapshot()
    assert isinstance(typed_model, CheckpointableImuModelProtocol)
    assert isinstance(typed_serializable_model, SerializableCheckpointableImuModelProtocol)
    testing.assert_array_equal(_measure(model, 0.1).delta_v, _measure(typed_model, 0.1).delta_v)
    testing.assert_array_equal(_measure(model, 0.2).delta_theta, _measure(typed_model, 0.2).delta_theta)
####


def test_checkpoint_restores_initial_baseline() -> None:
    model = ImuModel(_config(), default_rng(11))
    checkpoint = model.snapshot()
    resumed = ImuModel.from_checkpoint(checkpoint)

    expected = _measure(model, 1.0)
    actual = _measure(resumed, 1.0)
    testing.assert_array_equal(expected.delta_v, zeros(3))
    testing.assert_array_equal(actual.delta_v, zeros(3))
    assert expected.start_time == actual.start_time == 1.0
    assert expected.end_time == actual.end_time == 1.0
####


def test_checkpoint_handles_channels_without_flicker_components() -> None:
    model = ImuModel(rng=default_rng(13))
    checkpoint = model.snapshot()
    resumed = ImuModel.from_checkpoint(checkpoint)

    expected = _measure(model, 0.0)
    actual = _measure(resumed, 0.0)
    testing.assert_array_equal(expected.delta_v, actual.delta_v)
    testing.assert_array_equal(expected.delta_theta, actual.delta_theta)
####


def test_json_codec_supports_all_checkpointed_numpy_generators() -> None:
    for bit_generator_type in (PCG64, PCG64DXSM, Philox, SFC64, MT19937):
        model = ImuModel(_config(), Generator(bit_generator_type(23)))
        _measure(model, 0.0)
        checkpoint = model.snapshot()
        assert isinstance(model.checkpoint_codec, CheckpointCodecProtocol)

        payload = model.checkpoint_codec.encode(checkpoint)
        decoded = model.checkpoint_codec.decode(payload)
        resumed = ImuModel.from_checkpoint(decoded)
        expected = _measure(model, 0.1)
        actual = _measure(resumed, 0.1)

        testing.assert_array_equal(expected.delta_v, actual.delta_v)
        testing.assert_array_equal(expected.delta_theta, actual.delta_theta)
    ####
####


def test_file_helpers_accept_an_explicit_checkpoint_codec(tmp_path: Path) -> None:
    model = ImuModel(_config(), default_rng(31))
    _measure(model, 0.0)
    path = tmp_path / "custom-codec-checkpoint.bin"
    codec = PydanticJsonCheckpointCodec(ImuModelCheckpoint)

    model.save_checkpoint(path, codec=codec)
    resumed = ImuModel.load_checkpoint(path, codec=codec)
    expected = _measure(model, 0.1)
    actual = _measure(resumed, 0.1)

    testing.assert_array_equal(expected.delta_v, actual.delta_v)
    testing.assert_array_equal(expected.delta_theta, actual.delta_theta)
####
