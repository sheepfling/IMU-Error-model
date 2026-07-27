import json
from collections.abc import Mapping
from io import BytesIO, StringIO
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from numpy import array, corrcoef, eye, isclose, testing, var, zeros
from numpy.random import default_rng

from imu_error_model import (
    AxisConfig,
    ImuConfig,
    ImuModel,
    LoadedProfile,
    config_from_mapping,
    load_profile,
    load_profile_document,
    load_profile_stream,
    save_profile,
)


def _baseline(model: ImuModel) -> None:
    model.measure(0.0, zeros(3), eye(3))
####

def _assert_values_close(actual: Any, expected: Any) -> None:
    if isinstance(actual, bool) or isinstance(expected, bool):
        assert actual == expected
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        assert isclose(float(actual), float(expected))
    elif isinstance(actual, Mapping) and isinstance(expected, Mapping):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_values_close(actual[key], expected[key])
        ####
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_values_close(actual_item, expected_item)
        ####
    else:
        assert actual == expected
    ####
####
def test_profile_round_trip(tmp_path: Path) -> None:
    config = ImuConfig(accelerometer=AxisConfig(quantization_step=0.01))
    path = tmp_path / "profile.json"
    save_profile(config, path)
    loaded = load_profile(path)
    quantization_step = loaded.accelerometer.quantization_step
    assert quantization_step is not None
    assert isclose(quantization_step, 0.01)
####


def test_load_profile_dispatches_json_and_yaml(short_correlation_profile_path: Path,
                                               hg9900_profile: LoadedProfile) -> None:
    json_config = load_profile(short_correlation_profile_path)
    yaml_config = hg9900_profile.config
    json_bias_correlation_time = json_config.accelerometer.bias_correlation_time
    yaml_bias_correlation_time = yaml_config.accelerometer.bias_correlation_time
    assert json_bias_correlation_time is not None
    assert yaml_bias_correlation_time is not None
    assert isclose(json_bias_correlation_time, 0.5)
    assert isclose(yaml_bias_correlation_time, 3600.0)
####


def test_json_and_yaml_documents_share_canonical_validation(
        tmp_path: Path,
        hardware_profile_texts: dict[str, str],
        hardware_profile_payloads: dict[str, dict[str, Any]],
) -> None:
    yaml_path = tmp_path / "profile.yaml"
    yaml_path.write_text(hardware_profile_texts["hg9900.yaml"], encoding="utf-8")
    json_path = tmp_path / "profile.json"
    json_path.write_text(
        json.dumps(hardware_profile_payloads["hg9900.yaml"], default=str),
        encoding="utf-8",
    )

    yaml_document = load_profile_document(yaml_path)
    json_document = load_profile_document(json_path)
    _assert_values_close(json_document.config.model_dump(), yaml_document.config.model_dump())
    _assert_values_close(json_document.metadata.model_dump(), yaml_document.metadata.model_dump())
####


def test_load_profile_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "profile.toml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported profile extension"):
        load_profile(path)
    ####
####


def test_load_profile_supports_jsonc_comments_and_trailing_commas(tmp_path: Path) -> None:
    path = tmp_path / "profile.JSONC"
    path.write_text(
        "{\n"
        "  // A beginner-friendly comment.\n"
        '  "accelerometer": {"white_noise_density": 0.1,},\n'
        '  "gyroscope": {\n'
        "    /* Block comments are supported too. */\n"
        '    "bias_std": 0.2,\n'
        "  },\n"
        "}\n",
        encoding="utf-8",
    )
    config = load_profile(path)
    assert isclose(config.accelerometer.white_noise_density, 0.1)
    assert isclose(config.gyroscope.bias_std, 0.2)
####


def test_config_loading_accepts_mappings_and_text_or_binary_streams() -> None:
    mapping = {"accelerometer": {"white_noise_density": 0.25}}
    expected = config_from_mapping(MappingProxyType(mapping))

    json_text = json.dumps(mapping)
    _assert_values_close(load_profile_stream(StringIO(json_text)).model_dump(), expected.model_dump())
    _assert_values_close(load_profile_stream(BytesIO(json_text.encode()), fmt=".json").model_dump(),
                         expected.model_dump())

    yaml_text = "accelerometer:\n  white_noise_density: 0.25\n"
    _assert_values_close(load_profile_stream(StringIO(yaml_text), fmt="yaml").model_dump(), expected.model_dump())
####


def test_stream_loader_accepts_complete_profile_documents(
        hardware_profile_payloads: dict[str, dict[str, Any]],
) -> None:
    document = hardware_profile_payloads["hg9900.yaml"].copy()
    document["model_name"] = "stream-test"
    document["sample_period_s"] = 0.01
    document["accelerometer"] = {"white_noise_density": 0.25}
    loaded = load_profile_stream(StringIO(json.dumps(document)))
    assert isclose(loaded.accelerometer.white_noise_density, 0.25)
####


def test_temperature_changes_bias_only_when_configured() -> None:
    config = ImuConfig(accelerometer=AxisConfig(thermal_bias_coefficient=0.1))
    model = ImuModel(config, default_rng(1))
    _baseline(model)
    output = model.measure(0.1, zeros(3), eye(3), temperature_celsius=35.0)
    testing.assert_allclose(output.acceleration, [1.0, 1.0, 1.0])
####


def test_correlated_covariance_is_reproducible() -> None:
    covariance = ((1.0, 0.5, 0.0), (0.5, 1.0, 0.0), (0.0, 0.0, 1.0))
    config = ImuConfig(accelerometer=AxisConfig(noise_covariance=covariance))
    a, b = ImuModel(config, default_rng(5)), ImuModel(config, default_rng(5))
    _baseline(a)
    _baseline(b)
    first = a.measure(0.01, zeros(3), eye(3))
    second = b.measure(0.01, zeros(3), eye(3))
    testing.assert_array_equal(first.delta_v, second.delta_v)
####


def test_correlated_covariance_controls_cross_axis_noise() -> None:
    covariance = ((1.0, 0.6, 0.0), (0.6, 1.0, 0.0), (0.0, 0.0, 1.0))
    config = ImuConfig(
        accelerometer=AxisConfig(noise_covariance=covariance, apply_clipping=False),
    )
    model = ImuModel(config, default_rng(17))
    _baseline(model)
    samples = array([model.measure((index + 1) * 0.01, zeros(3), eye(3)).acceleration for index in range(5000)])
    measured_correlation = corrcoef(samples[:, 0], samples[:, 1])[0, 1]
    assert isclose(measured_correlation, 0.6, atol=0.05)
####


def test_thermal_noise_coefficient_changes_noise_level() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(
            white_noise_density=0.2,
            thermal_noise_coefficient=0.1,
        ),
    )
    model = ImuModel(config, default_rng(19))
    _baseline(model)
    samples = array(
        [
            model.measure((index + 1) * 0.01, zeros(3), eye(3), temperature_celsius=35.0).acceleration
            for index in range(4000)
        ]
    )
    expected_variance = (0.2 * 2.0) ** 2 / 0.01
    assert isclose(var(samples[:, 0]), expected_variance, rtol=0.12)
####


def test_quantization_is_applied_to_reported_rate() -> None:
    config = ImuConfig(accelerometer=AxisConfig(quantization_step=0.5))
    model = ImuModel(config)
    _baseline(model)
    output = model.measure(1.0, array([0.6, 0, 0]), eye(3))
    testing.assert_allclose(output.acceleration, [0.5, 0, 0])
####


def test_white_noise_variance_matches_density() -> None:
    density = 0.2
    dt = 0.01
    config = ImuConfig(accelerometer=AxisConfig(white_noise_density=density))
    model = ImuModel(config, default_rng(22))
    _baseline(model)
    samples = array([model.measure((index + 1) * dt, zeros(3), eye(3)).acceleration for index in range(4000)])
    variance = var(samples[:, 0])
    assert isclose(variance, density ** 2 / dt, rtol=0.12)
####


def test_yaml_profile_preserves_provenance(hg9900_profile: LoadedProfile) -> None:
    profile = hg9900_profile
    assert profile.model_name == "HG9900"
    assert isclose(profile.sample_period_s, 0.0033333333)
    assert profile.metadata.vendor == "Honeywell"
    assert profile.metadata.sources
    output_scale = profile.config.accelerometer.output_scale
    bias_correlation_time = profile.config.gyroscope.bias_correlation_time
    assert output_scale is not None
    assert bias_correlation_time is not None
    assert isclose(output_scale, 1048576.0)
    assert isclose(bias_correlation_time, 3600.0)
####


def test_temperature_vectors_are_retained(sbg_pulse_40_profile: LoadedProfile) -> None:
    profile = sbg_pulse_40_profile
    testing.assert_allclose(profile.config.accelerometer.thermal_bias_coefficient, (0.0005, 0.0005, 0.0005))
    testing.assert_allclose(profile.config.gyroscope.thermal_scale_factor_coefficient, (0.00003, 0.00003, 0.00003))
####


def test_all_reference_profiles_load(
        hardware_profile_paths: tuple[str, ...],
        hardware_profiles: dict[str, LoadedProfile],
) -> None:
    assert len(hardware_profile_paths) == 14
    profiles = list(hardware_profiles.values())
    assert {profile.model_name for profile in profiles} >= {"HG9900", "HG5700CA01", "HG1700AG61"}
####
