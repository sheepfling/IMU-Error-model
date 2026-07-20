import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import AxisConfig, ImuConfig


def _as_float(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _yaml_axis_mapping(data: dict[str, Any]) -> dict[str, Any]:
    temperature = data.get("temperature_model", {})
    return {
        "white_noise_density": _as_float(data.get("noise_spectral_density")),
        "turn_on_bias_std": _as_float(data.get("turn_on_bias_std")),
        "bias_std": _as_float(data.get("long_term_bias_std")),
        "bias_correlation_time": None if data.get("bias_correlation_time") is None else _as_float(data["bias_correlation_time"]),
        "flicker_bias_std": _as_float(data.get("flicker_bias_std")),
        "flicker_min_correlation_time": (
            None if data.get("flicker_min_correlation_time") is None else _as_float(data["flicker_min_correlation_time"])
        ),
        "flicker_max_correlation_time": (
            None if data.get("flicker_max_correlation_time") is None else _as_float(data["flicker_max_correlation_time"])
        ),
        "flicker_components": int(data.get("flicker_components", 8)),
        "scale_factor": _as_float(data.get("linear_scale_factor")),
        "nonlinear_factor": _as_float(data.get("nonlinear_scale_factor")),
        "misalignment_std": _as_float(data.get("misalignment_std")),
        "measurement_range": None if data.get("measurement_range") is None else _as_float(data["measurement_range"]),
        "reference_temperature": _as_float(temperature.get("reference_temp_C"), 25.0),
        "thermal_bias_coefficient": tuple(_as_float(value) for value in temperature.get("bias_linear_per_C", [0.0, 0.0, 0.0])),
        "thermal_scale_factor_coefficient": tuple(_as_float(value) for value in temperature.get("scale_linear_per_C", [0.0, 0.0, 0.0])),
        "apply_clipping": data.get("apply_clipping", True),
        "output_scale": _as_float(data.get("scale"), 1.0),
    }


@dataclass(frozen=True)
class LoadedImuProfile:
    """A YAML profile plus its preserved source and provenance information."""

    config: ImuConfig
    model_name: str
    sample_period_s: float
    metadata: dict[str, Any]
    reference_links: tuple[str, ...]
    source_path: Path


def load_yaml_profile(path: str | Path) -> LoadedImuProfile:
    """Load a YAML hardware estimate into the current model schema.

    Reference links in leading comments are extracted alongside the parsed
    configuration so provenance remains available to callers.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML profiles require the 'profiles' extra") from exc
    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("YAML profile root must be a mapping")
    return LoadedImuProfile(
        config=ImuConfig(
            accelerometer=AxisConfig(**_yaml_axis_mapping(data.get("accelerometer", {}))),
            gyroscope=AxisConfig(**_yaml_axis_mapping(data.get("gyro", {}))),
        ),
        model_name=data["model_name"],
        sample_period_s=float(data["sample_period_s"]),
        metadata=dict(data.get("metadata", {})),
        reference_links=tuple(re.findall(r"^#.*?(https?://\S+)", text, re.MULTILINE)),
        source_path=source_path,
    )


def config_from_mapping(data: dict[str, Any]) -> ImuConfig:
    """Construct a validated config from a JSON-compatible mapping."""
    return ImuConfig(
        accelerometer=AxisConfig(**data.get("accelerometer", {})),
        gyroscope=AxisConfig(**data.get("gyroscope", {})),
        output_scale_accelerometer=data.get("output_scale_accelerometer", 1.0),
        output_scale_gyroscope=data.get("output_scale_gyroscope", 1.0),
    )


def load_profile(path: str | Path) -> ImuConfig:
    with Path(path).open(encoding="utf-8") as stream:
        return config_from_mapping(json.load(stream))


def save_profile(config: ImuConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(asdict(config), stream, indent=2)
        stream.write("\n")
