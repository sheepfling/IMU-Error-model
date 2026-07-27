import json
import re
from pathlib import Path
from typing import Annotated, IO, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .config import ImuConfig, ProfileDocument, ProfileMetadata


def _strip_jsonc(text: str) -> str:
    """Remove JSONC comments and trailing commas without touching strings."""
    output: list[str] = []
    in_string = False
    escaped = False
    in_line_comment = False
    in_block_comment = False
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                output.append(char)
            ####
            index += 1
            continue
        ####
        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                index += 2
            else:
                if char == "\n":
                    output.append(char)
                ####
                index += 1
            ####
            continue
        ####
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            ####
            index += 1
            continue
        ####
        if char == '"':
            in_string = True
            output.append(char)
        elif char == "/" and next_char == "/":
            in_line_comment = True
            index += 2
            continue
        elif char == "/" and next_char == "*":
            in_block_comment = True
            index += 2
            continue
        else:
            output.append(char)
        ####
        index += 1
    ####
    return re.sub(r",\s*([}\]])", r"\1", "".join(output))
####

def _load_jsonc(text: str) -> object:
    data = json.loads(_strip_jsonc(text))
    return data
####


def _load_text(text: str, profile_format: str) -> object:
    """Parse profile text when no filename is available to select a format."""
    if profile_format in {"json", "jsonc"}:
        return _load_jsonc(text)
    ####
    data = yaml.safe_load(text)
    return data
####


def _read_stream(stream: IO[str] | IO[bytes]) -> str:
    """Read a text or binary stream and normalize it to Unicode text."""
    value = stream.read()
    return value.decode("utf-8") if isinstance(value, bytes) else value
####


def _normalize_format(profile_format: str) -> str:
    normalized = profile_format.lower().removeprefix(".")
    if normalized not in {"json", "jsonc", "yaml", "yml"}:
        raise ValueError("unsupported profile format; expected json, jsonc, yaml, or yml")
    ####
    return normalized
####


def _load_mapping(path: Path) -> dict[str, object]:
    """Parse a supported profile file into a mapping."""
    suffix = path.suffix.lower()
    if suffix not in {".json", ".jsonc", ".yaml", ".yml"}:
        raise ValueError(f"unsupported profile extension {path.suffix!r}; expected .json, .jsonc, .yaml, or .yml")
    ####
    out = _load_text(path.read_text(encoding="utf-8"), suffix[1:])
    if not isinstance(out, dict):
        raise ValueError("profile must be a mapping")
    ####
    return out
####


def _validate_document(data: dict[str, object]) -> ProfileDocument:
    """Validate a metadata-bearing profile through the canonical envelope model."""
    return ProfileDocument.model_validate(data)
####


def _config_from_document(document: ProfileDocument) -> ImuConfig:
    """Extract and revalidate the canonical config portion of a profile document."""
    config_fields = set(ImuConfig.model_fields)
    return ImuConfig.model_validate(document.model_dump(include=config_fields))
####


def _config_from_mapping(data: dict[str, object]) -> ImuConfig:
    """Validate either a config-only mapping or a complete profile document."""
    if {"model_name", "sample_period_s", "metadata"}.issubset(data):
        return _config_from_document(_validate_document(data))
    ####
    return ImuConfig.model_validate(data)
####


class LoadedProfile(BaseModel):
    """Validated profile configuration with preserved provenance information."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: Annotated[ImuConfig, Field(description="Validated canonical IMU configuration.")]
    model_name: Annotated[str, Field(min_length=1, description="Human-readable profile model name.")]
    sample_period_s: Annotated[
        float,
        Field(gt=0.0, allow_inf_nan=False, description="Nominal profile sample period in seconds."),
    ]
    metadata: Annotated[
        ProfileMetadata,
        Field(description="Validated provenance and classification metadata."),
    ]
    source_path: Annotated[
        Path,
        Field(description="Filesystem path or logical packaged-resource identifier from which the profile was loaded."),
    ]
####


def _loaded_profile_from_document(document: ProfileDocument, source_path: Path) -> LoadedProfile:
    """Build a validated loaded-profile object from a canonical document."""
    return LoadedProfile(
        config=_config_from_document(document),
        model_name=document.model_name,
        sample_period_s=document.sample_period_s,
        metadata=document.metadata,
        source_path=source_path,
    )
####


def load_profile_document_stream(
        stream: IO[str] | IO[bytes], *, fmt: str = "json", source_path: str | Path = "<stream>"
) -> LoadedProfile:
    """Load and validate a metadata-bearing profile from a text or binary stream.

    ``source_path`` is retained for diagnostics and may be a logical identifier
    when the profile comes from packaged resources rather than the filesystem.
    """
    data = _load_text(_read_stream(stream), _normalize_format(fmt))
    if not isinstance(data, dict):
        raise ValueError("profile must be a mapping")
    ####
    return _loaded_profile_from_document(_validate_document(data), Path(source_path))
####


def load_profile_document(path: str | Path) -> LoadedProfile:
    """Load a complete profile document from JSON, JSONC, YAML, or YML.

    Structured metadata is validated identically across all supported formats.
    Provenance is read from the validated `metadata.sources` fields.
    """
    source_path = Path(path)
    data = _load_mapping(source_path)
    return _loaded_profile_from_document(_validate_document(data), source_path)
####


def config_from_mapping(data: Mapping[str, object]) -> ImuConfig:
    """Construct a validated config from a canonical mapping."""
    if not isinstance(data, Mapping):
        raise ValueError("JSON profile root must be a mapping")
    ####
    return _config_from_mapping(dict(data))
####


def load_profile_stream(stream: IO[str] | IO[bytes], *, fmt: str = "json") -> ImuConfig:
    """Load a config or profile document from a text or binary stream.

    Streams have no filename extension, so `format` must be supplied when the
    content is not JSON. It accepts `json`, `jsonc`, `yaml`, `yml`, and the same
    values with a leading dot.
    """
    data = _load_text(_read_stream(stream), _normalize_format(fmt))
    if not isinstance(data, dict):
        raise ValueError("profile must be a mapping")
    ####
    return _config_from_mapping(data)
####


def load_profile(path: str | Path) -> ImuConfig:
    """Load a JSON or YAML profile based on its file extension.

    JSON and YAML profiles both return the validated `ImuConfig`.
    Use `load_profile_document()` when profile metadata is also needed.
    Extensions are case-insensitive; supported extensions are `.json`,
    `.jsonc`, `.yaml`, and `.yml`. JSONC supports `//` and `/* */`
    comments and trailing commas.
    """
    source_path = Path(path)
    data = _load_mapping(source_path)
    return _config_from_mapping(data)
####


def save_profile(config: ImuConfig, path: str | Path) -> None:
    """Write a canonical `ImuConfig` as formatted JSON."""
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(config.model_dump(), stream, indent=2)
        stream.write("\n")
    ####
####
