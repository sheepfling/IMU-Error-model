"""Access to the versioned example profiles shipped with the package."""

from collections.abc import Iterator
from importlib.resources import files
from importlib.resources.abc import Traversable
from io import StringIO
from pathlib import Path, PurePosixPath

from .profiles import LoadedProfile, load_profile_document_stream


_PROFILE_PACKAGE = "imu_error_model.data.example_profiles"
_DEFAULT_CATEGORY = "hardware_estimates"
_BASELINE_CATEGORY = "baselines"
_SUPPORTED_SUFFIXES = {".json", ".jsonc", ".yaml", ".yml"}

def _resource_part(value: str | Path, label: str) -> tuple[str, ...]:
    """Validate a resource selector and return its normalized path parts."""
    text = value.as_posix() if isinstance(value, Path) else value
    path = PurePosixPath(text)
    if path.is_absolute() or not text or "\\" in text or ".." in path.parts:
        raise ValueError(f"invalid {label} resource name: {value!r}")
    ####
    return path.parts
####


def _category_resource(category: str) -> Traversable:
    """Return a packaged profile category resource."""
    parts = _resource_part(category, "category")
    if len(parts) != 1:
        raise ValueError("profile category must be a single resource directory name")
    ####
    return files(_PROFILE_PACKAGE).joinpath(*parts)
####


def _category_names() -> tuple[str, ...]:
    """Return all packaged example-profile categories."""
    root = files(_PROFILE_PACKAGE)
    return tuple(
        sorted(
            item.name
            for item in root.iterdir()
            if item.is_dir()
            and not item.name.startswith((".", "_"))
            and any(_iter_profile_resources(item))
        )
    )
####


def _iter_profile_resources(resource: Traversable, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, Traversable]]:
    """Yield packaged profile paths and resources recursively."""
    for item in resource.iterdir():
        if item.is_dir():
            yield from _iter_profile_resources(item, (*prefix, item.name))
        elif PurePosixPath(item.name).suffix.lower() in _SUPPORTED_SUFFIXES:
            yield PurePosixPath(*prefix, item.name).as_posix(), item
        ####
    ####
####


def _all_profile_resources(category: str | None) -> list[tuple[str, str, Traversable]]:
    """Collect profile paths with their category names."""
    categories = (_resource_part(category, "category")[0],) if category is not None else _category_names()
    resources: list[tuple[str, str, Traversable]] = []
    for category_name in categories:
        category_resource = _category_resource(category_name)
        resources.extend((category_name, relative_name, resource) for relative_name, resource in
                         _iter_profile_resources(category_resource))
    ####
    return resources
####


def list_example_profile_categories() -> tuple[str, ...]:
    """List the packaged example-profile categories."""
    return _category_names()
####


def list_example_profiles(category: str | None = _DEFAULT_CATEGORY) -> tuple[str, ...]:
    """List packaged example profiles, optionally across all categories.

    With the default category, or an explicit category, names are relative to
    that category. When ``category`` is ``None``, names are fully qualified as
    ``"category/profile.yaml"`` so every returned value is directly addressable.
    """
    if category is not None:
        resource = _category_resource(category)
        if not resource.is_dir():
            raise FileNotFoundError(f"example profile category not found: {category}")
        ####
    ####
    resources = _all_profile_resources(category)
    if category is None:
        return tuple(sorted(f"{category_name}/{relative_name}" for category_name, relative_name, _ in resources))
    ####
    return tuple(sorted(relative_name for _, relative_name, _ in resources))
####


def _resolve_profile(name: str | Path, category: str | None) -> tuple[str, str, Traversable]:
    """Resolve an exact packaged path or an unambiguous short profile name."""
    selector = PurePosixPath(name)
    parts = _resource_part(name, "profile")
    if category is None:
        categories = _category_names()
    else:
        category_parts = _resource_part(category, "category")
        if len(category_parts) != 1:
            raise ValueError("profile category must be a single resource directory name")
        ####
        categories = (category_parts[0],)
        if len(parts) > 1 and parts[0] == category_parts[0]:
            selector = PurePosixPath(*parts[1:])
        ####
    ####
    if category is None and len(parts) > 1 and parts[0] in categories:
        categories = (parts[0],)
        selector = PurePosixPath(*parts[1:])
    ####
    resources = _all_profile_resources(categories[0]) if len(categories) == 1 else _all_profile_resources(None)
    selected = selector.as_posix()
    suffix = selector.suffix.lower()
    if suffix in _SUPPORTED_SUFFIXES:
        exact = [item for item in resources if item[1] == selected]
        matches = exact or [item for item in resources if PurePosixPath(item[1]).name == selector.name]
    else:
        exact = [item for item in resources if PurePosixPath(item[1]).with_suffix("").as_posix() == selected]
        matches = exact or [item for item in resources if PurePosixPath(item[1]).stem == selector.name]
    ####
    if not matches:
        raise FileNotFoundError(f"example profile not found: {name}")
    ####
    if len(matches) > 1:
        choices = ", ".join(f"{item[0]}/{item[1]}" for item in matches)
        raise ValueError(f"ambiguous example profile {name!r}; matches: {choices}")
    ####
    return matches[0]
####


def load_example_profile(name: str | Path, *, category: str | None = None) -> LoadedProfile:
    """Load one packaged example profile with canonical validation and metadata.

    ``name`` may be an exact packaged path, a filename with or without its
    extension, or an unambiguous short stem. When ``category`` is omitted, all
    packaged categories are searched and ambiguous matches raise ``ValueError``.
    The returned ``source_path`` is a logical package-resource identifier.
    """
    category_name, relative_name, resource = _resolve_profile(name, category)
    source_path = f"package://{_PROFILE_PACKAGE}/{category_name}/{relative_name}"
    return load_profile_document_stream(
        StringIO(resource.read_text(encoding="utf-8")),
        fmt=PurePosixPath(resource.name).suffix,
        source_path=source_path,
    )
####


def read_example_profile(name: str | Path, *, category: str | None = None) -> str:
    """Read one packaged example profile as UTF-8 text using the same resolver."""
    _, _, resource = _resolve_profile(name, category)
    return resource.read_text(encoding="utf-8")
####


def load_noiseless_profile() -> LoadedProfile:
    """Load the packaged deterministic zero-error IMU baseline."""
    return load_example_profile("ideal.yaml", category=_BASELINE_CATEGORY)
####
