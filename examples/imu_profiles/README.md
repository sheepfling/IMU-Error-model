# Example IMU profiles

These packaged resources are illustrative, best-effort estimates for representative
hardware and biological reference systems. They are documentation and example
inputs only: they are not package defaults, vendor configurations,
certification results, procurement specifications, or safety models.

The profiles are shipped inside the wheel under
`imu_error_model.data.example_profiles`, so installed users can access the same
versioned resources without assuming a repository checkout or filesystem layout.

Every numeric value is a notional and approximate estimate derived from
datasheets or other published sources; none is authoritative. The `baseline`
tag identifies a project simulation baseline; it does not indicate a vendor
baseline or validated hardware configuration.

## Profile structure

Hardware profiles use the canonical `ProfileDocument` shape. Each profile has
validated `metadata` containing:

- `family`, `vendor`, `grade`, and `revision`
- profile-level `source_date`
- searchable `tags` and an `active` flag
- one or more `sources`

Each source records its public `url`, review `date`, and optional backup links
in `archive_urls`:

```yaml
sources:
  - url: "https://example.com/datasheet.pdf"
    date: "2026-07-26"
    archive_urls:
      - "https://web.archive.org/web/20260726/https://example.com/datasheet.pdf"
```

These dates record provenance review dates, not asserted publication dates.
The source links are stored as structured profile data; nearby comments explain
assumptions but are not parsed as provenance.

## Loading a profile

Use the packaged-profile helpers when you want one of the bundled examples:

```python
from imu_error_model import (
    ImuModel,
    list_example_profile_categories,
    list_example_profiles,
    load_example_profile,
)

print(list_example_profile_categories())
print(list_example_profiles(None))  # category/profile.yaml paths
profile = load_example_profile("hg9900")
model = ImuModel(profile.config)
```

`load_example_profile()` accepts an exact packaged path, a filename, or an
unambiguous short name with or without its extension. If two categories contain
the same short name, it raises an ambiguity error; pass `category=` to select
one deliberately. It returns the validated `LoadedProfile`, including the model
name, sample period, typed metadata, and logical package-resource source
identifier. Use `load_profile()` or `load_profile_document()` for user-supplied
filesystem paths.

`list_example_profile_categories()` lists available categories. The default
`list_example_profiles()` call lists hardware estimates; passing a category lists
that category, while `list_example_profiles(None)` returns fully qualified names
for every packaged profile. Every name returned by the all-category form can be
passed directly to `load_example_profile()`.

For a deterministic zero-error baseline, use:

```python
from imu_error_model import ImuModel, load_noiseless_profile

model = ImuModel(load_noiseless_profile().config)
```

The noiseless profile is kept in the separate `baselines` category and is not
included in the hardware-estimate listing.

Included examples cover Honeywell HG1700/HG5700/HG9900 variants, ADIS16470,
ICM-42688-P, SBG Pulse-40, an iPhone-like consumer-MEMS benchmark, and a
human-vestibular reference profile. The latter two are comparison benchmarks,
not directly comparable physical IMUs.

Before using an estimate, verify its units, parameter interpretation, source
revision, and whether each value is measured data or a modeling assumption.

The repository-level analysis commands are documented in the root
[`README.md`](../../README.md). In particular:

```bash
python3 scripts/ci.py allan
python3 examples/dead_reckoning.py --duration 10
```
