# Example IMU profiles

These files are illustrative, best-effort estimates for representative
hardware and biological reference systems. They are documentation and example
inputs only: they are not package defaults, vendor configurations,
certification results, procurement specifications, or safety models.

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

`load_profile()` selects JSON, JSONC, YAML, or YML from the file extension and
returns a validated `ImuConfig`:

```python
from imu_error_model import ImuModel, load_profile

config = load_profile("examples/imu_profiles/hardware-estimates/hg9900.yaml")
model = ImuModel(config)
```

Use `load_profile_document()` when provenance is needed. It additionally preserves
the model name, sample period, metadata, and source path.

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
