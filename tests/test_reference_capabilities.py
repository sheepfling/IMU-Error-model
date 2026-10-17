from pathlib import Path

from playground.calibration.reference_capabilities import EVIDENCE, unclassified_reference_fields
from playground.calibration.reference_equipment import AngleLaw, TableGeometry
from playground.calibration.reference_observations import EncoderCorrection, EncoderErrorConfig


def test_all_reference_fields_have_explicit_evidence() -> None:
    assert unclassified_reference_fields()==()
    root=Path(__file__).resolve().parents[1]
    for status,path in EVIDENCE.values():
        assert status
        assert (root/path).is_file()
    ####
####


def test_reference_schemas_and_configurations_roundtrip() -> None:
    import json
    root=Path(__file__).resolve().parents[1]
    for model in (AngleLaw,TableGeometry,EncoderErrorConfig,EncoderCorrection):
        instance=model()
        assert model.model_validate_json(instance.model_dump_json())==instance
        schema=json.loads((root/"schemas"/f"{model.__name__}_v1.schema.json").read_text())
        assert schema==model.model_json_schema()
    ####
####
