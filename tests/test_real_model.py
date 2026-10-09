import os

import pytest

from app.phase3 import default_profiles, run_phase3
from test_loader import DEMO


@pytest.mark.real_model
@pytest.mark.skipif(not os.environ.get("INCIDENT_MODEL_PATH"), reason="Set INCIDENT_MODEL_PATH for optional offline real-model validation")
def test_offline_real_model_diagnosis(tmp_path):
    profiles = default_profiles("transformers", os.environ["INCIDENT_MODEL_PATH"])
    run = run_phase3(DEMO, profiles=profiles, execution_mode="local", save_root=tmp_path)
    assert run["diagnosis"]["execution_mode"] == "local"
    assert run["diagnosis"]["primary_root_cause"]["kind"] == "duplicate_models"
    assert run["diagnosis"]["evidence_grounding_rate"] == 1
    call = run["telemetry"]["inference_calls"][-1]
    assert call["input_tokens"] > 0 and call["output_tokens"] > 0
    assert call["output_tokens_per_second"] > 0
