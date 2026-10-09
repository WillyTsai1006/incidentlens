from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest

from inference.base import InferenceUnavailable
from inference.transformers_backend import TransformersBackend


@pytest.fixture
def fake_model_library(monkeypatch):
    calls = {}

    class Encoded(dict):
        def to(self, device):
            calls["input_device"] = device
            return self

    class Tokenizer:
        chat_template = "chat"
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            calls["chat"] = (messages, kwargs)
            return "formatted prompt"

        def __call__(self, text, **kwargs):
            calls["tokenize"] = (text, kwargs)
            return Encoded(input_ids=SimpleNamespace(shape=(1, 3)))

        def decode(self, tokens, **kwargs):
            calls["decode"] = (tokens, kwargs)
            return '{"hypotheses": []}'

    class Model:
        config = SimpleNamespace(max_position_embeddings=4096)

        def to(self, device):
            calls["model_device"] = device
            return self

        def eval(self):
            calls["eval"] = True

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return [[1, 2, 3, 9, 10]]

    model, tokenizer = Model(), Tokenizer()

    def load_model(path, **kwargs):
        calls["load_model_count"] = calls.get("load_model_count", 0) + 1
        calls["load_model"] = (path, kwargs)
        return model

    def load_tokenizer(path, **kwargs):
        calls["load_tokenizer"] = (path, kwargs)
        return tokenizer

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer)))
    return calls, model, tokenizer


def test_local_adapter_offline_flags_greedy_generation_and_prompt_stripping(tmp_path, fake_model_library):
    calls, model, tokenizer = fake_model_library
    backend = TransformersBackend(tmp_path)
    assert backend.generate("prompt", max_new_tokens=16) == '{"hypotheses": []}'
    assert backend.is_mock is False
    assert calls["load_model"][1] == calls["load_tokenizer"][1] == {
        "local_files_only": True, "trust_remote_code": False}
    assert calls["model_device"] == calls["input_device"] == "cpu"
    assert calls["generate"]["do_sample"] is False and calls["generate"]["max_new_tokens"] == 16
    assert calls["decode"][0] == [9, 10]
    assert calls["eval"]


def test_local_adapter_plain_tokenizer_path(tmp_path, fake_model_library):
    calls, model, tokenizer = fake_model_library
    tokenizer.chat_template = None
    TransformersBackend(tmp_path).generate("plain prompt", max_new_tokens=16)
    assert calls["tokenize"][0] == "plain prompt"
    assert calls["tokenize"][1]["add_special_tokens"] is True


def test_model_loaded_only_once(tmp_path, fake_model_library):
    calls, model, tokenizer = fake_model_library
    backend = TransformersBackend(tmp_path)
    backend.generate("one", max_new_tokens=16)
    backend.generate("two", max_new_tokens=16)
    assert backend._model is model
    assert calls["load_model_count"] == 1


def test_unavailable_requested_device_reports_clear_error(tmp_path, fake_model_library):
    calls, model, tokenizer = fake_model_library

    def fail(device):
        raise RuntimeError("device not available")

    model.to = fail
    with pytest.raises(InferenceUnavailable, match="device is unavailable"):
        TransformersBackend(tmp_path, device="xpu").generate("prompt")


def test_adapter_integrates_with_full_pipeline_using_library_stubs(tmp_path, fake_model_library, demo_graph):
    from core.diagnosis import diagnose
    from inference.mock_backend import MockBackend

    calls, model, tokenizer = fake_model_library

    def grounded_decode(tokens, **kwargs):
        prompt = calls["chat"][0][0]["content"]
        return MockBackend().generate(prompt)

    tokenizer.decode = grounded_decode
    result = diagnose(*demo_graph, TransformersBackend(tmp_path))
    assert result.primary_root_cause.kind == "duplicate_models"
    assert result.inference_backend == "transformers-local" and result.execution_mode == "local"
    # This is a library-stub contract test, not evidence of real model execution.


def test_context_overflow_is_rejected_without_silent_truncation(tmp_path, fake_model_library):
    calls, model, tokenizer = fake_model_library
    model.config.max_position_embeddings = 8
    with pytest.raises(InferenceUnavailable, match="not truncated"):
        TransformersBackend(tmp_path).generate("prompt", max_new_tokens=16)
    assert "generate" not in calls


def test_tokenizer_context_limit_is_respected(tmp_path, fake_model_library):
    calls, model, tokenizer = fake_model_library
    tokenizer.model_max_length = 8
    with pytest.raises(InferenceUnavailable, match="not truncated"):
        TransformersBackend(tmp_path).generate("prompt", max_new_tokens=16)
    assert "generate" not in calls


def test_unavailable_dependencies_reported_without_network(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(InferenceUnavailable, match="optional"):
        TransformersBackend(tmp_path).generate("prompt")


def test_missing_directory_checked_before_import(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(InferenceUnavailable, match="directory"):
        TransformersBackend(tmp_path / "missing").generate("prompt")


def test_local_generation_failure_never_silently_falls_back(tmp_path, fake_model_library):
    calls, model, tokenizer = fake_model_library

    def fail(**kwargs):
        raise RuntimeError("device failure")

    model.generate = fail
    with pytest.raises(InferenceUnavailable, match="no mock fallback"):
        TransformersBackend(tmp_path).generate("prompt", max_new_tokens=16)


@pytest.mark.parametrize("options", [{"temperature": 1}, {"max_new_tokens": 0}])
def test_deterministic_generation_options_required(tmp_path, options):
    with pytest.raises(ValueError, match="temperature=0"):
        TransformersBackend(tmp_path).generate("prompt", **options)
