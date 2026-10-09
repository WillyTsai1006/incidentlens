"""Optional offline Transformers adapter; model libraries stay outside core."""

from pathlib import Path
from time import perf_counter

from inference.base import InferenceUnavailable


class TransformersBackend:
    name = "transformers-local"
    is_mock = False

    def __init__(self, model_path: str | Path, *, device: str = "cpu", cpu_threads: int | None = None):
        self.model_path = Path(model_path).resolve()
        self.model_name = str(self.model_path)
        self.device = device
        self._model = self._tokenizer = self._torch = None
        self.cpu_threads = cpu_threads
        self.last_generation_metrics = None

    def _load(self):
        if not self.model_path.is_dir():
            raise InferenceUnavailable("Local model directory does not exist; no model will be downloaded.")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except (ImportError, OSError) as exc:
            raise InferenceUnavailable("Install the optional 'local' dependencies to run Transformers.") from exc
        try:
            if self.device == "cpu" and self.cpu_threads is not None and hasattr(torch, "set_num_threads"):
                torch.set_num_threads(self.cpu_threads)
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), local_files_only=True, trust_remote_code=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), local_files_only=True, trust_remote_code=False,
            ).to(self.device)
            model.eval()
        except (OSError, ValueError, RuntimeError) as exc:
            raise InferenceUnavailable("Local model or requested device is unavailable.") from exc
        self._torch, self._tokenizer, self._model = torch, tokenizer, model

    def generate(self, prompt: str, *, max_new_tokens: int = 2048,
                 temperature: float = 0.0) -> str:
        if temperature != 0 or max_new_tokens <= 0:
            raise ValueError("Diagnosis generation requires temperature=0 and a positive token limit")
        self.last_generation_metrics = None
        cold_start = self._model is None
        if self._model is None:
            self._load()
        try:
            formatted = prompt
            if self._tokenizer.chat_template:
                formatted = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                )
            encoded = self._tokenizer(formatted, return_tensors="pt", add_special_tokens=not bool(self._tokenizer.chat_template))
            encoded = encoded.to(self.device)
            prompt_length = encoded["input_ids"].shape[-1]
            limits = [getattr(self._model.config, "max_position_embeddings", None),
                      getattr(self._model.config, "n_positions", None),
                      getattr(self._tokenizer, "model_max_length", None)]
            limits = [limit for limit in limits if isinstance(limit, int) and limit > 0]
            if limits and prompt_length + max_new_tokens > min(limits):
                raise InferenceUnavailable("Evidence prompt exceeds local model context; evidence was not truncated.")
            options = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
            if self._tokenizer.eos_token_id is not None:
                options["pad_token_id"] = self._tokenizer.eos_token_id
            adapter = getattr(self._torch, self.device, None) if self.device != "cpu" else None
            if adapter is not None:
                adapter.synchronize()
            generation_start = perf_counter()
            with self._torch.inference_mode():
                tokens = self._model.generate(**encoded, **options)
            if adapter is not None:
                adapter.synchronize()
            generation_ms = (perf_counter() - generation_start) * 1000
            new_tokens = tokens[0][prompt_length:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            allocated = None
            if adapter is not None:
                try:
                    allocated = adapter.memory_allocated() / 2**20
                except (AttributeError, RuntimeError):
                    pass
            self.last_generation_metrics = dict(input_tokens=prompt_length, output_tokens=len(new_tokens),
                                                generation_latency_ms=generation_ms,
                                                device_allocated_memory_mb=allocated, cold_start=cold_start)
            return text
        except (OSError, ValueError, RuntimeError) as exc:
            if isinstance(exc, InferenceUnavailable):
                raise
            raise InferenceUnavailable("Local generation failed; no mock fallback was used.") from exc
