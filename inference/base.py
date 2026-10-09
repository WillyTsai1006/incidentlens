from typing import Protocol


class InferenceUnavailable(RuntimeError):
    """The requested local backend cannot execute."""


class InferenceBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def is_mock(self) -> bool: ...

    def generate(self, prompt: str, *, max_new_tokens: int = 2048,
                 temperature: float = 0.0) -> str: ...
