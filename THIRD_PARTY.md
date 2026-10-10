# Third-party components

IncidentLens's own code is covered by [LICENSE](LICENSE). External components
and model weights remain subject to their respective licenses and notices;
the repository license does not replace them. Weights are downloaded separately
and are not distributed in this repository. Consult the exact version/revision
you use; this inventory is not a legal compatibility determination.

| Project | Role | Official project / model |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct | Optional local diagnostic inference and plain-LLM comparison; pinned revision `7ae557604adf67be50417f59c2c2f167def9a775` | [Model repository and notices](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| PyTorch | Optional CPU/CUDA tensor execution and model generation | [PyTorch](https://pytorch.org/) |
| Hugging Face Transformers | Optional offline tokenizer/model adapter | [Transformers](https://github.com/huggingface/transformers) |
| Hugging Face Hub | Explicit one-time model download tooling | [huggingface_hub](https://github.com/huggingface/huggingface_hub) |
| Streamlit | Local demo UI and AppTest regression checks | [Streamlit](https://github.com/streamlit/streamlit) |
| Pydantic | Typed schemas and strict validation | [Pydantic](https://github.com/pydantic/pydantic) |
| PyYAML | Deterministic YAML parsing | [PyYAML](https://github.com/yaml/pyyaml) |
| psutil | Measured host/process memory snapshots | [psutil](https://github.com/giampaolo/psutil) |
| pytest | Unit/integration tests | [pytest](https://github.com/pytest-dev/pytest) |
| setuptools | Python package build backend | [setuptools](https://github.com/pypa/setuptools) |
| uv | Documented environment/package management tooling | [uv](https://github.com/astral-sh/uv) |

GitHub CI also uses [actions/checkout](https://github.com/actions/checkout) and
[actions/setup-python](https://github.com/actions/setup-python), each subject
to its own repository license. Transitive packages are not exhaustively listed;
dependency metadata and installed distributions provide their notices.
