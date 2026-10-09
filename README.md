# IncidentLens

**Evidence-Grounded Local Incident Diagnosis** — turn logs, code, configurations
and metrics into inspectable hypotheses entirely on local hardware.

**Implemented milestone: Phase 3 — hardware-aware local demo.** Local incident
sources become structured evidence and correlations, three candidate hypotheses,
explicit verification results, application-computed scores, and a ranked diagnosis.
The default CLI demo uses a clearly labeled deterministic mock backend. Phase 3 adds
inspectable routing, measured telemetry, complete run snapshots and a Streamlit
UI. Real offline Qwen2.5-0.5B-Instruct inference was validated on CPU and the
RTX 5060 Laptop GPU using CUDA. Intel Arc Pro B70 and XPU execution remain
unvalidated. See the [matched CPU/CUDA comparison](docs/rtx5060_validation.md).

## Setup and run

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Run from the repository
root. Installing dependencies needs network access; running the mock demo does not.
The first installation block installs the core, tests and UI, **not a real model
runtime or weights**. The CLI demo works immediately in mock mode. For a real
Streamlit competition run, complete the CUDA or CPU setup below before Analyze.
In a core/UI-only environment, select Mock under Advanced Runtime Settings.

```powershell
uv --cache-dir .uv-cache venv .venv --python python
uv --cache-dir .uv-cache pip install --python .venv/Scripts/python.exe -e ".[dev,ui]"
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m app.main --demo
.venv/Scripts/python.exe -m app.main --demo --save-run
.venv/Scripts/python.exe -m streamlit run app/ui.py --browser.gatherUsageStats false
```

On Linux/macOS use `.venv/bin/python` instead of `.venv/Scripts/python.exe` for
install, test, and CLI commands. With an activated environment the equivalent
commands are `pytest -q` and `python -m app.main --demo`.

To inspect your own directory using the supported scenario formats, or export
only the original Phase 1 artifacts:

```powershell
.venv/Scripts/python.exe -m app.main --incident path/to/incident
.venv/Scripts/python.exe -m app.main --demo --core-only
```

The CLI writes evidence, relations, and the full diagnosis as JSON to stdout and
operational logging to stderr. Invalid inputs return exit code 2. Directory
loading selects `.log`, `.yaml`, `.yml`, `.py`, and `.txt` files; JSON expectation
files are ignored. Explicit file loading rejects unsupported types. Source code
is parsed, never executed. Tests use `.pytest-tmp/` under the project to avoid
restricted Windows system temp directories; pytest manages and clears this
dedicated test directory.

`--demo` preserves the Phase 2 JSON contract. `--phase3` or `--save-run` adds
routing and telemetry and saves a timestamped directory under `runs/`.
In Streamlit, click **Run GPU OOM Demo**, then **Analyze Incident**. The UI defaults
to real local Qwen inference on CUDA when available. Otherwise it explicitly
selects CPU and displays a warning. Weights and compatible PyTorch must already
be installed; execution errors do not silently fall back to CPU or mock results.
The bundled CUDA model uses the existing validated RTX development profile.
Open **Advanced Runtime Settings** to select mock mode, a custom model path,
device or latency preference. **Upload Your Incident** reveals the input uploader.

The result leads with the diagnosis, cited evidence and independent hypothesis
scores. Verification, recommended next steps and live runtime measurements follow;
raw inputs, evidence, relations and metadata remain available in collapsed audit
sections. Missing metrics show **N/A**; **GPU Memory Snapshot** displays actual
post-call allocated memory rather than an unmeasured peak. Home capability cards
summarize evidence grounding, verification and private local inference.
Intel Arc Pro B70
remains the competition target and is unvalidated. See the
[UI polish and browser validation report](docs/ui_polish_status.md).
The [final demo polish and latency investigation](docs/final_demo_polish.md)
records why the archived 8.85-second benchmark is not a live-UI latency promise.

## Real local inference

For the validated **RTX 5060 development GPU**, use the CUDA build below in the
same environment created above. This configuration was tested on Windows with
Python 3.13; other operating systems/interpreters are not covered by that GPU
validation. Do not install the CPU wheel afterwards if you intend to use CUDA.

```powershell
uv --cache-dir .uv-cache pip install --python .venv/Scripts/python.exe 'torch==2.14.1+cu130' --index-url https://download.pytorch.org/whl/cu130
uv --cache-dir .uv-cache pip install --python .venv/Scripts/python.exe 'transformers==4.57.6' 'huggingface-hub==0.36.2'
.venv/Scripts/hf.exe download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-dir models/Qwen2.5-0.5B-Instruct --include '*.json' '*.safetensors' '*.txt' 'README.md' 'LICENSE*'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.venv/Scripts/python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
.venv/Scripts/python.exe -m app.main --demo --save-run --backend transformers --profiles profiles/demo_rtx5060.json --device cuda
.venv/Scripts/python.exe -m streamlit run app/ui.py --browser.gatherUsageStats false
```

Weights are deliberately excluded from source control. Downloading them needs
network access once; subsequent inference is offline. The release audit reused
the existing pinned local weights in its fresh environment, rather than claiming
a second model download. B70/XPU is a target architecture and remains unvalidated.

The **CPU alternative** uses the following setup instead of the CUDA wheel:

The earlier CPU configuration used PyTorch 2.14.1+cpu, Transformers 4.57.6,
four CPU threads, and the pinned Qwen model below. Download weights once; the
diagnosis itself runs offline. No incident contents are uploaded.

```powershell
uv --cache-dir .uv-cache pip install --python .venv/Scripts/python.exe 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
uv --cache-dir .uv-cache pip install --python .venv/Scripts/python.exe 'transformers==4.57.6' 'huggingface-hub==0.36.2'
.venv/Scripts/hf.exe download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-dir models/Qwen2.5-0.5B-Instruct --include '*.json' '*.safetensors' '*.txt' 'README.md' 'LICENSE*'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.venv/Scripts/python.exe -m app.main --demo --save-run --backend transformers --profiles profiles/demo_local.json --device cpu
```

The adapter loads with `local_files_only=True` and `trust_remote_code=False`,
generates greedily, and rejects context overflow without dropping evidence.
It never downloads a model or uploads incident contents. `--device cuda` or `xpu`
requires a compatible local PyTorch/device installation; B70 compatibility has
not been tested. CPU is the default. Missing packages, model, or device produce
an error. Invalid candidate JSON/grounding gets two attempts, then an error.
There is **no silent fallback from local execution to mock results**. Phase 3
uses a smaller per-candidate evidence-ID schema: three candidate families, up to
two attempts each. Known IDs expand into exact normalized evidence quotations;
extra factual prose and model scores are rejected. The legacy Phase 2 local
bundle schema is unchanged and was unreliable with this small model. Use the
Phase 3 command above for the validated real demo.
See the [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and
[Transformers generation documentation](https://huggingface.co/docs/transformers/llm_tutorial).

`profiles/demo_local.json` contains one enabled CPU model and a disabled,
unvalidated B70/XPU target profile. Profile memory requirements and preference
ranks are configuration estimates, not benchmarks. Routing checks task, context,
complexity, matching memory pool and latency preference; every decision explains
its selection and rejected candidates. No second real model or distributed
scheduler is implied when the preferred reasoning tier is unavailable.

Optional real tests require existing weights:

```powershell
$env:INCIDENT_MODEL_PATH=(Resolve-Path 'models/Qwen2.5-0.5B-Instruct').Path
.venv/Scripts/python.exe -m pytest -q
```

Without that variable, the two real-model tests skip. UI tests require the `ui`
extra. The release-audit suite contains **181 tests** including the two optional
real-model tests; without real-model weights, those two tests skip.
The original 161 tests remain covered. CLI, Streamlit AppTest and actual browser
paths have been exercised. Browser review covers 1920x1080 and 1366x768, with
real CUDA, real CPU and labeled mock runs. Physical projector presentation and
Intel Arc Pro B70 execution remain unvalidated.

## Bundled synthetic incident

Four workers load a synthetic `demo-4b-model` in fp16. GPU readings show a zero
baseline followed by 9.5, 19.0, and 28.5 GiB as three workers start. The fourth
worker's requested 9.5 GiB does not fit in the remaining 3.5 GiB; service startup
then fails. Projected demand is `4 × 9.5 = 38 GiB`, exceeding the 32 GiB ceiling.

The target Intel Arc Pro B70 is advertised with [32 GB GDDR6](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html).
The fixture uses a nominal synthetic 32 GiB ceiling, not measured usable VRAM or
a literal decimal GB conversion. Its 28.5 GiB successful startup footprint fits
within 32 decimal GB as well; 38 GiB exceeds both interpretations. Actual usable
capacity depends on device settings and runtime reservations. The 4B fp16 model's
9.5 GiB increment includes illustrative startup overhead, not measured weights.

The output includes 21 observed evidence items, 4 provenance-linked derived facts,
30 explicit relations, and three verified/ranked candidates:

| Hypothesis | Evidence score |
| --- | ---: |
| Duplicate model loading across inference workers | 0.866667 |
| Excessive runtime/batch-related allocation | 0.153333 |
| Memory leak | 0.113333 |

These scores are heuristic, not probabilities. The primary explanation still
requires process-level replica confirmation. The leak hypothesis lacks a
steady-state observation; startup steps weaken it without ruling out a leak.
Final factual claims quote existing evidence with validated IDs. The grounding
rate is 1.0 for those quotations, not a measure of RCA accuracy.
All incident fixture values are synthetic. The real diagnosis model is separate
from the incident's illustrative 4B model. The last independent RTX demo-polish
run measured **37.16 s end-to-end / 32.40 s generation**, **5.80 tokens/s**, and a
**1918.6 MiB allocation snapshot**; its fresh-server first analysis measured
**59.16 / 33.48 s**. These are development observations, not latency guarantees
or B70 benchmarks. The original **8.85 / 4.68 s** CLI result and initial CPU
130–295 s records are explicitly **historical**, with different startup/system
conditions. See [current measurement scopes](docs/final_demo_polish.md) and
[the release audit](RELEASE_READINESS.md) for the latest fresh-environment run.
The fresh-environment audit's first real CUDA analysis measured **152.56 s
end-to-end / 33.81 s generation**, **5.56 tokens/s**, and **1917.3 MiB** allocated
memory snapshot. Its substantial non-generation cold-start cost is retained in
the displayed end-to-end value; do not use the older 8.85-second observation as
a current demo promise. Allow startup/rehearsal time before recording.
Missing GPU metrics and mock token counts are
JSON null, never invented. See the [Phase 3 report](docs/phase3_status.md) for
hardware, timing scopes, limitations and an archived run.

## Phase boundaries

Phase 1 covers schemas, parsers, evidence repository, correlations, initial
confidence arithmetic, and tests. `--core-only` retains that output contract.
Phase 2 adds backend abstraction, mock inference, optional local adapter,
structured candidates, verification, scoring, final diagnosis, strict JSON and
grounding validation with retry, and full mock integration tests.

Candidate families, cautious wording, verification criteria, and recommendations
come from an explicit three-family catalog. Inference supplies candidate IDs and
evidence quotations. Free-form model factual prose and arbitrary scores are
rejected. This remains a targeted prototype, not general production RCA.

Phase 3 adds `app.phase3.run_phase3`, structured profiles, routed inference,
typed telemetry, run persistence and Streamlit. Evidence extraction, verification
and authoritative confidence remain deterministic and outside the model.
Artifacts include inputs, evidence, graph relations, hypotheses, verification,
diagnosis, routing, telemetry, environment/source metadata, profiles, prompts/raw
outputs and SHA-256 checksums. Failed candidate generation is saved as a failed
run instead of a successful diagnosis. Replay instructions and artifact layout
are in [docs/phase3_status.md](docs/phase3_status.md).
See [architecture](docs/architecture.md), [demo scenario](docs/demo_scenario.md),
[limitations](docs/limitations.md), [Phase 1 report](docs/phase1_status.md), and
[Phase 2 report](docs/phase2_status.md).

Incident contents remain local; no cloud API is used. JSON includes excerpts for
auditability and should be handled like the original files. Verification evaluates
available evidence and recommends further observations; it never executes shell
commands or remediation. This is not a production incident-remediation system.
