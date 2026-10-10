# Codex Implementation Specification

## Project
**Evidence-Grounded Local Incident Diagnosis Agent**  
Stage I MVP for **2026 ASUS UGen AI League — Battlefield Thunderstorm / Workplace AI**

**Version:** 0.1  
**Primary goal:** Build a working, reproducible MVP that can be demonstrated in a <=3 minute English video and linked from a public GitHub repository.

---

# 1. Product Definition

## 1.1 One-line pitch
A private, hardware-aware AI agent that diagnoses engineering incidents by extracting evidence from logs, code, configuration, and metrics; generating multiple root-cause hypotheses; verifying those hypotheses against evidence; and returning a ranked, auditable diagnosis using local inference.

## 1.2 Problem
Engineering incidents are rarely explained by a single log line. Root-cause analysis often requires correlating multiple sources such as:

- application logs
- system logs
- source code
- configuration files
- runtime metrics
- stack traces

A generic LLM can summarize these files, but a credible incident-diagnosis system should do more than generate a plausible answer. It should:

1. distinguish observed evidence from model inference;
2. generate multiple candidate hypotheses instead of jumping to one conclusion;
3. show supporting and contradicting evidence;
4. suggest a verification step;
5. rank hypotheses with a transparent confidence score;
6. keep sensitive operational data local;
7. select inference configurations according to workload complexity and hardware constraints.

## 1.3 MVP demo incident
The Stage I MVP must support **one polished end-to-end incident scenario**:

> GPU out-of-memory caused by duplicate model loading across multiple inference workers.

The system receives four files:

```text
app.log
system.log
config.yaml
inference.py
```

The correct diagnosis should be:

> Multiple worker processes independently load the same model, causing GPU memory saturation and an allocation failure during worker startup.

---

# 2. Success Criteria

The MVP is complete when a user can:

1. launch the app locally;
2. load the bundled GPU OOM incident example;
3. see structured evidence extracted from the four files;
4. see 2–3 generated root-cause hypotheses;
5. see supporting and contradicting evidence for each hypothesis;
6. see hypotheses ranked by confidence;
7. see one suggested verification action per hypothesis;
8. see a final root cause and recommended action;
9. see inference telemetry including selected model/backend, latency, and memory when available;
10. reproduce the full demo from a clean clone using documented commands.

The MVP must **not** depend on a cloud LLM API for its core diagnosis path.

---

# 3. Scope

## 3.1 Required for Stage I

### Product behavior
- Incident file ingestion.
- Evidence extraction.
- Evidence normalization.
- Evidence correlation representation.
- Candidate hypothesis generation.
- Evidence-based verification.
- Confidence ranking.
- Final diagnosis generation.
- Recommended action generation.
- Local model inference abstraction.
- Simple adaptive model router.
- Basic inference telemetry.
- Web UI suitable for video recording.
- Bundled reproducible demo dataset.
- Unit/integration tests for deterministic logic.

### Documentation
- README with setup and demo instructions.
- Architecture document.
- Demo scenario document.
- Known limitations.

## 3.2 Explicit non-goals for Stage I
Do **not** spend time implementing these unless all required MVP work is complete:

- Kubernetes integration
- automatic remediation
- production log streaming
- multi-node inference
- Ray cluster
- vLLM distributed serving
- VLM/image analysis
- Neo4j or another graph database
- enterprise authentication
- persistent user accounts
- production security hardening
- learned confidence model
- fully autonomous shell execution
- generalized support for arbitrary incidents

Stage I values a polished working prototype over broad but incomplete functionality.

---

# 4. Core Design Principles

## 4.1 Evidence first
Never treat an LLM-generated statement as evidence.

Every evidence item must point to an original source location.

## 4.2 Separate observation from inference
The UI and internal schema must distinguish:

- **Observed Evidence** — directly parsed or quoted from supplied files.
- **Derived Fact** — deterministic calculation from observed evidence.
- **Hypothesis** — model-generated possible explanation.
- **Recommendation** — proposed next step or fix.

## 4.3 No giant-prompt-only architecture
Do not implement the system as:

```text
all files -> one prompt -> final answer
```

The required pipeline is:

```text
raw files
  -> structured evidence
  -> evidence correlations
  -> candidate hypotheses
  -> hypothesis verification
  -> confidence ranking
  -> final diagnosis
```

LLMs may be used inside stages, but intermediate structured artifacts must exist.

## 4.4 Local-first inference
The core pipeline must work with a local model backend.

Hardware support must be implemented through an abstraction so development can proceed on available hardware while preserving an Intel Arc Pro B70 target path.

Suggested initial backends:

1. Hugging Face Transformers + PyTorch
2. optional llama.cpp backend if helpful
3. Intel/OpenVINO backend adapter stub or implementation when environment is available

Do not claim B70 benchmark results unless they were actually measured on B70 hardware.

---

# 5. System Architecture

```text
┌───────────────────────────────┐
│            Web UI             │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│        Incident Loader        │
│ log / yaml / py / text        │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│       Evidence Extractor      │
│ parsers + LLM-assisted parse  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│       Evidence Repository     │
│ structured JSON/Pydantic      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      Correlation Engine       │
│ graph-like relationships      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│     Diagnosis Orchestrator    │
│ ┌───────────────────────────┐ │
│ │ Hypothesis Generator      │ │
│ │ Evidence Verifier         │ │
│ │ Confidence Engine         │ │
│ └───────────────────────────┘ │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│     Adaptive Model Router     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│     Local Inference Layer     │
│ Transformers / OpenVINO / ... │
└───────────────────────────────┘
```

---

# 6. Recommended Repository Structure

```text
incident-diagnosis-agent/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── ui.py
├── core/
│   ├── __init__.py
│   ├── schemas.py
│   ├── incident_loader.py
│   ├── evidence_extractor.py
│   ├── evidence_repository.py
│   ├── correlation.py
│   ├── hypothesis.py
│   ├── verifier.py
│   ├── confidence.py
│   ├── diagnosis.py
│   └── router.py
├── inference/
│   ├── __init__.py
│   ├── base.py
│   ├── transformers_backend.py
│   ├── openvino_backend.py
│   └── telemetry.py
├── prompts/
│   ├── hypothesis_generation.txt
│   ├── evidence_verification.txt
│   └── final_diagnosis.txt
├── examples/
│   └── gpu_oom_duplicate_models/
│       ├── app.log
│       ├── system.log
│       ├── config.yaml
│       ├── inference.py
│       ├── expected_evidence.json
│       └── expected_diagnosis.json
├── benchmarks/
│   ├── benchmark_pipeline.py
│   └── benchmark_router.py
├── tests/
│   ├── test_loader.py
│   ├── test_evidence_extractor.py
│   ├── test_correlation.py
│   ├── test_confidence.py
│   └── test_demo_integration.py
└── docs/
    ├── architecture.md
    ├── demo_scenario.md
    └── limitations.md
```

---

# 7. Data Schemas

Use Pydantic models unless there is a strong reason not to.

## 7.1 SourceReference

```python
class SourceReference(BaseModel):
    file_path: str
    line_start: int | None = None
    line_end: int | None = None
    raw_excerpt: str
```

## 7.2 Evidence

```python
class Evidence(BaseModel):
    id: str
    category: Literal[
        "log_event",
        "config_value",
        "code_behavior",
        "system_metric",
        "derived_fact"
    ]
    statement: str
    value: str | int | float | bool | None = None
    unit: str | None = None
    timestamp: str | None = None
    source: SourceReference
    confidence: float = 1.0
```

`confidence=1.0` is acceptable for deterministic parser output. LLM-extracted evidence must have a lower extraction confidence unless validated.

## 7.3 EvidenceRelation

```python
class EvidenceRelation(BaseModel):
    source_evidence_id: str
    relation: Literal[
        "causes",
        "supports",
        "contradicts",
        "precedes",
        "correlates_with",
        "derived_from"
    ]
    target_evidence_id: str
    explanation: str
```

## 7.4 Hypothesis

```python
class Hypothesis(BaseModel):
    id: str
    title: str
    description: str
    supporting_evidence_ids: list[str] = []
    contradicting_evidence_ids: list[str] = []
    missing_evidence: list[str] = []
    verification_action: str
    confidence: float = 0.0
```

## 7.5 DiagnosisResult

```python
class DiagnosisResult(BaseModel):
    incident_summary: str
    hypotheses: list[Hypothesis]
    top_root_cause_id: str
    recommended_action: str
    selected_model: str
    inference_backend: str
    latency_ms: float | None = None
    memory_mb: float | None = None
```

---

# 8. Demo Dataset Specification

Create a synthetic but internally consistent incident.

## 8.1 `config.yaml`
Must show multiple workers, e.g.:

```yaml
service:
  workers: 4
model:
  name: demo-7b-model
  load_per_worker: true
  precision: fp16
runtime:
  gpu_memory_limit_gb: 8
```

Adjust values to be logically consistent with the demo environment and narrative.

## 8.2 `inference.py`
Must clearly show that each worker initializes its own model instance.

Example behavior:

```python
def worker_main(worker_id):
    model = load_model(MODEL_NAME)
    serve(model)
```

## 8.3 `system.log`
Must contain GPU-memory growth correlated with worker startup.

Example conceptual sequence:

```text
worker-1 started -> GPU memory 7.4 GB
worker-2 started -> GPU memory 14.8 GB
worker-3 started -> GPU memory 22.2 GB
worker-4 start attempted -> allocation failure
```

## 8.4 `app.log`
Must show service startup events followed by an OOM/allocation error and worker failure.

Do not make the root cause trivially explicit in one line. The system should need cross-file correlation.

---

# 9. Evidence Extraction Requirements

Use deterministic parsing wherever possible.

## 9.1 Config parser
Parse YAML values directly.

Expected evidence examples:

- `service.workers = 4`
- `model.load_per_worker = true`
- `model.precision = fp16`

## 9.2 Source-code parser
For Stage I, AST or regex-assisted extraction is acceptable.

Required behavior:
- detect model loading inside the worker lifecycle;
- produce an evidence statement such as:
  - `Each worker calls load_model() independently.`

Do not attempt a general-purpose static analyzer.

## 9.3 Log parser
Extract:
- worker startup sequence;
- memory readings;
- allocation/OOM event;
- service failure event.

## 9.4 Derived facts
Implement deterministic derived facts, for example:

```text
number of workers × approximate per-model memory > available GPU memory
```

If required data is not available, do not fabricate it.

---

# 10. Correlation Engine

For Stage I, implement the evidence graph in memory using one of:

- NetworkX
- custom adjacency lists
- plain Pydantic/JSON structures

No graph database is needed.

Required relations for demo:

```text
workers=4
  -> combined with
model loaded inside each worker
  -> implies
multiple concurrent model replicas
  -> correlated with
growing GPU memory per worker
  -> precedes
allocation failure
  -> causes
service startup failure
```

The relation list must be visible/debuggable in JSON and optionally visualized in the UI.

---

# 11. Hypothesis Generation

Generate 3 hypotheses for the demo incident.

Expected candidates:

1. Duplicate model loading across workers.
2. Memory leak during inference/service startup.
3. Batch size or another runtime allocation is too large.

The LLM may generate hypotheses, but:

- each hypothesis must be returned in structured JSON;
- it must not assign final confidence itself;
- confidence must be calculated by the application after evidence verification.

Suggested structured output:

```json
[
  {
    "title": "Duplicate model loading across workers",
    "description": "Each worker appears to create an independent model replica, exhausting GPU memory.",
    "verification_action": "Inspect process-level GPU allocations and worker initialization behavior."
  }
]
```

---

# 12. Evidence Verification

For each hypothesis, classify evidence as:

- supporting
- contradicting
- irrelevant

The verifier may use a local LLM, deterministic rules, or both.

For the primary demo hypothesis, expected support includes:

- multiple workers configured;
- model load occurs inside worker lifecycle;
- GPU memory grows as workers start;
- allocation fails during another worker startup.

Expected contradiction or weak support for memory-leak hypothesis:

- no continuous growth pattern after steady state;
- memory increases in discrete steps aligned with worker initialization.

The final UI must show supporting and contradicting evidence separately.

---

# 13. Confidence Engine

Stage I confidence is intentionally heuristic and must be documented as such.

Suggested formula:

```text
score =
  0.55 * support_ratio
+ 0.25 * source_diversity
+ 0.20 * causal_consistency
- contradiction_penalty
```

Where:

- `support_ratio`: relevant supporting evidence / expected support slots
- `source_diversity`: number of unique source types supporting hypothesis, normalized
- `causal_consistency`: whether temporal/causal ordering is coherent
- `contradiction_penalty`: weighted count of contradicting evidence

Clamp to `[0, 1]`.

The formula can be refined, but confidence must be **application-computed**, not blindly accepted from the LLM.

---

# 14. Adaptive Model Router

Stage I needs only a simple, demonstrable router.

## 14.1 Inputs

```python
class RoutingContext(BaseModel):
    task_type: Literal[
        "extraction",
        "classification",
        "hypothesis_generation",
        "verification",
        "final_diagnosis"
    ]
    context_chars: int
    evidence_count: int
    complexity_score: float
    available_memory_mb: int | None = None
```

## 14.2 Behavior
At minimum support two logical tiers:

- `small_model`
- `reasoning_model`

Suggested routing:

```text
simple extraction/classification -> small model
hypothesis generation / multi-file reasoning -> reasoning model
```

If only one local model is available during development, preserve the routing abstraction and record both logical route and actual backend. Do not fake execution on two different models.

## 14.3 Telemetry
Record per inference call:

- logical task type
- selected model
- actual backend
- prompt/input length
- output length
- latency
- memory usage if measurable

---

# 15. Inference Backend Interface

Create a backend protocol/ABC so runtime backends are replaceable.

```python
class InferenceBackend(Protocol):
    @property
    def name(self) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str: ...
```

Implement at least one working local backend.

Preferred development order:

1. local Transformers backend
2. optional llama.cpp backend
3. OpenVINO/Intel backend when environment is available

The project must not fail entirely if Intel-specific packages are unavailable during early development.

---

# 16. UI Specification

Use **Streamlit** unless another framework materially reduces implementation time.

The UI is optimized for a 3-minute demo, not production use.

## 16.1 Page layout

### Header
- Project name
- Subtitle: `Evidence-Grounded Local Incident Diagnosis`
- local/privacy badge

### Left/main section
1. Incident input files
2. `Load Demo Incident` button
3. `Analyze Incident` button

### Analysis progress
Show the stages:

```text
1. Parsing sources
2. Extracting evidence
3. Correlating evidence
4. Generating hypotheses
5. Verifying hypotheses
6. Ranking root causes
```

### Results
Show:

**Top Root Cause**
- title
- confidence

**Supporting Evidence**
- each item with file + line reference

**Contradicting Evidence**
- each item with file + line reference

**Verification Action**

**Recommended Action**

### Additional expandable sections
- All hypotheses
- Evidence table
- Evidence graph/relations
- Inference telemetry

## 16.2 Demo stability
The UI must support a deterministic `demo mode` using the bundled incident dataset.

A demo recording must not fail because of random model output.

Use:
- temperature 0
- schema validation
- retry-on-invalid-JSON
- deterministic fallback logic for the bundled demo if local-model output cannot be parsed

Fallbacks must be clearly implemented and documented; do not silently fake model output.

---

# 17. Final Demo Output Contract

For the bundled demo, the final page must contain roughly:

```text
Top Root Cause
Duplicate model loading across inference workers

Confidence
~0.8–0.95 (derived from evidence; exact value depends on implemented scoring)

Supporting Evidence
- config.yaml configures multiple workers.
- inference.py loads the model inside each worker.
- system.log shows stepwise GPU-memory growth as workers start.
- app.log shows allocation failure during worker startup.

Contradicting Evidence
- No progressive memory-growth pattern is observed after initialization.

Verification
Inspect process-level GPU allocations and confirm one model replica per worker.

Recommended Action
Use a shared inference server/model process, reduce worker-local replicas,
or use a memory-aware serving architecture.
```

Do not hardcode the complete final diagnosis as one static string in the UI. The result should be produced from pipeline artifacts.

---

# 18. Evaluation

## 18.1 Required Stage I metrics

### Functional
- correct top-1 root cause for bundled demo
- number of evidence items extracted
- percentage of displayed diagnosis claims linked to evidence

### Performance
- end-to-end latency
- inference latency by stage
- memory usage where measurable
- selected model/backend

## 18.2 Evidence Grounding Rate
Implement a simple MVP metric:

```text
Evidence Grounding Rate =
number of factual diagnosis claims linked to >=1 evidence item
/
total factual diagnosis claims
```

For the bundled demo, target `1.0` or close to it.

## 18.3 Do not overclaim
Do not claim broad production RCA accuracy from a single synthetic scenario.

Describe Stage I measurements as **prototype validation**.

---

# 19. Testing Requirements

## 19.1 Unit tests
Required:

- YAML config parsing
- log event extraction
- source-code evidence extraction for demo pattern
- derived fact generation
- confidence scoring
- router selection logic

## 19.2 Integration test
One test must execute the full bundled demo pipeline and assert:

- at least N evidence items are produced;
- 3 hypotheses are produced;
- duplicate model loading is ranked #1;
- top hypothesis references evidence from at least 3 source types;
- final diagnosis has a verification action;
- final diagnosis has a recommended action.

The integration test should be runnable without the UI.

## 19.3 Mock inference
Tests must be able to run with a deterministic mock backend so CI does not require a GPU or model download.

---

# 20. CLI / Developer Experience

Provide the following developer workflows.

## Install

Prefer `uv` if convenient; otherwise standard venv/pip is acceptable.

Example:

```bash
git clone <repo>
cd incident-diagnosis-agent
uv sync
```

or

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run tests

```bash
pytest -q
```

## Run demo pipeline in terminal

```bash
python -m app.main --demo
```

or provide an equivalent CLI.

## Run UI

```bash
streamlit run app/ui.py
```

README must contain the exact commands that work.

---

# 21. Logging and Reproducibility

Use Python `logging` rather than print statements for core modules.

Every diagnosis run should have a run ID and optionally save artifacts under:

```text
runs/<run_id>/
├── evidence.json
├── relations.json
├── hypotheses.json
├── diagnosis.json
└── telemetry.json
```

This is useful both for debugging and the competition demo.

---

# 22. Error Handling

The application must handle:

- missing input file
- invalid YAML
- unsupported file type
- invalid LLM JSON output
- local model unavailable
- GPU unavailable
- telemetry unavailable

When a feature is unavailable, degrade gracefully and show a clear message rather than crashing.

---

# 23. Security / Safety for MVP

- Do not automatically execute arbitrary shell commands from LLM output.
- Verification actions are **recommendations only** in Stage I.
- Do not upload incident files to an external service.
- Do not log secrets or full sensitive file contents unnecessarily.
- Add a README note that the Stage I prototype is not a production incident-remediation system.

---

# 24. Stage I Demo Storyboard

The implementation should support this exact recording flow.

## 0:00–0:25 — Problem
Show an AI service failure and explain that engineers normally inspect multiple sources manually.

## 0:25–0:45 — Inputs
Load:

- `app.log`
- `system.log`
- `config.yaml`
- `inference.py`

## 0:45–1:20 — Pipeline
Show progress:

- evidence extraction
- correlation
- hypothesis generation
- verification

Show selected local model/backend.

## 1:20–2:15 — Diagnosis
Display:

- top root cause
- confidence
- supporting evidence with source references
- contradicting evidence
- verification action
- recommended fix

## 2:15–2:40 — Why this is different
Show the three differentiators:

1. Evidence-grounded diagnosis
2. Hypothesis verification loop
3. Adaptive local inference

## 2:40–3:00 — Hardware / roadmap
Show local inference architecture and Intel Arc Pro B70 target.

---

# 25. Definition of Done

A Stage I candidate build is done only when all items below are true:

- [ ] Repository installs from documented steps.
- [ ] Unit tests pass.
- [ ] Full demo integration test passes.
- [ ] Streamlit UI runs locally.
- [ ] `Load Demo Incident` works.
- [ ] Evidence items contain source references.
- [ ] Evidence correlations are represented explicitly.
- [ ] At least 3 candidate hypotheses are produced.
- [ ] Supporting and contradicting evidence are shown separately.
- [ ] Confidence is application-computed.
- [ ] Duplicate model loading ranks #1 for demo scenario.
- [ ] Verification action is shown.
- [ ] Recommended action is shown.
- [ ] Local inference backend works.
- [ ] Router abstraction exists.
- [ ] Telemetry includes latency and backend/model name.
- [ ] No cloud API is required for the core path.
- [ ] README contains reproducible commands.
- [ ] `docs/architecture.md` explains system design.
- [ ] `docs/demo_scenario.md` explains expected demo flow.
- [ ] Claims about Intel B70 are separated from actually measured results.
- [ ] UI is stable enough for one-take demo recording.

---

# 26. Codex Execution Instructions

When implementing this specification:

1. **Start with a working vertical slice, not all modules at once.**
   - bundled incident files
   - deterministic parsers
   - evidence schema
   - hypotheses
   - verifier
   - final result
   - then local LLM integration
   - then UI
   - then telemetry/router refinement

2. **Run tests after every meaningful implementation step.**

3. **Do not invent benchmark numbers.**

4. **Do not require unavailable Intel hardware to develop the application.**
   Keep a backend abstraction and add Intel-specific support separately.

5. **Prefer simple implementations that are easy to explain in a competition presentation.**

6. **Avoid unnecessary frameworks.**
   This is a 6-day Stage I MVP.

7. **Do not replace evidence extraction and verification with one monolithic LLM prompt.**

8. **Preserve provenance.**
   Every displayed evidence item must link back to its source file and location.

9. **Make the bundled demo deterministic.**

10. **After the MVP works, produce a concise status report:**
    - implemented features
    - tests passing
    - run commands
    - missing features
    - known limitations
    - recommended next step

---

# 27. Suggested Implementation Order

## Phase 1 — Deterministic core
1. scaffold repository
2. create schemas
3. create bundled incident dataset
4. implement YAML/log/code parsers
5. implement evidence repository
6. implement deterministic correlations
7. implement heuristic confidence engine
8. write tests

## Phase 2 — AI reasoning
9. implement inference interface
10. implement local backend
11. implement structured hypothesis generation
12. implement evidence verification
13. add schema validation/retry
14. integration test

## Phase 3 — Product layer
15. implement model router
16. implement telemetry
17. implement Streamlit UI
18. add run artifact saving
19. polish README/docs
20. rehearse demo path

---

# 28. Future Work (Not Stage I)

If selected for Stage II, extend toward:

- Intel Arc Pro B70-specific optimization
- OpenVINO / Intel GPU execution tuning
- model quantization comparison
- richer hardware-aware routing
- real incident datasets
- Kubernetes / observability integrations
- automated evidence collection
- VLM support for dashboards/screenshots
- temporal evidence graphs
- learned hypothesis ranking
- multi-incident evaluation benchmark
- safe human-approved remediation workflows

---

# 29. References for README / Proposal

Use these as background references; verify bibliography formatting before final submission.

1. Chen et al. (2023), **Automatic Root Cause Analysis via Large Language Models for Cloud Incidents (RCACopilot)**. arXiv:2305.15778.
2. Zhang et al. (2024), **Automated Root Causing of Cloud Incidents using In-Context Learning with GPT-4**. arXiv:2401.13810.
3. Goel et al. (2024), **X-lifecycle Learning for Cloud Incident Management using LLMs**. arXiv:2404.03662.
4. Ding et al. (2024), **Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing**. arXiv:2404.14618.
5. Intel, **Intel Arc Pro B70 Graphics Specifications**.

---

# 30. Product Positioning Summary

The product should never be presented as merely:

> “an LLM that reads logs.”

Present it as:

> **An evidence-grounded, locally executed incident diagnosis system that forms multiple hypotheses, verifies them against heterogeneous engineering evidence, ranks root causes, and adapts inference to task complexity and available compute.**

This is the architectural and product identity that must remain visible throughout implementation.
