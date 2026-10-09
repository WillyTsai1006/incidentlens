# Phase 3 status

Phase 3 is complete. Phase 1 extraction/relations and Phase 2 verification,
ranking and legacy CLI contracts are preserved. No later phase or distributed
execution was implemented.

The later IncidentLens UI polish preserves this architecture and the original
161 tests, adding 13 presentation/default-setting tests. Current UI defaults,
browser checks and screenshots are recorded in [ui_polish_status.md](ui_polish_status.md).
Test counts and UI descriptions below document the original Phase 3 milestone.

Subsequent RTX 5060 CUDA validation succeeded without diagnosis code changes.
See [rtx5060_validation.md](rtx5060_validation.md) for the matched CPU/GPU
comparison, placement assertions, exact environment and new artifacts. The CPU
records below document the initial Phase 3 milestone; B70 remains unvalidated.

## Implemented

- Deterministic adaptive router with typed profiles, task/context/complexity,
  memory-pool and latency preferences, stable ties, rejected-profile explanations
  and human-readable decisions. One real CPU model is enabled in the demo;
  a disabled B70/XPU profile documents the target without pretending validation.
- Typed per-call and end-to-end telemetry: actual identity, token counts when
  available, latency, generation throughput, memory snapshots and routing cost.
  Mock/stub metrics are labeled; unavailable measurements remain null.
- Offline real Transformers inference through the existing backend abstraction.
  The small model selects existing evidence IDs for each catalog candidate.
  IDs become exact normalized quotations. Unknown IDs, model confidence and
  extra factual prose are rejected. Independent verification and deterministic
  scoring decide support and ranking; correlation is never promoted to proof.
- Streamlit inputs, evidence, graph relations, all ranked hypotheses, supporting
  and contradicting evidence, missing observations, verification, primary
  diagnosis, action, routing, telemetry and clear mock/real labels.
- Timestamped run snapshots, checksums, inputs, profiles, raw inference exchanges
  and environment/source metadata, including failed validation runs.
- CLI `--phase3` and `--save-run`, backend/profile/device selection and routing
  preferences. Existing `--demo` remains Phase 2; `--core-only` remains Phase 1.

## Validation and hardware

The complete suite passed: **161 tests**, including the two optional real-model
tests (API and Streamlit AppTest). All original 122 Phase 1/2 tests still pass.
With `INCIDENT_MODEL_PATH` absent and UI dependencies installed, the default
suite has **159 passed, 2 skipped**. Default tests need no model download or GPU.

Both mock and real CLI demos ran. Streamlit AppTest exercised loading, diagnosis,
full result rendering and local-model error handling; the real UI integration
also passed. A live server on `127.0.0.1:8501` returned `ok` from its health
endpoint. This is execution validation, not a browser screenshot/layout review.

| Item | Actual validation |
| --- | --- |
| Real model | Qwen/Qwen2.5-0.5B-Instruct |
| Weight revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Inference device | CPU, four PyTorch threads |
| CPU | Intel Core i5-13450HX, 10 cores / 16 logical CPUs |
| RAM | 48755.07 MiB host total, measured with psutil |
| OS / Python | Windows 11 / Python 3.13.13 |
| Runtime | torch 2.14.1+cpu, Transformers 4.57.6, Streamlit 1.65.0 |
| Physically present GPU | NVIDIA RTX 5060 Laptop GPU, 8151 MiB; Intel UHD |
| GPU inference | Subsequently validated on RTX 5060; see CUDA comparison |
| Intel Arc Pro B70 | Unavailable; target architecture only |

Hardware inventory was checked separately from inference. CPU PyTorch reporting
CUDA unavailable does not mean the physical NVIDIA GPU is absent. A 1.5B Qwen
model was downloaded during investigation but was not executed or validated.
There are no B70 benchmark numbers or OpenVINO measurements. RTX development
measurements are in the separate validation report. Stub telemetry exists only
as a labeled test contract; no stub result
is presented as an actual model run.

## Recorded result and metrics

Representative real CLI run:
`runs/20261009T075858.536828Z-95caa8a4/`, generated with the final code and
`profiles/demo_local.json`. Its stored inputs, outputs and checksums were
verified. An earlier successful cold run under
`runs/20261008T094842.763263Z-2f4b5b65/` took about 130 seconds; the final run
below took about 295 seconds. CPU latency varies substantially between runs.

| Ranked hypothesis | Application evidence score |
| --- | ---: |
| Duplicate model loading across inference workers | 0.866667 |
| Excessive runtime/batch-related allocation | 0.153333 |
| Memory leak | 0.113333 |

There are 25 evidence items and 30 relations. The synthetic demand is 38 GiB
against a nominal 32 GiB ceiling. Scores are heuristics, not probabilities.
Process-level replica confirmation remains missing; the primary explanation is
best supported by supplied evidence, not proven causality.

Example routing: hypothesis generation, 6322 input characters, complexity 0.80,
balanced latency preference, host budget snapshot 31139 MiB. Preferred tier is
`reasoning_model`; selected `qwen-cpu-small` is the actual enabled small CPU model.
Its 8192 MiB footprint is a conservative configuration estimate, not a measured
benchmark. The reason explicitly explains the unavailable preferred tier.

| Real call | Input/output tokens | Call latency | Generation tokens/sec | Process RSS |
| --- | --- | ---: | ---: | ---: |
| 1, cold | 1805 / 56 | 170849.92 ms | 0.9266 | 2291.25 MiB |
| 2 | 1806 / 76 | 67459.70 ms | 1.1269 | 2291.08 MiB |
| 3 | 1805 / 56 | 56584.61 ms | 0.9901 | 2290.51 MiB |

End-to-end diagnosis: **295170.78 ms**. Total routing overhead: **0.2568 ms**.
All three candidate responses validated. GPU allocated memory is **null**.
Generation throughput excludes model loading; first-call latency includes it.
RSS is a process snapshot, not peak memory or VRAM. End-to-end timing excludes
artifact writing and UI rendering. Values vary with host workload and caches.
Mock calls have actual measured wall time but null token counts/TPS and explicit
mock labels. No missing metric is filled with a synthetic hardware number.

## Demo and reproducibility

Install dependencies and pinned weights using README. From the project root:

```powershell
.venv/Scripts/python.exe -m app.main --demo --save-run
.venv/Scripts/python.exe -m app.main --demo --save-run --backend transformers --profiles profiles/demo_local.json --device cpu
.venv/Scripts/python.exe -m streamlit run app/ui.py --browser.gatherUsageStats false
```

The UI uses an existing local model directory and explicit device; it never
downloads weights. Run GPU OOM Demo, select mode and Analyze Incident. Local
failures remain errors with saved failure artifacts, without mock substitution.
`INCIDENT_RUNS_DIR` changes the UI artifact root; `INCIDENT_MODEL_PATH` changes
its default local directory and enables the optional real tests.

Each successful run contains:

```text
runs/<UTC timestamp>-<unique suffix>/
  inputs/{app.log,config.yaml,inference.py,system.log}
  evidence.json
  relations.json
  hypotheses.json
  verification.json
  diagnosis.json
  routing.json
  telemetry.json
  metadata.json
  profiles.json
  inference_io.json
  manifest.json
```

Metadata records package versions, source hashes, input hashes, model/profile
settings and requested generation parameters. Actual capped settings, prompts,
raw outputs, retries and validation status are in `inference_io.json`. Profiles
include the model revision. Weights are separate; preserve that revision.

Inspect checksums and replay the representative input snapshot:

```powershell
.venv/Scripts/python.exe -c "from core.run_artifacts import verify_manifest; verify_manifest('runs/20261009T075858.536828Z-95caa8a4')"
.venv/Scripts/python.exe -m app.main --incident runs/20261009T075858.536828Z-95caa8a4/inputs --save-run --backend transformers --profiles runs/20261009T075858.536828Z-95caa8a4/profiles.json --available-memory-mb 31139 --complexity-score 0.8 --latency-preference balanced
```

Saved model paths must exist on the replay machine. For a deterministic mock
replay, omit the backend/profile overrides. Matching source, package versions,
weights and device makes a run reproducible for inspection; hardware-dependent
timings and cross-device generation are not promised bit-identical. Input text
is stored as the UTF-8 content actually supplied to extraction.

## Deviations and limitations

The small model failed the legacy all-candidate quotation bundle. Phase 3 uses
per-candidate evidence-ID selection with bounded retries instead. It still
generates three structured candidates through the existing abstraction and
enforces the same grounding/verifier/confidence boundary. The Phase 2 contract
is unchanged. Model selections can be irrelevant: in the recorded run some
worker-start IDs were selected for the leak candidate. The verifier does not
automatically treat those as leak support. This demonstrates execution and
grounding, not broad RCA accuracy or high-quality model reasoning.

The catalog remains limited to three hypothesis families. Only hypothesis
generation invokes a model; parsing, verification and confidence are application
logic. No speculative second model, distributed scheduler or production
dashboard was added. Router footprint estimates and one-time memory snapshots
cannot guarantee avoiding OOM. B70/XPU compatibility needs actual target testing.
Artifacts contain incident excerpts and should be handled as sensitive data.
No verification plan, recommendation, incident code or remediation is executed.

## Remaining competition submission work

- Validate model loading/generation and actual usable memory on Intel Arc Pro
  B70; record driver/runtime/model revisions and measured telemetry.
- Rehearse on the competition machine, including offline weights, failure mode,
  artifact replay and latency; replace estimates only with measured profiles.
- Review UI visually in a browser and record the three-minute demo with explicit
  mock/real and development/target-hardware labels.
- Publish the reviewed source and submission materials, confirm model licensing,
  and prepare the architecture explanation and known limitations.

No public publication or competition submission was performed. **Stop point:
Phase 3 implemented and tested.**
