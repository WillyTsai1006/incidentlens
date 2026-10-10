# IncidentLens competition UI polish

The subsequent [final demo polish](final_demo_polish.md) adds the home capability
cards and verification count metrics, replaces the prominent unavailable peak
with GPU Memory Snapshot, and documents the CLI/Streamlit latency investigation.
The screenshots and timings below retain the preceding polish-pass history.

Implemented and reviewed on 2026-10-09. This is presentation work on the existing
Phase 3 pipeline. Diagnosis architecture, candidate logic, evidence validation,
verification, authoritative scoring, routing algorithms, telemetry schemas and
run-artifact contracts are unchanged. Core and inference source hashes were
checked against the preceding CUDA-validation run metadata.

## Changes

| File | Purpose |
| --- | --- |
| `app/ui.py` | IncidentLens branding, simpler controls, overview, prominent diagnosis, cited evidence, score bars, verification, runtime summary and collapsed audit details |
| `app/ui_presenter.py` | Pure presentation helpers for evidence-derived overview, supporting-evidence ancestry, verification labels and current-run telemetry aggregation; original serialization preserved |
| `app/ui_runtime.py` | CUDA availability/default selection and wiring the bundled CUDA model to the existing validated RTX profile |
| `.streamlit/config.toml` | Native light theme, teal primary color, readable heading sizes and explicit sidebar text color |
| `tests/test_ui.py` | Preserve existing UI coverage; check new real/CUDA defaults, unavailable-CUDA warning and upload controls |
| `tests/test_ui_presentation.py` | Evidence provenance, non-mutating summaries, conservative status labels, unknown metrics, aggregation and profile-selection checks |
| `README.md`, `docs/phase3_status.md`, `docs/streamlit_demo_review.md`, this report | Current launch instructions and validation status; earlier milestone/review records retained as historical |

The main action is **Run GPU OOM Demo**, followed by **Analyze Incident**. Loading
is deliberately separate from inference so the presenter can show the incident
before starting the real model. **Upload Your Incident** reveals the uploader.

The four overview values come from extracted/derived evidence, with evidence-ID
tooltips: nominal synthetic capacity **32 GiB**, **4 workers**, **9.5 GiB** per
worker and **38 GiB** projected demand. They are incident data, not measurements
of the inference device. Supporting facts retain evidence IDs and derived-fact
provenance. The main view presents six concise supporting facts; the complete
evidence, graph and original result remain inspectable.

The primary hypothesis is **Duplicate model loading across inference workers**,
scored **86.7%**. Runtime/batch allocation scores **15.3%** and a memory leak
scores **11.3%**. Bars represent independent heuristic scores, not normalized
probabilities. **PARTIALLY VERIFIED** describes five supported criteria and one
missing process-level allocation check. Correlation is not presented as proven
causality, and the recommended action executes no remediation.

## Defaults and advanced settings

- Default inference: **Local Transformers (real)**, existing local Qwen2.5-0.5B-Instruct weights.
- Default device: **cuda** if PyTorch reports CUDA available; otherwise **cpu** with an explicit warning. A selected CUDA execution failure remains an error, without CPU/mock fallback.
- Default optimization preference: **balanced**.
- **Advanced Runtime Settings** is initially collapsed and contains mode, model directory, device and latency preference.
- **View Extracted Evidence**, **View Evidence Relations**, **View Raw Verification Details**, **View Run Metadata** and **View Incident Inputs** retain the audit trail in collapsed sections. The result JSON download remains available.

Only the bundled model's CUDA path uses `profiles/demo_rtx5060.json`. This was
already validated before this task; its 4096 MiB footprint is a routing estimate,
not a new benchmark. Other model paths retain existing conservative defaults.
No model caching, routing policy or inference configuration was changed.

## Validation

The final complete suite passed **174 tests**, including the two optional
real-model tests: **174 passed in 467.96 seconds** on the final rerun. An earlier
complete run passed in 75.20 seconds; these are test durations, not benchmarks.
The original 161 tests remain
covered; 13 new tests cover presentation and default-setting behavior. The normal
offline/mock suite also passed **172 tests, 2 skipped**; optional tests require
already downloaded weights, with no downloads or GPU requirement in default tests.

Real Chrome browser interactions exercised loading and analyzing the demo on
CUDA and CPU. Both produced the unchanged three ranked hypotheses and scores.
The browser review covered **1920x1080** and **1366x768**; the primary diagnosis
and its score fit in the initial result viewport at both sizes. The original
long technical tables are collapsed. Screenshots:

- CUDA result, 1920x1080 (local-only, ignored): `../output/playwright/incidentlens-final-cuda-1920.png`
- CUDA result, 1366x768 (local-only, ignored): `../output/playwright/incidentlens-final-cuda-1366.png`
- CPU result (local-only, ignored): `../output/playwright/incidentlens-final-cpu-1920.png`
- Mock result (local-only, ignored): `../output/playwright/incidentlens-final-mock-1920.png`
- Mock runtime: unavailable model metrics (local-only, ignored): `../output/playwright/incidentlens-final-mock-runtime-1920.png`

The fresh CUDA browser session reported zero console errors and warnings.
AppTest also verifies mock execution and the unavailable-CUDA warning, while
mock presentation helpers verify that unavailable model measurements remain null.
An actual browser mock run completed with the same three rankings, an explicit
**MOCK inference — no language model executed** label and **N/A** model metrics;
its browser console reported zero errors and warnings. The unchanged CLI demo
also completed successfully and saved a normal artifact.

These are UI validation runs, **not a controlled CPU-versus-GPU benchmark**.
Cold loading and other concurrent machine activity affect timings. Refer to
[the earlier matched benchmark](rtx5060_validation.md) for that comparison.

| Actual UI run | End-to-end | Generation, summed | Output throughput | GPU allocation snapshot | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | --- |
| Real RTX 5060 CUDA | 59.51 s | 33.21 s | 5.66 tokens/s | 1917.3 MiB | N/A |
| Real CPU | 36.40 s | 35.73 s | 5.26 tokens/s | N/A | N/A |

CUDA artifact: `runs/20261009T090400.757198Z-e9884aa7/`.
CPU artifact: `runs/20261009T090828.571794Z-13ec4e89/`.
Final mock UI artifact: `runs/20261009T131201.026692Z-8aa82b4e/`.
CLI demo artifact: `runs/20261009T130442.364117Z-25cbef6d/`.
These are normal pipeline artifacts containing inputs, evidence, relations,
hypotheses, verification, diagnosis, routing, telemetry, profiles, inference I/O,
metadata and manifest. No artifact format was added or changed.

The runtime section aggregates the current run only. End-to-end timing excludes
artifact writing and UI rendering; inference-call timing includes lazy model
loading. Throughput is total generated tokens divided by summed generation
time. Recorded GPU allocated memory is a snapshot, not peak VRAM. Peak GPU
memory is unavailable in the existing schema and therefore displays **N/A**.
Mock model token/generation/GPU metrics also remain unavailable and are labeled.
The mock execution-device label says **No model executed (mock)**; detecting an
installed GPU does not imply that the mock backend executed a model on it.

## Launch and rehearsal

From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe -m streamlit run app/ui.py --browser.gatherUsageStats false
```

Open `http://localhost:8501`, click **Run GPU OOM Demo**, inspect the overview,
then click **Analyze Incident**. After completion, present the primary diagnosis,
supporting/missing evidence, alternative scores, partial verification and next
step. Scroll to the actual device/routing/telemetry and optionally expand an audit
section. For a deterministic rehearsal, select **Mock (deterministic)** under
advanced settings before analyzing; show its mock label.

## Remaining limitations

- Development hardware: **NVIDIA GeForce RTX 5060 Laptop GPU**, Windows and the existing CUDA PyTorch installation. CPU and CUDA paths were actually executed.
- Competition target: **Intel Arc Pro B70 32 GB**. B70/XPU execution and performance remain **pending / unvalidated**; no B70 benchmark numbers are shown.
- The small model's existing constrained candidate catalog is unchanged; this is a targeted prototype rather than general production RCA.
- The real model must exist locally, and cold inference waits remain variable. No fake loading result or hard-coded runtime value is used.
- Physical projector visibility, competition display scaling and a final live-stage rehearsal remain to be checked by the presenter. Browser screenshots do not replace that check.
- Competition recording/submission and target-hardware rehearsal remain outside this UI-only task. No new phase, autonomous commands or remediation was implemented.

Public curated screenshot: [recorded verification view](assets/incidentlens_demo.png).
