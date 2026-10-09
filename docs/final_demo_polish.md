# Final competition-demo polish and latency investigation

For the subsequent fresh-environment installation and submission audit, see
[RELEASE_READINESS.md](../RELEASE_READINESS.md). Generated screenshots and raw
records below are local audit evidence under ignored `output/` and `runs/`;
they are not automatically included in a public source checkout. Curate demo
materials separately when publishing.

This pass changes native Streamlit presentation only. Diagnosis/evidence logic,
scoring, verification semantics, inference settings, routing, telemetry schemas
and artifact contracts are unchanged. Intel Arc Pro B70 remains the unvalidated
competition target; all GPU measurements here use the RTX 5060 Laptop GPU.

## Exact measured discrepancy

The archived RTX benchmark (`20261009T081505.525124Z-77b1d06e`) took **8.851 s**
end-to-end and **4.679 s** generation. The referenced Streamlit run
(`20261009T132513.282746Z-ad4e2e59`) took **32.437 s** and **28.194 s**.
The difference is **23.585686 s** end-to-end, of which **23.514326 s** is in
generation. The remaining **0.071361 s** is outside generation. Thus the saved
data identify slower execution of the same generation work, rather than larger
inputs, extra model calls, retries, or significant rendering overhead.

| Comparison | Archived CLI benchmark | Referenced Streamlit run |
| --- | --- | --- |
| Prompt characters per call | 6322 / 6314 / 6320 | Identical, byte-for-byte prompts |
| Input tokens | 1805 / 1806 / 1805 (5416 total) | Identical |
| Output tokens | 56 / 76 / 56 (188 total) | Identical |
| Calls / retries | 3 / 0 | 3 / 0 |
| Generation seconds by call | 1.661 / 1.723 / 1.295 | 8.457 / 11.324 / 8.412 |
| Cold-model flags | true / false / false | Identical |
| Generation configuration | Greedy, temperature 0, effective limit 512 tokens | Identical |
| Local weights / revision | Qwen2.5-0.5B-Instruct, pinned revision `7ae557604adf67be50417f59c2c2f167def9a775` | Identical |
| Backend / device | Transformers, CUDA | Identical; no CPU fallback |
| Confidence / verification | Three deterministic ranked hypotheses | Exactly identical hypotheses and verification results |

Prompts, evidence and relations were compared directly, not inferred from their
counts. Evidence and relation JSON files matched byte-for-byte. The primary
score remains 0.866667, with runtime allocation 0.153333 and leak 0.113333.
Raw comparisons are in `output/final_polish/historical_comparison.json`.

### What is established, and what cannot be reconstructed

- **Model loading:** a fresh backend/model is created for each Analyze click; it is reused across the three calls within that diagnosis. There is no cross-run model cache. Loading is included in the first call and end-to-end timing, but excluded from generation timing. The historical non-generation totals are 4.172 s versus 4.243 s, so loading does not explain their 23.51 s generation gap.
- **CUDA initialization and warm-up:** the original harness initialized CUDA before pipeline timing. Streamlit's home-screen availability probe also initializes CUDA before analysis. The current probes measured explicit initialization at approximately 0.088–0.104 s outside the pipeline. Neither path runs an intentional model-generation warm-up. The first call still includes first-use kernels. This does not demonstrate a historical 23-second initialization delay.
- **Settings and threads:** the original harness explicitly called `torch.set_num_threads(4)`. The existing CUDA adapter receives `cpu_threads=4` but applies it only for CPU execution. Current fresh CUDA processes report 10 threads unless explicitly set. Existing request metadata describes the configured value, not a measurement of effective CUDA-process threads. Four-versus-ten-thread controls below show that this difference alone does not reproduce the historical speed gap.
- **Streamlit reruns:** inference is guarded by Analyze; completion reruns render the saved result. Expander/uploader interactions do not regenerate hypotheses or create new inference calls. AppTest verifies the saved run stays identical after a non-analysis interaction. Rendering after completion and artifact writing are excluded from displayed pipeline latency. Progress callbacks run outside individual generation calls.
- **Device placement:** current probes assert CUDA model parameters and forward inputs, with float32 weights. Existing benchmark validation also checked CUDA logits and generated tensors. Live artifacts select CUDA and record actual CUDA allocated memory. No CPU fallback occurred.
- **Current power state:** AC is online, battery full, and Windows reports the active plan named **Silent**. A CPU snapshot reported 2111 MHz and 72% processor performance. NVIDIA samples fluctuate between idle P8 and active P2/P4 states, with changing clocks/utilization. A detailed snapshot reports a 95 W current power limit and no active thermal/power slowdown flags. The CSV query's power-limit field is N/A; the separate detailed query is the source for 95 W. None of these settings were changed.
- **Historical power/system state:** the old runs did not capture power plan, GPU clock/utilization history, effective CUDA CPU threads, CPU scheduling or competing workloads. Therefore an exact historical hardware/OS cause cannot be proven retrospectively. **Silent mode, thermal throttling, thread count, or Streamlit overhead must not be presented as proven causes.**

The precise time-accounting cause is established: the same three generations
ran more slowly. The underlying historical execution-state cause is unresolved
because the required contemporaneous measurements do not exist. The original
8.85 s benchmark remains a valid single observation, not a promised live-UI
latency. No measured number was scaled, substituted or hidden.

## Current controls and live runs

Sequential, single-process CUDA probes ran the unchanged pipeline with the same
profile and incident. A one-off subclass records placement, actual thread count
and NVIDIA state; it does not alter model generation settings. A fresh model is
loaded once per run. CUDA initialization precedes pipeline timing, and cold
dependency imports inside the first model call remain included in these rows.

| Current control | Effective CPU threads | End-to-end | Generation |
| --- | ---: | ---: | ---: |
| Main thread, defaults | 10 | 49.681 s | 30.612 s |
| Main thread, original benchmark thread setting | 4 | 46.734 s | 28.694 s |
| Background thread, original benchmark thread setting | 4 | 48.641 s | 29.427 s |
| Main thread, four threads set before CUDA initialization | 4 | 48.794 s | 29.884 s |
| Main thread, dependency classes preimported (separate timing scope) | 4 | 34.051 s | 29.263 s |

These reproduce slow generation outside Streamlit. Moving the same task to a
background thread does not create the historic sixfold generation difference.
Setting the thread count before CUDA initialization, matching the original
harness's order, also fails to reproduce the old 4.68-second generation result.
These are small diagnostic controls, not statistical benchmarks or proof that
all machine conditions were held fixed. Dependency import/loading overhead in
fresh processes contributes another roughly 18–19 s in these current runs.
Preimporting dependency classes before timing brings end-to-end latency to
34.051 s while generation remains 29.263 s. This aligns startup scope with a
reused Streamlit process; the model remains freshly loaded, with no generation
warm-up. This is an explicitly labeled measurement control, not a product change.

Current live observations are retained:

| Live Streamlit run | End-to-end | Generation | Conditions |
| --- | ---: | ---: | --- |
| First analysis in a restarted server | 57.612 s | 32.359 s | Fresh dependency imports and fresh model |
| Subsequent analysis | 62.637 s | 56.545 s | Overlapped optional CPU-model tests; excluded from controlled latency comparison, retained as observed |
| Final restart, first analysis | 59.160 s | 33.480 s | Independently run after tests and other inference finished; fresh dependencies and model |
| Final same-session analysis | **37.164 s** | **32.402 s** | Independently run; dependency imports already complete, model freshly loaded |

Normal artifacts for these runs are `runs/20261009T134231.677101Z-01950e31/`
and `runs/20261009T134704.106698Z-b6117b92/`. They retain the standard schema and
manifest. No investigation data were injected into those artifacts. Probe JSON,
power-state snapshots, GPU sampling and the reproducible measurement script are
separate under `output/final_polish/`. Test/inference overlap must not be used to
claim a controlled CPU-versus-GPU or CLI-versus-UI comparison.

The final two artifacts are `runs/20261009T135659.701765Z-da96afcd/` and
`runs/20261009T135826.005327Z-4ba66c3d/`. The latest live run shows **37.16 s**
end-to-end, **32.40 s** generation, **5.80 tokens/s** and **1918.6 MiB GPU Memory
Snapshot**, with routing overhead **0.2416 ms**. All three calls passed grounding
validation with the same 5416 input / 188 output tokens and cold flags
true/false/false. No model cache, warm-up, changed token limit, power-plan change
or other product performance fix was introduced. Both the startup and repeated
observations are reported to avoid presenting a warm-import number as cold start.
The latest artifact manifest was verified, and its hypotheses (including scores
and verification), evidence and relations match the archived benchmark exactly.

For repeating the diagnostic controls from the repository root, with existing
weights and compatible CUDA PyTorch, run these **sequentially**, without tests
or other inference running concurrently:

```powershell
.venv/Scripts/python.exe output/final_polish/latency_probe.py main default
.venv/Scripts/python.exe output/final_polish/latency_probe.py main 4
.venv/Scripts/python.exe output/final_polish/latency_probe.py worker 4
```

The optional `warm-imports` argument preimports dependency classes before timing,
without loading or warming the model. Such a row must be labeled separately;
it changes measurement scope, not product behavior.

## UI changes

- Three home-state native capability cards: **Evidence Grounded**, **Verification First**, **Private Local AI**, with the exact requested descriptions. They disappear once an incident is loaded so results retain their hierarchy.
- Advanced Runtime Settings explicitly defaults to collapsed.
- Verification counts now use four compact native metrics: Supported, Contradicted, Mixed, Missing. Their existing counting logic and criterion meanings are unchanged.
- **GPU Memory Snapshot** replaces the prominent unavailable peak metric. It shows the largest recorded post-call PyTorch allocated-memory snapshot in the current run, with an explicit tooltip: not peak and not whole-device VRAM. CPU/mock/unavailable measurements remain N/A.
- True run-isolated peak allocator tracking would require backend/reset lifecycle changes and can be contaminated by concurrent sessions. This presentation-only pass keeps the existing trustworthy snapshots and does not add peak telemetry or artifact fields.

IncidentLens branding, overview, diagnosis hierarchy, Evidence Score wording,
supporting/contradicting/missing evidence, independent alternative scores,
causality disclaimer, recommendations, runtime section, B70 disclaimer and all
audit expanders remain in place. No new product functionality was added.

Changed product/test files: `app/ui.py`, `tests/test_ui.py`. Documentation:
`README.md`, `docs/ui_polish_status.md`, and this report. Diagnostic outputs under
`output/final_polish/` are measurement records, not an application feature.

## Tests and browser review

The complete suite passed **175 tests in 302.83 seconds**, including both optional
real-model tests. The original 161 tests remain covered; this final pass adds one
home-state test and strengthens existing UI assertions for counts, snapshot
wording and non-analysis reruns. No default test requires CUDA or model downloads.
Twenty core/inference/CLI/pipeline files retain the original benchmark's source
hashes. One sandboxed focused invocation stalled before collecting tests and
was stopped; the focused checks passed with local runtime access (**16 passed,
1 real-model test deselected**), followed by the complete passing suite.

The actual browser shows real CUDA execution, the unchanged ranking, 5 supported,
0 contradicted, 0 mixed and 1 missing criterion. Native home cards and the
default-collapsed sidebar were reviewed at 1366x768. Runtime memory is an actual
post-call snapshot, not peak VRAM. Screenshots:

- [Home capabilities](../output/playwright/final-polish-home-1366.png)
- [Final diagnosis, 1366x768](../output/playwright/final-polish-diagnosis-1366.png)
- [Verification count metrics](../output/playwright/final-polish-verification-1920.png)
- [Final live RTX runtime and GPU memory snapshot](../output/playwright/final-polish-runtime-1920.png)

The fresh release browser session reported **zero console errors and warnings**.

Launch from the repository root:

```powershell
.venv/Scripts/python.exe -m streamlit run app/ui.py --browser.gatherUsageStats false
```

Physical competition display/projector rehearsal and B70 execution remain
unvalidated. Model inference is local, and no remediation is executed.
