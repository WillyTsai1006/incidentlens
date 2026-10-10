# Synthetic GPU OOM scenario

The four incident sources are under `examples/gpu_oom_duplicate_models/`.
`expected_evidence.json` and `expected_diagnosis.json` are independent test oracles;
the pipeline never reads either one.

| Source | Observation |
| --- | --- |
| config.yaml | Four workers; worker-local 4B fp16 model; nominal 32 GiB capacity |
| inference.py | `worker_main` directly calls `load_model(MODEL_NAME)` |
| system.log | Baseline 0 GiB; startup readings 9.5, 19.0, 28.5 GiB |
| app.log | Fourth worker starts, allocation fails, service startup then fails |

The fixture targets the [Intel Arc Pro B70's advertised 32 GB VRAM](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html).
A nominal synthetic ceiling of 32 GiB is used to exceed the user's 32 GiB
threshold conservatively; this is not measured usable hardware memory. Even
interpreting 32 GB literally as decimal bytes, three successful 9.5 GiB steps
fit (28.5 GiB < approximately 29.8 GiB), whereas 38 GiB does not. Real headroom
must eventually be measured; [Intel notes ECC can reduce reported VRAM](https://www.intel.com/content/www/us/en/support/articles/000102907/graphics.html).
No settings are changed by this application.

The synthetic 4B fp16 model fits the narrative of approximately 7.45 GiB weights
plus illustrative startup overhead for a 9.5 GiB total per worker. This is an
assumption, not a measured model profile. The source snippet is for inspection
only: `load_model` and `serve` are deliberately undefined, and no incident source
is executed.

Chronology uses synthetic UTC timestamps on 2026-10-08. Three workers are ready
before worker 4 attempts a 9.5 GiB allocation with 3.5 GiB remaining under the
nominal 32 GiB ceiling. Phase 1 derives replicas=4, per-worker increment=9.5 GiB,
projected demand=38 GiB, and exceeds-capacity=true. There are 21 observations,
4 derived facts and 30 relations, all retaining provenance.

`python -m app.main --demo --core-only` exports those artifacts.
`python -m app.main --demo` runs the complete Phase 2 mock diagnosis:

1. Generate three candidates with evidence quotation IDs.
2. Evaluate each application's verification criteria against the repository/graph.
3. Compute scores and rank duplicate models (0.866667), runtime allocation
   (0.153333), and memory leak (0.113333).
4. Return the primary explanation, alternatives, evidence, verification results,
   missing observations, and a suggested shared-model/fewer-replica action.

The primary explanation is the best-supported inference, not proven causality.
Its process-level replica confirmation criterion remains missing. Startup-aligned
memory steps weaken a leak explanation but the fixture has no steady-state
period; the verifier does not invent a 'no leak' observation. Initialization
failure weakens a request-time batch explanation, without excluding every
possible startup allocation.

## Phase 3 demonstration

`python -m app.main --demo --save-run` prints the same ranked mock diagnosis with
an inspectable routing decision, typed telemetry and a saved artifact directory.
The validated real CPU command is:

```powershell
.venv/Scripts/python.exe -m app.main --demo --save-run --backend transformers --profiles profiles/demo_local.json --device cpu
.venv/Scripts/python.exe -m streamlit run app/ui.py --browser.gatherUsageStats false
```

Download the pinned weights first using README instructions. In the UI, load the
synthetic incident, choose mock or real local inference, then analyze. Real mode
uses the same independently verified ranking; its evidence selections are not
automatically accepted as supporting observations. The latest independent RTX
demo-polish run took 37.16 s end-to-end / 32.40 s generation; fresh-server first
analysis took 59.16 / 33.48 s. These are observations, not guarantees. CPU
130–295 s and the original 8.85 s CLI GPU benchmark are historical records.
See [the measurement report](final_demo_polish.md) and
[historical release audit](development/release_readiness_2026-10-10.md) before rehearsing a recording.

Suggested three-minute recording: show inputs and the 38 GiB projection; inspect
the observed evidence and conservative relations; compare all three hypotheses
and the missing process-level check; show routing reason and actual/mock metric
labels; inspect the saved JSON run and conclude with the proposed verification
and shared-model/fewer-replica action. Record actual execution beforehand if the
CPU wait is cut for time, and disclose the cut and validation hardware.

Qwen2.5-0.5B-Instruct was exercised on this development CPU through both CLI and
Streamlit AppTest. RTX 5060 CUDA inference was subsequently validated; see the
[CPU/GPU comparison](rtx5060_validation.md). B70 and XPU remain unvalidated.
Competition recording, public repository publication and target-hardware
rehearsal remain.
