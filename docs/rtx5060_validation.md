# RTX 5060 development validation

**Historical hardware-validation benchmark, not current live-demo performance.**
For the newer independent RTX demo measurements (37.16 s end-to-end / 32.40 s
generation, with a 59.16 / 33.48 s fresh-server first run), see
[final_demo_polish.md](final_demo_polish.md). See [the release audit](../RELEASE_READINESS.md)
for the latest fresh-environment check. The 8.851-second result below remains an
archived single observation and is not a current performance claim.

CUDA inference succeeded on **NVIDIA GeForce RTX 5060 Laptop GPU** on
2026-10-09. This is development validation hardware. **Intel Arc Pro B70 remains
the competition target and is unvalidated; no B70 measurements are reported.**

The initial blocker was the installed `torch 2.14.1+cpu` package, which had no
CUDA runtime. Replacing it with `2.14.1+cu130` enabled actual CUDA execution.
The [official CUDA wheel index](https://download.pytorch.org/whl/cu130/torch/)
contains the Windows Python 3.13 wheel used here. No diagnosis code or hypothesis
logic needed modification.

## Matched comparison

Both runs used Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, float32 weights, four CPU threads,
greedy generation, a 512-token limit per call, balanced routing preference and
the same synthetic GPU OOM inputs. There were three successful calls, with no
retries. Prompts, extracted evidence, relations and final ranked scores were
checked for equality. Model placement was the only inference setting changed.

| Metric | CPU: i5-13450HX | RTX 5060: CUDA |
| --- | ---: | ---: |
| End-to-end diagnosis | 31.897 s | 8.851 s |
| Model generation, sum of three calls | 26.637 s | 4.679 s |
| Total input / output tokens | 5416 / 188 | 5416 / 188 |
| Aggregate generated tokens/sec | 7.06 | 40.18 |
| Largest recorded process RSS snapshot | 2615.86 MiB | 2080.22 MiB |
| GPU allocated memory, post-call snapshot | Unavailable / not used | 1917.32 MiB |
| GPU peak allocated memory | Not used | 2087.25 MiB |
| GPU peak reserved memory | Not used | 2252.00 MiB |
| Total routing overhead | 0.0708 ms | 0.0732 ms |

End-to-end speedup: **3.60x**. Generation speedup: **5.69x**. Aggregate throughput
is total generated tokens divided by the sum of measured generation times;
it is not a decode-only throughput benchmark.

| Call | Input / output tokens | CPU generation | CUDA generation | CPU / CUDA tokens/sec |
| --- | --- | ---: | ---: | ---: |
| 1, cold model | 1805 / 56 | 8.434 s | 1.661 s | 6.64 / 33.71 |
| 2 | 1806 / 76 | 9.888 s | 1.723 s | 7.69 / 44.11 |
| 3 | 1805 / 56 | 8.315 s | 1.295 s | 6.74 / 43.24 |

The runtime was Windows 11, Python 3.13.13, PyTorch **2.14.1+cu130**, CUDA
runtime **13.0**, Transformers **4.57.6**, NVIDIA driver **591.91**, compute
capability **12.0**, and approximately **8151 MiB** physical VRAM. The driver's
`nvidia-smi` CUDA 13.1 capability label is distinct from PyTorch's CUDA 13.0
runtime. Both CPU and GPU benchmarks used this same CUDA-enabled PyTorch build.

## GPU execution evidence and artifacts

The one-off validation harness used the existing `run_phase3` entry point and
Transformers adapter. Assertions checked every model parameter, tensor keyword
input on all 188 forward calls, output logits and generated token tensors. GPU
checks recorded `cuda:0`; CPU checks recorded `cpu`. The GPU parameter dtype was
`torch.float32`. CUDA operations were synchronized by the existing adapter when
measuring generation. No CPU fallback occurred.

Normal artifact directories:

- CPU: `runs/20261009T081407.335272Z-3ddb94a4/`
- CUDA: `runs/20261009T081505.525124Z-77b1d06e/`

Both contain the standard inputs, evidence, hypotheses, verification, diagnosis,
routing, telemetry, metadata, profiles and raw inference exchanges. Additional
`benchmark_validation.json` records placement, versions, GPU identity and peak
allocator measurements; `benchmark_harness.py` preserves the measurement code.
Both additions are included in the normal SHA-256 manifest. Checksums were
verified for both runs. Standard telemetry remains unchanged and CPU GPU
metrics remain null, not zero-valued fictional measurements.

Final ranking in both runs:

| Hypothesis | Deterministic evidence score |
| --- | ---: |
| Duplicate model loading across inference workers | 0.866667 |
| Excessive runtime/batch-related allocation | 0.153333 |
| Memory leak | 0.113333 |

Evidence grounding remained 1.0. Verification and scoring remained independent
of the model. Scores are heuristics, not probabilities; correlation is not proof
of causality, and process-level replica confirmation is still missing.

## Reproduction and changes

Install the exact CUDA build in the existing virtual environment, keep the
pinned local weights, and run the unchanged CLI:

```powershell
uv --cache-dir .uv-cache pip install --python .venv/Scripts/python.exe 'torch==2.14.1+cu130' --index-url https://download.pytorch.org/whl/cu130
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.venv/Scripts/python.exe -m app.main --demo --save-run --backend transformers --profiles profiles/demo_rtx5060.json --device cuda
```

That uninstrumented CLI command also succeeded and its manifest was verified:
`runs/20261009T081834.656813Z-f7ed30b8/`. It selected CUDA, reported positive GPU
allocation and returned the same primary diagnosis and score.

The added profile uses a conservative **4096 MiB estimate**, not a benchmark
number. The existing default 8192 MiB estimate exceeds this laptop's GPU capacity
and is correctly rejected by the router. The explicit development profile avoids
changing that default or bypassing memory checks. Its estimate exceeded the
observed 2252 MiB peak reserved allocation in this incident.

For the exact instrumented comparison, copy `benchmark_harness.py` from the GPU
artifact into `.uv-cache/benchmark_cuda.py` under the repository root and run it
with `cpu`, then `cuda`. Do not run the benchmarks concurrently. Only the
development profile and documentation were added to tracked project files;
application, evidence, hypothesis, routing, scoring and artifact code are unchanged.

## Scope and limitations

This is one cold-model run per device, sequentially CPU then GPU, without an OS
file-cache reset or repeated-sample statistics. PyTorch imports and CUDA device
initialization occur before the measured pipeline in both the validation setup
and harness. Model loading is inside the first call; timing excludes artifact
writing and UI rendering. Placement hooks add small measurement overhead.
Earlier CPU records used different startup scopes/environment conditions and
are not the baseline for this speedup calculation.

RSS values are snapshots, not host memory peaks. GPU allocation/reservation is
measured by PyTorch, not whole-device VRAM usage; desktop/WDDM allocations are
outside those values. The validated 0.5B float32 workload does not establish
larger-model capacity, sustained performance, RCA accuracy or B70 compatibility.
Memory estimates cannot guarantee avoiding OOM on a busy machine.

The complete suite passed after the environment/profile changes: **161 passed
in 85.24 seconds**, including both optional real-model API and Streamlit tests.
All existing Phase 1/2 behavior and tests remain intact. No new application
feature or hypothesis change was introduced.
