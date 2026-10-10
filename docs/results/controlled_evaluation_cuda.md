# Controlled three-case evaluation

**Synthetic incidents, n=3. Not a production-accuracy benchmark.**

Existing local CUDA observations, curated without rerunning inference. Development hardware: **NVIDIA GeForce RTX 5060 Laptop GPU**. Intel Arc Pro B70 remains the competition target and is **unvalidated**.

Model: **Qwen/Qwen2.5-0.5B-Instruct**, revision `7ae557604adf67be50417f59c2c2f167def9a775`. Execution mode: local; device: CUDA. Greedy generation, temperature 0, maximum 512 new tokens per call. IncidentLens made three evidence-selection calls per case; the plain baseline made one call per case. Each backend started cold independently.

**Plain local LLM: 0/3 unambiguous correct top hypotheses.** Two responses listed multiple causes; the remaining response selected the wrong family. This is not a 0% production-accuracy claim.

**Accepted-claim traceability: 100%** (57/57 IncidentLens accepted claims), with zero unsupported accepted claims. Traceability is not semantic accuracy. Baseline source matching found 6/8 valid quotations; two cited incorrect source lines. Free explanations were unverified. The two traceability denominators describe different units.

## Outcomes and measured observations

| Case | Expected / IncidentLens top | Ranked evidence scores | IncidentLens end-to-end (ms) | Generation total (ms) | Baseline end-to-end (ms) | Baseline generation (ms) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A | `duplicate_models` / `duplicate_models` | duplicate_models: 0.866667, runtime_allocation: 0.153333, memory_leak: 0.113333 | 50626.431300072 | 29336.44889993593 | 43391.805200022645 | 38227.65839996282 |
| B | `runtime_allocation` / `runtime_allocation` | runtime_allocation: 0.85, duplicate_models: 0.291667, memory_leak: 0.083333 | 27413.767100078985 | 23321.02979987394 | 34872.77920008637 | 29810.27430004906 |
| C | `memory_leak` / `memory_leak` | memory_leak: 0.85, runtime_allocation: 0.35, duplicate_models: 0.216667 | 31848.87049999088 | 27708.458899869584 | 21848.78709993791 | 17036.0014999751 |

These are recorded development observations, not performance guarantees or a benchmark ranking. End-to-end includes cold loading and other overhead; generation totals sum the recorded calls. Full token counts, per-call latency, allocated-memory snapshots, routing and hardware metadata are preserved in [the public JSON](controlled_evaluation_cuda.json). GPU snapshots are not peak VRAM.

## Case evidence and baseline outputs

### Case A

Logical run: `artifact:20261010T123947.113574Z-2cab9577`. Baseline interpreted top: `None`; correct top: `False`.

Verification criteria (supported/contradicted/missing are evaluations of available observations, not causal proof):

```json
{
  "duplicate_models": {
    "multiple_workers": "supported",
    "worker_model": "supported",
    "startup_steps": "supported",
    "capacity": "supported",
    "startup_alignment": "supported",
    "process_replicas": "missing"
  },
  "runtime_allocation": {
    "oom": "supported",
    "batch_pressure": "missing",
    "runtime_timing": "contradicted"
  },
  "memory_leak": {
    "oom": "supported",
    "steady_growth": "missing",
    "leak_pattern": "contradicted"
  }
}
```

Unmodified baseline raw output:

````text
```json
{
    "root_cause": [
        "Duplicate models",
        "Runtime allocation failure",
        "Memory leak detected"
    ],
    "evidence": [
        {
            "file": "app.log",
            "line": 1,
            "quote": "INFO service starting\n"
        },
        {
            "file": "config.yaml",
            "line": 1,
            "quote": "service:\n"
        },
        {
            "file": "inference.py",
            "line": 1,
            "quote": "def worker_main(worker_id):\n"
        }
    ],
    "explanation": "The error message indicates that there is a conflict between multiple instances of the `worker` function in the `inference.py` script. This could be due to duplicate models being loaded into the GPU, leading to insufficient GPU memory.",
    "uncertainty": ["Duplicate models", "Runtime allocation failure", "Memory leak"],
    "contradicting_evidence": []
}
```
````

Source SHA-256 values:

```json
{
  "app.log": "162801e4866254a6ba7429452f73be379b661887c8e12bb2faa21fa0950c761f",
  "config.yaml": "372923c14b552133ca082220416d9351fb22621f291ea267cd9233cf4de63436",
  "expected_diagnosis.json": "fd28a90555a2b6afd0e430fcca6f458467c5e34fbdcb5f2aac99d62b2e044ecf",
  "expected_evidence.json": "a694161847c212793440a263d12c395347cc5f7a0d9a37e4a7f9888c8c1eff81",
  "inference.py": "117f689b4102520983024cc645d4fc1f2d0f2af76b1c176baa6980f97dc8f42b",
  "system.log": "2ee89efc7e0cb921b4e5a161f1cda6c1a00c8c793df8dde41abff6c130348348"
}
```

### Case B

Logical run: `artifact:20261010T124121.554684Z-e5aefa8b`. Baseline interpreted top: `None`; correct top: `False`.

Verification criteria (supported/contradicted/missing are evaluations of available observations, not causal proof):

```json
{
  "runtime_allocation": {
    "oom": "supported",
    "batch_pressure": "supported",
    "runtime_timing": "supported"
  },
  "duplicate_models": {
    "multiple_workers": "supported",
    "worker_model": "supported",
    "startup_steps": "missing",
    "capacity": "missing",
    "startup_alignment": "contradicted",
    "process_replicas": "missing"
  },
  "memory_leak": {
    "oom": "supported",
    "steady_growth": "contradicted",
    "leak_pattern": "missing"
  }
}
```

Unmodified baseline raw output:

````text
```json
{
    "root_cause": [
        "Duplicate models",
        "Runtime allocation exceeded GPU memory limit"
    ],
    "evidence": [
        {
            "file": "app.log",
            "line": 1,
            "quote": "INFO service starting\n"
        },
        {
            "file": "config.yaml",
            "line": 1,
            "quote": "service:\n"
        },
        {
            "file": "inference.py",
            "line": 1,
            "quote": "def worker_main(worker_id):\n"
        }
    ],
    "explanation": "The most likely root cause is a conflict between multiple instances of the same model running in parallel on the GPU, leading to a memory allocation error due to exceeding the GPU's allocated memory capacity.",
    "uncertainty": ["duplicate_models", "runtime_allocation"],
    "contradicting_evidence": []
}
```
````

Source SHA-256 values:

```json
{
  "app.log": "504e340cdc780c5851a2d7e6773f19202aa3aaa9411a912260e04af36b190339",
  "config.yaml": "372c90312558b3432b797321669ba540c9d3957ee9b371848b4bfe847b30a047",
  "expected_diagnosis.json": "0372a77d17d40d292643ec7571ffd21bab8939948121d308e7b9a96e75e4f959",
  "expected_evidence.json": "8ec2db6a5ac8f347e293bac0f0565d8486d63c945d4b617be12ad11a12d88d4e",
  "inference.py": "12b769c0964a1b5073f0877a3d074be4b766acb32a460d4ef21b08576ebef4a0",
  "system.log": "2657c80cbfa635229293de93e65e1558fce4a53ac4c37f667e6391d5d440eeb9"
}
```

### Case C

Logical run: `artifact:20261010T124224.433808Z-81c52807`. Baseline interpreted top: `duplicate_models`; correct top: `False`.

Verification criteria (supported/contradicted/missing are evaluations of available observations, not causal proof):

```json
{
  "memory_leak": {
    "oom": "supported",
    "steady_growth": "supported",
    "leak_pattern": "supported"
  },
  "runtime_allocation": {
    "oom": "supported",
    "batch_pressure": "contradicted",
    "runtime_timing": "supported"
  },
  "duplicate_models": {
    "multiple_workers": "supported",
    "worker_model": "supported",
    "startup_steps": "contradicted",
    "capacity": "missing",
    "startup_alignment": "contradicted",
    "process_replicas": "missing"
  }
}
```

Unmodified baseline raw output:

````text
```json
{
    "root_cause": ["Duplicate models"],
    "evidence": [
        {
            "file": "app.log",
            "line": 1,
            "quote": "INFO service starting\n"
        },
        {
            "file": "config.yaml",
            "line": 1,
            "quote": "service:\n"
        }
    ],
    "explanation": "Duplicate models are causing the out-of-memory error.",
    "uncertainty": [],
    "contradicting_evidence": []
}
```
````

Source SHA-256 values:

```json
{
  "app.log": "5f39f2df644def872b7481dacd6ca83bebb214f0539112d02fb14d9a420d872c",
  "config.yaml": "00a2ff391931286228bd5867e2adea49c8eea68cb0c1c614f8346acd8433fa63",
  "expected_diagnosis.json": "662b71ebcb547ce89b7507de3c346b2c65fd41e6f2ba4aaec087208884c2edfc",
  "expected_evidence.json": "6d4b39ac5f0db0d64ecbd91b74ad124ab48ec635bd5c4c01d6ee8bad2c8d0c57",
  "inference.py": "12b769c0964a1b5073f0877a3d074be4b766acb32a460d4ef21b08576ebef4a0",
  "system.log": "838f6ebd1f02b8bdd4a200f1e5576391477283f015d3793d7e28851e04fec169"
}
```

## Provenance and limitations

Recorded timestamp: `2026-10-10T12:39:47.113496+00:00`. Saved-output audit timestamp: `2026-10-10T12:44:55.528384+00:00`. The final lexical missing-evidence rubric rescored saved, unmodified outputs without additional generation; audit details and rubric hash are retained in JSON.

Original report SHA-256: `4d29e5f581ecca02ef6b61c92471ea98fdab84f916b1b2f5e49c897c0be54c84`. Private model paths are now repository-relative; local run paths are logical artifact IDs. Original ignored run artifacts and their provenance remain untouched. This curated report is not a replacement replay bundle; locally retained artifacts contain full inputs, prompts and manifests.

The scenarios were designed around three predefined families, not sampled from production or unseen incidents. Model-selected evidence is deterministically validated; authoritative scoring and verification remain outside the LLM. Correlation alone does not prove causality, and missing process/profiling evidence still matters. No B70 results, production accuracy, general root-cause discovery, multi-model routing performance or remediation capability are claimed.

See [evaluation method](../controlled_evaluation.md), [limitations](../limitations.md) and [README](../../README.md).
