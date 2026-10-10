# Controlled three-case evaluation

Public snapshot: [sanitized real CUDA results](results/controlled_evaluation_cuda.md) and [JSON](results/controlled_evaluation_cuda.json). Original replay artifacts remain local and ignored.

## Objective

Check whether the existing IncidentLens evidence → candidate hypotheses →
verification → evidence scoring pipeline distinguishes three intentionally
controlled patterns. Compare it with one plain prompt over the raw files using
the same local Qwen2.5-0.5B-Instruct model. All incidents are **synthetic**;
**sample size = 3**. This is **not a scientific benchmark**.

The baseline independently reads raw files and makes one greedy generation
call. It receives no normalized evidence IDs, graph, verifier rules, verification
plans or application scores. Minimal JSON requests a root cause, source-line
observations, explanation, uncertainty and contradicting observations. Expected
JSON fixture manifests are not inputs to either model.

## Incident definitions and selection

| Case | Intended hypothesis | Controlled observations |
| --- | --- | --- |
| A | `duplicate_models` | Original fixture unchanged: four configured workers, 9.5 GiB startup steps, projected demand 38 GiB versus 32 GiB capacity, OOM during initialization. |
| B | `runtime_allocation` | Two 8 GiB replicas initialize successfully; three steady-state samples remain at 16 GiB with two workers. A request batch needs 20 GiB against 16 GiB pre-batch headroom; inference OOM then service failure. |
| C | `memory_leak` | Two workers initialize successfully; five steady-state samples rise 16 → 20 → 24 → 28 → 31 GiB with constant active_workers=2 and no additional starts in the recorded window. A sampled 1 GiB request allocation initially fits 16 GiB headroom; later a 2 GiB allocation fails against 1 GiB remaining. |

These cases exercise the three existing hypothesis families and existing
`batch_pressure`, `runtime_timing`, `steady_growth` and `leak_pattern` criteria.
They were designed to supply observations the prototype understands, not as
unseen production incidents. Growing memory does not establish an actual leak:
retained allocations, workload changes or caches require further profiling.
Temporal relationships do not prove causality.

Each fixture contains logs, configuration, inspectable Python source and
independent expected evidence/diagnosis manifests. Fixture Python is never
executed. B/C preserve a 32 GiB synthetic capacity; the small diagnostic model
does not allocate the simulated 32 GiB incident workload on the development GPU.

## IncidentLens results

Both mock evaluation and real CUDA evaluation ranked the intended hypothesis
first in all three cases without changing any core diagnosis code or weights.

| Case | Ranked evidence scores | Main verification |
| --- | --- | --- |
| A | duplicate 0.866667; runtime 0.153333; leak 0.113333 | Five supported duplicate criteria; process-level replica observation missing. |
| B | runtime 0.850000; duplicate 0.291667; leak 0.083333 | OOM, batch pressure and runtime timing supported; steady-state stability contradicts leak growth. |
| C | leak 0.850000; runtime 0.350000; duplicate 0.216667 | OOM, steady growth and leak pattern supported; the sampled batch allocation fits its initial headroom. |

Final diagnosis grounding is 100% for each case. Every accepted candidate/final
factual statement was also independently checked against its known evidence ID
and exact normalized statement. Unsupported accepted statements: **zero**.
There are nine missing criterion observations across all competing hypotheses
(A: 3, B: 4, C: 2); primary missing observations are 1, 0, 0. Missing observations
are preserved, not filled by the model. B/C alternative duplicate capacity
verification remains missing: the existing correlator requires a coherent
startup-step sequence and does not isolate it from subsequent steady-state
samples. This limitation was not changed to improve the evaluation.

## Plain local-LLM baseline and comparison

Real execution used **NVIDIA GeForce RTX 5060 Laptop GPU / CUDA**, development
validation hardware, Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, PyTorch 2.14.1+cu130 and CUDA 13.0.
Every IncidentLens inference call selected the CUDA profile; each baseline
backend also used the same model/device. No model downloads, cloud APIs or
CPU/mock fallback were used.

| Metric | Plain Local LLM | IncidentLens (real CUDA) |
| --- | --- | --- |
| Correct top hypothesis | 0/3 **unambiguous** expected tops: A/B ambiguous lists; C wrong family | 3/3 |
| Evidence-grounded factual claims | 6/8 source quotations (75%); explanation entailment unverified | 57/57 accepted candidate/final statement instances; final grounding rate 100% per case |
| Unsupported accepted claims | 2 unsupported source citations; total for free explanations unknown | 0 |
| Explicit missing evidence | 0 explicit observation gaps; 5 uncertainty labels were just family names | 9 criterion gaps across all hypotheses; 1 across primary hypotheses |
| Contradicting evidence exposed | 0 quotations | 14 unique evidence IDs summed per case across competing hypotheses |
| Deterministic diagnosis verification | No | Yes |

The baseline's A answer lists duplicate models, runtime allocation failure and
memory leak; B lists duplicate models and runtime allocation; C answers duplicate
models. Its A/B citations to `inference.py` line 1 contain a worker definition
that is actually on another line. The six valid quotations are generic service
startup/configuration headings, not decisive GPU observations. Quotation
traceability is deliberately not reported as causal support. Its three free
explanations were not automatically accepted as factual claims.

| Case | IncidentLens end-to-end / generation | Baseline end-to-end / generation | Baseline result |
| --- | --- | --- | --- |
| A | 50.626 / 29.336 s | 43.392 / 38.228 s | Ambiguous three-family list; 2/3 traceable quotations |
| B | 27.414 / 23.321 s | 34.873 / 29.810 s | Ambiguous two-family list; 2/3 traceable quotations |
| C | 31.849 / 27.708 s | 21.849 / 17.036 s | `duplicate_models`; 2/2 traceable quotations |

These are recorded execution observations, not expected demo timings or speedup
claims. Default tests ran concurrently with part of this evaluation; machine
load and startup state were not controlled as a performance benchmark.
IncidentLens artifact writing is outside its reported diagnosis timing.
Baseline timing includes raw-source reading, prompt building and its generation
call; it also excludes report writing.

Final JSON and Markdown:
`output/controlled_evaluation/local_final/report.json` and `report.md`.
Original baseline prompts/outputs and measured metrics are preserved there.
The finalized lexical missing-evidence rubric was applied to those saved outputs
without additional generation; `evaluation_audit` records that assessment and
its rubric hash, while `execution_report.json` retains the execution-time report.

Normal unmodified diagnosis artifacts:

- A: `output/controlled_evaluation/local_final/runs/20261010T123947.113574Z-2cab9577/`
- B: `output/controlled_evaluation/local_final/runs/20261010T124121.554684Z-e5aefa8b/`
- C: `output/controlled_evaluation/local_final/runs/20261010T124224.433808Z-81c52807/`

Mock results are separately labeled at `output/controlled_evaluation/mock/`:
3/3 expected tops, 36/36 accepted statement instances, no real baseline run.
Baseline differences therefore come from a real CUDA execution, not mocked
answers. These results do not support a general claim of superiority over plain
LLMs: the model/prompt, ambiguity rule and three hand-designed cases constrain
what can be concluded.

## Interpretation and rubric

Root-cause mapping uses exact family labels and a small published name-alias
table in `evaluation/baseline.py`. A multi-cause list has no unambiguous top;
the evaluator does not pick whichever cause would match the expected answer.
Wrong or ambiguous baseline predictions are comparison outcomes; wrong
IncidentLens rankings fail the controlled-case contract.

Baseline observations are traceable only when their named file and 1-based
line exist and the whitespace-trimmed quotation occurs on that line. A partial
quote can be traceable; an incorrect file, line or quotation is unsupported.
An empty observation list has unavailable grounding, not 100%. This audit does
not prove that a quoted line supports the proposed cause. Free explanations
remain unverified; their unsupported-claim total is **unknown**, not zero.
Missing-evidence counts use explicit lexical signals (such as “missing” or
“not measured”) in uncertainty entries; family names alone do not count.
Contradiction counts audit quotation traceability, not contradiction entailment.
No second LLM is used as an evaluator. Manual semantic review remains necessary.

IncidentLens's facts are exact application-normalized evidence statements;
its hypothesis explanations remain conditional. Its evidence score is a
heuristic score, **not a probability**. Counts in the comparison have different
units and do not establish relative production accuracy or observation quality.

The first strict-format real attempt failed visibly because all three baseline
responses used root-cause lists instead of strings. That report is retained at
`output/controlled_evaluation/local/`. The evaluator was then made conservative
about single/multiple cause lists and source quote fragments, without changing
model prompts, generation settings, fixture values or IncidentLens semantics.
The final real execution and original outputs are recorded separately above;
baseline prompts and outputs were identical between those two executions.

## Reproduction and reports

From the repository root, after the existing README installation:

```powershell
.venv/Scripts/python.exe -m evaluation.run_evaluation
```

This uses the existing mock pipeline, writes JSON/Markdown reports under
`output/controlled_evaluation/mock/` and normal schema-preserving run artifacts
under that report directory's `runs/`. The plain local LLM is explicitly **not
executed** in mock mode. `--no-artifacts` disables diagnosis artifact persistence
while retaining evaluation reports.

After installing the README CUDA dependencies and pinned local Qwen weights:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.venv/Scripts/python.exe -m evaluation.run_evaluation --backend local --device cuda --profiles profiles/demo_rtx5060.json --output-dir output/controlled_evaluation/local_final
```

The same profile/model/device is used for both systems. IncidentLens normally
makes three evidence-selection calls; the baseline makes one call. Both use
temperature=0, greedy generation, float32 and a 512-token cap per call. Separate
backend instances preserve their respective cold loading costs. These unequal
tasks/call counts are not a speed comparison. Runtime metrics are measured;
unavailable metrics remain unavailable.

Runner failures return exit code 2 and record failed cases when possible:
unexpected IncidentLens ranking, missing expected observations, unknown accepted
evidence IDs, invalid claims, real-mode/device mismatch, unparseable baseline
structure, or artifact/report persistence failure. Report-write failures cannot
necessarily save an error report and propagate explicitly. Model/generation
failure does not trigger mock/CPU fallback.

Reports include raw baseline prompts/responses, source SHA-256 values, profile,
device, execution times, all scores, verification statuses and artifact paths.
Generated output/run artifacts are ignored by Git; curate selected reports for
submission separately. The previous Git audit inventory predates these new
evaluation files and should be refreshed before staging.

## Validation

220 tests collected: full default suite **218 passed, 2 optional skipped** in
23.39 s; the two existing optional offline-model tests then **both passed** in
397.91 s (CPU test configuration). All 220 tests were therefore executed and
passed across those two invocations. The 39 evaluation tests pass without a GPU
or model downloads. The existing missing-model UI test's model-path setting was
moved to the optional real-model test, fixing its accidental dependency on an
environment variable; application UI code was untouched.

Mock and real CUDA evaluation CLI paths returned success. The first strict
baseline-format attempt returned exit code 2 and is retained as a failure.
An editable reinstall includes the new evaluation package; its CLI help was
also verified outside the repository working directory. Artifact manifests and
source hashes were rechecked: all existing `app/`, `core/`, `inference/` files
and the original Case A sources remain unchanged. No scoring weights, verifier
rules, routing semantics, artifact schemas or UI behavior changed.

**CONTROLLED EVALUATION PASSED WITHOUT CHANGING CORE DIAGNOSIS LOGIC**

## Limitations

- Synthetic incidents only; only three cases and predefined hypothesis families.
- Current local model is Qwen2.5-0.5B-Instruct; deterministic mock tests are not real-LLM results.
- No claim of production RCA accuracy or general incident coverage.
- Intel Arc Pro B70 remains an unvalidated target; RTX 5060 Laptop is development validation hardware.
- Scores are heuristic evidence scores, not probabilities or causal proof.
- Traceability does not establish entailment, relevance or proof of a leak.
- The existing generic service-failure normalization still uses startup wording; raw B/C logs explicitly place failure after inference. Known normalization limitations were not changed for this evaluation.
- Baseline explanations need manual review; strict source-line matching can miss valid paraphrases or inaccurate citations to otherwise real observations.
- Timing depends on cold startup, call count and current machine load; no B70 performance estimates are provided.
