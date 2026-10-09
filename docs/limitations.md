# Known limitations

- Phase 3 is a targeted local demo, not a production RCA system. It has routing,
  typed telemetry, Streamlit and run snapshots; no production telemetry dashboard,
  distributed execution, autonomous shell execution or automatic remediation.
- Hypothesis families are limited to duplicate models, runtime/batch allocation,
  and memory leak. Inference proposes all three and supplies exact factual
  quotations; application-owned descriptions and criteria enforce a conservative
  contract. Arbitrary free-form hypothesis reasoning is intentionally unsupported.
- Default CLI inference is explicitly mock. Phase 3 real CPU inference and UI
  execution were tested with Qwen2.5-0.5B-Instruct. RTX 5060 CUDA inference was
  subsequently validated; see [the comparison](rtx5060_validation.md). XPU and
  Intel B70 are unvalidated. Weights are downloaded separately; inference never downloads them
  or sends incident data to a cloud API. The legacy Phase 2 bundle JSON was
  unreliable with the small model; Phase 3 uses per-candidate evidence-ID selection.
- Small-model selections can be weak or irrelevant, even when IDs are valid.
  Independent verification evaluates all evidence; it does not treat model
  selections as demonstrated support. This is not an accuracy benchmark.
- Routing uses estimated memory requirements and a budget snapshot, not memory
  reservations or a guarantee against OOM. Host RAM is never substituted for a
  GPU memory pool. One real CPU model is enabled; tier preference can fall back
  to that model with an explicit reason. The separate RTX development profile
  enables one CUDA model; the B70 profile is disabled.
- Token counts/TPS exist only for actual model generations. Mock token counts
  and unavailable device metrics are null. Process RSS is a snapshot, not peak
  VRAM. Cold call latency includes model loading; generation TPS excludes it.
  End-to-end timing excludes artifact serialization/writing and UI rendering.
- Run snapshots support inspection and replay with matching code, dependencies,
  model revision and hardware. They do not bundle weights or promise identical
  generation/timings across devices. Raw prompts and outputs are included and
  must be treated as sensitive incident data.
- Every generated factual claim must exactly quote one normalized evidence
  statement and cite its existing ID. This stricter contract rejects paraphrases
  as well as hallucinations. Grounding rate measures these quoted factual claims,
  not semantic validity of hypotheses or broad diagnosis accuracy.
- Verification strengthens/weakens hypotheses against explicit observations; it
  does not run experiments, collect process-level allocations, or prove causality.
  Missing observations stay missing. Shared OOM evidence alone cannot select a
  root cause. An empty or nondiscriminating incident returns a null primary cause.
- Scores and contradiction penalties are hand-authored, not calibrated or learned.
  Criteria are domain heuristics and some are logically related. Scores are not
  probabilities or statistical estimates of real-world RCA accuracy.
- Memory capacity is configured. The fixture uses nominal synthetic 32 GiB for a
  target advertised as 32 GB; it does not detect usable VRAM, reservations, or ECC.
  Startup memory projections assume observed uniform increments continue.
- YAML supports nested mappings and scalars. Lists, recursive mappings, arbitrary
  tags, duplicate/non-string keys, and complex aliases are unsupported. SafeLoader
  validates inputs; Python source is parsed without execution.
- Code detection targets a top-level `worker_main` with direct `load_model` calls.
  It is not a call graph analyzer or proof a conditional call actually runs.
- Log GPU readings use `gpu_memory_used_gib=<number>`, starts use `worker-N
  starting/started`. Runtime verification recognizes `batch_allocation_gib` and
  `batch_headroom_gib` on the same line and OOM with `phase=inference` or 'during
  inference'. Leak observations require at least three ordered `phase=steady_state`
  readings with constant `active_workers=N` and no additional starts in the window.
  Other log lines are ignored. Missing timestamps cannot establish ordering.
- Sources require unique basenames. Loading is nonrecursive and UTF-8 only.
  Unsupported/empty input, bad YAML/Python, invalid JSON/IDs, model/device
  unavailability and context overflow produce clear errors rather than mock
  substitution or evidence truncation.
- Exported JSON contains source excerpts; handle it as sensitive incident data.
  No recommendations, shell commands, verification experiments or remediation
  actions are executed. Production security and remediation are out of scope.
