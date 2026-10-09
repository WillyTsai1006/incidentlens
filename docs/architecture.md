# Phase 1, Phase 2 and Phase 3 architecture

```text
local UTF-8 sources
  -> incident_loader (read only)
  -> evidence_extractor (YAML / AST / demo-format logs)
  -> EvidenceRepository (unique stable IDs)
  -> correlate (derived facts + explicit relations)
  -> validated repository and relation JSON (Phase 1)
  -> backend-proposed candidate JSON and grounded quotations
  -> application-owned verification criteria and results
  -> evidence coverage / contradiction / verification scoring
  -> ranked hypotheses and final diagnosis JSON (Phase 2)
```

`app.main.analyze_core()` and `--core-only` preserve the Phase 1 contract.
`analyze_incident()` runs the complete pipeline and is exposed by
`python -m app.main --demo`. `core.diagnosis.diagnose()` also accepts an existing
repository and relation list. Expected fixtures are used only by tests.

## Evidence and provenance

`SourceReference`, `Evidence`, `EvidenceRelation`, `Hypothesis`, and
`DiagnosisResult` use the specification's Pydantic fields. Validation constrains
line ranges and confidence values. Collection defaults are independent per model.

YAML scalar evidence uses a dotted key and retains its original key/value lines.
Python AST extraction looks for direct `load_model()` calls within a top-level
`worker_main` function, ignoring nested function/class definitions. Log extraction
normalizes worker starts, GPU readings in GiB, OOM/allocation events, and service
failure events. All observed evidence has deterministic extraction confidence
1.0; that value describes parsing certainty, not root-cause certainty.

Evidence IDs use source basenames and normalized keys or line/event identifiers.
The repository rejects duplicates and validates all relation endpoints before
export. Loading a directory is nonrecursive and preserves original UTF-8 contents.
Other source files can be selected with `load_files()`.

`correlate()` returns derived evidence and relations without mutating the input
repository. The orchestrator explicitly adds the derived evidence and validates
references. Derived facts carry one original source anchor plus `derived_from`
edges to all inputs; follow these edges recursively for complete multi-file
provenance. A calculation statement is not an observed source quotation.

## Correlation rules

- Configured workers > 1, `load_per_worker: true`, and worker-local code imply
  configured independent replicas.
- A baseline plus at least two distinct startup-aligned, consistently increasing
  memory steps estimates an approximate per-worker GPU memory increment.
- With replicas and explicit positive GiB capacity, calculate projected demand
  including the baseline and compare with capacity. Configured worker count must
  cover the number of observed distinct worker startups.
- Timestamped worker starts correlate with readings from the same source line.
  Earlier memory growth precedes OOM; OOM precedes later service failure.

Timestamps are compared as actual instants, including timezone offsets. Missing
or incomparable timestamps yield no temporal edge. Missing inputs, non-stepwise
growth, or unknown units yield no corresponding memory-derived fact. Temporal
co-occurrence is not sufficient proof for a `causes` edge; Phase 1 emits
`precedes`, `correlates_with`, `supports`, and `derived_from`.

## Confidence engine

The independently callable `score_confidence(ConfidenceInputs(...))` returns its
components and score:

```text
support_ratio = min(supporting_slots / expected_support_slots, 1)
source_diversity = min(supporting_source_types / expected_source_types, 1)
contradiction_penalty = sum(contradiction_weights)
score = clamp(0.55 * support_ratio + 0.25 * source_diversity
              + 0.20 * causal_consistency - contradiction_penalty, 0, 1)
```

The caller supplies distinct expected support slots, unique observed source types
(for this demo: app log, system log, YAML, code), a [0, 1] causal consistency
assessment, and explicit [0, 1] contradiction penalties. Missing support slots
stay in the denominator. More evidence lines must not be counted as additional
support for the same slot, and derived facts must not count as independent source
types. No LLM confidence is accepted by this function. This original Phase 1
formula remains for compatibility. Phase 2 uses the verified formula below
instead of trusting caller/model-declared support counts.

This heuristic is not calibrated, learned, or a probability. A fully supported
hypothetical input can score 1.0 without establishing real-world RCA accuracy.

## Candidate generation and grounding boundary

`InferenceBackend` is a protocol with `generate`, backend/model labels, and an
explicit mock indicator. `core` has no dependency on torch or Transformers.
`MockBackend` derives quotations from the incoming evidence prompt; it never
reads expected diagnosis fixtures or produces a final answer. The optional
`TransformersBackend` imports model packages lazily, loads only a local directory,
uses greedy generation, removes prompt tokens before decoding, and rejects an
oversized context. Device choice is explicit, not an adaptive router.

The generator sends evidence, relations and a strict Pydantic JSON schema to the
backend. It requires three distinct hypothesis kinds/IDs. `CandidateHypothesis`
contains only ID, kind and factual quotations. Each `GroundedClaim` cites one
existing evidence ID and exactly matches its normalized statement. Unknown IDs,
unquoted prose, unsupported fields (including model scores), duplicate IDs,
missing candidates and invalid JSON are rejected. One retry gives two total
attempts; failure returns an error, never a fabricated diagnosis or hidden mock.

The application expands candidates into `Hypothesis` objects with catalog-owned
titles, conditional descriptions, verification plans and recommended actions.
This is limited to duplicate models, runtime/batch allocation and memory leak.
It prevents arbitrary factual assertions from entering free-form descriptions.
It is not an open-ended incident hypothesis generator. Proposed quotations are
not automatically support: the verifier evaluates the repository independently.

## Verification

Each criterion defines supporting and contradicting observations, support weight,
contradiction penalty, and whether it discriminates between hypotheses. A result
is `supported`, `contradicted`, `mixed`, or `missing`, with evidence IDs. Absence
of an observation yields `missing`, not evidence that its opposite occurred.

- Duplicate models: multiple workers, worker-local loading code/flag, distinct
  startup memory steps, projected capacity pressure, startup/OOM/failure ordering,
  and process-level replica confirmation. The last remains missing in the fixture.
- Runtime allocation: shared OOM evidence, measured batch allocation versus
  pre-batch headroom on the same source line, and explicit request/inference
  failure timing. Initialization failure weakens the request-time explanation;
  it does not exclude every possible startup allocation.
- Leak: shared OOM evidence, three ordered `phase=steady_state` GPU readings with
  fixed `active_workers=N`, and growth independent of additional worker starts.
  Stable sampled memory weakens the leak hypothesis only within that window.
  Startup steps provide a competing explanation, not proof of no leak.

Verification consumes `precedes` edges for ordering; `causes` labels are not
accepted as substitutes. It does not promote correlations to proven causality,
perform counterfactual experiments, or execute verification plans. Additional
process-level measurements and repeat-startup experiments remain recommendations.

## Verified confidence formula

`score_verified_hypothesis()` requires exactly one result per unique criterion
and checks result statuses against their evidence references. It computes:

```text
support_ratio = supported criterion weight / total criterion weight
verification_strength = supported discriminating weight / total discriminating weight
evidence_coverage = evaluated (non-missing) weight / total criterion weight
source_diversity = unique original source types / 4, clamped to 1
score = clamp(0.45 * support_ratio + 0.20 * source_diversity
              + 0.20 * verification_strength + 0.15 * evidence_coverage
              - sum(contradicted criterion penalties), 0, 1)
```

Mixed results receive half support weight and the full applicable contradiction
penalty. Missing criteria remain in the denominator. Shared OOM evidence has
weight 0.25 and is not discriminating. Repeating IDs does not improve support.
Source diversity follows `derived_from` to original evidence and counts YAML,
code, app log and system log once each; derived facts never add independent
source types. Missing or cyclic derivation provenance is rejected. All components
and the caveat are returned as `ConfidenceMetadata`.

The demo satisfies five of six primary criteria: support/verification/coverage
are 5/6, source diversity is 1, and score is 0.866667. These components are domain
heuristics, not statistically independent measurements or calibrated probabilities.

## Final output

`DiagnosisResult` exposes the primary hypothesis, ranked alternatives, application
confidence, evidence with sources, verification results/plans, missing observations,
a suggested action, factual quotations and backend labels. Root selection uses
computed score, with deterministic kind/ID tie-breaks. Without discriminating
support, primary root cause is null, confidence is 0, all candidates are alternatives,
and the conclusion requests more evidence.

The grounding metric covers final factual quotations validated against repository
evidence. It does not count hypotheses/recommendations as observations or prove
their truth. Legacy Phase 2 latency/memory fields remain null for compatibility;
Phase 3 adds a separate typed telemetry envelope.

## Phase 3 orchestration

`app.phase3.run_phase3()` composes the existing parsers, repository, graph and
diagnosis engine with `RoutedBackend`. CLI `--phase3`/`--save-run` and Streamlit
call this entry point. Existing APIs and default demo retain their old contracts.

`core.router` contains structured model profiles, routing context and decisions.
It filters task, context limit, execution mode, enabled status and estimated
memory against the matching host/CUDA/XPU pool. Tier preference comes from task,
context and deterministic complexity; latency/quality ranks and stable profile-ID
ties finish selection. Reasons and rejected profiles are serialized. No model
library is imported by the domain router, verifier or confidence engine.

`RoutedBackend` routes each actual generation attempt, caches adapters per profile
and records prompt/output/settings plus typed metrics. Local profiles invoke the
existing Transformers adapter; mock remains deterministic. Binding checks prevent
a factory from masquerading as a different execution mode/backend. There is no
local-to-mock fallback. Disabled target profiles cannot execute.

For Phase 3 local execution the generator asks for evidence IDs separately for
each of the three catalog families. Pydantic rejects extra fields and wrong
types; repository validation rejects unknown IDs. Valid selections expand into
exact evidence statements, then pass through the same independent verification
and application confidence code as Phase 2. Retry is bounded per candidate.
The model supplies neither the primary cause nor authoritative scores.

`inference.telemetry` records hardware snapshots and each generation's identity,
mode, timing, optional actual token counts/TPS, RSS/device allocation snapshots,
routing overhead and validation status. Successful generation and candidate
validity are distinct. Mock timings are measured but explicitly mock; unavailable
token/device values remain null. Timing scopes are serialized with each run.

`core.run_artifacts` creates unique UTC timestamp/UUID directories without
overwriting prior runs. It stores input snapshots, all pipeline artifacts,
profiles, actual prompt/output records, environment versions and source hashes,
plus file checksums. Generation/validation failures retain inspectable partial
artifacts with failed metadata and null diagnosis.

`app.ui_presenter` validates UI-facing schemas and all evidence references before
rendering; `app.ui` displays the complete pipeline and reports errors without
substitution. Progress reflects actual pipeline stages. It never executes plans,
recommendations, incident code or shell commands.
