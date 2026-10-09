# Phase 2 implementation and validation

This is the historical Phase 2 milestone report. Current routing, telemetry,
UI and real CPU model validation are documented in [phase3_status.md](phase3_status.md).

## Implemented

The Phase 1 fixture was corrected before Phase 2 implementation and its original
69-test suite passed. Four synthetic 4B fp16 worker-local allocations of 9.5 GiB
project to **38 GiB > 32 GiB**. Three starts use 28.5 GiB; the next 9.5 GiB request
cannot fit in 3.5 GiB. Config, logs, code, expected evidence, tests and documentation
use these values consistently. They are synthetic, not B70 measurements.

Phase 2 adds:

1. A model-library-independent inference protocol and explicit deterministic mock.
2. An optional offline local Transformers adapter suitable for later real inference.
3. Strict structured candidate generation from repository evidence and relations.
4. Extended hypotheses with grounded claims, support/contradiction/missing IDs,
   verification criteria/results, recommended actions and confidence metadata.
5. Application-owned verification for three competing hypotheses.
6. Coverage, source diversity, verified support and contradiction scoring.
7. A complete pipeline and final primary/alternatives/evidence/verification output.
8. Full CLI diagnosis, with `--core-only` preserving Phase 1 artifacts.
9. Unit, adversarial, counterfactual and full mock integration tests.

Relations remain temporal, correlational, support or derivation edges. They are
not promoted to proven causal claims. Verification identifies the best-supported
explanation and still requests process-level confirmation.

## Validation evidence

Validated on Windows / Python 3.13.13 on 2026-10-08:

- **122 tests passed**, including all original 69 Phase 1 tests.
- Full mock CLI demo produces 25 evidence items and 30 relations plus diagnosis.
- The editable package was refreshed to include `inference`; an import/pipeline
  smoke check from outside the project root passed.
- Primary: duplicate model loading, score **0.866667**.
- Alternatives: runtime/batch allocation **0.153333**, leak **0.113333**.
- Grounding rate **1.0** for final factual quotations; no broad RCA accuracy claim.
- Alternate raw incidents rank a leak or batch allocation first, proving selection
  is evidence-dependent rather than a hard-coded primary label.
- Unknown IDs, fabricated text with valid IDs, ungrounded fields, model scores,
  duplicate candidates, invalid JSON and invalid provenance are rejected.
- Missing evidence and shared OOM-only incidents do not produce a claimed cause.
- Local adapter checks use library stubs, including pipeline integration. No real
  local model generation, B70 execution, GPU validation or benchmark occurred.

Run from the installed repository root:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m app.main --demo
.venv/Scripts/python.exe -m app.main --demo --core-only
```

See README for installing optional local dependencies and selecting an existing
model directory. Installation instructions for model packages are provided;
those optional packages and a real model were not installed during this phase.

## Sample diagnosis

```text
Primary: Duplicate model loading across inference workers
Confidence: 0.866667 (heuristic, not a probability)
Support: multiple workers; worker-local load; startup-aligned memory steps;
         38 GiB projected demand > 32 GiB; startup/OOM/service-failure ordering
Contradictions to primary: none in supplied evidence
Missing: process-level confirmation of one model replica per worker
Verification: inspect process allocations; repeat startup with one shared model
Action: consider a shared model process or fewer replicas after confirming allocations
Alternatives: runtime/batch allocation 0.153333; memory leak 0.113333
Conclusion: best-supported explanation; causality not proven
Backend: deterministic-mock / mock-candidate-catalog-v1
```

## Architectural choices and deviations

- An explicit three-family candidate catalog replaces unrestricted model prose.
  Backend JSON supplies IDs, family and exact evidence quotations; application
  templates own conditional descriptions, plans and actions. This stronger
  grounding boundary is a deliberate restriction on open-ended generation.
- Verification is deterministic. Model-proposed quotations do not become support
  automatically. No local LLM verifier or final-answer prompt is needed.
- The initial Phase 1 scoring function remains for compatibility; the final
  pipeline uses a new documented verification/coverage-aware formula. Model scores
  are forbidden in candidate JSON and never determine the authoritative score.
- The CLI defaults to explicit mock mode for reproducibility. Invalid local output
  retries once then fails clearly; no silent deterministic fallback impersonates
  model execution. The user's 'suitable for later real inference' local adapter
  goal is implemented; the original full-MVP real-inference validation gate remains.
- B70 advertised memory is 32 GB; the fixture models a nominal 32 GiB ceiling and
  does not claim actual device reservations/ECC settings or usable VRAM.
- No additional run persistence, router, latency collection or UI was added.

## Limitations and remaining work

The pipeline covers one incident family with synthetic examples and hand-authored
criteria. Process-level replica confirmation, real model JSON quality, offline
model loading on an actual runtime, and target GPU compatibility remain unmeasured.
Consult `docs/limitations.md` before treating prototype scores as operational RCA.

Phase 3 remains: adaptive router, inference telemetry, Streamlit UI, run artifact
saving, product documentation polish and demo recording rehearsal. Ray, vLLM
clusters, Kubernetes, VLM and automatic remediation remain out of scope.
**Historical stop point: Phase 2 complete.** The statements above describe that
milestone. Phase 3 is now implemented and CPU real inference is validated; see
[phase3_status.md](phase3_status.md) for current scope and validation.
