# Phase 1 implementation and validation

This report records the deterministic core milestone, with its synthetic fixture
corrected to the B70 target before Phase 2. Phase 2 and Phase 3 are now
implemented; see [phase2_status.md](phase2_status.md) for that milestone and
[the current report](phase3_status.md) for current demo validation.

## Implemented

All eight Phase 1 items in specification section 27 are implemented:

1. Installable Python package and test/developer entry points.
2. Pydantic source, evidence, relation, hypothesis, and diagnosis schemas.
3. Internally consistent four-file synthetic GPU OOM fixture with expected evidence.
4. YAML/log/Python AST parsers preserving original source locations and excerpts.
5. In-memory evidence repository, duplicate validation, and JSON serialization.
6. Deterministic correlations and provenance-linked memory/replica calculations.
7. Auditable heuristic confidence formula with bounds and input validation.
8. Unit tests and bundled deterministic core integration/CLI tests.

## Verification

Validated locally on Windows with Python 3.13.13 on 2026-10-08:

- **69 tests passed**, including the bundled core and actual CLI subprocess.
- A fresh virtual environment installed the declared package and dev dependencies
  from the downloaded cache and ran the entire suite successfully.
- Demo output: **25 evidence items** (21 observed, 4 derived), **30 relations**;
  projected demand **38.0 GiB > 32 GiB**.
- No local model inference, GPU execution, or UI validation was performed because
  those features are outside Phase 1.

Commands from the project root on Windows:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m app.main --demo --core-only
```

Tests cover schema validation, YAML types/source locations/errors, code patterns,
startup/memory/OOM/service events, repository integrity, complete derivation
provenance, missing evidence, mismatched units, higher capacity, nonuniform growth,
temporal ordering/timezone offsets, confidence arithmetic/clamping, deterministic
artifacts, and CLI success/failure behavior. Test results apply only to this core.

## Specification choices and deviations

- Section 27's explicit phase boundary takes precedence over section 26's wider
  vertical slice suggestion. Hypotheses, verification, final results, and the
  complete demo integration test remain Phase 2 work.
- The conceptual 8 GB configuration example is replaced by 32 GiB so the three
  successful 9.5 GiB startup steps fit. `gpu_memory_unit: GiB` is added to make
  units explicit while retaining the specification's `gpu_memory_limit_gb` key.
- Relations use temporal/support/correlation labels rather than asserting that
  startup failure is proven caused by OOM from log ordering alone.
- A CLI and minimal documentation are provided now to make Phase 1 independently
  reviewable; UI, router, telemetry and persistent run artifacts remain Phase 3.
- `expected_diagnosis.json` was deferred at Phase 1 and has now been added in
  Phase 2. No benchmark figures or real model execution are simulated.

## Remaining

The Phase 2 pipeline is implemented and mock-tested; actual local model execution
and target hardware validation remain unmeasured. Phase 3 remains: routing,
telemetry, Streamlit UI, run artifacts, documentation polish and demo rehearsal.
